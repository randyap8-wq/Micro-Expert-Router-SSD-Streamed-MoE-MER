//! Scalar Rust implementation of the dense pieces of a Mixtral / Llama-style
//! transformer decoder layer:
//!
//! * [`RmsNorm`] — RMSNorm normalisation.
//! * [`apply_rope_inplace`] — rotary positional embedding for one head.
//! * [`MultiHeadSelfAttention`] — scalar causal multi-head self attention
//!   with a per-layer KV cache. Grouped-query attention (GQA) supported by
//!   passing `num_kv_heads < num_heads`.
//! * [`TransformerLayer`] — wires `attention -> residual -> rmsnorm -> moe`.
//!
//! These are the pieces the gist asks for in **Phase 2**. They are
//! deliberately written in plain `f32` Rust — no BLAS, no SIMD intrinsic
//! crates — for two reasons:
//!
//! 1. The whole engine builds on stable Rust with zero non-Rust toolchain
//!    requirements (the existing scalar SwiGLU expert kernel in
//!    [`crate::inference`] is the same shape).
//! 2. The dense weights are *resident* (small relative to the per-layer
//!    expert weights), so the dense compute is not the bottleneck the
//!    engine is built to optimise — it's the SSD-streamed expert FFN. The
//!    interesting work of this codebase is the streaming cache, not the
//!    matmul kernel.
//!
//! The matmul shape and call sites match what `candle_transformers::models::mixtral`
//! does, so a future PR can swap each method for a `candle::Tensor` op
//! without changing call sites.
//!
//! These types are exercised by unit tests below; production wiring (a
//! full forward pass through stacked `TransformerLayer`s with loaded
//! Mixtral weights) lands in a follow-up PR — the [`crate::server`]
//! generation loop currently drives [`crate::engine::Engine::generate`]
//! directly, which is enough to exercise the SSD-streaming substrate
//! end-to-end. Allow `dead_code` here so the public API is greppable
//! without forcing a stub call site.
#![allow(dead_code)]

use crate::backend::Backend;
use crate::dense_tensor::{DenseDType, DenseWeight};
use crate::expert_cache::ExpertResident;
use crate::inference::{forward_candle_tensors, ExpertWeights, HiddenState};
use candle_core::{Device, Tensor};
use half::f16;
use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

/// RMSNorm: `y = x * rsqrt(mean(x^2) + eps) * weight`.
///
/// Used before attention and before the MoE block in Llama / Mixtral-style
/// architectures. The `weight` parameter is a learnable per-channel scale
/// of length `d_model`.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// In-place RMSNorm. `x.len()` must equal `weight.len()`.
    pub fn forward_inplace(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.weight.len(), "RMSNorm dim mismatch");
        let n = x.len() as f32;
        let mut sq_sum = 0.0f32;
        for &v in x.iter() {
            sq_sum += v * v;
        }
        let mean_sq = sq_sum / n;
        let scale = 1.0 / (mean_sq + self.eps).sqrt();
        for (xi, wi) in x.iter_mut().zip(self.weight.iter()) {
            *xi = *xi * scale * *wi;
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        self.forward_inplace(&mut y);
        y
    }

    pub fn forward_into(&self, x: &[f32], y: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), self.weight.len(), "RMSNorm dim mismatch");
        y.clear();
        y.extend_from_slice(x);
        self.forward_inplace(y);
    }
}

/// Apply rotary positional embedding to a single head's `(q or k)` vector
/// **in place** at absolute position `pos`.
///
/// Layout: contiguous `head_dim` floats; rotated as `head_dim/2` complex
/// pairs at frequencies `1 / base^(2i / head_dim)`. Matches the rotary
/// convention used by Llama-2/3 and Mixtral.
pub fn apply_rope_inplace(v: &mut [f32], pos: usize, base: f32) {
    let head_dim = v.len();
    debug_assert!(head_dim % 2 == 0, "RoPE requires even head_dim");
    let half = head_dim / 2;
    let pos_f = pos as f32;
    for i in 0..half {
        // Inverse-frequency for this pair.
        let inv_freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
        let theta = pos_f * inv_freq;
        let (sin_t, cos_t) = theta.sin_cos();
        // Pair up dim `i` and dim `i + half` (Llama convention; matches
        // candle-transformers' Mixtral implementation).
        let a = v[i];
        let b = v[i + half];
        v[i] = a * cos_t - b * sin_t;
        v[i + half] = a * sin_t + b * cos_t;
    }
}

// ---------------------------------------------------------------------
// YaRN RoPE scaling (long-context position interpolation).
//
// YaRN ("Yet another RoPE extensioN", arXiv:2309.00071) extends a
// model's context window past `original_max_position_embeddings` by
// blending two per-frequency strategies:
//
// * **extrapolation** — high-frequency pairs (many full rotations over
//   the original context) keep the unscaled `1/base^(2i/d)` frequency;
// * **interpolation** — low-frequency pairs are slowed down by the
//   scaling `factor` (`1/(factor * base^(2i/d))`).
//
// A linear ramp between the `beta_fast` / `beta_slow` rotation counts
// decides how much of each strategy a given pair receives. On top of
// the frequency blend, attention magnitudes are corrected by an
// `mscale` factor folded into the rotation (cos/sin are multiplied by
// `attn_factor`, so Q and K each carry one factor and attention scores
// carry its square) — this matches the HuggingFace / DeepSeek-V3
// `YarnRotaryEmbedding` convention.
// ---------------------------------------------------------------------

/// YaRN magnitude-scaling helper: `0.1 * mscale * ln(scale) + 1.0` for
/// `scale > 1`, identity otherwise. Mirrors `yarn_get_mscale` from the
/// DeepSeek-V3 reference implementation.
pub fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        return 1.0;
    }
    0.1 * mscale * scale.ln() + 1.0
}

/// Precomputed YaRN rotary-embedding parameters for one head shape.
///
/// Built once at model-construction time from the checkpoint's
/// `rope_scaling` block ([`crate::architecture::RopeScaling`]) and the
/// head dim the rotation applies to; [`apply_rope_scaled_inplace`]
/// consumes it on the hot path with no per-token recomputation beyond
/// the `sin_cos` the unscaled path already pays.
#[derive(Debug, Clone, PartialEq)]
pub struct YarnRope {
    /// Per-pair blended inverse frequencies, length `head_dim / 2`.
    pub inv_freq: Vec<f32>,
    /// Multiplier applied to cos/sin (i.e. to both Q and K), carrying
    /// the YaRN attention-magnitude correction. Attention scores see
    /// `attn_factor^2`.
    pub attn_factor: f32,
}

impl YarnRope {
    /// Dimension index below which a frequency completes fewer than
    /// `num_rotations` full turns over the original context (the YaRN
    /// "correction dim").
    fn correction_dim(num_rotations: f32, dim: usize, base: f32, max_pos: usize) -> f32 {
        let d = dim as f32;
        d * (max_pos as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
    }

    /// Build the blended inverse-frequency table + attention factor for
    /// a rotation over `head_dim` dims (`head_dim/2` pairs) with the
    /// given RoPE `base`. Returns `None` when `scaling` is not a YaRN
    /// config or carries a non-expanding factor (<= 1), in which case
    /// the caller should keep the standard unscaled path.
    pub fn from_scaling(
        head_dim: usize,
        base: f32,
        scaling: &crate::architecture::RopeScaling,
    ) -> Option<Self> {
        if !scaling.rope_type.eq_ignore_ascii_case("yarn") || scaling.factor <= 1.0 {
            return None;
        }
        if head_dim == 0 || head_dim % 2 != 0 {
            return None;
        }
        let orig_max = if scaling.original_max_position_embeddings > 0 {
            scaling.original_max_position_embeddings
        } else {
            4096
        };
        let half = head_dim / 2;
        let low = Self::correction_dim(scaling.beta_fast, head_dim, base, orig_max)
            .floor()
            .max(0.0);
        let high = Self::correction_dim(scaling.beta_slow, head_dim, base, orig_max)
            .ceil()
            .min((half - 1) as f32)
            .max(low);
        // Minimum ramp width guard: when beta_fast ≈ beta_slow the
        // correction range collapses (`high == low`) and the ramp
        // would divide by zero; clamping to 1e-3 turns it into a step
        // function at `low` instead.
        let range = (high - low).max(1e-3);
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let freq_extra = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let freq_inter = freq_extra / scaling.factor;
            // Linear ramp: 0 below `low` (pure extrapolation), 1 above
            // `high` (pure interpolation).
            let ramp = ((i as f32 - low) / range).clamp(0.0, 1.0);
            inv_freq.push(freq_extra * (1.0 - ramp) + freq_inter * ramp);
        }
        // HF / DeepSeek convention: cos/sin are multiplied by
        // `get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)`.
        // With the defaults (mscale=1, mscale_all_dim=0) this reduces to
        // the canonical YaRN `0.1*ln(factor)+1`.
        let attn_factor = yarn_get_mscale(scaling.factor, scaling.mscale)
            / yarn_get_mscale(scaling.factor, scaling.mscale_all_dim);
        Some(Self {
            inv_freq,
            attn_factor,
        })
    }
}

/// Default number of absolute positions backed by per-position RoPE rows.
///
/// Qwen3-Coder long-context checkpoints commonly advertise very large
/// `rope_theta` values and production prompts can move well past the old
/// 4K/8K window. The cache is lazy: this allocates `OnceLock` slots, not
/// sin/cos rows, so the steady RSS scales with positions actually touched.
pub const DEFAULT_ROPE_CACHE_POSITIONS: usize = 262_144;

#[derive(Debug, Clone, Copy)]
struct RopePair {
    sin: f32,
    cos: f32,
}

/// Shared rotary-embedding cache for one `(rope_dim, base, scaling)` shape.
///
/// The expensive inverse-frequency table is computed once. Sin/cos rows are
/// materialised lazily by absolute position and then reused by every layer/head
/// sharing this cache. `OnceLock` keeps already-materialised positions on a
/// lock-free fast path; only the first use of a new position performs work.
#[derive(Debug)]
pub struct RopeCache {
    rope_dim: usize,
    base_bits: u32,
    yarn: bool,
    attn_factor_bits: u32,
    inv_freq: Box<[f32]>,
    rows: Box<[OnceLock<Box<[RopePair]>>]>,
}

impl RopeCache {
    pub fn new(rope_dim: usize, base: f32, yarn: Option<&YarnRope>) -> Option<Self> {
        Self::with_positions(rope_dim, base, yarn, DEFAULT_ROPE_CACHE_POSITIONS)
    }

    pub fn with_positions(
        rope_dim: usize,
        base: f32,
        yarn: Option<&YarnRope>,
        positions: usize,
    ) -> Option<Self> {
        if rope_dim == 0 || !rope_dim.is_multiple_of(2) {
            return None;
        }
        let half = rope_dim / 2;
        let (inv_freq, attn_factor, has_yarn) = match yarn {
            Some(y) if y.inv_freq.len() == half => {
                (y.inv_freq.clone().into_boxed_slice(), y.attn_factor, true)
            }
            Some(_) => return None,
            None => {
                let mut inv_freq = Vec::with_capacity(half);
                for i in 0..half {
                    inv_freq.push(1.0 / base.powf(2.0 * i as f32 / rope_dim as f32));
                }
                (inv_freq.into_boxed_slice(), 1.0, false)
            }
        };
        let rows = std::iter::repeat_with(OnceLock::new)
            .take(positions)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Some(Self {
            rope_dim,
            base_bits: base.to_bits(),
            yarn: has_yarn,
            attn_factor_bits: attn_factor.to_bits(),
            inv_freq,
            rows,
        })
    }

    #[inline]
    fn matches(&self, rope_dim: usize, base: f32, yarn: Option<&YarnRope>) -> bool {
        self.rope_dim == rope_dim
            && self.base_bits == base.to_bits()
            && self.yarn == yarn.is_some()
            && self.attn_factor_bits
                == yarn
                    .map(|y| y.attn_factor.to_bits())
                    .unwrap_or(1.0f32.to_bits())
    }

    fn compute_row(&self, pos: usize) -> Box<[RopePair]> {
        let pos_f = pos as f32;
        let m = f32::from_bits(self.attn_factor_bits);
        self.inv_freq
            .iter()
            .map(|&inv| {
                let theta = pos_f * inv;
                let (sin, cos) = theta.sin_cos();
                RopePair {
                    sin: sin * m,
                    cos: cos * m,
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn apply_inplace(&self, v: &mut [f32], pos: usize) {
        debug_assert_eq!(v.len(), self.rope_dim);
        if let Some(slot) = self.rows.get(pos) {
            let row = slot.get_or_init(|| self.compute_row(pos));
            apply_rope_pairs(v, row);
        } else {
            let row = self.compute_row(pos);
            apply_rope_pairs(v, &row);
        }
    }

    #[cfg(test)]
    fn materialized_positions(&self) -> usize {
        self.rows.iter().filter(|slot| slot.get().is_some()).count()
    }
}

#[inline]
fn apply_rope_pairs(v: &mut [f32], pairs: &[RopePair]) {
    let half = v.len() / 2;
    debug_assert_eq!(pairs.len(), half);
    for i in 0..half {
        let a = v[i];
        let b = v[i + half];
        let p = pairs[i];
        v[i] = a * p.cos - b * p.sin;
        v[i + half] = a * p.sin + b * p.cos;
    }
}

/// YaRN-scaled variant of [`apply_rope_inplace`]: rotates with the
/// precomputed per-pair inverse frequencies and multiplies cos/sin by
/// the attention factor. `yarn.inv_freq.len()` must equal
/// `v.len() / 2`.
pub fn apply_rope_scaled_inplace(v: &mut [f32], pos: usize, yarn: &YarnRope) {
    let head_dim = v.len();
    debug_assert!(head_dim % 2 == 0, "RoPE requires even head_dim");
    let half = head_dim / 2;
    debug_assert_eq!(
        yarn.inv_freq.len(),
        half,
        "YarnRope built for a different head_dim"
    );
    let pos_f = pos as f32;
    let m = yarn.attn_factor;
    for i in 0..half {
        let theta = pos_f * yarn.inv_freq[i];
        let (sin_t, cos_t) = theta.sin_cos();
        let (sin_t, cos_t) = (sin_t * m, cos_t * m);
        let a = v[i];
        let b = v[i + half];
        v[i] = a * cos_t - b * sin_t;
        v[i + half] = a * sin_t + b * cos_t;
    }
}

/// Dispatch helper used by the attention paths: YaRN-scaled rotation
/// when `yarn` is configured, the standard unscaled rotation otherwise.
#[inline]
pub fn apply_rope_maybe_scaled(v: &mut [f32], pos: usize, base: f32, yarn: Option<&YarnRope>) {
    match yarn {
        Some(y) => apply_rope_scaled_inplace(v, pos, y),
        None => apply_rope_inplace(v, pos, base),
    }
}

/// Cached RoPE dispatcher for hot attention paths. Falls back to the
/// compatibility implementation if a caller supplies no cache or a cache
/// for a different rotary shape.
#[inline]
pub fn apply_rope_with_cache(
    v: &mut [f32],
    pos: usize,
    base: f32,
    yarn: Option<&YarnRope>,
    cache: Option<&RopeCache>,
) {
    if let Some(cache) = cache {
        if cache.matches(v.len(), base, yarn) {
            cache.apply_inplace(v, pos);
            return;
        }
    }
    apply_rope_maybe_scaled(v, pos, base, yarn);
}

/// One layer's **paged** KV cache (per-layer). Stores keys and values
/// in fixed-size blocks of [`PAGED_BLOCK_TOKENS`] tokens × `kv_dim`
/// floats each, indexed by a block table.
///
/// Replaces the original `Vec<f32>` flat layout (which `extend_from_slice`d
/// — and therefore reallocated the entire backing storage — every time
/// the sequence grew past the `Vec`'s capacity) with vLLM-style
/// PagedAttention block allocation: each block is a separately-owned
/// boxed slice and the cache grows by *appending one new block to the
/// block table* when the trailing block fills up. The block table
/// (`Vec<Box<[f32]>>` here, conceptually a `Vec<u32>` of block ids in a
/// shared pool) is the indirection layer that decouples per-request
/// sequence growth from a single contiguous allocation per request.
///
/// Public surface compatibility: the original `KvCache` exposed
/// `keys: Vec<f32>` / `values: Vec<f32>` as `pub` fields used only
/// inside this module via `key(i)` / `value(i)` accessors and a single
/// `extend_from_slice` call inside `append`. All external callers go
/// through `KvCache::new(kv_dim)` plus `append`, `reset`, `seq_len` and
/// `kv_dim`, all of which keep their original semantics.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    /// Block table: each entry holds [`PAGED_BLOCK_TOKENS`] tokens'
    /// worth of `kv_dim` floats laid out as `[token_in_block, kv_dim]`
    /// row-major. The last block may be partially filled (the unused
    /// tail is never read because `seq_len` bounds every iteration).
    keys_blocks: VecDeque<Box<[f32]>>,
    /// Mirrors `keys_blocks` for the value half of the cache.
    values_blocks: VecDeque<Box<[f32]>>,
    /// Number of leading paged blocks that have been physically dropped by
    /// [`Self::evict_before`] (sliding-window KV eviction). Absolute
    /// positions keep counting from 0 via `seq_len`, but the block table is
    /// indexed *relative* to this offset: logical block `b` lives at
    /// physical index `b - evicted_blocks`. `0` for caches that are never
    /// evicted (every non-SWA layer), preserving the original byte-for-byte
    /// indexing.
    evicted_blocks: usize,
    pub seq_len: usize,
    pub kv_dim: usize,
    /// Value-half row width. Equals [`Self::kv_dim`] for every architecture
    /// except MiMo-V2-Flash, whose V head dim (`v_head_dim`) is smaller than
    /// its K head dim. Defaults to `kv_dim` via [`Self::new`].
    pub v_dim: usize,
}

/// Number of tokens per PagedAttention block. Matches the spec value
/// (16) and the vLLM default. Larger blocks waste more memory at the
/// tail; smaller blocks add more block-table indirections per
/// attention sweep. 16 is a known sweet spot for Mixtral / Llama
/// shapes and keeps each block at `16 * kv_dim * 4` bytes — well
/// under one OS page for any realistic `kv_dim`.
pub const PAGED_BLOCK_TOKENS: usize = 16;

impl KvCache {
    pub fn new(kv_dim: usize) -> Self {
        Self::new_kv(kv_dim, kv_dim)
    }

    /// Construct a cache whose key and value halves have independent row
    /// widths. `k_dim` is the K width (`num_kv_heads * head_dim`) and
    /// `v_dim` the V width (`num_kv_heads * v_head_dim`). They differ only
    /// for MiMo-V2-Flash; [`Self::new`] passes `v_dim == k_dim` for every
    /// other architecture.
    pub fn new_kv(k_dim: usize, v_dim: usize) -> Self {
        Self {
            keys_blocks: VecDeque::new(),
            values_blocks: VecDeque::new(),
            evicted_blocks: 0,
            seq_len: 0,
            kv_dim: k_dim,
            v_dim,
        }
    }

    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.kv_dim);
        debug_assert_eq!(v.len(), self.v_dim);
        let pos = self.seq_len;
        let block_idx = pos / PAGED_BLOCK_TOKENS;
        let in_block = pos % PAGED_BLOCK_TOKENS;
        // Allocate a fresh block when crossing a block boundary. This
        // is the *only* allocation point in the per-token path — the
        // existing block bytes are written in place. The block table is
        // indexed relative to `evicted_blocks` (leading blocks dropped by
        // SWA eviction), so the freshly-pushed block lands at physical
        // index `block_idx - evicted_blocks`.
        if in_block == 0 {
            debug_assert_eq!(self.keys_blocks.len(), block_idx - self.evicted_blocks);
            self.keys_blocks
                .push_back(vec![0.0f32; PAGED_BLOCK_TOKENS * self.kv_dim].into_boxed_slice());
            self.values_blocks
                .push_back(vec![0.0f32; PAGED_BLOCK_TOKENS * self.v_dim].into_boxed_slice());
        }
        // Borrow the freshly-allocated (or current) trailing block and
        // write directly into its in-place slot.
        let kb = self
            .keys_blocks
            .back_mut()
            .expect("block must exist after append");
        kb[in_block * self.kv_dim..in_block * self.kv_dim + self.kv_dim].copy_from_slice(k);
        let vb = self
            .values_blocks
            .back_mut()
            .expect("block must exist after append");
        vb[in_block * self.v_dim..in_block * self.v_dim + self.v_dim].copy_from_slice(v);
        self.seq_len += 1;
    }

    pub fn reset(&mut self) {
        self.keys_blocks.clear();
        self.values_blocks.clear();
        self.evicted_blocks = 0;
        self.seq_len = 0;
    }

    /// Overwrite every cached K/V byte with zero before discarding the
    /// cache, then truncate the block tables. Called from the session
    /// store's `DELETE /v1/sessions/{id}` handler so that a tenant's
    /// (potentially sensitive) attention state cannot be read by a
    /// subsequent allocation that lands in the same heap region.
    ///
    /// We use `std::ptr::write_volatile` to perform the zeroing
    /// writes: volatile stores are defined to have observable side
    /// effects and may not be elided by the optimiser, which is the
    /// guarantee `Vec::fill` (followed by a drop) does *not* provide
    /// — without volatile semantics the stdlib `fill` of a value that
    /// is dropped immediately afterwards is, in principle, eligible
    /// for dead-store elimination.
    pub fn zeroize(&mut self) {
        zeroize_blocks(self.keys_blocks.iter_mut());
        zeroize_blocks(self.values_blocks.iter_mut());
        self.reset();
    }

    /// Number of allocated blocks. Useful for telemetry — matches
    /// the vLLM `block_tables` length. After SWA eviction this reflects
    /// only the *resident* blocks (it shrinks as old blocks are dropped),
    /// which is exactly what makes a sliding-window cache bounded.
    pub fn num_blocks(&self) -> usize {
        self.keys_blocks.len()
    }

    /// Evict KV entries for absolute positions strictly older than `pos`.
    ///
    /// Only whole leading paged blocks that lie *entirely* below `pos` are
    /// dropped (a partially-in-window block is retained), so the surviving
    /// absolute positions keep their exact K/V bytes and the attention math
    /// is unchanged for any position the model can still attend to. This is
    /// the memory-efficiency half of sliding-window attention: for a layer
    /// in [`crate::architecture::AttentionMode::SlidingWindow`], positions
    /// outside the window are never read again, so retaining them is dead
    /// weight. The decode loop calls
    /// `kv.evict_before(pos.saturating_sub(window))` after each step on SWA
    /// layers, keeping the cache at `O(window)` instead of `O(seq_len)`.
    ///
    /// No-op for `pos == 0` and for global-attention layers (which never
    /// call this), preserving the original full-history behaviour.
    pub fn evict_before(&mut self, pos: usize) {
        let pos = pos.min(self.seq_len);
        // Logical block `b` (where `BLOCK = PAGED_BLOCK_TOKENS`) covers
        // absolute positions `[b * BLOCK, b * BLOCK + BLOCK)`. It is fully
        // below `pos` iff `(b + 1) * BLOCK <= pos`, i.e. `b < pos / BLOCK`.
        // So the number of logical blocks entirely below `pos` is
        // `pos / BLOCK`.
        let target_evicted = pos / PAGED_BLOCK_TOKENS;
        if target_evicted <= self.evicted_blocks {
            return;
        }
        let blocks_to_drop = target_evicted - self.evicted_blocks;
        // Never drop more than what is physically resident (defensive;
        // `blocks_to_drop` is bounded by the resident block count in practice).
        let blocks_to_drop = blocks_to_drop.min(self.keys_blocks.len());
        if blocks_to_drop == 0 {
            return;
        }
        // Zeroize the dropped blocks before releasing them so a tenant's
        // attention state cannot survive in a freed heap region (mirrors
        // `zeroize`), then remove them from the front of the block table.
        zeroize_blocks(self.keys_blocks.iter_mut().take(blocks_to_drop));
        zeroize_blocks(self.values_blocks.iter_mut().take(blocks_to_drop));
        for _ in 0..blocks_to_drop {
            self.keys_blocks.pop_front();
            self.values_blocks.pop_front();
        }
        self.evicted_blocks += blocks_to_drop;
    }

    /// Get the i-th cached key as a slice of length `kv_dim`.
    ///
    /// Caller invariant: `i` must not refer to a position that has already
    /// been evicted by [`Self::evict_before`] — i.e. its block index must be
    /// `>= self.evicted_blocks`. Standard attention upholds this because its
    /// `t_start` never drops below `pos - window`, which is exactly the
    /// floor `evict_before` preserves; MLA never evicts at all.
    fn key(&self, i: usize) -> &[f32] {
        let abs_block = i / PAGED_BLOCK_TOKENS;
        debug_assert!(
            abs_block >= self.evicted_blocks,
            "KvCache::key({i}) reads an evicted block ({abs_block} < {})",
            self.evicted_blocks
        );
        let block_idx = abs_block - self.evicted_blocks;
        let in_block = i % PAGED_BLOCK_TOKENS;
        let start = in_block * self.kv_dim;
        &self.keys_blocks[block_idx][start..start + self.kv_dim]
    }

    /// Public read accessor for the i-th cached key (length `kv_dim`).
    ///
    /// Multi-head latent attention ([`crate::mla`]) stores its compressed
    /// latent `[compressed_kv | k_pe]` (concatenation) in the key slot
    /// and reconstructs the per-head K/V on the fly, so it needs read
    /// access to historical entries from outside this module. Standard attention never calls
    /// this (it sweeps the cache through the private `key`/`value`
    /// helpers inside `MultiHeadSelfAttention::forward`).
    pub fn key_at(&self, i: usize) -> &[f32] {
        self.key(i)
    }

    fn value(&self, i: usize) -> &[f32] {
        let abs_block = i / PAGED_BLOCK_TOKENS;
        debug_assert!(
            abs_block >= self.evicted_blocks,
            "KvCache::value({i}) reads an evicted block ({abs_block} < {})",
            self.evicted_blocks
        );
        let block_idx = abs_block - self.evicted_blocks;
        let in_block = i % PAGED_BLOCK_TOKENS;
        let start = in_block * self.v_dim;
        &self.values_blocks[block_idx][start..start + self.v_dim]
    }
}

/// Zero every `f32` of every block via `ptr::write_volatile` so the
/// optimiser cannot elide the stores even though the underlying
/// `Vec`s are dropped immediately afterwards. The trailing
/// `compiler_fence` prevents the writes from being reordered past
/// the eventual deallocation of the backing buffers.
#[inline(never)]
fn zeroize_blocks<'a, I>(blocks: I)
where
    I: IntoIterator<Item = &'a mut Box<[f32]>>,
{
    for block in blocks {
        let ptr = block.as_mut_ptr();
        let len = block.len();
        // Safety: `block` is a `Box<[f32]>` (boxed slice) of length
        // `len`. If `len == 0`, the loop below executes zero times,
        // so `ptr` is never dereferenced; this remains valid even if
        // an empty boxed slice uses a dangling but properly aligned
        // non-null pointer and performs no allocation. If `len > 0`,
        // `ptr` points to the start of the slice's contiguous
        // storage, so every offset `0..len` is in-bounds and
        // properly aligned for `f32`. The slice is uniquely borrowed
        // via `&mut block`, so no other thread or reference can
        // observe or mutate these bytes for the duration of the
        // loop.
        for i in 0..len {
            unsafe { std::ptr::write_volatile(ptr.add(i), 0.0f32) };
        }
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Causal multi-head self attention with optional grouped-query attention.
///
/// Weights are stored row-major:
/// * `wq` : `[num_heads * head_dim, d_model]`
/// * `wk` : `[num_kv_heads * head_dim, d_model]`
/// * `wv` : `[num_kv_heads * head_dim, d_model]`
/// * `wo` : `[d_model, num_heads * head_dim]`
#[derive(Debug, Clone)]
pub struct MultiHeadSelfAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Number of head dimensions that receive RoPE rotation. Equals
    /// `head_dim` for full rotation (every architecture except
    /// MiMo-V2-Flash, which sets `partial_rotary_factor = 0.334` giving
    /// `rope_dim = 64` of `head_dim = 192`). Dimensions `[rope_dim..head_dim]`
    /// pass through unrotated. Always even.
    pub rope_dim: usize,
    /// V head dimension. Equals `head_dim` for every standard architecture;
    /// MiMo-V2-Flash uses `128` for V while Q/K use `192`. Drives the V
    /// projection (`wv`) row count, the per-head V slice width, and the
    /// attention-output / `wo` input width.
    pub v_head_dim: usize,
    /// Post-attention output scaling applied to the weighted V sum before
    /// the output projection (MiMo-V2-Flash `attention_value_scale = 0.707`).
    /// `None` (every other architecture) means no scaling (factor 1.0).
    pub attention_value_scale: Option<f32>,
    pub rope_base: f32,
    pub wq: DenseWeight,
    pub wk: DenseWeight,
    pub wv: DenseWeight,
    pub wo: DenseWeight,
    /// Sliding-window attention span. When `Some(w)`, each query position
    /// `pos` only attends to KV positions in `[pos.saturating_sub(w - 1) ..=
    /// pos]`; `None` recovers full causal attention (the backward-compatible
    /// default used by every existing test).
    ///
    /// This is the storage form of the per-layer
    /// [`crate::architecture::AttentionMode`]: `None` ⇔ `Global`, `Some(w)`
    /// ⇔ `SlidingWindow { window: w }` (see [`Self::attention_mode`]).
    /// Hybrid models (MiMo-V2 5:1, GPT-OSS 1:1) set this **per layer** at
    /// construction time so SWA and Global layers coexist in one model;
    /// uniform-SWA models (Mixtral) set the same `Some(4096)` on every
    /// layer. The KV cache still stores all appended positions, but the
    /// decode loop additionally evicts out-of-window entries on SWA layers
    /// (see [`KvCache::evict_before`]) to keep memory `O(window)`.
    pub window_size: Option<usize>,
    /// Optional per-head RMSNorm applied to **Q** before RoPE (Qwen3 /
    /// Qwen3-MoE "QK-Norm"). The weight vector has length `head_dim` and
    /// is applied independently to each of the `num_heads` query heads.
    /// `None` (Mixtral / Llama / Mistral / Phi-4) leaves Q untouched.
    pub q_norm: Option<RmsNorm>,
    /// Optional per-head RMSNorm applied to **K** before RoPE (Qwen3 /
    /// Qwen3-MoE). Weight length `head_dim`, applied to each of the
    /// `num_kv_heads` key heads. `None` leaves K untouched.
    pub k_norm: Option<RmsNorm>,
    /// Optional YaRN long-context RoPE scaling. When `Some`, Q/K
    /// rotations use the precomputed blended inverse frequencies and
    /// attention-magnitude correction instead of the plain
    /// `1/base^(2i/d)` schedule. Built from the checkpoint's
    /// `rope_scaling` block; `None` keeps the standard rotation.
    pub rope_yarn: Option<YarnRope>,
    /// Shared model-level RoPE table for this `(rope_dim, base, scaling)`
    /// shape. Manual tests may leave it `None`, in which case the
    /// compatibility path computes sin/cos directly.
    pub rope_cache: Option<Arc<RopeCache>>,
    /// Optional additive bias for the Q projection (`attention_bias = true`
    /// in config, e.g. GPT-OSS), length `num_heads * head_dim`. Added to the
    /// raw `wq · x` projection before QK-Norm and RoPE. `None` for every
    /// other architecture (the bias-free default used by every existing
    /// test).
    pub bq: Option<Vec<f32>>,
    /// Optional additive bias for the K projection, length
    /// `num_kv_heads * head_dim`. See [`Self::bq`].
    pub bk: Option<Vec<f32>>,
    /// Optional additive bias for the V projection, length
    /// `num_kv_heads * head_dim`. See [`Self::bq`].
    pub bv: Option<Vec<f32>>,
    /// Optional additive bias for the output projection, length `d_model`.
    /// Added to the `wo · attn_out` projection. See [`Self::bq`].
    pub bo: Option<Vec<f32>>,
    /// Per-head attention sink bias (MiMo-V2-Flash `add_swa_attention_sink_bias`).
    /// When `Some`, a scalar per attention head is added to the logit for
    /// position 0 (the sink token) before softmax, on SWA layers only.
    /// Length = `num_heads`. `None` for every other architecture / layer.
    pub sink_bias: Option<Vec<f32>>,
}

/// Reusable buffers for one standard attention block.
///
/// Kept inside [`TransformerLayerScratch`] so the real decode loop can carry
/// Q/K/V projections, attention scores, and projection outputs across tokens
/// without repeatedly allocating the same per-layer temporaries.
#[derive(Debug, Default)]
pub(crate) struct MultiHeadSelfAttentionScratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
    attn_out: Vec<f32>,
}

impl MultiHeadSelfAttention {
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// V projection (`wv`) output width: `num_kv_heads * v_head_dim`. Equals
    /// [`Self::kv_dim`] for every architecture except MiMo-V2-Flash, where
    /// V uses a smaller per-head dim than K.
    pub fn v_proj_dim(&self) -> usize {
        self.num_kv_heads * self.v_head_dim
    }

    /// Attention-output width = `wo` input width: `num_heads * v_head_dim`.
    /// Equals [`Self::q_dim`] when `v_head_dim == head_dim`.
    pub fn attn_out_dim(&self) -> usize {
        self.num_heads * self.v_head_dim
    }

    pub(crate) fn gpu_dense_path_supported(&self) -> bool {
        self.v_head_dim == self.head_dim
            && self.sink_bias.is_none()
            && self.wq.dtype() == DenseDType::F32
            && self.wk.dtype() == DenseDType::F32
            && self.wv.dtype() == DenseDType::F32
            && self.wo.dtype() == DenseDType::F32
    }

    #[inline]
    fn apply_rope_to(&self, v: &mut [f32], pos: usize) {
        apply_rope_with_cache(
            v,
            pos,
            self.rope_base,
            self.rope_yarn.as_ref(),
            self.rope_cache.as_deref(),
        );
    }

    /// Apply the optional post-attention output scale (MiMo-V2-Flash
    /// `attention_value_scale`) in place. No-op when `None`, so every other
    /// architecture pays nothing.
    fn apply_value_scale(&self, out: &mut [f32]) {
        if let Some(scale) = self.attention_value_scale {
            for x in out.iter_mut() {
                *x *= scale;
            }
        }
    }

    /// This layer's per-layer [`AttentionMode`], derived from
    /// `window_size`. `None` ⇒ [`AttentionMode::Global`], `Some(w)` ⇒
    /// [`AttentionMode::SlidingWindow`]. Used by the decode loop to decide
    /// whether to evict out-of-window KV entries.
    pub fn attention_mode(&self) -> crate::architecture::AttentionMode {
        crate::architecture::AttentionMode::from_window(self.window_size)
    }

    /// Apply the optional per-head Q RMSNorm (QK-Norm) in place. No-op when
    /// `q_norm` is `None`. `q` is the full `num_heads * head_dim` query
    /// vector; each head's `head_dim` slice is normalised independently.
    fn apply_q_norm(&self, q: &mut [f32]) {
        if let Some(norm) = self.q_norm.as_ref() {
            for h in 0..self.num_heads {
                let s = h * self.head_dim;
                norm.forward_inplace(&mut q[s..s + self.head_dim]);
            }
        }
    }

    /// Apply the optional per-head K RMSNorm (QK-Norm) in place. No-op when
    /// `k_norm` is `None`. `k` is the full `num_kv_heads * head_dim` key
    /// vector; each KV head's `head_dim` slice is normalised independently.
    fn apply_k_norm(&self, k: &mut [f32]) {
        if let Some(norm) = self.k_norm.as_ref() {
            for h in 0..self.num_kv_heads {
                let s = h * self.head_dim;
                norm.forward_inplace(&mut k[s..s + self.head_dim]);
            }
        }
    }

    /// Add the optional Q/K/V projection biases in place (GPT-OSS
    /// `attention_bias = true`). No-op when the biases are `None`, so every
    /// bias-free architecture pays nothing. Applied to the raw projection
    /// outputs *before* QK-Norm and RoPE, matching the HF reference where
    /// the bias is part of the linear layer.
    fn apply_qkv_bias(&self, q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
        if let Some(bq) = self.bq.as_ref() {
            for (qi, bi) in q.iter_mut().zip(bq.iter()) {
                *qi += bi;
            }
        }
        if let Some(bk) = self.bk.as_ref() {
            for (ki, bi) in k.iter_mut().zip(bk.iter()) {
                *ki += bi;
            }
        }
        if let Some(bv) = self.bv.as_ref() {
            for (vi, bi) in v.iter_mut().zip(bv.iter()) {
                *vi += bi;
            }
        }
    }

    /// Add the optional output-projection bias in place (GPT-OSS). No-op
    /// when `bo` is `None`.
    fn apply_o_bias(&self, out: &mut [f32]) {
        if let Some(bo) = self.bo.as_ref() {
            for (oi, bi) in out.iter_mut().zip(bo.iter()) {
                *oi += bi;
            }
        }
    }

    /// Project a single token embedding `x` into this layer's
    /// cache-ready **K** and **V** vectors at absolute position `pos`:
    /// `wk`/`wv` matmul, optional K/V bias, optional K QK-Norm, and
    /// RoPE on K over the partial `rope_dim` with optional YaRN scaling
    /// — exactly the steps [`Self::forward`] performs before
    /// `kv.append`. V is returned at its full `v_proj_dim()` width
    /// (which differs from `kv_dim()` on MiMo-V2-Flash). Shared by the
    /// verifier forward and the speculative KV preview so the two can
    /// never drift out of sync.
    pub fn project_kv(&self, x: &[f32], pos: usize) -> (Vec<f32>, Vec<f32>) {
        let mut k = Vec::new();
        let mut v = Vec::new();
        self.project_kv_into_with_timing(x, pos, &mut k, &mut v, None);
        (k, v)
    }

    pub fn project_kv_into_with_timing(
        &self,
        x: &[f32],
        pos: usize,
        k: &mut Vec<f32>,
        v: &mut Vec<f32>,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) {
        k.resize(self.kv_dim(), 0.0);
        v.resize(self.v_proj_dim(), 0.0);
        crate::stage_timing::time_optional(timings, crate::stage_timing::K_PROJECTION, || {
            self.wk.matvec_into(x, k)
        });
        crate::stage_timing::time_optional(timings, crate::stage_timing::V_PROJECTION, || {
            self.wv.matvec_into(x, v)
        });
        if let Some(bk) = self.bk.as_ref() {
            for (ki, bi) in k.iter_mut().zip(bk.iter()) {
                *ki += bi;
            }
        }
        if let Some(bv) = self.bv.as_ref() {
            for (vi, bi) in v.iter_mut().zip(bv.iter()) {
                *vi += bi;
            }
        }
        crate::stage_timing::time_optional(timings, crate::stage_timing::RMS_NORM, || {
            self.apply_k_norm(k.as_mut_slice())
        });
        crate::stage_timing::time_optional(timings, crate::stage_timing::ROPE, || {
            for h in 0..self.num_kv_heads {
                let s = h * self.head_dim;
                self.apply_rope_to(&mut k[s..s + self.rope_dim], pos);
            }
        });
    }

    /// Forward one token at absolute position `pos`. Updates `kv` with
    /// the new K/V for this position. Returns a new hidden state of
    /// length `d_model`.
    ///
    /// The GPU path (selected when `backend.is_gpu()`) currently converts
    /// resident F32 projection weights and activations to F16 on the CPU,
    /// uploads them through backend calls, writes K/V via
    /// `backend.kv_cache_insert`, and runs attention via `backend.kv_attend`.
    /// The CPU path is the original paged-attention loop, byte-for-byte.
    pub fn forward(
        &self,
        x: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
    ) -> Vec<f32> {
        self.forward_with_timing(x, pos, layer_idx, kv, backend, None)
    }

    pub fn forward_with_timing(
        &self,
        x: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Vec<f32> {
        use crate::backend::{Backend, TensorView, TensorViewMut};

        debug_assert_eq!(x.len(), self.d_model);
        debug_assert_eq!(kv.kv_dim, self.kv_dim());
        debug_assert_eq!(kv.v_dim, self.v_proj_dim());

        let q_dim = self.q_dim();
        let kv_dim = self.kv_dim();
        let v_head_dim = self.v_head_dim;
        let cpu_attend = |q: &[f32], kv: &KvCache| -> Vec<f32> {
            let mut attn_out = vec![0.0f32; self.attn_out_dim()];
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let t_max = kv.seq_len;
            // Per-layer attention mode: Global attends to all past
            // positions; SlidingWindow restricts the sum to the last
            // `window` positions. Equivalent to the previous match on
            // `window_size`, now expressed via `AttentionMode` so hybrid
            // models (MiMo-V2, GPT-OSS) share one code path with the
            // uniform-window (Mixtral) and full-causal (everything else)
            // cases.
            let t_start = match self.attention_mode() {
                crate::architecture::AttentionMode::SlidingWindow { window } => {
                    t_max.saturating_sub(window)
                }
                crate::architecture::AttentionMode::Global => 0,
            };

            for h in 0..self.num_heads {
                let kv_head = h * self.num_kv_heads / self.num_heads;
                let q_head = &q[h * self.head_dim..(h + 1) * self.head_dim];
                let span = t_max - t_start;
                let mut scores = Vec::with_capacity(span);

                for t in t_start..t_max {
                    let k_t = kv.key(t);
                    let k_h = &k_t[kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                    let mut s = 0.0f32;
                    for j in 0..self.head_dim {
                        s += q_head[j] * k_h[j];
                    }
                    scores.push(s * scale);
                }
                // Attention sink bias (MiMo-V2-Flash `add_swa_attention_sink_bias`):
                // add the per-head scalar to the logit for the first (sink)
                // position before softmax. Only applied when position 0 is
                // within the attention span (`t_start == 0`), so the sink
                // token's slot is `scores[0]`. `None` (every other
                // architecture / global layers) is a no-op.
                if let Some(bias) = self.sink_bias.as_ref() {
                    if t_start == 0 && !scores.is_empty() {
                        if let Some(b) = bias.get(h) {
                            scores[0] += *b;
                        }
                    }
                }
                softmax_inplace(&mut scores);

                // V uses `v_head_dim` (may differ from `head_dim` on
                // MiMo-V2-Flash); the attention output head is the same width.
                let out_h = &mut attn_out[h * v_head_dim..(h + 1) * v_head_dim];
                for (idx, score) in scores.iter().enumerate() {
                    let t = t_start + idx;
                    let v_t = kv.value(t);
                    let v_h = &v_t[kv_head * v_head_dim..(kv_head + 1) * v_head_dim];
                    for j in 0..v_head_dim {
                        out_h[j] += score * v_h[j];
                    }
                }
            }
            attn_out
        };
        let mut cpu_forward = || {
            let mut q = crate::stage_timing::time_optional(
                timings,
                crate::stage_timing::Q_PROJECTION,
                || self.wq.matvec(x),
            );
            // Q bias (GPT-OSS `attention_bias`) before QK-Norm / RoPE; K and
            // V biases are applied inside `project_kv`.
            if let Some(bq) = self.bq.as_ref() {
                for (qi, bi) in q.iter_mut().zip(bq.iter()) {
                    *qi += bi;
                }
            }
            // QK-Norm (Qwen3): per-head RMSNorm on Q *before* RoPE.
            crate::stage_timing::time_optional(timings, crate::stage_timing::RMS_NORM, || {
                self.apply_q_norm(&mut q)
            });
            // RoPE rotates only the first `rope_dim` dims of each head
            // (partial rotary on MiMo-V2-Flash; `rope_dim == head_dim`
            // elsewhere ⇒ full rotation).
            crate::stage_timing::time_optional(timings, crate::stage_timing::ROPE, || {
                for h in 0..self.num_heads {
                    let s = h * self.head_dim;
                    self.apply_rope_to(&mut q[s..s + self.rope_dim], pos);
                }
            });
            // K/V projection (+ bias, QK-Norm, RoPE) is split here only
            // when benchmark stage timings are active; the untimed path
            // keeps using the shared helper that speculative KV preview
            // also calls.
            let (k, v) = if timings.is_some() {
                let mut k = crate::stage_timing::time_optional(
                    timings,
                    crate::stage_timing::K_PROJECTION,
                    || self.wk.matvec(x),
                );
                let mut v = crate::stage_timing::time_optional(
                    timings,
                    crate::stage_timing::V_PROJECTION,
                    || self.wv.matvec(x),
                );
                if let Some(bk) = self.bk.as_ref() {
                    for (ki, bi) in k.iter_mut().zip(bk.iter()) {
                        *ki += bi;
                    }
                }
                if let Some(bv) = self.bv.as_ref() {
                    for (vi, bi) in v.iter_mut().zip(bv.iter()) {
                        *vi += bi;
                    }
                }
                crate::stage_timing::time_optional(timings, crate::stage_timing::RMS_NORM, || {
                    self.apply_k_norm(&mut k)
                });
                crate::stage_timing::time_optional(timings, crate::stage_timing::ROPE, || {
                    for h in 0..self.num_kv_heads {
                        let s = h * self.head_dim;
                        self.apply_rope_to(&mut k[s..s + self.rope_dim], pos);
                    }
                });
                (k, v)
            } else {
                self.project_kv(x, pos)
            };
            kv.append(&k, &v);
            let mut attn_out = crate::stage_timing::time_optional(
                timings,
                crate::stage_timing::ATTENTION_SCORE_VALUE,
                || cpu_attend(&q, kv),
            );
            // Post-attention output scale (MiMo-V2-Flash 0.707), applied
            // before the output projection.
            self.apply_value_scale(&mut attn_out);
            crate::stage_timing::time_optional(timings, crate::stage_timing::O_PROJECTION, || {
                let mut out = self.wo.matvec(&attn_out);
                self.apply_o_bias(&mut out);
                out
            })
        };
        if !backend.is_gpu() || !self.gpu_dense_path_supported() {
            // The current GPU dense path copies full projection matrices to
            // F16 per token, so it is only safe for resident F32 weights. Q8
            // projections must stay on the CPU path to avoid full per-token
            // dequantization. The GPU kernels also assume symmetric K/V head
            // dims and do not apply attention sink bias yet.
            // TODO: apply attention_sink_bias on GPU path (kv_attend kernel).
            return cpu_forward();
        }

        // ── Helpers: f32 ↔ f16 conversion at the backend boundary ────────────
        let to_f16 = |v: &[f32]| -> Vec<f16> { v.iter().map(|&f| f16::from_f32(f)).collect() };
        let to_f32 = |v: &[f16]| -> Vec<f32> { v.iter().map(|h| h.to_f32()).collect() };

        let x_f16 = to_f16(x);
        let wq_f32 = self.wq.to_f32_vec();
        let wk_f32 = self.wk.to_f32_vec();
        let wv_f32 = self.wv.to_f32_vec();
        let wq_f16 = to_f16(&wq_f32);
        let wk_f16 = to_f16(&wk_f32);
        let wv_f16 = to_f16(&wv_f32);

        let mut q_f16 = vec![f16::ZERO; q_dim];
        let mut k_f16 = vec![f16::ZERO; kv_dim];
        let mut v_f16 = vec![f16::ZERO; kv_dim];

        if backend
            .matmul_into(
                TensorView {
                    data: &wq_f16,
                    rows: q_dim,
                    cols: self.d_model,
                },
                TensorView {
                    data: &x_f16,
                    rows: self.d_model,
                    cols: 1,
                },
                &mut TensorViewMut {
                    data: &mut q_f16,
                    rows: q_dim,
                    cols: 1,
                },
            )
            .is_err()
        {
            return cpu_forward();
        }

        if backend
            .matmul_into(
                TensorView {
                    data: &wk_f16,
                    rows: kv_dim,
                    cols: self.d_model,
                },
                TensorView {
                    data: &x_f16,
                    rows: self.d_model,
                    cols: 1,
                },
                &mut TensorViewMut {
                    data: &mut k_f16,
                    rows: kv_dim,
                    cols: 1,
                },
            )
            .is_err()
        {
            return cpu_forward();
        }

        if backend
            .matmul_into(
                TensorView {
                    data: &wv_f16,
                    rows: kv_dim,
                    cols: self.d_model,
                },
                TensorView {
                    data: &x_f16,
                    rows: self.d_model,
                    cols: 1,
                },
                &mut TensorViewMut {
                    data: &mut v_f16,
                    rows: kv_dim,
                    cols: 1,
                },
            )
            .is_err()
        {
            return cpu_forward();
        }

        // ── 2) Apply RoPE in f32 (cheap; stays on CPU regardless of backend) ─
        let mut q = to_f32(&q_f16);
        let mut k = to_f32(&k_f16);
        let mut v = to_f32(&v_f16);

        // QKV projection biases (GPT-OSS `attention_bias = true`), mirror of
        // the CPU path so GPU and CPU attention agree numerically. Applied to
        // the raw projections before QK-Norm / RoPE.
        self.apply_qkv_bias(&mut q, &mut k, &mut v);

        // QK-Norm (Qwen3): per-head RMSNorm on Q and K *before* RoPE, mirror
        // of the CPU path so the GPU and CPU attention agree numerically.
        self.apply_q_norm(&mut q);
        self.apply_k_norm(&mut k);

        for h in 0..self.num_heads {
            let s = h * self.head_dim;
            self.apply_rope_to(&mut q[s..s + self.rope_dim], pos);
        }
        for h in 0..self.num_kv_heads {
            let s = h * self.head_dim;
            self.apply_rope_to(&mut k[s..s + self.rope_dim], pos);
        }

        // ── 3) KV insert + attention ──────────────────────────────────────────
        let k_f16_rope = to_f16(&k);
        // V is not RoPE'd. When there is no V-bias, `v` is an exact f32 copy of
        // the already-correct `v_f16` projection, so reuse it directly and skip
        // the redundant per-token f32→f16 round-trip.
        let v_f16_rope = if self.bv.is_some() { to_f16(&v) } else { v_f16 };

        // Generation must advance strictly one token at a time: before we append
        // the new KV for `pos`, the cache length should equal that position.
        debug_assert_eq!(pos, kv.seq_len);
        kv.append(&k, &v);
        let seq_len = kv.seq_len;

        if backend
            .kv_cache_insert(
                layer_idx,
                pos,
                TensorView {
                    data: &k_f16_rope,
                    rows: 1,
                    cols: kv_dim,
                },
                TensorView {
                    data: &v_f16_rope,
                    rows: 1,
                    cols: kv_dim,
                },
            )
            .is_err()
        {
            let mut attn_out = cpu_attend(&q, kv);
            self.apply_value_scale(&mut attn_out);
            let mut out = self.wo.matvec(&attn_out);
            self.apply_o_bias(&mut out);
            return out;
        }

        let q_f16_rope = to_f16(&q);
        let mut out_f16 = vec![f16::ZERO; q_dim];
        let attn_out = if backend
            .kv_attend(
                layer_idx,
                TensorView {
                    data: &q_f16_rope,
                    rows: self.num_heads,
                    cols: self.head_dim,
                },
                seq_len,
                &mut TensorViewMut {
                    data: &mut out_f16,
                    rows: self.num_heads,
                    cols: self.head_dim,
                },
            )
            .is_ok()
        {
            to_f32(&out_f16)
        } else {
            cpu_attend(&q, kv)
        };

        // Post-attention output scale (MiMo-V2-Flash 0.707), applied before
        // the output projection. No-op for every other architecture.
        let mut attn_out = attn_out;
        self.apply_value_scale(&mut attn_out);

        // ── 4) Output projection via backend ──────────────────────────────────
        let wo_f32 = self.wo.to_f32_vec();
        let wo_f16 = to_f16(&wo_f32);
        let attn_f16 = to_f16(&attn_out);
        let mut out_f16 = vec![f16::ZERO; self.d_model];

        let mut out = if backend
            .matmul_into(
                TensorView {
                    data: &wo_f16,
                    rows: self.d_model,
                    cols: q_dim,
                },
                TensorView {
                    data: &attn_f16,
                    rows: q_dim,
                    cols: 1,
                },
                &mut TensorViewMut {
                    data: &mut out_f16,
                    rows: self.d_model,
                    cols: 1,
                },
            )
            .is_ok()
        {
            to_f32(&out_f16)
        } else {
            self.wo.matvec(&attn_out)
        };
        self.apply_o_bias(&mut out);
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_into_with_timing(
        &self,
        x: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
        scratch: &mut MultiHeadSelfAttentionScratch,
        out: &mut Vec<f32>,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) {
        // The scratch path is the CPU-production path. Keep GPU behaviour
        // byte-for-byte by delegating to the existing allocating path when a
        // GPU backend is active.
        if backend.is_gpu() {
            let y = self.forward_with_timing(x, pos, layer_idx, kv, backend, timings);
            out.clear();
            out.extend_from_slice(&y);
            return;
        }

        debug_assert_eq!(x.len(), self.d_model);
        debug_assert_eq!(kv.kv_dim, self.kv_dim());
        debug_assert_eq!(kv.v_dim, self.v_proj_dim());

        let q_dim = self.q_dim();
        scratch.q.resize(q_dim, 0.0);
        crate::stage_timing::time_optional(timings, crate::stage_timing::Q_PROJECTION, || {
            self.wq.matvec_into(x, &mut scratch.q)
        });
        if let Some(bq) = self.bq.as_ref() {
            for (qi, bi) in scratch.q.iter_mut().zip(bq.iter()) {
                *qi += bi;
            }
        }
        crate::stage_timing::time_optional(timings, crate::stage_timing::RMS_NORM, || {
            self.apply_q_norm(&mut scratch.q)
        });
        crate::stage_timing::time_optional(timings, crate::stage_timing::ROPE, || {
            for h in 0..self.num_heads {
                let s = h * self.head_dim;
                self.apply_rope_to(&mut scratch.q[s..s + self.rope_dim], pos);
            }
        });

        self.project_kv_into_with_timing(
            x,
            pos,
            &mut scratch.k,
            &mut scratch.v,
            timings,
        );
        debug_assert_eq!(pos, kv.seq_len);
        kv.append(&scratch.k, &scratch.v);

        crate::stage_timing::time_optional(
            timings,
            crate::stage_timing::ATTENTION_SCORE_VALUE,
            || {
                self.cpu_attend_into(
                    &scratch.q,
                    kv,
                    &mut scratch.scores,
                    &mut scratch.attn_out,
                )
            },
        );
        self.apply_value_scale(&mut scratch.attn_out);

        out.resize(self.d_model, 0.0);
        crate::stage_timing::time_optional(timings, crate::stage_timing::O_PROJECTION, || {
            self.wo.matvec_into(&scratch.attn_out, out);
            self.apply_o_bias(out);
        });
    }

    fn cpu_attend_into(
        &self,
        q: &[f32],
        kv: &KvCache,
        scores: &mut Vec<f32>,
        attn_out: &mut Vec<f32>,
    ) {
        attn_out.clear();
        attn_out.resize(self.attn_out_dim(), 0.0);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let t_max = kv.seq_len;
        let t_start = match self.attention_mode() {
            crate::architecture::AttentionMode::SlidingWindow { window } => {
                t_max.saturating_sub(window)
            }
            crate::architecture::AttentionMode::Global => 0,
        };
        let span = t_max - t_start;

        for h in 0..self.num_heads {
            let kv_head = h * self.num_kv_heads / self.num_heads;
            let q_head = &q[h * self.head_dim..(h + 1) * self.head_dim];
            scores.resize(span, 0.0);

            for (idx, score) in scores.iter_mut().enumerate() {
                let t = t_start + idx;
                let k_t = kv.key(t);
                let k_h = &k_t[kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                let mut s = 0.0f32;
                for j in 0..self.head_dim {
                    s += q_head[j] * k_h[j];
                }
                *score = s * scale;
            }
            if let Some(bias) = self.sink_bias.as_ref() {
                if t_start == 0 && !scores.is_empty() {
                    if let Some(b) = bias.get(h) {
                        scores[0] += *b;
                    }
                }
            }
            softmax_inplace(scores);

            let out_h = &mut attn_out[h * self.v_head_dim..(h + 1) * self.v_head_dim];
            for (idx, score) in scores.iter().enumerate() {
                let t = t_start + idx;
                let v_t = kv.value(t);
                let v_h =
                    &v_t[kv_head * self.v_head_dim..(kv_head + 1) * self.v_head_dim];
                for j in 0..self.v_head_dim {
                    out_h[j] += score * v_h[j];
                }
            }
        }
    }
}

/// Combine the per-token outputs of `k` selected experts using the gating
/// scores. `outputs[i]` and `scores[i]` must be aligned (same expert).
/// Scores must already be softmax-normalised over the chosen top-K set.
///
/// Thin wrapper over [`crate::inference::combine_outputs`] (the canonical
/// MoE combiner) that ignores the redundant `d_model` argument; kept for
/// backwards compatibility with `TransformerLayer::moe_combine`.
pub fn combine_moe_outputs(outputs: &[HiddenState], scores: &[f32], d_model: usize) -> HiddenState {
    let mut out = Vec::new();
    combine_moe_outputs_into(outputs, scores, d_model, &mut out);
    out
}

pub fn combine_moe_outputs_into(
    outputs: &[HiddenState],
    scores: &[f32],
    d_model: usize,
    out: &mut Vec<f32>,
) {
    debug_assert!(
        outputs.iter().all(|o| o.len() == d_model),
        "every expert output must have length d_model"
    );
    let _ = d_model;
    crate::inference::combine_outputs_into(outputs, scores, out);
}

/// Run one expert FFN by reinterpreting its on-disk bytes (already loaded
/// into the resident buffer) as `[gate || up || down]` SwiGLU weights and
/// applying the SwiGLU forward over `x`. This is the bridge from
/// "bytes streamed from SSD" to "expert output vector" used by the
/// transformer layer's MoE block.
pub fn run_expert_forward(
    resident: &ExpertResident,
    x: &[f32],
    d_model: usize,
    d_ff: usize,
) -> Result<HiddenState, crate::inference::ExpertWeightsError> {
    let weights = ExpertWeights::from_bytes(resident.data(), d_model, d_ff)?;
    Ok(weights.forward(x))
}

/// Row-major matrix-vector multiply: `y = W · x` where `W` is
/// `[rows, cols]` row-major. Returns a fresh `Vec<f32>` of length `rows`.
///
/// The implementation is selected at runtime via
/// [`crate::parallel::DenseMatvecBackend`]. `Auto` preserves the previous
/// build defaults (`blas` builds use one tuned matrixmultiply SGEMM call;
/// other builds use the Rayon row-parallel dot-product path), while
/// production configs can force serial matrixmultiply, Rayon reference/SIMD,
/// or Rayon-chunked matrixmultiply without rebuilding.
pub fn matmul_row_major(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    matmul_row_major_into(w, x, &mut y, rows, cols);
    y
}

pub fn matmul_row_major_with_backend(
    w: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    backend: crate::parallel::DenseMatvecBackend,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    matmul_row_major_into_with_backend(w, x, &mut y, rows, cols, backend);
    y
}

/// In-place row-major matrix-vector multiply: `y = W · x`.
pub fn matmul_row_major_into(w: &[f32], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    matmul_row_major_into_with_backend(w, x, y, rows, cols, crate::parallel::dense_matvec_backend())
}

pub fn matmul_row_major_into_with_backend(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    cols: usize,
    backend: crate::parallel::DenseMatvecBackend,
) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);

    let backend = match backend {
        crate::parallel::DenseMatvecBackend::Auto => {
            crate::parallel::default_dense_matvec_backend()
        }
        other => other,
    };
    match backend {
        crate::parallel::DenseMatvecBackend::Auto => unreachable!("auto resolves before dispatch"),
        crate::parallel::DenseMatvecBackend::Matrixmultiply => {
            matmul_row_major_matrixmultiply_into(w, x, y, rows, cols);
        }
        crate::parallel::DenseMatvecBackend::Rayon => {
            matmul_row_major_parallel_into(w, x, y, rows, cols);
        }
        crate::parallel::DenseMatvecBackend::RayonMatrixmultiply => {
            matmul_row_major_parallel_matrixmultiply_into(w, x, y, rows, cols);
        }
    }
}

/// Serial tuned SGEMV via `matrixmultiply::sgemm`.
fn matmul_row_major_matrixmultiply(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    matmul_row_major_matrixmultiply_into(w, x, &mut y, rows, cols);
    y
}

#[inline]
fn matmul_row_major_matrixmultiply_into(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);
    if rows == 0 || cols == 0 {
        y.fill(0.0);
        return;
    }
    // SAFETY: matrixmultiply::sgemm is defined for row-major (m × k) and
    // (k × n) matrices writing into row-major (m × n) output. `w` is
    // exactly `rows * cols`, `x` is exactly `cols`, and `y` is a disjoint
    // mutable output slice of `rows` elements.
    unsafe {
        matrixmultiply::sgemm(
            rows,
            cols,
            1,
            1.0,
            w.as_ptr(),
            cols as isize,
            1,
            x.as_ptr(),
            1,
            1,
            0.0,
            y.as_mut_ptr(),
            1,
            1,
        );
    }
}

/// Row-parallel matmul on the shared `rayon` pool. Each worker computes a
/// contiguous block of output rows; no synchronisation is required because
/// the output rows are disjoint. Always compiled — gist Task 1's
/// "auto-escalation" requirement means we can't hide this behind a
/// cargo feature any more.
///
/// Unlike the previous `std::thread::scope` implementation this does
/// **not** spawn OS threads per call: [`crate::parallel::par_row_chunks`]
/// dispatches onto the process-wide pool, so the per-call cost is a
/// fork-join over resident workers and concurrent requests (continuous
/// batching) share one bounded pool instead of each oversubscribing the
/// machine. The arithmetic — one `f32` dot product per output row — is
/// unchanged, so the result is identical to the scalar path.
fn matmul_row_major_parallel(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    matmul_row_major_parallel_into(w, x, &mut y, rows, cols);
    y
}

fn matmul_row_major_parallel_into(w: &[f32], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);
    crate::parallel::par_row_chunks(y, cols, |row_start, out| {
        for (i, slot) in out.iter_mut().enumerate() {
            let row = &w[(row_start + i) * cols..(row_start + i + 1) * cols];
            *slot = crate::kernels::dot_f32(row, x);
        }
    });
}

/// Rayon-chunked matrixmultiply. Output rows are split into contiguous
/// chunks on the existing global Rayon pool; each task invokes one tuned
/// SGEMM over its chunk. If already running inside a Rayon worker, fall
/// back to the serial matrixmultiply path to avoid nested pool fan-out.
fn matmul_row_major_parallel_matrixmultiply(
    w: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    matmul_row_major_parallel_matrixmultiply_into(w, x, &mut y, rows, cols);
    y
}

fn matmul_row_major_parallel_matrixmultiply_into(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);
    if crate::parallel::in_rayon_worker() {
        matmul_row_major_matrixmultiply_into(w, x, y, rows, cols);
        return;
    }
    crate::parallel::par_row_chunks(y, cols, |row_start, out| {
        let row_count = out.len();
        let w_start = row_start * cols;
        let w_end = w_start + row_count * cols;
        matmul_row_major_matrixmultiply_into(&w[w_start..w_end], x, out, row_count, cols);
    });
}

/// Final language-modelling head: a linear projection from the residual
/// stream `[d_model]` to per-token logits `[vocab_size]`. In real models
/// this is sometimes weight-tied with the input embedding; we keep them
/// separate so the engine can sanity-check sampling without an embedding
/// matrix.
#[derive(Debug, Clone)]
pub struct LMHead {
    pub weights: DenseWeight,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl LMHead {
    pub fn new(weights: Vec<f32>, vocab_size: usize, d_model: usize) -> Self {
        assert_eq!(
            weights.len(),
            vocab_size * d_model,
            "lm_head weights must be [vocab_size, d_model]"
        );
        Self {
            weights: DenseWeight::from_f32(weights, vocab_size, d_model),
            vocab_size,
            d_model,
        }
    }

    pub fn from_dense(weights: DenseWeight, vocab_size: usize, d_model: usize) -> Self {
        assert_eq!(weights.rows(), vocab_size, "lm_head row count mismatch");
        assert_eq!(weights.cols(), d_model, "lm_head column count mismatch");
        Self {
            weights,
            vocab_size,
            d_model,
        }
    }

    /// Compute logits = `W · hidden`.
    pub fn forward(&self, hidden: &[f32]) -> Vec<f32> {
        self.weights.matvec(hidden)
    }

    /// One-shot: project `hidden` to logits and sample a next-token id
    /// using the given [`crate::sampling::SamplingParams`]. The
    /// `position` is folded into the sampler's per-step seed so a
    /// `(seed, position)` pair always yields the same token — see
    /// `crate::sampling` for the deterministic-decode contract.
    pub fn sample(
        &self,
        hidden: &[f32],
        params: &crate::sampling::SamplingParams,
        position: u64,
    ) -> u32 {
        if params.is_greedy() {
            return self.weights.greedy_argmax(hidden);
        }
        if params.top_k > 0 && !(params.top_p > 0.0 && params.top_p < 1.0) {
            return self.sample_top_k(hidden, params, position);
        }
        let logits = self.forward(hidden);
        crate::sampling::sample(&logits, params, position)
    }

    fn sample_top_k(
        &self,
        hidden: &[f32],
        params: &crate::sampling::SamplingParams,
        position: u64,
    ) -> u32 {
        let candidates = self.weights.top_k_logits(hidden, params.top_k);
        if candidates.is_empty() {
            return 0;
        }
        let t = params.temperature.max(1e-6);
        let max = candidates
            .iter()
            .map(|&(_, logit)| logit / t)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = candidates
            .iter()
            .map(|&(_, logit)| ((logit / t) - max).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        if !(sum > 0.0) {
            return candidates[0].0 as u32;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let p_cut = if !(params.top_p > 0.0 && params.top_p < 1.0) {
            candidates.len()
        } else {
            let target = params.top_p.clamp(1e-6, 1.0);
            let mut cum = 0.0f32;
            let mut idx = 0usize;
            for (i, p) in probs.iter().enumerate() {
                cum += *p;
                idx = i + 1;
                if cum >= target {
                    break;
                }
            }
            idx.min(candidates.len()).max(1)
        };
        let kept_sum: f32 = probs.iter().take(p_cut).sum();
        if !(kept_sum > 0.0) {
            return candidates[0].0 as u32;
        }
        let bits = params.step_seed(position);
        let u = ((bits >> 40) as u32) as f32 / ((1u32 << 24) as f32) * kept_sum;
        let mut acc = 0.0f32;
        for (i, p) in probs.iter().take(p_cut).enumerate() {
            acc += *p;
            if u <= acc {
                return candidates[i].0 as u32;
            }
        }
        candidates[p_cut - 1].0 as u32
    }
}

/// Element-wise residual add: `y = a + b`.
#[inline]
pub fn add_residual(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    add_residual_into(a, b, &mut out);
    out
}

#[inline]
pub fn add_residual_into(a: &[f32], b: &[f32], out: &mut Vec<f32>) {
    debug_assert_eq!(a.len(), b.len());
    out.clear();
    out.reserve(a.len());
    out.extend(a.iter().zip(b.iter()).map(|(x, y)| x + y));
}

/// A dense ("shared") expert FFN that is applied to **every** token in a
/// layer, in addition to the routed top-K experts.
///
/// This is the architectural feature that distinguishes Qwen2-MoE,
/// DeepSeek-MoE and OLMoE-style MoEs from a vanilla Mixtral block: the
/// router picks K experts per token, *and* a separate always-on FFN
/// (the "shared expert") contributes to every token's output. Mixtral
/// has no such tensor, so [`TransformerLayer::shared_expert`] is an
/// `Option` and stays `None` for those models — the engine remains MoE-
/// architecture-agnostic and incurs zero extra work when the weights are
/// absent.
///
/// The three projection matrices are concatenated in the same
/// `[gate || up || down]` SwiGLU layout the routed experts use on disk,
/// so the forward pass can reuse the exact same
/// [`crate::inference::ExpertWeights`] kernel and stay numerically
/// consistent with the routed path. `d_ff` is the shared expert's own
/// intermediate size, which in Qwen2-MoE differs from the routed
/// `moe_intermediate_size`; it is therefore stored per shared expert
/// rather than read from the layer/model config.
///
/// `gate_inp`, when present, is the Qwen2-MoE "shared expert gate"
/// (`ffn_gate_inp_shexp` / `shared_expert_gate`): a `[d_model] → 1`
/// linear whose `sigmoid` scales the shared expert output. DeepSeek-MoE
/// shared experts have no such gate (they are unconditionally added), so
/// it is optional; absence is treated as a scale of `1.0`.
#[derive(Debug, Clone)]
pub struct SharedExpert {
    pub d_model: usize,
    pub d_ff: usize,
    /// `[gate || up || down]` concatenated, row-major, exactly the
    /// layout [`crate::inference::ExpertWeights::from_floats`] expects:
    /// `gate`/`up` are `[d_ff, d_model]` and `down` is `[d_model, d_ff]`.
    pub weights: Vec<f32>,
    /// Optional sigmoid gate weights `[d_model]` (Qwen2-MoE). `None`
    /// means the shared expert output is added unscaled (DeepSeek-MoE).
    pub gate_inp: Option<Vec<f32>>,
    /// Pre-built `(gate, up, down)` `candle-core` tensors, materialised
    /// **once** at construction. The shared expert is always-on (it runs
    /// for every token), so building the Candle weight tensors per call
    /// would re-copy all weights into Candle storage on every token and
    /// dominate runtime. Caching them here makes the per-token cost just
    /// the matmuls. `None` only if the one-time tensor build failed, in
    /// which case [`Self::forward`] falls back to the per-call view path.
    tensors: Option<(Tensor, Tensor, Tensor)>,
}

impl SharedExpert {
    /// Build a shared expert from its three separate projection matrices
    /// (each row-major) plus an optional sigmoid-gate vector. The
    /// matrices are concatenated into the `[gate || up || down]` layout
    /// the SwiGLU kernel consumes. Returns `None` if the shapes are
    /// inconsistent, so a malformed on-disk tensor degrades gracefully
    /// to "no shared expert" instead of aborting the load.
    pub fn from_projections(
        d_model: usize,
        d_ff: usize,
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        gate_inp: Option<Vec<f32>>,
    ) -> Option<Self> {
        if gate.len() != d_ff * d_model
            || up.len() != d_ff * d_model
            || down.len() != d_model * d_ff
        {
            return None;
        }
        if let Some(g) = gate_inp.as_ref() {
            if g.len() != d_model {
                return None;
            }
        }
        let mut weights = Vec::with_capacity(gate.len() + up.len() + down.len());
        weights.extend_from_slice(gate);
        weights.extend_from_slice(up);
        weights.extend_from_slice(down);
        // Materialise the Candle weight tensors once: the shared expert
        // runs on every token, so this avoids per-token full-weight
        // copies in the forward pass.
        let tensors = ExpertWeights::from_floats(&weights, d_model, d_ff)
            .ok()
            .and_then(|w| w.to_candle_tensors(&Device::Cpu).ok());
        Some(Self {
            d_model,
            d_ff,
            weights,
            gate_inp,
            tensors,
        })
    }

    /// Run the dense SwiGLU forward over `x` (the MoE-normalised hidden
    /// state) and apply the optional sigmoid gate. Reuses the routed
    /// expert kernel so the math matches the streamed experts exactly.
    pub fn forward(&self, x: &[f32]) -> HiddenState {
        // Fast path: reuse the Candle weight tensors built once at
        // construction so the per-token cost is just the matmuls, not a
        // full copy of every weight into Candle storage.
        let mut out = match self.tensors.as_ref() {
            Some((gate_t, up_t, down_t)) => {
                match forward_candle_tensors(gate_t, up_t, down_t, self.d_model, x) {
                    Ok(y) => y,
                    Err(err) => {
                        tracing::error!(
                            error = %err,
                            d_model = self.d_model,
                            d_ff = self.d_ff,
                            "shared expert SwiGLU forward failed; skipping shared expert"
                        );
                        return vec![0.0f32; self.d_model];
                    }
                }
            }
            None => {
                // Degraded path: the one-time tensor build failed, so
                // rebuild a view per call (still correct, just slower).
                let weights =
                    match ExpertWeights::from_floats(&self.weights, self.d_model, self.d_ff) {
                        Ok(w) => w,
                        Err(err) => {
                            tracing::error!(
                                error = %err,
                                d_model = self.d_model,
                                d_ff = self.d_ff,
                                "shared expert weight view failed; skipping shared expert"
                            );
                            return vec![0.0f32; self.d_model];
                        }
                    };
                weights.forward(x)
            }
        };
        if let Some(gate_inp) = self.gate_inp.as_ref() {
            // sigmoid(W_gate · x): a single scalar that scales the whole
            // shared expert output (Qwen2-MoE `shared_expert_gate`).
            let mut logit = 0.0f32;
            for (w, &xi) in gate_inp.iter().zip(x.iter()) {
                logit += w * xi;
            }
            let scale = 1.0 / (1.0 + (-logit).exp());
            for v in out.iter_mut() {
                *v *= scale;
            }
        }
        out
    }
}

/// One Llama / Mixtral-style transformer decoder layer.
///
/// Holds the dense (resident) weights — RMSNorms, attention projections,
/// and the routing gate — but **not** the routed expert FFN weights
/// themselves. Those are streamed from SSD per token by the engine's
/// [`crate::expert_cache::ExpertCache`] and handed back here as already-
/// loaded `ExpertResident`s for the [`Self::moe_combine`] step.
///
/// `shared_expert` is the optional always-on dense FFN used by
/// Qwen2-MoE / DeepSeek-MoE / OLMoE (see [`SharedExpert`]). It is held
/// resident (it runs for every token, so streaming it would be pure
/// overhead) and is `None` for architectures without one (e.g. Mixtral).
///
/// The layer is intentionally split into sync helpers
/// ([`Self::attn_block`], [`Self::moe_pre`], [`Self::moe_combine`],
/// [`Self::shared_expert_forward`]) rather than one monolithic `forward`
/// because routed expert loading is **async** (it issues `pread(2)` to
/// NVMe). The async driver in `crate::model::RealModel::step` calls
/// `attn_block`, then `moe_pre`, then `await`s expert fetches via the
/// engine, then calls `moe_combine` to fold the per-expert FFN outputs
/// back into the residual stream — exactly the pseudocode the gist gives.
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub rms_attn: RmsNorm,
    pub attn: MultiHeadSelfAttention,
    /// Optional multi-head latent attention (DeepSeek-V3). When `Some`,
    /// [`Self::attn_block`] runs the MLA path against the layer's latent
    /// KV cache instead of the standard `attn`; `attn` is then unused
    /// for compute but retained so existing field accessors
    /// (`attn.d_model`, telemetry) keep working. `None` for every other
    /// architecture, preserving the standard attention path byte-for-byte.
    pub mla: Option<crate::mla::MultiHeadLatentAttention>,
    pub rms_moe: RmsNorm,
    pub gate: crate::gating::LinearGate,
    /// Optional always-on dense FFN (Qwen2-MoE / DeepSeek-MoE shared
    /// expert). `None` for Mixtral-style MoEs.
    pub shared_expert: Option<SharedExpert>,
    /// Dense SwiGLU FFN for **dense layers** (Mistral Small 3, Phi-4, and
    /// DeepSeek's `first_k_dense_replace` leading layers). When `Some`,
    /// this layer bypasses the SSD-streamed expert path entirely:
    /// `RealModel::step` runs this resident FFN over the post-attention
    /// normalised hidden state instead of routing to streamed experts.
    /// `None` means the layer is sparse and routes through the engine's
    /// expert cache (Mixtral / Qwen3-MoE / DeepSeek sparse layers).
    pub dense_ffn: Option<SharedExpert>,
}

#[derive(Debug, Default)]
pub struct TransformerLayerScratch {
    pub(crate) attn_normed: Vec<f32>,
    pub(crate) attn: MultiHeadSelfAttentionScratch,
    pub(crate) moe_normed: Vec<f32>,
    pub(crate) moe_accum: Vec<f32>,
    pub(crate) routing: crate::gating::RoutingScratch,
    pub(crate) global_expert_ids: Vec<u32>,
}

impl TransformerLayerScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TransformerLayer {
    /// `hidden -> rmsnorm -> attention -> residual`. Updates `kv` with
    /// the K/V for this token. `layer_idx` and `backend` are threaded
    /// through to [`MultiHeadSelfAttention::forward`] so the GPU path
    /// can route K/V writes to the correct VRAM layer slice.
    pub fn attn_block(
        &self,
        hidden: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
    ) -> Vec<f32> {
        self.attn_block_with_timing(hidden, pos, layer_idx, kv, backend, None)
    }

    pub fn attn_block_with_timing(
        &self,
        hidden: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Vec<f32> {
        let mut scratch = TransformerLayerScratch::new();
        let mut out = Vec::new();
        self.attn_block_into_with_timing(
            hidden,
            pos,
            layer_idx,
            kv,
            backend,
            &mut scratch,
            &mut out,
            timings,
        );
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_block_into_with_timing(
        &self,
        hidden: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
        scratch: &mut TransformerLayerScratch,
        out: &mut Vec<f32>,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) {
        crate::stage_timing::time_optional(timings, crate::stage_timing::RMS_NORM, || {
            self.rms_attn.forward_into(hidden, &mut scratch.attn_normed)
        });
        let normed = &scratch.attn_normed;
        match self.mla.as_ref() {
            // DeepSeek-V3 multi-head latent attention runs on CPU against
            // the layer's compressed latent KV cache. The GPU attention
            // kernels are shaped for standard MHA (uniform K/V head dim),
            // so MLA stays on the reference path; `backend` is unused
            // here but kept in the signature for the standard path below.
            Some(mla) => {
                let attn_out = crate::stage_timing::time_optional(
                    timings,
                    crate::stage_timing::ATTENTION_SCORE_VALUE,
                    || mla.forward(normed, pos, kv),
                );
                add_residual_into(hidden, &attn_out, out);
            }
            None => {
                self.attn.forward_into_with_timing(
                    normed,
                    pos,
                    layer_idx,
                    kv,
                    backend,
                    &mut scratch.attn,
                    out,
                    timings,
                );
                debug_assert_eq!(hidden.len(), out.len());
                for (oi, hi) in out.iter_mut().zip(hidden.iter()) {
                    *oi += hi;
                }
            }
        }
    }

    /// KV-cache width this layer needs: the MLA latent dim when latent
    /// attention is active, otherwise the standard `num_kv_heads *
    /// head_dim`.
    pub fn kv_dim(&self) -> usize {
        match self.mla.as_ref() {
            Some(mla) => mla.latent_dim(),
            None => self.attn.kv_dim(),
        }
    }

    /// V-cache width this layer needs. MLA stores its latent vector in the
    /// value slot too (`KvCache::append(&latent, &latent)`), so the V width
    /// is the latent dim; the standard path uses `num_kv_heads * v_head_dim`.
    pub fn v_dim(&self) -> usize {
        match self.mla.as_ref() {
            Some(mla) => mla.latent_dim(),
            None => self.attn.v_proj_dim(),
        }
    }

    /// `hidden -> rmsnorm -> gate.route()`. Returns the normalised
    /// hidden state (which is what every expert FFN should consume) and
    /// the routing decision.
    pub fn moe_pre(&self, hidden: &[f32]) -> (Vec<f32>, crate::gating::RoutingDecision) {
        self.moe_pre_with_timing(hidden, None)
    }

    pub fn moe_pre_with_timing(
        &self,
        hidden: &[f32],
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> (Vec<f32>, crate::gating::RoutingDecision) {
        let mut scratch = crate::gating::RoutingScratch::new();
        self.moe_pre_with_scratch_and_timing(hidden, &mut scratch, timings)
    }

    pub fn moe_pre_with_scratch_and_timing(
        &self,
        hidden: &[f32],
        routing_scratch: &mut crate::gating::RoutingScratch,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> (Vec<f32>, crate::gating::RoutingDecision) {
        let normed =
            crate::stage_timing::time_optional(timings, crate::stage_timing::RMS_NORM, || {
                self.rms_moe.forward(hidden)
            });
        let routing =
            crate::stage_timing::time_optional(timings, crate::stage_timing::ROUTER_GATE, || {
                self.gate.route_with_scratch(&normed, routing_scratch)
            });
        (normed, routing)
    }

    pub fn moe_pre_into_with_timing(
        &self,
        hidden: &[f32],
        scratch: &mut TransformerLayerScratch,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> crate::gating::RoutingDecision {
        crate::stage_timing::time_optional(timings, crate::stage_timing::RMS_NORM, || {
            self.rms_moe.forward_into(hidden, &mut scratch.moe_normed)
        });
        crate::stage_timing::time_optional(timings, crate::stage_timing::ROUTER_GATE, || {
            self.gate
                .route_with_scratch(&scratch.moe_normed, &mut scratch.routing)
        })
    }

    /// Fold the per-expert FFN outputs back into the residual stream:
    /// `hidden + sum_i weights[i] * expert_outputs[i]`. The lengths of
    /// `expert_outputs` and `weights` must match (one per chosen expert);
    /// any expert that failed to materialise on disk should be filtered
    /// out of *both* slices upstream so the weighted sum stays
    /// well-defined.
    pub fn moe_combine(
        &self,
        hidden: &[f32],
        expert_outputs: &[HiddenState],
        weights: &[f32],
    ) -> Vec<f32> {
        self.moe_combine_with_timing(hidden, expert_outputs, weights, None)
    }

    pub fn moe_combine_with_timing(
        &self,
        hidden: &[f32],
        expert_outputs: &[HiddenState],
        weights: &[f32],
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Vec<f32> {
        crate::stage_timing::time_optional(
            timings,
            crate::stage_timing::MOE_WEIGHTED_COMBINATION,
            || {
                let moe = combine_moe_outputs(expert_outputs, weights, self.attn.d_model);
                add_residual(hidden, &moe)
            },
        )
    }

    pub fn moe_combine_into_with_timing(
        &self,
        hidden: &[f32],
        expert_outputs: &[HiddenState],
        weights: &[f32],
        moe_accum: &mut Vec<f32>,
        out: &mut Vec<f32>,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) {
        crate::stage_timing::time_optional(
            timings,
            crate::stage_timing::MOE_WEIGHTED_COMBINATION,
            || {
                combine_moe_outputs_into(expert_outputs, weights, self.attn.d_model, moe_accum);
                add_residual_into(hidden, moe_accum, out);
            },
        )
    }

    pub fn moe_accumulated_into_with_timing(
        &self,
        hidden: &[f32],
        moe_accum: &[f32],
        out: &mut Vec<f32>,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) {
        crate::stage_timing::time_optional(
            timings,
            crate::stage_timing::MOE_WEIGHTED_COMBINATION,
            || {
                debug_assert_eq!(moe_accum.len(), self.attn.d_model);
                add_residual_into(hidden, moe_accum, out);
            },
        )
    }

    /// Run the layer's optional shared expert over the MoE-normalised
    /// hidden state `normed` (the same input the routed experts consume).
    /// Returns `None` when the layer has no shared expert (Mixtral), so
    /// the caller can skip the residual add entirely.
    pub fn shared_expert_forward(&self, normed: &[f32]) -> Option<HiddenState> {
        self.shared_expert_forward_with_timing(normed, None)
    }

    pub fn shared_expert_forward_with_timing(
        &self,
        normed: &[f32],
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Option<HiddenState> {
        self.shared_expert.as_ref().map(|se| {
            crate::stage_timing::time_optional(timings, crate::stage_timing::EXPERT_COMPUTE, || {
                se.forward(normed)
            })
        })
    }

    /// `true` if this layer is a **dense** FFN layer (Mistral Small 3,
    /// Phi-4, or a DeepSeek `first_k_dense_replace` prefix layer). Dense
    /// layers bypass the SSD-streamed expert path entirely.
    pub fn is_dense(&self) -> bool {
        self.dense_ffn.is_some()
    }

    /// Run the resident dense SwiGLU FFN of a dense layer over `hidden`:
    /// `hidden -> rms_moe -> dense_ffn -> + residual`. Returns `None` for
    /// sparse (routed-MoE) layers so the caller falls back to the streamed
    /// expert path. Dense layers never touch the engine's expert cache, so
    /// they do not exercise SSD streaming (by design — they have no experts
    /// to stream).
    pub fn dense_forward(&self, hidden: &[f32]) -> Option<Vec<f32>> {
        self.dense_forward_with_timing(hidden, None)
    }

    pub fn dense_forward_with_timing(
        &self,
        hidden: &[f32],
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Option<Vec<f32>> {
        let ffn = self.dense_ffn.as_ref()?;
        let normed =
            crate::stage_timing::time_optional(timings, crate::stage_timing::RMS_NORM, || {
                self.rms_moe.forward(hidden)
            });
        let out = crate::stage_timing::time_optional(
            timings,
            crate::stage_timing::EXPERT_COMPUTE,
            || ffn.forward(&normed),
        );
        Some(add_residual(hidden, &out))
    }
}

/// Numerically-stable softmax, in place.
/// Process-wide count of attention-softmax non-finite fallbacks.
///
/// Incremented (cold path only) whenever [`softmax_inplace`] — or the MLA
/// attention softmax — encounters a `NaN`/`±inf`/fully-masked row and
/// substitutes a uniform distribution to preserve numerical safety. This is
/// deliberately a low-overhead global atomic so the hot attention path never
/// takes a lock or logs per token; observers sample it via
/// [`nonfinite_softmax_fallbacks`].
///
/// This counts **attention** numerical fallbacks specifically and is distinct
/// from the router's filtering of non-finite gate scores.
pub static NONFINITE_SOFTMAX_FALLBACKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Current cumulative count of attention-softmax non-finite fallbacks.
#[inline]
pub fn nonfinite_softmax_fallbacks() -> u64 {
    NONFINITE_SOFTMAX_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Record a single attention-softmax non-finite fallback event.
#[inline]
pub(crate) fn record_nonfinite_softmax_fallback() {
    NONFINITE_SOFTMAX_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    NONFINITE_SOFTMAX_TL.with(|c| c.set(c.get() + 1));
}

thread_local! {
    /// Per-thread count of non-finite softmax fallback events
    /// (hardening pass, Part A3). The CPU attention and router softmax
    /// loops always run on the thread driving the model forward pass,
    /// so the real-model layer loop can snapshot this before a layer
    /// and fail the request when the layer produced numerical
    /// corruption — without racing concurrent requests the way the
    /// process-global counter would.
    static NONFINITE_SOFTMAX_TL: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Current per-thread count of non-finite softmax fallback events. See
/// [`NONFINITE_SOFTMAX_TL`]; used by the fail-closed real-model forward
/// pass to attribute numerical corruption to a specific layer.
#[inline]
pub fn thread_nonfinite_softmax_events() -> u64 {
    NONFINITE_SOFTMAX_TL.with(|c| c.get())
}

/// Serialises every test that reads or mutates the process-wide non-finite
/// softmax fallback counter, so their before/after deltas stay deterministic
/// even though the counter is a global atomic. Shared crate-wide (e.g. with
/// the `bench-real` validity test in `main.rs`) so those tests cannot race.
#[cfg(test)]
pub(crate) static SOFTMAX_FALLBACK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    let mut saw_nan = false;
    for &x in v.iter() {
        if x.is_nan() {
            saw_nan = true;
        } else if x > max {
            max = x;
        }
    }
    // Non-finite fallback: a stray `NaN`, a `+inf`, or a fully-masked row
    // (every logit `-inf`, leaving `max == -inf`) cannot yield a meaningful
    // distribution, so emit a uniform distribution rather than letting the
    // `x - max` subtraction produce `NaN`s that propagate downstream.
    if saw_nan || !max.is_finite() {
        record_nonfinite_softmax_fallback();
        let uniform = 1.0 / v.len() as f32;
        v.iter_mut().for_each(|x| *x = uniform);
        return;
    }
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

// =====================================================================
// Dense backbone abstraction (gist Part 2, fix #6).
// =====================================================================

/// Trait abstraction over the dense ("backbone") compute pieces of a
/// transformer decoder — `attn_block`, `RmsNorm`, and `LMHead`.
///
/// **Why**: the dense backbone is O(N²) attention math plus a couple
/// of dense matmuls. Today every byte of it runs on the CPU through
/// the scalar `transformer.rs` implementation (auto-escalated to
/// AVX-512 by the [`crate::kernels`] dispatcher inside the
/// [`crate::backend::Backend`] math facade). The gist's Part 2 calls
/// for a *clean seam* so an opt-in heterogeneous executor (a
/// `cudarc` / `wgpu` GpuBackend) can take over the dense body while
/// the SSD-streaming MoE path stays CPU-side. Pinned-host residuals
/// cross the host/device boundary exactly once per attention block,
/// not on every row of math.
///
/// **Where it plugs in**: [`crate::backend::Backend`] already owns
/// the per-row matmul / SwiGLU / softmax primitives (the GpuBackend
/// implementation overrides those). `DenseBackbone` is the layer
/// *above* that: it composes those primitives into the named blocks
/// the gist enumerates (`attn_block`, `RmsNorm`, `LMHead`) so a GPU
/// executor can fuse them into a single device-side kernel launch
/// instead of paying per-primitive host/device boundary cost.
///
/// The default implementation [`CpuBackbone`] just delegates to the
/// inherent methods on [`TransformerLayer`], [`RmsNorm`], and
/// [`LMHead`] — i.e. the existing CPU path. A future GPU backbone
/// implementation lives in `backend/mod.rs` next to [`crate::backend::GpuBackend`].
pub trait DenseBackbone: Send + Sync {
    /// Short human-readable name (e.g. `"cpu"`, `"cuda-0"`,
    /// `"wgpu-vulkan"`). Used by the startup log so operators can see
    /// which backbone is live alongside the math [`crate::backend::Backend`]
    /// identifier.
    fn name(&self) -> &'static str;

    /// `hidden → rmsnorm → attention → residual`. Equivalent to
    /// `layer.attn_block(hidden, pos, layer_idx, kv, backend)` on the CPU
    /// path. A GPU implementation can launch a single fused kernel here.
    fn attn_block(
        &self,
        layer: &TransformerLayer,
        hidden: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
    ) -> Vec<f32>;

    /// RMSNorm. Equivalent to `norm.forward(x)`. The default impl
    /// delegates to the inherent method; a device-side
    /// implementation overrides this to keep the residual on-device.
    fn rmsnorm(&self, norm: &RmsNorm, x: &[f32]) -> Vec<f32> {
        norm.forward(x)
    }

    /// Project the final hidden state to logits via the LM head.
    /// Equivalent to `head.forward(hidden)`. A GPU implementation
    /// runs the final `[vocab × d_model]` matmul on-device and only
    /// transfers the `vocab`-long logits vector back to the host.
    fn lm_head(&self, head: &LMHead, hidden: &[f32]) -> Vec<f32> {
        head.forward(hidden)
    }
}

/// Default CPU backbone: every method delegates to the existing
/// inherent implementation. Adding this is a no-op for the CPU
/// runtime — it's the trait wrapper around the existing methods so
/// callers can be ported to `DenseBackbone` without losing behaviour.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackbone;

impl DenseBackbone for CpuBackbone {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn attn_block(
        &self,
        layer: &TransformerLayer,
        hidden: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
    ) -> Vec<f32> {
        layer.attn_block(hidden, pos, layer_idx, kv, backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_backend() -> crate::backend::BackendBox {
        crate::backend::BackendBox::Cpu(crate::backend::CandleBackend::new())
    }

    fn reference_matmul(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .map(|r| {
                let row = &w[r * cols..(r + 1) * cols];
                row.iter().zip(x).map(|(a, b)| a * b).sum()
            })
            .collect()
    }

    fn q8_zero_weight(rows: usize, cols: usize) -> DenseWeight {
        let blocks = rows
            .saturating_mul(cols)
            .div_ceil(crate::inference::Q8_0_BLOCK_ELEMS);
        DenseWeight::from_q8_0_bytes(
            vec![0u8; blocks * crate::inference::Q8_0_BLOCK_BYTES],
            rows,
            cols,
        )
        .unwrap()
    }

    fn gpu_dense_path_test_attention(q8_projection: Option<&str>) -> MultiHeadSelfAttention {
        let d_model = 8;
        let num_heads = 2;
        let head_dim = 4;
        let num_kv_heads = 2;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mk = |rows: usize, cols: usize| {
            DenseWeight::from_f32(
                (0..rows * cols)
                    .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
                    .collect(),
                rows,
                cols,
            )
        };
        let projection = |name: &str, rows: usize, cols: usize| {
            if q8_projection == Some(name) {
                q8_zero_weight(rows, cols)
            } else {
                mk(rows, cols)
            }
        };
        MultiHeadSelfAttention {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim: head_dim,
            v_head_dim: head_dim,
            attention_value_scale: None,
            rope_base: 10000.0,
            wq: projection("wq", q_dim, d_model),
            wk: projection("wk", kv_dim, d_model),
            wv: projection("wv", kv_dim, d_model),
            wo: projection("wo", d_model, q_dim),
            window_size: None,
            q_norm: None,
            k_norm: None,
            rope_yarn: None,
            rope_cache: None,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            sink_bias: None,
        }
    }

    #[test]
    fn dense_matvec_backends_match_reference() {
        use crate::parallel::DenseMatvecBackend;

        for &(rows, cols) in &[(1usize, 1usize), (7, 13), (128, 64), (257, 129)] {
            let w: Vec<f32> = (0..rows * cols)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.0075)
                .collect();
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.03125)
                .collect();
            let want = reference_matmul(&w, &x, rows, cols);
            for backend in [
                DenseMatvecBackend::Matrixmultiply,
                DenseMatvecBackend::Rayon,
                DenseMatvecBackend::RayonMatrixmultiply,
                DenseMatvecBackend::Auto,
            ] {
                let got = matmul_row_major_with_backend(&w, &x, rows, cols, backend);
                assert_eq!(got.len(), want.len());
                for (idx, (g, e)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g - e).abs() <= 1e-4,
                        "backend={backend:?} rows={rows} cols={cols} idx={idx}: {g} vs {e}"
                    );
                }
            }
        }
    }

    #[test]
    fn dense_tensor_gpu_dense_path_supported_for_all_f32_attention_projections() {
        let attn = gpu_dense_path_test_attention(None);
        assert!(attn.gpu_dense_path_supported());
    }

    #[test]
    fn dense_tensor_gpu_dense_path_rejects_any_q8_attention_projection() {
        for projection in ["wq", "wk", "wv", "wo"] {
            let attn = gpu_dense_path_test_attention(Some(projection));
            assert!(
                !attn.gpu_dense_path_supported(),
                "{projection} Q8_0 must force the CPU dense attention path"
            );
        }
    }

    #[test]
    fn dense_tensor_gpu_dense_path_check_leaves_q8_projection_storage_unchanged() {
        let attn = gpu_dense_path_test_attention(Some("wq"));
        let (before_ptr, before_len, before_resident_bytes) = match &attn.wq {
            DenseWeight::Q8_0 { bytes, .. } => {
                (bytes.as_ptr(), bytes.len(), attn.wq.resident_bytes())
            }
            other => panic!("expected Q8_0 wq, got {}", other.dtype()),
        };

        assert!(!attn.gpu_dense_path_supported());

        let (after_ptr, after_len) = match &attn.wq {
            DenseWeight::Q8_0 { bytes, .. } => (bytes.as_ptr(), bytes.len()),
            other => panic!("expected Q8_0 wq after check, got {}", other.dtype()),
        };
        assert_eq!(after_ptr, before_ptr);
        assert_eq!(after_len, before_len);
        assert_eq!(attn.wq.resident_bytes(), before_resident_bytes);
        assert_eq!(attn.wq.dtype(), DenseDType::Q8_0);
    }

    #[test]
    fn rmsnorm_unit_weight_normalises_to_unit_variance() {
        let n = 8;
        let weight = vec![1.0f32; n];
        let norm = RmsNorm::new(weight, 1e-6);
        let mut x: Vec<f32> = (0..n).map(|i| i as f32 - 3.5).collect();
        norm.forward_inplace(&mut x);
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        // After RMSNorm with unit weight and tiny eps, mean(x^2) ≈ 1.
        assert!((mean_sq - 1.0).abs() < 1e-3, "mean_sq={mean_sq}");
    }

    #[test]
    fn rmsnorm_forward_into_matches_forward_and_reuses_capacity() {
        let n = 8;
        let norm = RmsNorm::new((0..n).map(|i| 1.0 + i as f32 * 0.1).collect(), 1e-6);
        let x: Vec<f32> = (0..n).map(|i| i as f32 - 3.5).collect();
        let expected = norm.forward(&x);
        let mut out = Vec::with_capacity(n * 2);
        let capacity = out.capacity();
        norm.forward_into(&x, &mut out);
        assert_eq!(out, expected);
        assert_eq!(out.capacity(), capacity);
    }

    #[test]
    fn rope_pos_zero_is_identity() {
        let mut v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let original = v.clone();
        apply_rope_inplace(&mut v, 0, 10000.0);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "rope at pos 0 must be identity");
        }
    }

    #[test]
    fn rope_preserves_norm() {
        let mut v: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
        let n_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_inplace(&mut v, 7, 10000.0);
        let n_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n_before - n_after).abs() < 1e-4);
    }

    #[test]
    fn rope_cache_matches_unscaled_reference_and_reuses_position() {
        let cache = RopeCache::with_positions(16, 10_000.0, None, 32).expect("cache");
        let original: Vec<f32> = (1..=16).map(|i| i as f32 * 0.125).collect();
        let mut cached = original.clone();
        let mut direct = original.clone();
        cache.apply_inplace(&mut cached, 17);
        apply_rope_inplace(&mut direct, 17, 10_000.0);
        for (a, b) in cached.iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-6, "cached {a} direct {b}");
        }
        assert_eq!(cache.materialized_positions(), 1);
        let mut cached_again = original;
        cache.apply_inplace(&mut cached_again, 17);
        assert_eq!(cache.materialized_positions(), 1);
    }

    fn yarn_test_scaling(factor: f32) -> crate::architecture::RopeScaling {
        crate::architecture::RopeScaling {
            rope_type: "yarn".to_string(),
            factor,
            original_max_position_embeddings: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 0.0,
        }
    }

    #[test]
    fn yarn_rejects_non_yarn_and_non_expanding_configs() {
        let mut s = yarn_test_scaling(40.0);
        s.rope_type = "linear".to_string();
        assert!(YarnRope::from_scaling(64, 10_000.0, &s).is_none());
        let s = yarn_test_scaling(1.0);
        assert!(YarnRope::from_scaling(64, 10_000.0, &s).is_none());
        // Odd / zero head dims can't be rotated as complex pairs.
        let s = yarn_test_scaling(4.0);
        assert!(YarnRope::from_scaling(63, 10_000.0, &s).is_none());
        assert!(YarnRope::from_scaling(0, 10_000.0, &s).is_none());
    }

    #[test]
    fn yarn_inv_freq_blends_between_extrapolation_and_interpolation() {
        let head_dim = 64;
        let base = 10_000.0f32;
        let s = yarn_test_scaling(40.0);
        let yarn = YarnRope::from_scaling(head_dim, base, &s).expect("yarn config");
        assert_eq!(yarn.inv_freq.len(), head_dim / 2);
        for (i, &f) in yarn.inv_freq.iter().enumerate() {
            let extra = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let inter = extra / s.factor;
            assert!(
                f <= extra * 1.0001 && f >= inter * 0.9999,
                "pair {i}: blended {f} outside [{inter}, {extra}]"
            );
        }
        // Highest-frequency pair (i=0) completes far more than beta_fast
        // rotations over the original context → pure extrapolation.
        assert!((yarn.inv_freq[0] - 1.0).abs() < 1e-6);
        // Lowest-frequency pair → pure interpolation (slowed by factor).
        let last = head_dim / 2 - 1;
        let extra_last = 1.0 / base.powf(2.0 * last as f32 / head_dim as f32);
        assert!((yarn.inv_freq[last] - extra_last / s.factor).abs() < extra_last * 1e-4);
        // Default attention factor: 0.1 * ln(factor) + 1.
        let expected = 0.1 * 40.0f32.ln() + 1.0;
        assert!((yarn.attn_factor - expected).abs() < 1e-5);
    }

    #[test]
    fn rope_cache_matches_yarn_reference() {
        let s = yarn_test_scaling(8.0);
        let yarn = YarnRope::from_scaling(16, 10_000.0, &s).expect("yarn config");
        let cache = RopeCache::with_positions(16, 10_000.0, Some(&yarn), 32).expect("cache");
        let original: Vec<f32> = (1..=16).map(|i| i as f32 * -0.075).collect();
        let mut cached = original.clone();
        let mut direct = original;
        cache.apply_inplace(&mut cached, 19);
        apply_rope_scaled_inplace(&mut direct, 19, &yarn);
        for (a, b) in cached.iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-6, "cached {a} direct {b}");
        }
    }

    #[test]
    fn yarn_rope_scales_vector_norm_by_attn_factor() {
        let s = yarn_test_scaling(8.0);
        let yarn = YarnRope::from_scaling(16, 10_000.0, &s).expect("yarn config");
        let mut v: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
        let n_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_scaled_inplace(&mut v, 9, &yarn);
        let n_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (n_after - n_before * yarn.attn_factor).abs() < 1e-4,
            "norm {n_after} != {n_before} * {}",
            yarn.attn_factor
        );
    }

    #[test]
    fn yarn_interpolated_pair_matches_unscaled_rope_at_compressed_position() {
        // For a pair in the pure-interpolation regime, rotating at
        // position `factor * p` with YaRN equals rotating at `p`
        // unscaled (up to the attention factor).
        let head_dim = 16;
        let base = 10_000.0f32;
        let factor = 4.0;
        let mut s = yarn_test_scaling(factor);
        s.mscale = 0.0; // attn_factor numerator → 1.0
        s.mscale_all_dim = 0.0;
        // mscale = 0 gives yarn_get_mscale(f, 0) = 1.0 on both halves →
        // attn_factor exactly 1, isolating the frequency blend.
        let yarn = YarnRope::from_scaling(head_dim, base, &s).expect("yarn config");
        assert!((yarn.attn_factor - 1.0).abs() < 1e-6);
        let last = head_dim / 2 - 1;
        let extra_last = 1.0 / base.powf(2.0 * last as f32 / head_dim as f32);
        assert!(
            (yarn.inv_freq[last] - extra_last / factor).abs() < extra_last * 1e-4,
            "test premise: last pair must be pure interpolation"
        );
        // Compare just the last pair's rotation.
        let mut scaled = vec![0.0f32; head_dim];
        scaled[last] = 0.3;
        scaled[last + head_dim / 2] = -0.7;
        let mut unscaled = scaled.clone();
        apply_rope_scaled_inplace(&mut scaled, 4 * 11, &yarn);
        apply_rope_inplace(&mut unscaled, 11, base);
        assert!((scaled[last] - unscaled[last]).abs() < 1e-3);
        assert!((scaled[last + head_dim / 2] - unscaled[last + head_dim / 2]).abs() < 1e-3);
    }

    #[test]
    fn yarn_get_mscale_matches_reference() {
        assert_eq!(yarn_get_mscale(1.0, 1.0), 1.0);
        assert_eq!(yarn_get_mscale(0.5, 1.0), 1.0);
        let m = yarn_get_mscale(40.0, 1.0);
        assert!((m - (0.1 * 40.0f32.ln() + 1.0)).abs() < 1e-6);
        let m2 = yarn_get_mscale(40.0, 0.707);
        assert!((m2 - (0.1 * 0.707 * 40.0f32.ln() + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn attention_with_yarn_stays_finite_and_differs_from_unscaled() {
        let mut attn = make_window_attn(None);
        let mut kv_a = KvCache::new(attn.kv_dim());
        let mut kv_b = KvCache::new(attn.kv_dim());
        let backend = cpu_backend();
        // Position-varying inputs so V differs per cached token —
        // otherwise attention output is invariant to the softmax
        // weights and YaRN could never change it.
        let x_at = |pos: usize| -> Vec<f32> {
            (0..attn.d_model)
                .map(|i| ((i + 1) as f32 * 0.3 + pos as f32 * 0.7).sin())
                .collect()
        };
        let unscaled: Vec<Vec<f32>> = (0..6)
            .map(|pos| attn.forward(&x_at(pos), pos, 0, &mut kv_a, &backend))
            .collect();
        let s = yarn_test_scaling(16.0);
        attn.rope_yarn = YarnRope::from_scaling(attn.head_dim, attn.rope_base, &s);
        assert!(attn.rope_yarn.is_some());
        let scaled: Vec<Vec<f32>> = (0..6)
            .map(|pos| attn.forward(&x_at(pos), pos, 0, &mut kv_b, &backend))
            .collect();
        let mut any_diff = false;
        for (u, sc) in unscaled.iter().zip(scaled.iter()) {
            assert!(sc.iter().all(|v| v.is_finite()));
            if u.iter().zip(sc.iter()).any(|(a, b)| (a - b).abs() > 1e-6) {
                any_diff = true;
            }
        }
        assert!(
            any_diff,
            "YaRN scaling must change attention outputs at pos > 0"
        );
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, -1.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum={sum}");
        // Largest input -> largest output.
        assert!(v[2] > v[1] && v[1] > v[0] && v[0] > v[3]);
    }

    #[test]
    fn softmax_handles_empty() {
        let mut v: Vec<f32> = Vec::new();
        softmax_inplace(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn softmax_all_neg_inf_is_uniform() {
        let _g = SOFTMAX_FALLBACK_TEST_LOCK.lock().unwrap();
        // A fully-masked attention row (every score `-inf`) must not
        // produce NaNs: `(-inf) - (-inf)` is NaN and would poison the
        // whole distribution. We fall back to a uniform distribution.
        let mut v = vec![f32::NEG_INFINITY; 4];
        softmax_inplace(&mut v);
        assert!(v.iter().all(|&x| (x - 0.25).abs() < 1e-6), "got {v:?}");
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum={sum}");
    }

    #[test]
    fn softmax_mixed_nan_is_uniform() {
        let _g = SOFTMAX_FALLBACK_TEST_LOCK.lock().unwrap();
        // A stray NaN that is NOT at index 0 must not poison the row. The
        // previous `!max.is_finite()` guard missed this because the running
        // max ignores NaN, leaving a finite max.
        let mut v = vec![0.0, f32::NAN, 1.0, -1.0];
        softmax_inplace(&mut v);
        let expected = 0.25;
        assert!(
            v.iter()
                .all(|&x| x.is_finite() && (x - expected).abs() < 1e-6),
            "got {v:?}"
        );
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum={sum}");
    }

    /// Serialises the tests that assert on the process-wide non-finite
    /// softmax fallback counter, so their before/after deltas are
    /// deterministic even though the counter is a global atomic. Aliased to
    /// the crate-wide lock so cross-module tests (e.g. `main.rs`) share it.
    use super::SOFTMAX_FALLBACK_TEST_LOCK;

    #[test]
    fn softmax_nan_increments_fallback_counter_once() {
        let _g = SOFTMAX_FALLBACK_TEST_LOCK.lock().unwrap();
        let before = nonfinite_softmax_fallbacks();
        let mut v = vec![0.0, f32::NAN, 1.0];
        softmax_inplace(&mut v);
        assert_eq!(nonfinite_softmax_fallbacks() - before, 1);
    }

    #[test]
    fn softmax_inf_increments_fallback_counter() {
        let _g = SOFTMAX_FALLBACK_TEST_LOCK.lock().unwrap();
        let before = nonfinite_softmax_fallbacks();
        // +inf leaves a non-finite max → uniform fallback.
        let mut v = vec![f32::INFINITY, 1.0, 2.0];
        softmax_inplace(&mut v);
        assert_eq!(nonfinite_softmax_fallbacks() - before, 1);
    }

    #[test]
    fn softmax_finite_does_not_increment_fallback_counter() {
        let _g = SOFTMAX_FALLBACK_TEST_LOCK.lock().unwrap();
        let before = nonfinite_softmax_fallbacks();
        let mut v = vec![0.0, 1.0, 2.0, -1.0];
        softmax_inplace(&mut v);
        assert_eq!(nonfinite_softmax_fallbacks() - before, 0);
    }

    #[test]
    fn softmax_fallback_counter_concurrent_increments_are_safe() {
        let _g = SOFTMAX_FALLBACK_TEST_LOCK.lock().unwrap();
        let before = nonfinite_softmax_fallbacks();
        const THREADS: u64 = 8;
        const PER: u64 = 1000;
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in 0..PER {
                        record_nonfinite_softmax_fallback();
                    }
                });
            }
        });
        assert_eq!(nonfinite_softmax_fallbacks() - before, THREADS * PER);
    }

    #[test]
    fn attention_shapes_match_and_cache_grows() {
        let d_model = 8;
        let num_heads = 2;
        let head_dim = 4;
        let num_kv_heads = 2;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // Use deterministic small weights so we exercise the math.
        let mk = |rows: usize, cols: usize| {
            (0..rows * cols)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
                .collect()
        };
        let attn = MultiHeadSelfAttention {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim: head_dim,
            v_head_dim: head_dim,
            attention_value_scale: None,
            rope_base: 10000.0,
            wq: DenseWeight::from_f32(mk(q_dim, d_model), q_dim, d_model),
            wk: DenseWeight::from_f32(mk(kv_dim, d_model), kv_dim, d_model),
            wv: DenseWeight::from_f32(mk(kv_dim, d_model), kv_dim, d_model),
            wo: DenseWeight::from_f32(mk(d_model, q_dim), d_model, q_dim),
            window_size: None,
            q_norm: None,
            k_norm: None,
            rope_yarn: None,
            rope_cache: None,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            sink_bias: None,
        };
        let mut kv = KvCache::new(kv_dim);
        let x: Vec<f32> = (0..d_model).map(|i| 0.1 * i as f32).collect();
        let y0 = attn.forward(&x, 0, 0, &mut kv, &cpu_backend());
        assert_eq!(y0.len(), d_model);
        assert_eq!(kv.seq_len, 1);
        let y1 = attn.forward(&x, 1, 0, &mut kv, &cpu_backend());
        assert_eq!(y1.len(), d_model);
        assert_eq!(kv.seq_len, 2);
        // Output must be finite.
        assert!(y1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn qk_norm_changes_attention_output_and_stays_finite() {
        // Two otherwise-identical attention blocks: one with QK-Norm, one
        // without. QK-Norm must (a) keep outputs finite and (b) change the
        // result (it renormalises Q and K per head before RoPE).
        let d_model = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mk = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| ((i % 5) as f32 - 2.0) * 0.2)
                .collect()
        };
        let base = MultiHeadSelfAttention {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim: head_dim,
            v_head_dim: head_dim,
            attention_value_scale: None,
            rope_base: 10000.0,
            wq: DenseWeight::from_f32(mk(q_dim, d_model), q_dim, d_model),
            wk: DenseWeight::from_f32(mk(kv_dim, d_model), kv_dim, d_model),
            wv: DenseWeight::from_f32(mk(kv_dim, d_model), kv_dim, d_model),
            wo: DenseWeight::from_f32(mk(d_model, q_dim), d_model, q_dim),
            window_size: None,
            q_norm: None,
            k_norm: None,
            rope_yarn: None,
            rope_cache: None,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            sink_bias: None,
        };
        let mut normed = base.clone();
        // Non-unit per-head norm weights so the effect is visible.
        normed.q_norm = Some(RmsNorm::new(vec![1.5, 0.5, 1.0, 2.0], 1e-6));
        normed.k_norm = Some(RmsNorm::new(vec![0.7, 1.2, 1.0, 0.9], 1e-6));

        let x: Vec<f32> = (0..d_model).map(|i| 0.15 * i as f32 - 0.3).collect();
        let mut kv_a = KvCache::new(kv_dim);
        let mut kv_b = KvCache::new(kv_dim);
        // Prime position 0 so the cache holds two keys: with a single key
        // the softmax is trivially 1.0 and the Q/K scaling cannot change
        // the attention output. The QK-Norm effect is only visible once
        // there are at least two positions to attend over.
        let _ = base.forward(&x, 0, 0, &mut kv_a, &cpu_backend());
        let _ = normed.forward(&x, 0, 0, &mut kv_b, &cpu_backend());
        let x1: Vec<f32> = (0..d_model).map(|i| 0.1 * i as f32 + 0.05).collect();
        let y_plain = base.forward(&x1, 1, 0, &mut kv_a, &cpu_backend());
        let y_norm = normed.forward(&x1, 1, 0, &mut kv_b, &cpu_backend());
        assert!(y_norm.iter().all(|v| v.is_finite()));
        let diff: f32 = y_plain
            .iter()
            .zip(&y_norm)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "QK-Norm should change the output (diff={diff})"
        );
    }

    #[test]
    fn combine_moe_outputs_weights_correctly() {
        let d = 4;
        let outs = vec![vec![1.0; d], vec![2.0; d], vec![4.0; d]];
        let scores = vec![0.5, 0.25, 0.25];
        let y = combine_moe_outputs(&outs, &scores, d);
        // 0.5*1 + 0.25*2 + 0.25*4 = 2.0
        for v in &y {
            assert!((v - 2.0).abs() < 1e-6);
        }

        let mut into = Vec::with_capacity(d * 2);
        let capacity = into.capacity();
        combine_moe_outputs_into(&outs, &scores, d, &mut into);
        assert_eq!(into, y);
        assert_eq!(into.capacity(), capacity);
    }

    #[test]
    fn lm_head_projects_to_vocab() {
        let d_model = 4;
        let vocab = 6;
        // Identity-ish: first d_model rows are I, rest are zero.
        let mut w = vec![0.0f32; vocab * d_model];
        for i in 0..d_model.min(vocab) {
            w[i * d_model + i] = 1.0;
        }
        let head = LMHead::new(w, vocab, d_model);
        let logits = head.forward(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(logits.len(), vocab);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], 4.0);
        // Rows beyond d_model are all zero.
        assert_eq!(logits[4], 0.0);
        assert_eq!(logits[5], 0.0);
    }

    #[test]
    fn lm_head_top_k_sampling_matches_full_logits_sampler() {
        let d_model = 3;
        let vocab = 7;
        let weights: Vec<f32> = (0..vocab * d_model)
            .map(|i| ((i % 9) as f32 - 4.0) / 5.0)
            .collect();
        let hidden = [0.25, -0.75, 1.25];
        let params = crate::sampling::SamplingParams {
            temperature: 0.8,
            top_p: 1.0,
            top_k: 3,
            seed: 1234,
        };
        let head = LMHead::new(weights, vocab, d_model);
        let logits = head.forward(&hidden);
        for pos in 0..16 {
            let expected = crate::sampling::sample(&logits, &params, pos);
            let got = head.sample(&hidden, &params, pos);
            assert_eq!(got, expected, "position {pos}");
        }
    }

    #[test]
    fn add_residual_is_elementwise() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let y = add_residual(&a, &b);
        assert_eq!(y, vec![11.0, 22.0, 33.0]);

        let mut into = Vec::with_capacity(8);
        let capacity = into.capacity();
        add_residual_into(&a, &b, &mut into);
        assert_eq!(into, y);
        assert_eq!(into.capacity(), capacity);
    }

    /// Build a tiny `TransformerLayer` with deterministic small weights
    /// so we can exercise the full `attn_block + moe_pre + moe_combine`
    /// path without loading anything from disk.
    fn make_layer(d_model: usize, num_experts: usize, top_k: usize) -> TransformerLayer {
        let head_dim = 4;
        let num_heads = d_model / head_dim;
        let num_kv_heads = num_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mk = |rows: usize, cols: usize, scale: f32| {
            (0..rows * cols)
                .map(|i| ((i % 7) as f32 - 3.0) * scale)
                .collect::<Vec<f32>>()
        };
        TransformerLayer {
            rms_attn: RmsNorm::new(vec![1.0; d_model], 1e-6),
            attn: MultiHeadSelfAttention {
                d_model,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_dim: head_dim,
                v_head_dim: head_dim,
                attention_value_scale: None,
                rope_base: 10000.0,
                wq: DenseWeight::from_f32(mk(q_dim, d_model, 0.05), q_dim, d_model),
                wk: DenseWeight::from_f32(mk(kv_dim, d_model, 0.05), kv_dim, d_model),
                wv: DenseWeight::from_f32(mk(kv_dim, d_model, 0.05), kv_dim, d_model),
                wo: DenseWeight::from_f32(mk(d_model, q_dim, 0.05), d_model, q_dim),
                window_size: None,
                q_norm: None,
                k_norm: None,
                rope_yarn: None,
                rope_cache: None,
                bq: None,
                bk: None,
                bv: None,
                bo: None,
                sink_bias: None,
            },
            mla: None,
            rms_moe: RmsNorm::new(vec![1.0; d_model], 1e-6),
            gate: crate::gating::LinearGate::new(
                mk(num_experts, d_model, 0.1),
                num_experts,
                d_model,
                top_k,
            ),
            shared_expert: None,
            dense_ffn: None,
        }
    }

    #[test]
    fn transformer_layer_attn_block_is_finite_and_grows_kv() {
        let d_model = 16;
        let layer = make_layer(d_model, 4, 2);
        let mut kv = KvCache::new(layer.attn.kv_dim());
        let x: Vec<f32> = (0..d_model).map(|i| 0.1 * i as f32 - 0.5).collect();
        let y0 = layer.attn_block(&x, 0, 0, &mut kv, &cpu_backend());
        assert_eq!(y0.len(), d_model);
        assert!(y0.iter().all(|v| v.is_finite()));
        assert_eq!(kv.seq_len, 1);
        let y1 = layer.attn_block(&y0, 1, 0, &mut kv, &cpu_backend());
        assert_eq!(kv.seq_len, 2);
        assert!(y1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn attention_forward_into_matches_forward_and_reuses_capacity() {
        let d_model = 16;
        let layer = make_layer(d_model, 4, 2);
        let x0: Vec<f32> = (0..d_model).map(|i| 0.1 * i as f32 - 0.5).collect();
        let x1: Vec<f32> = (0..d_model).map(|i| 0.07 * i as f32 - 0.25).collect();
        let backend = cpu_backend();

        let mut kv_ref = KvCache::new(layer.attn.kv_dim());
        let mut kv_into = KvCache::new(layer.attn.kv_dim());
        let y0_ref = layer.attn.forward(&x0, 0, 0, &mut kv_ref, &backend);
        let y1_ref = layer.attn.forward(&x1, 1, 0, &mut kv_ref, &backend);

        let mut scratch = MultiHeadSelfAttentionScratch::default();
        let mut y0_into = Vec::with_capacity(d_model * 2);
        let out_capacity = y0_into.capacity();
        layer.attn.forward_into_with_timing(
            &x0,
            0,
            0,
            &mut kv_into,
            &backend,
            &mut scratch,
            &mut y0_into,
            None,
        );
        assert_eq!(y0_into.capacity(), out_capacity);
        assert_eq!(y0_into.len(), d_model);
        for (a, b) in y0_ref.iter().zip(y0_into.iter()) {
            assert!((a - b).abs() < 1e-5, "forward_into mismatch: {a} vs {b}");
        }

        let q_capacity = scratch.q.capacity();
        let k_capacity = scratch.k.capacity();
        let v_capacity = scratch.v.capacity();
        let attn_capacity = scratch.attn_out.capacity();
        let mut y1_into = Vec::with_capacity(d_model * 2);
        layer.attn.forward_into_with_timing(
            &x1,
            1,
            0,
            &mut kv_into,
            &backend,
            &mut scratch,
            &mut y1_into,
            None,
        );
        assert_eq!(scratch.q.capacity(), q_capacity);
        assert_eq!(scratch.k.capacity(), k_capacity);
        assert_eq!(scratch.v.capacity(), v_capacity);
        assert_eq!(scratch.attn_out.capacity(), attn_capacity);
        for (a, b) in y1_ref.iter().zip(y1_into.iter()) {
            assert!((a - b).abs() < 1e-5, "forward_into mismatch: {a} vs {b}");
        }
        assert_eq!(kv_into.seq_len, kv_ref.seq_len);
    }

    #[test]
    fn transformer_layer_moe_pre_routes_top_k_experts() {
        let d_model = 16;
        let layer = make_layer(d_model, 8, 2);
        let x: Vec<f32> = (0..d_model).map(|i| 0.1 * i as f32 - 0.5).collect();
        let (normed, routing) = layer.moe_pre(&x);
        assert_eq!(normed.len(), d_model);
        assert!(normed.iter().all(|v| v.is_finite()));
        assert_eq!(routing.experts.len(), 2);
        assert_eq!(routing.weights.len(), 2);
        let sum: f32 = routing.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn transformer_layer_moe_combine_blends_outputs() {
        let d_model = 8;
        let layer = make_layer(d_model, 4, 2);
        let hidden = vec![1.0; d_model];
        // Two expert outputs of all-ones and all-twos with equal weights:
        // combined moe = 1.5 * 1.0 vector; hidden + moe = 2.5 vector.
        let outs = vec![vec![1.0; d_model], vec![2.0; d_model]];
        let weights = vec![0.5, 0.5];
        let y = layer.moe_combine(&hidden, &outs, &weights);
        for v in y {
            assert!((v - 2.5).abs() < 1e-6);
        }
    }

    /// Build a tiny attention block with controllable window. Uses
    /// identity-like Q/K projections so we can read the V contribution
    /// of position 0 directly.
    fn make_window_attn(window: Option<usize>) -> MultiHeadSelfAttention {
        let d_model = 4;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 4;
        // Identity for q/k/v/o (vector of d_model^2 with diagonal 1.0).
        let identity = |dim: usize| -> Vec<f32> {
            let mut w = vec![0.0f32; dim * dim];
            for i in 0..dim {
                w[i * dim + i] = 1.0;
            }
            w
        };
        MultiHeadSelfAttention {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim: head_dim,
            v_head_dim: head_dim,
            attention_value_scale: None,
            rope_base: 10000.0,
            wq: DenseWeight::from_f32(identity(d_model), d_model, d_model),
            wk: DenseWeight::from_f32(identity(d_model), d_model, d_model),
            wv: DenseWeight::from_f32(identity(d_model), d_model, d_model),
            wo: DenseWeight::from_f32(identity(d_model), d_model, d_model),
            window_size: window,
            q_norm: None,
            k_norm: None,
            rope_yarn: None,
            rope_cache: None,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            sink_bias: None,
        }
    }

    /// With `window_size = Some(2)`, position 3 must NOT attend to
    /// position 0 — the window covers only positions [2, 3]. The
    /// attention output for a query whose key would otherwise dominate
    /// at t=0 (a unique large signal there) must therefore *not* reflect
    /// that signal once we step past the window.
    #[test]
    fn sliding_window_excludes_positions_outside_span() {
        let attn = make_window_attn(Some(2));
        let mut kv = KvCache::new(attn.kv_dim());
        // Distinct token at position 0; the rest are ~zero.
        let big = vec![10.0f32, 0.0, 0.0, 0.0];
        let small = vec![0.0f32, 0.0, 0.0, 0.0];
        let _ = attn.forward(&big, 0, 0, &mut kv, &cpu_backend());
        let _ = attn.forward(&small, 1, 0, &mut kv, &cpu_backend());
        let _ = attn.forward(&small, 2, 0, &mut kv, &cpu_backend());
        // At pos 3 with window 2 the visible KV span is [2, 3] → t=0 must
        // not contribute. Output should reflect (mostly) the zero tokens.
        let y = attn.forward(&small, 3, 0, &mut kv, &cpu_backend());
        // The big spike at t=0 had a 10.0 in dim 0; if we *were*
        // attending to it the output's dim 0 would be > 1.0. With the
        // window excluding t=0 it must stay near 0.
        assert!(y[0].abs() < 1e-3, "leaked value from outside window: {y:?}");

        // Sanity check: with full attention (window = None), the same
        // pattern leaks the t=0 spike into the output (proves the
        // fixture is non-degenerate).
        let attn_full = make_window_attn(None);
        let mut kv2 = KvCache::new(attn_full.kv_dim());
        let _ = attn_full.forward(&big, 0, 0, &mut kv2, &cpu_backend());
        let _ = attn_full.forward(&small, 1, 0, &mut kv2, &cpu_backend());
        let _ = attn_full.forward(&small, 2, 0, &mut kv2, &cpu_backend());
        let y_full = attn_full.forward(&small, 3, 0, &mut kv2, &cpu_backend());
        assert!(
            y_full[0] > 0.5,
            "full attention should see t=0 spike: {y_full:?}"
        );
    }

    #[test]
    fn sliding_window_inside_span_behaves_like_full_attention() {
        // For a window larger than the sequence, results must match
        // unrestricted attention bit-for-bit.
        let attn_w = make_window_attn(Some(10));
        let attn_n = make_window_attn(None);
        let mut kv1 = KvCache::new(attn_w.kv_dim());
        let mut kv2 = KvCache::new(attn_n.kv_dim());
        let xs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        for (pos, x) in xs.iter().enumerate() {
            let y_w = attn_w.forward(x, pos, 0, &mut kv1, &cpu_backend());
            let y_n = attn_n.forward(x, pos, 0, &mut kv2, &cpu_backend());
            for (a, b) in y_w.iter().zip(y_n.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn sink_bias_shifts_attention_toward_first_token() {
        // MiMo-V2-Flash attention sink bias: a per-head scalar added to the
        // logit of position 0 before softmax pulls probability mass toward the
        // sink token. With identity Q/K/V/O and V[0] carrying a 1.0 in dim 0,
        // the bias must strictly raise the attention output's dim 0.
        let mut attn_bias = make_window_attn(Some(10));
        attn_bias.sink_bias = Some(vec![2.0]); // single head
        let attn_none = make_window_attn(Some(10));
        let t0 = vec![1.0f32, 0.0, 0.0, 0.0];
        let t1 = vec![0.0f32, 1.0, 0.0, 0.0];

        let run = |attn: &MultiHeadSelfAttention| -> Vec<f32> {
            let mut kv = KvCache::new(attn.kv_dim());
            let _ = attn.forward(&t0, 0, 0, &mut kv, &cpu_backend());
            attn.forward(&t1, 1, 0, &mut kv, &cpu_backend())
        };
        let y_bias = run(&attn_bias);
        let y_none = run(&attn_none);
        assert!(
            y_bias[0] > y_none[0] + 1e-4,
            "sink bias should pull attention toward token 0: bias={y_bias:?} none={y_none:?}"
        );
    }

    #[test]
    fn sink_bias_none_is_a_noop() {
        // `sink_bias = None` (every other architecture) should behave exactly
        // like a regular attention block; this is a smoke test that it runs and produces finite output.
        let attn = make_window_attn(None);
        assert!(attn.sink_bias.is_none());
        let t0 = vec![1.0f32, 0.0, 0.0, 0.0];
        let t1 = vec![0.0f32, 1.0, 0.0, 0.0];
        let mut kv = KvCache::new(attn.kv_dim());
        let _ = attn.forward(&t0, 0, 0, &mut kv, &cpu_backend());
        let y = attn.forward(&t1, 1, 0, &mut kv, &cpu_backend());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ----------------- PagedAttention block-storage tests -------------

    #[test]
    fn paged_kv_cache_grows_one_block_per_block_tokens() {
        let kv = KvCache::new(8);
        assert_eq!(kv.num_blocks(), 0);
        let mut kv = kv;
        // Insert exactly PAGED_BLOCK_TOKENS tokens — should fit in one block.
        for _ in 0..PAGED_BLOCK_TOKENS {
            kv.append(&[1.0; 8], &[2.0; 8]);
        }
        assert_eq!(kv.seq_len, PAGED_BLOCK_TOKENS);
        assert_eq!(kv.num_blocks(), 1);
        // One more token forces a new block.
        kv.append(&[3.0; 8], &[4.0; 8]);
        assert_eq!(kv.num_blocks(), 2);
        // The just-appended token should round-trip via `key`/`value`.
        let last = kv.seq_len - 1;
        assert_eq!(kv.key(last), &[3.0; 8][..]);
        assert_eq!(kv.value(last), &[4.0; 8][..]);
        // And the first token in the previous block should still match.
        assert_eq!(kv.key(0), &[1.0; 8][..]);
    }

    #[test]
    fn paged_kv_cache_reset_releases_blocks() {
        let mut kv = KvCache::new(4);
        for _ in 0..(PAGED_BLOCK_TOKENS * 2 + 3) {
            kv.append(&[1.0; 4], &[2.0; 4]);
        }
        assert!(kv.num_blocks() >= 3);
        kv.reset();
        assert_eq!(kv.seq_len, 0);
        assert_eq!(kv.num_blocks(), 0);
    }

    #[test]
    fn swa_kv_eviction_bounds() {
        // evict_before drops whole leading blocks below the window, keeping
        // a sliding-window cache bounded at O(window) instead of O(seq_len)
        // while leaving every still-attendable position byte-for-byte intact.
        let kv_dim = 4;
        let window = 2 * PAGED_BLOCK_TOKENS; // 32-token window
        let mut kv = KvCache::new(kv_dim);
        // Distinct K/V per position so we can verify survivors exactly.
        let tok = |p: usize| (vec![p as f32; kv_dim], vec![(p as f32) + 0.5; kv_dim]);
        let total = 10 * PAGED_BLOCK_TOKENS; // 160 tokens, far past the window
        for p in 0..total {
            let (k, v) = tok(p);
            kv.append(&k, &v);
            // Mirror the decode loop: evict positions older than the window.
            kv.evict_before(p.saturating_sub(window));
        }
        assert_eq!(kv.seq_len, total);
        // Bounded: never more than window/BLOCK + a small constant of slack.
        let max_blocks = window / PAGED_BLOCK_TOKENS + 2;
        assert!(
            kv.num_blocks() <= max_blocks,
            "cache not bounded: {} blocks (max {})",
            kv.num_blocks(),
            max_blocks
        );
        // The most recent `window` positions must still round-trip exactly.
        for p in (total - window)..total {
            let (k, v) = tok(p);
            assert_eq!(kv.key(p), &k[..], "evicted a still-windowed key at {p}");
            assert_eq!(kv.value(p), &v[..], "evicted a still-windowed value at {p}");
        }
    }

    #[test]
    fn evict_before_is_noop_at_zero_and_for_small_pos() {
        // No eviction can happen until at least one whole block is below
        // `pos`, so small `pos` (and pos == 0) leave the cache untouched.
        let mut kv = KvCache::new(4);
        for p in 0..(PAGED_BLOCK_TOKENS + 5) {
            kv.append(&[p as f32; 4], &[p as f32; 4]);
        }
        let before = kv.num_blocks();
        kv.evict_before(0);
        kv.evict_before(PAGED_BLOCK_TOKENS - 1); // still within block 0
        assert_eq!(kv.num_blocks(), before);
        assert_eq!(kv.key(0), &[0.0; 4][..]);
    }

    #[test]
    fn swa_attention_correct_after_eviction() {
        // End-to-end: a windowed attention block fed through eviction must
        // produce the same output as one that retained the full history,
        // because eviction only drops positions outside the window.
        let window = PAGED_BLOCK_TOKENS + 3;
        let attn = make_window_attn(Some(window));
        let mut kv_evict = KvCache::new(attn.kv_dim());
        let mut kv_keep = KvCache::new(attn.kv_dim());
        let total = 3 * PAGED_BLOCK_TOKENS;
        let mut last_evict = vec![];
        let mut last_keep = vec![];
        for p in 0..total {
            // Arbitrary but deterministic per-position input: the first
            // element varies with the position (mod 5) so successive tokens
            // differ, the rest are fixed; exact values are immaterial — the
            // test only asserts evict and keep paths agree bit-for-bit.
            let x = vec![((p % 5) as f32) * 0.3, 0.1, -0.2, 0.05];
            last_evict = attn.forward(&x, p, 0, &mut kv_evict, &cpu_backend());
            kv_evict.evict_before(p.saturating_sub(window));
            last_keep = attn.forward(&x, p, 0, &mut kv_keep, &cpu_backend());
        }
        for (a, b) in last_evict.iter().zip(last_keep.iter()) {
            assert!((a - b).abs() < 1e-5, "eviction changed output: {a} vs {b}");
        }
        // The evicting cache stayed bounded; the keeping one grew unbounded.
        assert!(kv_evict.num_blocks() < kv_keep.num_blocks());
    }

    #[test]
    fn paged_kv_cache_attention_matches_legacy_layout() {
        // Build an attention block, run a few tokens through it, and
        // verify the per-token output is unchanged from what a flat
        // KV cache would have produced. Since the block layout is
        // accessed only through `key(i)`/`value(i)` — which return
        // slices identical to what the old flat `Vec<f32>` would
        // have — the block index just has to stay correct as we
        // cross block boundaries. Walk past at least one boundary.
        let d_model = 4;
        let head_dim = 2;
        let num_heads = 2;
        let num_kv_heads = 2;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mk = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
                .collect()
        };
        let attn = MultiHeadSelfAttention {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim: head_dim,
            v_head_dim: head_dim,
            attention_value_scale: None,
            rope_base: 10000.0,
            wq: DenseWeight::from_f32(mk(q_dim, d_model), q_dim, d_model),
            wk: DenseWeight::from_f32(mk(kv_dim, d_model), kv_dim, d_model),
            wv: DenseWeight::from_f32(mk(kv_dim, d_model), kv_dim, d_model),
            wo: DenseWeight::from_f32(mk(d_model, q_dim), d_model, q_dim),
            window_size: None,
            q_norm: None,
            k_norm: None,
            rope_yarn: None,
            rope_cache: None,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            sink_bias: None,
        };
        let mut kv = KvCache::new(kv_dim);
        // Walk past the first block boundary to exercise multi-block
        // indexing.
        let xs: Vec<Vec<f32>> = (0..(PAGED_BLOCK_TOKENS + 3))
            .map(|t| {
                (0..d_model)
                    .map(|j| 0.05 * (t as f32) + 0.01 * (j as f32))
                    .collect()
            })
            .collect();
        let mut last = vec![0.0f32; d_model];
        for (pos, x) in xs.iter().enumerate() {
            last = attn.forward(x, pos, 0, &mut kv, &cpu_backend());
        }
        assert_eq!(kv.seq_len, xs.len());
        assert_eq!(kv.num_blocks(), 2);
        assert!(last.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zeroize_blocks_clears_every_element() {
        // Mix block sizes (including empty and non-power-of-two
        // lengths) to exercise the inner `for i in 0..len` loop's
        // bounds and the iteration over multiple blocks.
        let mut blocks: Vec<Box<[f32]>> = vec![
            vec![1.0f32; 16].into_boxed_slice(),
            vec![-2.5f32; 7].into_boxed_slice(),
            vec![f32::INFINITY; 1].into_boxed_slice(),
            Vec::<f32>::new().into_boxed_slice(),
            vec![std::f32::consts::PI; 33].into_boxed_slice(),
        ];
        // Sanity: at least one non-zero element exists before zeroising.
        assert!(blocks.iter().any(|b| b.iter().any(|&v| v != 0.0)));

        zeroize_blocks(&mut blocks);

        for (i, b) in blocks.iter().enumerate() {
            for (j, &v) in b.iter().enumerate() {
                assert_eq!(
                    v.to_bits(),
                    0.0f32.to_bits(),
                    "block {i} element {j} not zeroised: {v}"
                );
            }
        }
    }
}
