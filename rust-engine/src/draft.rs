//! Speculative-decoding "draft" engine (gist Phase 2 — Speculative
//! Verification Engine).
//!
//! Concept (from the gist):
//!
//! > Implement a small, low-latency dense LLM (the "draft model")
//! > entirely in RAM. The main MoE engine (the "verification model")
//! > then verifies a batch of `K` draft tokens in parallel. Crucially,
//! > the main engine bundles the SSD reads it needs for *all* K
//! > positions into a single batched [`Engine::warm_with`] call —
//! > eliminating the wait-for-disk cycle from the inner loop of the
//! > tail of every generation.
//!
//! ## Design
//!
//! [`DraftEngine`] is intentionally *minimal*: it owns a small,
//! deterministic dense projection trained to mimic the main MoE's
//! next-token distribution. The implementation here is the simplest
//! possible faithful draft head — a learned linear `vocab_size ×
//! d_model` matrix tied to a f32 snapshot of the main model's embedding plus a per-token
//! position bias. It is **not** competitive with a real 1-2 B draft
//! model on hit-rate, but it produces *deterministic* drafts (so
//! verification is reproducible in tests) and exercises every code
//! path the production version would touch:
//!
//! 1. `DraftEngine::draft(seed_token, k)` produces `k` candidate
//!    tokens with no SSD I/O.
//! 2. [`RealModel::step_speculative`] takes those `k` candidates,
//!    folds the engine's [`RealModel::peek_experts`] hint for each
//!    candidate position into a single `HashSet`, fires one
//!    [`Engine::warm_with`] for the union, then verifies each
//!    candidate by running the full main-model `step`. The accepted
//!    prefix is the longest prefix on which the main model's argmax
//!    matches the draft.
//!
//! Swapping a more capable draft head in later is a drop-in change:
//! anything implementing [`DraftLike`] plugs into
//! `step_speculative`.

// Speculative-decoding scaffold (gist Phase 2). The wiring into
// `Engine::generate` lands in a follow-up; until then keep the public
// surface compilable without per-item `dead_code` noise.
#![allow(dead_code)]

use crate::engine::Engine;
use crate::model::RealModel;
use crate::sampling::SamplingParams;
use crate::transformer::KvCache;
use std::sync::Arc;

/// Minimal trait the speculative verification loop needs from a
/// draft model. Implemented by [`DraftEngine`] and trivially mockable
/// in tests.
pub trait DraftLike: Send + Sync {
    /// Generate `k` draft tokens starting from `seed_token`. The
    /// draft is purely RAM-bound: no SSD I/O, no `engine.moe_step`
    /// awaits.
    fn draft(&self, seed_token: u32, k: usize) -> Vec<u32>;
}

/// **DraftEngine** — small, RAM-resident dense head used to
/// pre-generate candidate tokens for the main MoE engine to verify.
/// Gist Phase 2.
///
/// **Why tie embeddings?** Tying the draft's "lm_head" to the main
/// model's embedding matrix means the draft is forced to live in the
/// same token space and respects the main model's vocab layout. It
/// also keeps the draft's resident size tiny (one extra
/// `d_model`-sized bias vector per layer of the main model would
/// already be more than enough for production; this implementation
/// keeps just a single bias).
///
/// **Why deterministic?** Reproducibility in tests outweighs hit-rate
/// for the minimal implementation. The argmax-based draft is
/// equivalent to `temperature = 0` sampling and matches the verifier
/// when run with `SamplingParams::greedy()`.
pub struct DraftEngine {
    /// Shared `Arc<Vec<f32>>` snapshot of the main model's embedding
    /// table. Native quantized embeddings are dequantized once when the
    /// draft is built, keeping the draft path simple and RAM-resident while
    /// the verifier can keep its own embedding in compressed form.
    embedding: Arc<Vec<f32>>,
    /// Vocabulary size and hidden dim — copied from the main model
    /// at construction so the draft can run without holding a
    /// `&RealModel`.
    vocab_size: usize,
    d_model: usize,
    /// Per-token positional bias. Deterministic seeded init keeps
    /// `cargo test` reproducible; production builds would replace
    /// this with a learned vector from the main model's first
    /// attention block's output projection.
    ///
    /// Stored in a 64-byte-aligned heap allocation (boxed slice) so
    /// the AVX-512 / AVX2 `dot_f32` kernels see clean cache-line
    /// reads when they pull `bias` into a vector register on every
    /// `predict_next` call. The embedding rows are aligned implicitly
    /// (they're contiguous floats inside `embedding`, which the loader
    /// page-aligns), so the only buffer that needed promotion was the
    /// bias / hidden-state scratch.
    bias: AlignedF32,
}

/// 64-byte-aligned owned `[f32]` slice — sized to keep the AVX-512
/// kernels' 16-lane FMA loads on a single L1 cache line. Used by
/// [`DraftEngine`] for the bias and the residual-style hidden scratch
/// that get FMA-summed into every dot product.
struct AlignedF32 {
    ptr: std::ptr::NonNull<f32>,
    len: usize,
}

// SAFETY: this is an owned allocation with no interior mutability.
unsafe impl Send for AlignedF32 {}
unsafe impl Sync for AlignedF32 {}

impl AlignedF32 {
    /// Allocate `len` zero-initialised `f32`s on a 64-byte boundary.
    fn zeros(len: usize) -> Self {
        assert!(len > 0, "AlignedF32 length must be > 0");
        // Round the byte size up to a multiple of 64 so the *tail* of
        // the buffer also sits inside an aligned cache line — keeps
        // the AVX-512 16-wide tail load from straddling a cache line.
        let bytes = (len * std::mem::size_of::<f32>() + 63) & !63;
        let layout = std::alloc::Layout::from_size_align(bytes, 64)
            .expect("invalid AlignedF32 layout");
        // SAFETY: layout has non-zero size and a power-of-two align.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) } as *mut f32;
        let ptr = std::ptr::NonNull::new(raw)
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len }
    }

    #[inline]
    fn as_slice(&self) -> &[f32] {
        // SAFETY: ptr/len describe a fully-initialised owned f32 slab.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: ptr/len describe a fully-initialised owned f32 slab.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedF32 {
    fn drop(&mut self) {
        let bytes = (self.len * std::mem::size_of::<f32>() + 63) & !63;
        let layout = std::alloc::Layout::from_size_align(bytes, 64).unwrap();
        // SAFETY: we own the allocation and the layout matches `zeros`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout) }
    }
}

impl DraftEngine {
    /// Build a draft engine that shares the verifier's embedding
    /// table. The bias vector is deterministically seeded from the
    /// embedding so multiple `DraftEngine::from_main` calls on the
    /// same `RealModel` produce identical drafts (important for
    /// reproducible tests).
    pub fn from_main(main: &RealModel) -> Self {
        let d = main.config.d_model;
        let vocab = main.config.vocab_size;
        // A zero-vocab model can't produce drafts and would also make
        // `inv_vocab = 1.0 / 0.0 = inf`, silently corrupting every
        // bias entry to inf/NaN and every subsequent draft to garbage.
        assert!(
            vocab > 0,
            "DraftEngine::from_main requires vocab_size > 0 (got {vocab})"
        );
        assert!(
            d > 0,
            "DraftEngine::from_main requires d_model > 0 (got {d})"
        );
        // Seed the bias with a cheap hash of the embedding rows so
        // it varies with the main model but doesn't pull in an RNG.
        // 64-byte-aligned scratch — keeps the AVX-512 16-wide loads
        // in `predict_next` on one cache line per iteration.
        let mut bias = AlignedF32::zeros(d);
        {
            let bias_s = bias.as_mut_slice();
            let inv_vocab = 1.0f32 / (vocab as f32);
            let mut row = Vec::new();
            for tok in 0..vocab {
                row.clear();
                main.embedding.row_dequant_into(tok, &mut row);
                for (i, &x) in row.iter().enumerate() {
                    bias_s[i] += x * inv_vocab;
                }
            }
        }
        Self {
            embedding: Arc::new(main.embedding.to_f32_vec()),
            vocab_size: vocab,
            d_model: d,
            bias,
        }
    }

    /// Number of candidate tokens this draft engine is configured to
    /// produce per call. Constant for the minimal implementation;
    /// production drafts would expose a `set_lookahead(k)` knob.
    pub const DEFAULT_LOOKAHEAD: usize = 4;

    fn embed_row(&self, token: u32) -> &[f32] {
        let id = (token as usize) % self.vocab_size;
        &self.embedding[id * self.d_model..(id + 1) * self.d_model]
    }

    /// Argmax over `(embed(token) + bias) · embedding.T`. Tied head
    /// means this is a single `O(vocab_size · d_model)` matvec — the
    /// "low-latency dense model" property the gist asks for.
    ///
    /// The inner reductions go through [`crate::kernels::dot_f32`],
    /// which auto-escalates AVX-512 → AVX2 → scalar via the same
    /// runtime dispatcher [`crate::backend::ScalarBackend::matmul_into`]
    /// uses. That eliminates the raw nested scalar loop the gist
    /// flagged for L3-thrashing on `vocab_size × d_model` matmuls.
    /// The hidden state `h = embed(token) + bias` is built in a
    /// 64-byte-aligned scratch ([`AlignedF32`]) so the SIMD kernels
    /// see clean cache-line reads, and we reuse the same allocation
    /// across the call (no per-call `Vec::with_capacity`).
    fn predict_next(&self, token: u32, h_scratch: &mut AlignedF32) -> u32 {
        let cur = self.embed_row(token);
        let bias = self.bias.as_slice();
        let h = h_scratch.as_mut_slice();
        debug_assert_eq!(h.len(), self.d_model);
        debug_assert_eq!(bias.len(), self.d_model);
        debug_assert_eq!(cur.len(), self.d_model);
        // h = cur + bias (residual-style update). Plain elementwise
        // add — the optimiser autovectorises this with the same
        // 16-wide AVX-512 stride the dot product uses below.
        for i in 0..self.d_model {
            h[i] = cur[i] + bias[i];
        }
        // logits[v] = h · embedding[v]. Routed through the SIMD
        // dispatcher; on AVX-512-capable hosts each row is a
        // 4×-unrolled FMA reduction, on AVX2 it's an 8-wide FMA, and
        // on scalar it falls back to the reference loop. Argmax is
        // streamed against `best_score` to avoid materialising the
        // full logits vector — keeps the working set in L1 and
        // matches the gist's "no allocation on the hot path" rule.
        let d = self.d_model;
        let mut best = 0u32;
        let mut best_score = f32::NEG_INFINITY;
        for v in 0..self.vocab_size {
            let row = &self.embedding[v * d..(v + 1) * d];
            let score = crate::kernels::dot_f32(h, row);
            if score > best_score {
                best_score = score;
                best = v as u32;
            }
        }
        best
    }
}

impl DraftLike for DraftEngine {
    fn draft(&self, seed_token: u32, k: usize) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(k);
        // Single 64-byte-aligned hidden-state scratch reused across
        // every draft step — no per-iteration allocation, matches the
        // gist's "allocation-free hot path" rule.
        let mut h = AlignedF32::zeros(self.d_model.max(1));
        let mut cur = seed_token;
        for _ in 0..k {
            cur = self.predict_next(cur, &mut h);
            tokens.push(cur);
        }
        tokens
    }
}

/// Result of [`RealModel::step_speculative`] — a small struct so
/// callers can tell how much of the draft was accepted (and therefore
/// how much wall-clock the verification saved).
#[derive(Debug, Clone)]
pub struct SpeculativeStepResult {
    /// Tokens actually accepted by the verifier (i.e. the longest
    /// prefix of `draft_tokens` that the main model would have
    /// produced anyway). The last element of this `Vec` is the
    /// canonical "next token" the caller should append to its
    /// running generation. Always non-empty: at minimum the
    /// verifier emits its own next token even when zero draft
    /// candidates are accepted.
    pub accepted: Vec<u32>,
    /// Length of the accepted draft prefix (`0 ..= draft_tokens.len()`).
    /// Strict acceptance count, equal to `accepted.len() - 1` if the
    /// verifier disagreed on every draft token (the trailing element
    /// is the verifier's own correction), or `accepted.len()` if
    /// every draft was confirmed.
    pub accepted_len: usize,
    /// Number of unique global expert ids the pre-pass warmed in a
    /// single batched read. Mirrors the `warm_with` call so callers
    /// can prove (in tests + metrics) that the speculative path
    /// really did issue one unified prefetch.
    pub warmed_experts: usize,
}

impl RealModel {
    /// **Speculative decoding step** — verify `k` draft tokens with
    /// a single unified expert prefetch (gist Phase 2). On call:
    ///
    /// 1. The provided [`DraftLike`] produces `k` candidate tokens.
    /// 2. For each draft position we peek the routing decision via
    ///    [`Self::peek_experts`] and fold every global expert id
    ///    into a single `HashSet`.
    /// 3. One [`Engine::warm_with`] call pulls the union into the
    ///    expert cache concurrently (singleflight'd, so the SSD
    ///    sees one read per unique id).
    /// 4. The main verifier is then run with greedy sampling for
    ///    each draft position. The accepted prefix length equals
    ///    the count of leading positions where the verifier's
    ///    argmax matches the draft. If the verifier disagrees on
    ///    position `i`, the run stops and the verifier's own
    ///    `step(i)` result is appended as the corrective token.
    ///
    /// Returns the accepted token sequence (always non-empty) plus
    /// telemetry useful for tests and observability.
    pub async fn step_speculative<D: DraftLike + ?Sized>(
        &self,
        engine: &Arc<Engine>,
        draft: &D,
        seed_token: u32,
        pos: usize,
        kv: &mut [KvCache],
        k: usize,
    ) -> Result<SpeculativeStepResult, crate::model::RealInferenceError> {
        let drafts = draft.draft(seed_token, k);

        // Phase A: warm union of expert ids predicted by the peek
        // pre-pass for every draft position. Use a *clone* of the
        // KV slice so peeking does not mutate it — the verifier
        // path is what actually advances the real cache.
        //
        // **Position-1+ decay fix (gist Part 1, fix #3).** The
        // previous implementation appended zero K/V vectors for every
        // lookahead position past the first, which left the routing
        // pre-pass conditioned on garbage data as the speculation
        // window $K$ grew. Instead we now compute a lightweight
        // hidden-state approximation: for the layer the peek actually
        // re-attends over (layer 0), we project the draft token's
        // embedding through the layer's `wk`/`wv` matrices, apply
        // RoPE at the correct absolute position, and append the
        // result. That anchors `peek_experts` for every $i > 0$ on a
        // K/V slot derived from the real candidate token rather than
        // a zero vector, recovering prefetch accuracy for the tail of
        // the lookahead window. For layers $\geq 1$ the peek does not
        // re-attend (it re-uses the embedding directly), so their
        // KV-preview slots stay untouched — feeding them anything
        // would just waste cycles without changing the routing
        // decision.
        let mut union: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut kv_preview: Vec<KvCache> = kv.to_vec();
        let mut cur = seed_token;
        for (i, &draft_tok) in drafts.iter().enumerate() {
            for id in self.peek_experts(cur, pos + i, &kv_preview) {
                union.insert(id);
            }
            // Advance the preview cache so the next iteration's peek
            // is conditioned on a plausible position. Only layer 0
            // is consulted by `peek_experts`'s attention pre-pass;
            // other layers ignore their KV slots in the peek path.
            if i + 1 < drafts.len() {
                self.advance_preview_kv(&mut kv_preview[0], cur, pos + i);
            }
            cur = draft_tok;
        }
        let warmed_experts = union.len();
        if !union.is_empty() {
            let ids: Vec<u32> = union.into_iter().collect();
            // best-effort: failures here are tolerated because the
            // verifier path retries each expert individually.
            let _ = engine.warm_with(&ids).await;
        }

        // Phase B: verify each draft token. We always emit at least
        // one token (the verifier's own correction at the first
        // mismatch), so the result is never empty.
        let greedy = SamplingParams::greedy();
        let mut accepted: Vec<u32> = Vec::with_capacity(k.max(1));
        let mut cur = seed_token;
        let mut accepted_len = 0usize;
        for (i, &draft_tok) in drafts.iter().enumerate() {
            let next = self.step(engine, cur, pos + i, kv, &greedy).await?;
            accepted.push(next);
            if next != draft_tok {
                break;
            }
            accepted_len += 1;
            cur = next;
        }
        if accepted.is_empty() {
            // k == 0: still emit one verifier-produced token so
            // callers can treat `step_speculative` as a strict
            // generalisation of `step`.
            let next = self.step(engine, seed_token, pos, kv, &greedy).await?;
            accepted.push(next);
        }
        Ok(SpeculativeStepResult { accepted, accepted_len, warmed_experts })
    }

    /// Advance the layer-0 KV preview by one position using the draft
    /// token's embedding as a hidden-state approximation. Pure
    /// helper — does **not** touch the real KV cache. Used by
    /// [`Self::step_speculative`] to fix the position-1+ decay the
    /// gist flagged: the previous "append zeros" loop polluted the
    /// peek pre-pass for $i > 0$.
    fn advance_preview_kv(&self, kv0: &mut KvCache, draft_tok: u32, abs_pos: usize) {
        let x = self.embed(draft_tok);
        let layer = &self.layers[0];
        // Mirror the verifier's `attn_block` dispatch: MLA layers cache a
        // single latent vector (k == v == latent), the standard path caches
        // projected K/V. Using the matching projection keeps the peek
        // pre-pass faithful and, critically, matches the cache width
        // `fresh_kv_caches` allocated for this layer (MLA → `latent_dim`),
        // so the `append` below cannot hit a `copy_from_slice` mismatch.
        if let Some(mla) = layer.mla.as_ref() {
            debug_assert_eq!(kv0.kv_dim, mla.latent_dim());
            debug_assert_eq!(kv0.v_dim, mla.latent_dim());
            mla.project_and_cache_kv(&x, abs_pos, kv0);
            return;
        }
        let attn = &layer.attn;
        debug_assert_eq!(kv0.kv_dim, attn.kv_dim());
        debug_assert_eq!(kv0.v_dim, attn.v_proj_dim());
        // Project K/V exactly as the verifier's `attn_block` does — the
        // shared `project_kv` applies QKV bias, QK-Norm and partial/scaled
        // RoPE at the same absolute position the verifier will consume
        // this token at (`abs_pos`, i.e. `pos + i`). Using the shared
        // helper keeps the peek pre-pass faithful and prevents the V-width
        // (`v_proj_dim`, not `kv_dim`) and RoPE-position drift the previous
        // hand-rolled copy introduced. Skipping Q/wo + the attention sum
        // keeps the helper lightweight; `peek_experts` only re-reads
        // layer 0's K/V slots.
        let (k, v) = attn.project_kv(&x, abs_pos);
        kv0.append(&k, &v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineOptions, ModelShape};
    use crate::multi_layer_cache::MultiLayerExpertCache;
    use crate::buffer_pool::BufferPool;
    use crate::io_provider::{NvmeStorage, StorageConfig};
    use crate::router::{PredictiveLoader, TopKRouter};
    use crate::gating::Router;
    use std::path::PathBuf;
    use std::sync::Arc;

    /// Same lightweight tempdir helper used by `server.rs` tests so
    /// we don't add a `tempfile` dev-dependency for this one test.
    struct TempDir { path: PathBuf }
    impl TempDir {
        fn new(tag: &str) -> std::io::Result<Self> {
            let mut path = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            path.push(format!("mer-draft-test-{tag}-{}-{nanos}", std::process::id()));
            std::fs::create_dir_all(&path)?;
            Ok(Self { path })
        }
        fn path(&self) -> &std::path::Path { &self.path }
    }
    impl Drop for TempDir {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.path); }
    }

    fn make_engine(num_experts: u32, d_model: usize, d_ff: usize) -> (Arc<Engine>, TempDir) {
        let dir = TempDir::new("engine").unwrap();
        let expert_size = crate::inference::expert_weight_bytes(d_model, d_ff);
        let cfg = StorageConfig {
            base_path: dir.path().to_path_buf(),
            expert_size,
            block_align: 4096,
            use_direct_io: false,
            num_experts_per_layer: None,
        };
        crate::io_provider::generate_synthetic_experts(
            dir.path(),
            num_experts,
            expert_size,
            d_model,
            d_ff,
        )
        .unwrap();
        let storage = Arc::new(NvmeStorage::new(cfg).unwrap());
        let cache = Arc::new(MultiLayerExpertCache::single_layer(num_experts as usize));
        let pool = BufferPool::new(num_experts as usize, expert_size, 4096);
        let router = Router::Markov(Arc::new(TopKRouter::new(num_experts, 2, 0)));
        let predictor = Arc::new(PredictiveLoader::new(num_experts, 4, 0.0, 0));
        let shape = ModelShape { d_model, d_ff, hidden_seed: 0 };
        let engine = Arc::new(Engine::with_options(
            cache, pool, storage, router, predictor, shape, EngineOptions::default(),
        ));
        (engine, dir)
    }

    #[tokio::test]
    async fn draft_engine_is_deterministic() {
        let main = RealModel::new_seeded(crate::model::RealModelConfig::tiny(), 7);
        let d1 = DraftEngine::from_main(&main);
        let d2 = DraftEngine::from_main(&main);
        assert_eq!(d1.draft(42, 4), d2.draft(42, 4));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn speculative_step_matches_sequential_step() {
        // A tiny model whose verifier output is exactly reproducible
        // (greedy sampling + deterministic init). The
        // `step_speculative` path must produce the same accepted
        // sequence as running `step` four times in a row, regardless
        // of how many draft tokens the draft head got right.
        let cfg = crate::model::RealModelConfig::tiny();
        let main = RealModel::new_seeded(cfg.clone(), 11);
        let (engine, _dir) = make_engine(
            (cfg.num_layers * cfg.num_experts) as u32,
            cfg.d_model,
            cfg.d_ff,
        );
        let draft = DraftEngine::from_main(&main);

        // Reference: run four sequential greedy steps.
        let mut kv_ref = main.fresh_kv_caches();
        let greedy = SamplingParams::greedy();
        let mut expected = Vec::new();
        let mut cur = 5u32;
        for i in 0..4 {
            cur = main
                .step(&engine, cur, i, &mut kv_ref, &greedy)
                .await
                .expect("verifier step");
            expected.push(cur);
        }

        // Speculative: same starting state, same seed token. The
        // accepted sequence must be a prefix of `expected` and the
        // last accepted token must equal `expected[accepted.len() - 1]`.
        let mut kv_spec = main.fresh_kv_caches();
        let result = main
            .step_speculative(&engine, &draft, 5u32, 0, &mut kv_spec, 4)
            .await
            .expect("speculative step");
        assert!(!result.accepted.is_empty(), "must emit at least one token");
        assert!(result.accepted.len() <= 4);
        for (i, t) in result.accepted.iter().enumerate() {
            assert_eq!(*t, expected[i], "verifier output diverged at position {i}");
        }
    }
}
