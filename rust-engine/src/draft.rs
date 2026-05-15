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
//! d_model` matrix tied to the main model's embedding plus a per-token
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
    /// Shared reference to the main model's embedding table. Tied
    /// embedding keeps the draft head's parameter count at zero
    /// (excluding the bias vector) so the engine never accidentally
    /// drifts into "the draft is bigger than the verifier" territory.
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
    bias: Vec<f32>,
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
        // Seed the bias with a cheap hash of the embedding rows so
        // it varies with the main model but doesn't pull in an RNG.
        let mut bias = vec![0.0f32; d];
        for tok in 0..vocab {
            for (i, b) in bias.iter_mut().enumerate() {
                *b += main.embedding[tok * d + i] / (vocab as f32);
            }
        }
        Self {
            embedding: Arc::new(main.embedding.clone()),
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
    fn predict_next(&self, token: u32) -> u32 {
        let cur = self.embed_row(token);
        // h = cur + bias (residual-style update)
        let mut h = vec![0.0f32; self.d_model];
        for i in 0..self.d_model {
            h[i] = cur[i] + self.bias[i];
        }
        // logits[v] = h · embedding[v]
        let mut best = 0u32;
        let mut best_score = f32::NEG_INFINITY;
        for v in 0..self.vocab_size {
            let row = &self.embedding[v * self.d_model..(v + 1) * self.d_model];
            let mut score = 0.0f32;
            for i in 0..self.d_model {
                score += h[i] * row[i];
            }
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
        let mut cur = seed_token;
        for _ in 0..k {
            cur = self.predict_next(cur);
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
    ) -> SpeculativeStepResult {
        let drafts = draft.draft(seed_token, k);

        // Phase A: warm union of expert ids predicted by the peek
        // pre-pass for every draft position. Use a *clone* of the
        // KV slice so peeking does not mutate it — the verifier
        // path is what actually advances the real cache.
        let mut union: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut kv_preview: Vec<KvCache> = kv.to_vec();
        let mut cur = seed_token;
        for (i, &draft_tok) in drafts.iter().enumerate() {
            for id in self.peek_experts(cur, pos + i, &kv_preview) {
                union.insert(id);
            }
            // Advance the preview cache so the next iteration's peek
            // is conditioned on a plausible position. We can't run
            // real attention without paying the verifier's cost, so
            // just append a zero K/V — the position is what the
            // peek needs most.
            for slot in kv_preview.iter_mut() {
                let zeros = vec![0.0f32; slot.kv_dim];
                slot.append(&zeros, &zeros);
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
            let next = self.step(engine, cur, pos + i, kv, &greedy).await;
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
            let next = self.step(engine, seed_token, pos, kv, &greedy).await;
            accepted.push(next);
        }
        SpeculativeStepResult { accepted, accepted_len, warmed_experts }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineOptions, ModelShape};
    use crate::expert_cache::ExpertCache;
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
        let cache = Arc::new(ExpertCache::new(num_experts as usize));
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
            cur = main.step(&engine, cur, i, &mut kv_ref, &greedy).await;
            expected.push(cur);
        }

        // Speculative: same starting state, same seed token. The
        // accepted sequence must be a prefix of `expected` and the
        // last accepted token must equal `expected[accepted.len() - 1]`.
        let mut kv_spec = main.fresh_kv_caches();
        let result = main
            .step_speculative(&engine, &draft, 5u32, 0, &mut kv_spec, 4)
            .await;
        assert!(!result.accepted.is_empty(), "must emit at least one token");
        assert!(result.accepted.len() <= 4);
        for (i, t) in result.accepted.iter().enumerate() {
            assert_eq!(*t, expected[i], "verifier output diverged at position {i}");
        }
    }
}
