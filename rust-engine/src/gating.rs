//! Real **gating network** for a Mixtral-style MoE layer.
//!
//! The existing `crate::router::TopKRouter` is a *deterministic Markov chain*
//! over expert ids — useful for benchmarking the SSD-streaming substrate
//! without a trained model, but not what production MoE inference uses.
//! Production routing computes:
//!
//! ```text
//!     router_logits = x @ W_gate.T          // [num_experts]
//!     scores        = softmax(router_logits)
//!     (top_ids, top_scores) = top_k(scores, k)
//!     // optionally re-normalise top_scores to sum to 1
//! ```
//!
//! That is exactly what [`LinearGate`] does. The interface returns both the
//! chosen ids *and* the per-id weights, because the MoE block needs the
//! weights to take the weighted sum of the per-expert FFN outputs.
//!
//! The module also exposes a small [`Router`] enum that the engine can hold
//! polymorphically: production paths use [`Router::Linear`]; benchmarks /
//! `--io-only` runs use [`Router::Markov`] backed by the existing
//! `TopKRouter`. Both produce a uniform `RoutingDecision`.

//! `LinearGate` is the production routing path; `Router::Markov` keeps
//! the existing benchmark behaviour. Both are exercised by unit tests
//! below; the `serve` command currently wires the `Markov` variant
//! directly into `Engine`. `dead_code` is allowed at module scope so the
//! production-path API stays greppable until the full transformer
//! wiring (which will replace the engine's internal router) lands.
#![allow(dead_code)]

use crate::router::TopKRouter;
use crate::transformer::{matmul_row_major, softmax_inplace};
use std::sync::Arc;

/// Output of a routing decision for a single token at a single layer.
#[derive(Clone, Debug)]
pub struct RoutingDecision {
    /// Chosen expert ids (length `top_k`).
    pub experts: Vec<u32>,
    /// Mixing weights, one per chosen expert. Sum to 1.0 after
    /// re-normalisation. The MoE block computes
    /// `y = sum_i weights[i] * expert_i(x)`.
    pub weights: Vec<f32>,
}

/// Linear gating network: `W_gate @ x -> softmax -> top-K`.
///
/// Weight layout: `weights` is row-major `[num_experts, d_model]` (i.e.
/// the same layout HuggingFace `safetensors` ships for
/// `block_sparse_moe.gate.weight`).
#[derive(Debug, Clone)]
pub struct LinearGate {
    pub weights: Vec<f32>,
    pub num_experts: usize,
    pub d_model: usize,
    pub top_k: usize,
    /// If true, re-normalise top-K scores to sum to 1.0 after selection
    /// (Mixtral does this; some other MoE architectures don't).
    pub normalise_topk: bool,
}

impl LinearGate {
    pub fn new(weights: Vec<f32>, num_experts: usize, d_model: usize, top_k: usize) -> Self {
        assert!(top_k > 0 && top_k <= num_experts, "top_k must be in 1..=num_experts");
        assert_eq!(
            weights.len(),
            num_experts * d_model,
            "gate weight matrix must be [num_experts, d_model]"
        );
        Self {
            weights,
            num_experts,
            d_model,
            top_k,
            normalise_topk: true,
        }
    }

    /// Compute the routing decision for a single token's hidden state.
    pub fn route(&self, x: &[f32]) -> RoutingDecision {
        debug_assert_eq!(x.len(), self.d_model);
        // logits = W_gate · x  (length: num_experts)
        let mut logits = matmul_row_major(&self.weights, x, self.num_experts, self.d_model);
        softmax_inplace(&mut logits);

        // Top-K selection. For typical MoE sizes (8, 16, 64 experts) a
        // simple `O(N log N)` sort is fine — the cost is dwarfed by the
        // expert FFN matmul, never mind by the SSD read.
        let mut idx: Vec<u32> = (0..self.num_experts as u32).collect();
        idx.sort_by(|&a, &b| logits[b as usize].partial_cmp(&logits[a as usize]).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(self.top_k);
        let mut weights: Vec<f32> = idx.iter().map(|&i| logits[i as usize]).collect();
        if self.normalise_topk {
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }
        RoutingDecision { experts: idx, weights }
    }
}

/// Polymorphic router used by the engine. Production: `Linear` (real gate
/// from the model). Benchmarks: `Markov` (the existing `TopKRouter`).
#[derive(Clone)]
pub enum Router {
    Linear(Arc<LinearGate>),
    Markov(Arc<TopKRouter>),
}

impl Router {
    /// Route one token. `hidden` is required for `Linear`; ignored for
    /// `Markov` (which is stateful internally and uses `token_idx` only
    /// as a placeholder argument).
    pub fn route(&self, hidden: &[f32], token_idx: u64) -> RoutingDecision {
        match self {
            Router::Linear(gate) => gate.route(hidden),
            Router::Markov(r) => {
                let experts = r.route(token_idx);
                let n = experts.len() as f32;
                let weights = if n > 0.0 { vec![1.0 / n; experts.len()] } else { Vec::new() };
                RoutingDecision { experts, weights }
            }
        }
    }

    pub fn num_experts(&self) -> u32 {
        match self {
            Router::Linear(g) => g.num_experts as u32,
            Router::Markov(r) => r.num_experts(),
        }
    }

    pub fn top_k(&self) -> usize {
        match self {
            Router::Linear(g) => g.top_k,
            Router::Markov(r) => r.k(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_gate_picks_largest_logit_first() {
        let num_experts = 4;
        let d_model = 3;
        // Construct gate so that logit[i] = sum(x) * (i+1):
        // row i = [(i+1), (i+1), (i+1)].
        let mut w = Vec::with_capacity(num_experts * d_model);
        for i in 0..num_experts {
            for _ in 0..d_model {
                w.push((i + 1) as f32);
            }
        }
        let gate = LinearGate::new(w, num_experts, d_model, 2);
        let x = vec![1.0, 1.0, 1.0];
        let dec = gate.route(&x);
        assert_eq!(dec.experts.len(), 2);
        // The two largest logits are experts 3 and 2 (in that order).
        assert_eq!(dec.experts[0], 3);
        assert_eq!(dec.experts[1], 2);
        // Weights re-normalised to sum to 1.
        let sum: f32 = dec.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        // Top expert weight > second.
        assert!(dec.weights[0] > dec.weights[1]);
    }

    #[test]
    fn linear_gate_top_k_equals_num_experts_returns_all() {
        let gate = LinearGate::new(vec![0.0; 4 * 2], 4, 2, 4);
        let dec = gate.route(&[0.0, 0.0]);
        assert_eq!(dec.experts.len(), 4);
        // Uniform softmax over zeros => 0.25 each.
        for w in dec.weights {
            assert!((w - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn router_enum_dispatches_correctly() {
        let r = Router::Markov(Arc::new(TopKRouter::clustered(8, 2, 4, 0.9, 1)));
        let dec = r.route(&[], 0);
        assert_eq!(dec.experts.len(), 2);
        assert_eq!(dec.weights.len(), 2);
        // Markov path uses uniform 1/k weights.
        for w in dec.weights {
            assert!((w - 0.5).abs() < 1e-6);
        }
    }
}
