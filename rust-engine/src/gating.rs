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
//! below. The `serve` command instantiates `Router::Linear` from the
//! loaded model's per-layer gate weights when
//! `[real_transformer].enabled = true`, and falls back to
//! `Router::Markov` (over a clustered `TopKRouter`) for the
//! benchmark / `--io-only` path that has no real gating network.

use crate::router::TopKRouter;
use crate::transformer::{matmul_row_major_into, softmax_inplace};
use std::cmp::Ordering;
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

/// Reusable per-call routing buffers.
///
/// A request path can keep one scratch value and pass it through each layer
/// so router logits, selection scores, and small top-K work buffers retain
/// their allocation capacity across layers. The returned [`RoutingDecision`]
/// remains owned because downstream engine calls may await on I/O.
#[derive(Debug, Default)]
pub struct RoutingScratch {
    logits: Vec<f32>,
    selection: Vec<f32>,
    top_groups: Vec<(usize, f32)>,
    top_experts: Vec<(u32, f32)>,
}

impl RoutingScratch {
    pub fn new() -> Self {
        Self::default()
    }

    fn resize_for(&mut self, num_experts: usize) {
        self.logits.resize(num_experts, 0.0);
        self.selection.resize(num_experts, 0.0);
        self.top_groups.clear();
        self.top_experts.clear();
    }
}

/// How the router turns raw gate logits into per-expert scores.
///
/// * `Softmax` — the classic Mixtral / Qwen3-MoE path: a softmax over all
///   experts, so scores already sum to 1 before top-K selection.
/// * `Sigmoid` — DeepSeek-V3's `scoring_func = "sigmoid"`: each expert is
///   scored independently with a logistic, *without* a cross-expert
///   normalisation. The selected weights are re-normalised afterwards iff
///   [`LinearGate::normalise_topk`] is set (`norm_topk_prob`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringFunc {
    Softmax,
    Sigmoid,
}

impl Default for ScoringFunc {
    fn default() -> Self {
        ScoringFunc::Softmax
    }
}

/// Linear gating network: `W_gate @ x -> score -> (grouped) top-K`.
///
/// Weight layout: `weights` is row-major `[num_experts, d_model]` (i.e.
/// the same layout HuggingFace `safetensors` ships for
/// `block_sparse_moe.gate.weight` / `mlp.gate.weight`).
///
/// The default constructor ([`LinearGate::new`]) reproduces the original
/// Mixtral behaviour exactly (softmax, no bias, no groups, unit scaling,
/// top-K renormalisation). DeepSeek-V3's aux-loss-free routing
/// (sigmoid scoring, a selection-only correction bias, group-limited
/// top-K, and a routed scaling factor) is configured through
/// [`LinearGate::with_routing`].
#[derive(Debug, Clone)]
pub struct LinearGate {
    pub weights: Vec<f32>,
    pub num_experts: usize,
    pub d_model: usize,
    pub top_k: usize,
    /// If true, re-normalise top-K scores to sum to 1.0 after selection
    /// (Mixtral does this; some other MoE architectures don't).
    pub normalise_topk: bool,
    /// Logit-to-score function. `Softmax` for Mixtral / Qwen3-MoE,
    /// `Sigmoid` for DeepSeek-V3.
    pub scoring_func: ScoringFunc,
    /// Optional per-expert bias **added to the selection score only**
    /// (DeepSeek-V3 `e_score_correction_bias`, the aux-loss-free load
    /// balancer). It steers *which* experts win the top-K race but is
    /// never folded into the mixing weights. Length must be `num_experts`.
    pub correction_bias: Option<Vec<f32>>,
    /// Number of expert groups for group-limited routing
    /// (DeepSeek `n_group`). `0` or `1` disables grouping.
    pub n_group: usize,
    /// How many groups survive the group pre-selection (DeepSeek
    /// `topk_group`). Ignored when grouping is disabled.
    pub topk_group: usize,
    /// Final multiplier applied to the mixing weights (DeepSeek
    /// `routed_scaling_factor`). `1.0` is a no-op.
    pub routed_scaling_factor: f32,
}

impl LinearGate {
    /// Mixtral / Qwen3-MoE gate: softmax scoring, top-K renormalisation,
    /// no correction bias, no grouping, unit scaling. Preserves the
    /// original behaviour for every existing call site.
    pub fn new(weights: Vec<f32>, num_experts: usize, d_model: usize, top_k: usize) -> Self {
        assert!(
            top_k > 0 && top_k <= num_experts,
            "top_k must be in 1..=num_experts"
        );
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
            scoring_func: ScoringFunc::Softmax,
            correction_bias: None,
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor: 1.0,
        }
    }

    /// Builder for the full DeepSeek-V3-style routing surface. `n_group`
    /// of `0`/`1` disables group-limited selection; `correction_bias`, if
    /// present, must be length `num_experts`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_routing(
        weights: Vec<f32>,
        num_experts: usize,
        d_model: usize,
        top_k: usize,
        scoring_func: ScoringFunc,
        normalise_topk: bool,
        correction_bias: Option<Vec<f32>>,
        n_group: usize,
        topk_group: usize,
        routed_scaling_factor: f32,
    ) -> Self {
        let mut gate = Self::new(weights, num_experts, d_model, top_k);
        gate.scoring_func = scoring_func;
        gate.normalise_topk = normalise_topk;
        if let Some(bias) = correction_bias.as_ref() {
            assert_eq!(
                bias.len(),
                num_experts,
                "correction_bias must have length num_experts"
            );
        }
        gate.correction_bias = correction_bias;
        gate.n_group = n_group.max(1);
        gate.topk_group = topk_group;
        gate.routed_scaling_factor = routed_scaling_factor;
        gate
    }

    /// Turn raw gate logits into per-expert scores per [`Self::scoring_func`].
    fn score(&self, logits: &mut [f32]) {
        match self.scoring_func {
            ScoringFunc::Softmax => softmax_inplace(logits),
            ScoringFunc::Sigmoid => {
                for v in logits.iter_mut() {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
            }
        }
    }

    #[inline]
    fn score_id_is_better(
        candidate_score: f32,
        candidate_id: usize,
        incumbent_score: f32,
        incumbent_id: usize,
    ) -> bool {
        match candidate_score.total_cmp(&incumbent_score) {
            Ordering::Greater => true,
            Ordering::Equal => candidate_id < incumbent_id,
            Ordering::Less => false,
        }
    }

    fn insert_top_group(groups: &mut Vec<(usize, f32)>, group: usize, score: f32, limit: usize) {
        if limit == 0 {
            return;
        }
        let pos = groups
            .iter()
            .position(|&(g, s)| Self::score_id_is_better(score, group, s, g));
        match pos {
            Some(pos) => {
                groups.insert(pos, (group, score));
                if groups.len() > limit {
                    groups.pop();
                }
            }
            None if groups.len() < limit => groups.push((group, score)),
            None => {}
        }
    }

    fn insert_top_expert(experts: &mut Vec<(u32, f32)>, expert: usize, score: f32, limit: usize) {
        if limit == 0 || !score.is_finite() {
            return;
        }
        let pos = experts
            .iter()
            .position(|&(id, s)| Self::score_id_is_better(score, expert, s, id as usize));
        let entry = (expert as u32, score);
        match pos {
            Some(pos) => {
                experts.insert(pos, entry);
                if experts.len() > limit {
                    experts.pop();
                }
            }
            None if experts.len() < limit => experts.push(entry),
            None => {}
        }
    }

    fn group_filter_shape(&self) -> Option<(usize, usize)> {
        let n_group = self.n_group.max(1);
        if n_group <= 1
            || self.topk_group == 0
            || self.topk_group >= n_group
            || self.num_experts % n_group != 0
        {
            return None;
        }
        Some((n_group, self.num_experts / n_group))
    }

    /// Group-limited expert pre-selection (DeepSeek `n_group` /
    /// `topk_group`). `scratch_groups` receives the top group ids in
    /// deterministic score/id order and its capacity is reused across calls.
    /// Each group's score is the sum of its top-2 selection scores, exactly
    /// as in the reference DeepSeek-V3 implementation.
    fn select_top_groups(
        &self,
        selection: &[f32],
        scratch_groups: &mut Vec<(usize, f32)>,
    ) -> Option<usize> {
        let (n_group, group_size) = self.group_filter_shape()?;
        scratch_groups.clear();
        // Score each group by the sum of its two best selection scores.
        for g in 0..n_group {
            let slice = &selection[g * group_size..(g + 1) * group_size];
            let mut top1 = f32::NEG_INFINITY;
            let mut top2 = f32::NEG_INFINITY;
            for &s in slice {
                if s > top1 {
                    top2 = top1;
                    top1 = s;
                } else if s > top2 {
                    top2 = s;
                }
            }
            let sum = top1 + if top2.is_finite() { top2 } else { 0.0 };
            Self::insert_top_group(scratch_groups, g, sum, self.topk_group);
        }
        Some(group_size)
    }

    #[inline]
    fn expert_group_allowed(
        expert: usize,
        group_size: usize,
        selected_groups: &[(usize, f32)],
    ) -> bool {
        let group = expert / group_size;
        selected_groups.iter().any(|&(g, _)| g == group)
    }

    /// Compute the routing decision for a single token's hidden state.
    pub fn route(&self, x: &[f32]) -> RoutingDecision {
        let mut scratch = RoutingScratch::new();
        self.route_with_scratch(x, &mut scratch)
    }

    /// Compute the routing decision using caller-owned reusable scratch.
    pub fn route_with_scratch(&self, x: &[f32], scratch: &mut RoutingScratch) -> RoutingDecision {
        debug_assert_eq!(x.len(), self.d_model);
        scratch.resize_for(self.num_experts);
        // logits = W_gate · x  (length: num_experts)
        matmul_row_major_into(
            &self.weights,
            x,
            &mut scratch.logits,
            self.num_experts,
            self.d_model,
        );
        // scores: the values that become mixing weights (no bias folded in).
        self.score(&mut scratch.logits);

        // selection scores: scores (+ correction bias) used only to choose
        // experts, never to weight them. Kept separate so DeepSeek's
        // correction bias cannot leak into the MoE mixing weights.
        match self.correction_bias.as_ref() {
            Some(bias) => {
                for ((selection, &score), &bias) in scratch
                    .selection
                    .iter_mut()
                    .zip(scratch.logits.iter())
                    .zip(bias.iter())
                {
                    *selection = score + bias;
                }
            }
            None => {
                scratch.selection.copy_from_slice(&scratch.logits);
            }
        }

        let selected_group_size =
            self.select_top_groups(&scratch.selection, &mut scratch.top_groups);

        // Top-K selection over the (optionally grouped) selection scores.
        // This is O(num_experts * top_k) with top_k small (8 for Qwen3-MoE),
        // avoids allocating/sorting every expert id, and has explicit
        // deterministic tie handling: higher score first, then lower id.
        for expert in 0..self.num_experts {
            if let Some(group_size) = selected_group_size {
                if !Self::expert_group_allowed(expert, group_size, &scratch.top_groups) {
                    continue;
                }
            }
            Self::insert_top_expert(
                &mut scratch.top_experts,
                expert,
                scratch.selection[expert],
                self.top_k,
            );
        }

        // Mixing weights are the *original* scores at the chosen experts.
        let mut experts = Vec::with_capacity(scratch.top_experts.len());
        let mut weights = Vec::with_capacity(scratch.top_experts.len());
        for &(expert, _) in &scratch.top_experts {
            experts.push(expert);
            weights.push(scratch.logits[expert as usize]);
        }
        if self.normalise_topk {
            let sum: f32 = weights.iter().sum();
            // Guard against `0.0`, negatives (impossible post-softmax but
            // cheap to defend), and non-finite values (`NaN`/`±inf`) that
            // can arise from a broken gate weight load. In any of those
            // degenerate cases we leave the unnormalised top-k weights
            // alone rather than producing `NaN`s the downstream mixture
            // would silently propagate.
            if sum.is_finite() && sum > 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }
        // DeepSeek `routed_scaling_factor`: scale the final mixing weights.
        if self.routed_scaling_factor != 1.0 && self.routed_scaling_factor.is_finite() {
            for w in &mut weights {
                *w *= self.routed_scaling_factor;
            }
        }
        RoutingDecision { experts, weights }
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
                let weights = if n > 0.0 {
                    vec![1.0 / n; experts.len()]
                } else {
                    Vec::new()
                };
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

    /// Identity gate (W = I) so that logit[i] = x[i]. Makes the routing
    /// arithmetic exactly predictable for the DeepSeek-style tests below.
    fn identity_gate(num_experts: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; num_experts * num_experts];
        for i in 0..num_experts {
            w[i * num_experts + i] = 1.0;
        }
        w
    }

    #[test]
    fn sigmoid_scoring_does_not_normalise_scores_before_topk() {
        // With sigmoid scoring and normalise_topk = false, the per-expert
        // weights are independent logistics, not a softmax simplex.
        let n = 4;
        let gate = LinearGate::with_routing(
            identity_gate(n),
            n,
            n,
            2,
            ScoringFunc::Sigmoid,
            /*normalise_topk=*/ false,
            None,
            1,
            1,
            1.0,
        );
        // x picks experts 0 and 1 (largest logits).
        let dec = gate.route(&[2.0, 1.0, -5.0, -5.0]);
        assert_eq!(dec.experts, vec![0, 1]);
        // sigmoid(2) ≈ 0.8808, sigmoid(1) ≈ 0.7311 — kept un-normalised.
        assert!(
            (dec.weights[0] - 0.880_797).abs() < 1e-4,
            "{:?}",
            dec.weights
        );
        assert!(
            (dec.weights[1] - 0.731_058).abs() < 1e-4,
            "{:?}",
            dec.weights
        );
    }

    #[test]
    fn correction_bias_steers_selection_but_not_weights() {
        let n = 4;
        // Logits favour expert 0, but a large correction bias on expert 3
        // makes it win selection. Its *weight*, however, is still the
        // unbiased sigmoid score of its own logit.
        let bias = vec![0.0, 0.0, 0.0, 10.0];
        let gate = LinearGate::with_routing(
            identity_gate(n),
            n,
            n,
            1,
            ScoringFunc::Sigmoid,
            false,
            Some(bias),
            1,
            1,
            1.0,
        );
        let dec = gate.route(&[3.0, 0.0, 0.0, 1.0]);
        assert_eq!(dec.experts, vec![3], "bias should pull expert 3 to the top");
        // Weight is sigmoid(logit[3]) = sigmoid(1.0), NOT sigmoid(1.0 + 10).
        assert!(
            (dec.weights[0] - 0.731_058).abs() < 1e-4,
            "{:?}",
            dec.weights
        );
    }

    #[test]
    fn grouped_topk_restricts_selection_to_surviving_groups() {
        // 8 experts, 2 groups of 4. Group 0 = experts 0..4, group 1 = 4..8.
        // Group 1 holds the two strongest logits, so with topk_group = 1
        // only experts from group 1 may be picked even though expert 0 in
        // group 0 also has a high-ish logit.
        let n = 8;
        let gate = LinearGate::with_routing(
            identity_gate(n),
            n,
            n,
            2,
            ScoringFunc::Sigmoid,
            false,
            None,
            /*n_group=*/ 2,
            /*topk_group=*/ 1,
            1.0,
        );
        // expert 0 strong, but group 1 (experts 5,6) collectively strongest.
        let x = vec![3.0, -9.0, -9.0, -9.0, 4.0, 5.0, -9.0, -9.0];
        let dec = gate.route(&x);
        for e in &dec.experts {
            assert!(*e >= 4, "selected expert {e} must be in group 1 (>=4)");
        }
        assert_eq!(dec.experts.len(), 2);
    }

    #[test]
    fn routed_scaling_factor_scales_final_weights() {
        let n = 4;
        let gate = LinearGate::with_routing(
            identity_gate(n),
            n,
            n,
            2,
            ScoringFunc::Sigmoid,
            /*normalise_topk=*/ true,
            None,
            1,
            1,
            /*routed_scaling_factor=*/ 2.5,
        );
        let dec = gate.route(&[2.0, 1.0, -5.0, -5.0]);
        // Normalised weights sum to 1, then scaled by 2.5 → sum ≈ 2.5.
        let sum: f32 = dec.weights.iter().sum();
        assert!((sum - 2.5).abs() < 1e-4, "sum={sum}");
    }

    fn reference_route_stable(gate: &LinearGate, x: &[f32]) -> RoutingDecision {
        let mut scores = vec![0.0f32; gate.num_experts];
        matmul_row_major_into(
            &gate.weights,
            x,
            &mut scores,
            gate.num_experts,
            gate.d_model,
        );
        gate.score(&mut scores);

        let mut selection = scores.clone();
        if let Some(bias) = gate.correction_bias.as_ref() {
            for (s, b) in selection.iter_mut().zip(bias.iter()) {
                *s += *b;
            }
        }

        if let Some((n_group, group_size)) = gate.group_filter_shape() {
            let mut group_scores: Vec<(usize, f32)> = (0..n_group)
                .map(|g| {
                    let slice = &selection[g * group_size..(g + 1) * group_size];
                    let mut top1 = f32::NEG_INFINITY;
                    let mut top2 = f32::NEG_INFINITY;
                    for &s in slice {
                        if s > top1 {
                            top2 = top1;
                            top1 = s;
                        } else if s > top2 {
                            top2 = s;
                        }
                    }
                    (g, top1 + if top2.is_finite() { top2 } else { 0.0 })
                })
                .collect();
            group_scores.sort_by(|a, b| {
                let by_score = b.1.total_cmp(&a.1);
                if by_score == Ordering::Equal {
                    a.0.cmp(&b.0)
                } else {
                    by_score
                }
            });
            let selected_groups: Vec<usize> = group_scores
                .iter()
                .take(gate.topk_group)
                .map(|&(g, _)| g)
                .collect();
            for (expert, s) in selection.iter_mut().enumerate() {
                let group = expert / group_size;
                if !selected_groups.contains(&group) {
                    *s = f32::NEG_INFINITY;
                }
            }
        }

        let mut idx: Vec<u32> = (0..gate.num_experts as u32).collect();
        idx.sort_by(|&a, &b| {
            let by_score = selection[b as usize].total_cmp(&selection[a as usize]);
            if by_score == Ordering::Equal {
                a.cmp(&b)
            } else {
                by_score
            }
        });
        idx.retain(|&i| selection[i as usize].is_finite());
        idx.truncate(gate.top_k);

        let mut weights: Vec<f32> = idx.iter().map(|&i| scores[i as usize]).collect();
        if gate.normalise_topk {
            let sum: f32 = weights.iter().sum();
            if sum.is_finite() && sum > 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }
        if gate.routed_scaling_factor != 1.0 && gate.routed_scaling_factor.is_finite() {
            for w in &mut weights {
                *w *= gate.routed_scaling_factor;
            }
        }
        RoutingDecision {
            experts: idx,
            weights,
        }
    }

    fn assert_routing_close(actual: &RoutingDecision, expected: &RoutingDecision) {
        assert_eq!(actual.experts, expected.experts);
        assert_eq!(actual.weights.len(), expected.weights.len());
        for (a, e) in actual.weights.iter().zip(expected.weights.iter()) {
            assert!(
                (a - e).abs() < 1e-6,
                "actual={actual:?} expected={expected:?}"
            );
        }
    }

    #[test]
    fn topk_ties_are_deterministic_by_lower_expert_id() {
        let gate = LinearGate::new(vec![0.0; 6 * 2], 6, 2, 4);
        let dec = gate.route(&[1.0, -1.0]);
        assert_eq!(dec.experts, vec![0, 1, 2, 3]);
        for w in dec.weights {
            assert!((w - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn scratch_route_matches_stable_reference_across_routing_modes() {
        let n = 8;
        let softmax_gate = LinearGate::new(identity_gate(n), n, n, 3);
        let biased_group_gate = LinearGate::with_routing(
            identity_gate(n),
            n,
            n,
            3,
            ScoringFunc::Sigmoid,
            true,
            Some(vec![0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.25, 0.0]),
            /*n_group=*/ 2,
            /*topk_group=*/ 1,
            /*routed_scaling_factor=*/ 1.5,
        );
        let sigmoid_gate = LinearGate::with_routing(
            identity_gate(n),
            n,
            n,
            4,
            ScoringFunc::Sigmoid,
            false,
            None,
            1,
            1,
            1.0,
        );
        let cases = [
            (
                &softmax_gate,
                vec![0.5, 2.0, 2.0, -1.0, 3.0, 0.0, 3.0, -2.0],
            ),
            (
                &biased_group_gate,
                vec![3.0, -1.0, 1.0, 2.0, 4.0, 5.0, 0.5, -9.0],
            ),
            (
                &sigmoid_gate,
                vec![-2.0, 4.0, 1.0, 4.0, 0.5, -3.0, 0.0, 2.0],
            ),
        ];

        let mut scratch = RoutingScratch::new();
        for (gate, x) in cases {
            let expected = reference_route_stable(gate, &x);
            let actual = gate.route_with_scratch(&x, &mut scratch);
            assert_routing_close(&actual, &expected);
        }
    }

    #[test]
    fn routing_scratch_reuses_internal_capacity() {
        let n = 8;
        let gate = LinearGate::with_routing(
            identity_gate(n),
            n,
            n,
            2,
            ScoringFunc::Sigmoid,
            true,
            None,
            /*n_group=*/ 2,
            /*topk_group=*/ 1,
            1.0,
        );
        let x = vec![3.0, -1.0, 1.0, 2.0, 4.0, 5.0, 0.5, -9.0];
        let mut scratch = RoutingScratch::new();
        let first = gate.route_with_scratch(&x, &mut scratch);
        assert_eq!(first.experts.len(), 2);
        let capacities = (
            scratch.logits.capacity(),
            scratch.selection.capacity(),
            scratch.top_groups.capacity(),
            scratch.top_experts.capacity(),
        );
        let second = gate.route_with_scratch(&x, &mut scratch);
        assert_routing_close(&second, &first);
        assert_eq!(
            capacities,
            (
                scratch.logits.capacity(),
                scratch.selection.capacity(),
                scratch.top_groups.capacity(),
                scratch.top_experts.capacity(),
            )
        );
    }

    #[test]
    fn softmax_default_routing_is_unchanged_by_new_fields() {
        // The plain `new` constructor must behave exactly as before:
        // softmax, renormalised top-K, unit scaling.
        let n = 4;
        let gate = LinearGate::new(identity_gate(n), n, n, 2);
        let dec = gate.route(&[5.0, 4.0, 0.0, 0.0]);
        assert_eq!(dec.experts, vec![0, 1]);
        let sum: f32 = dec.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
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

    #[test]
    fn router_linear_variant_dispatches_to_real_gate() {
        // Pin (gist Part 1, finding #4): `Router::Linear` is the
        // production wiring path when `[real_transformer].enabled =
        // true`. This test asserts the enum variant actually
        // dispatches `route()` to `LinearGate::route` (a `softmax(W·x)
        // → top-K`) rather than to the legacy Markov chain. Without
        // it a refactor that swaps the variant arms would compile
        // silently.
        let num_experts = 4;
        let d_model = 3;
        let mut w = Vec::with_capacity(num_experts * d_model);
        for i in 0..num_experts {
            for _ in 0..d_model {
                w.push((i + 1) as f32);
            }
        }
        let gate = LinearGate::new(w, num_experts, d_model, 2);
        let router = Router::Linear(Arc::new(gate));
        assert_eq!(router.num_experts(), num_experts as u32);
        assert_eq!(router.top_k(), 2);
        let dec = router.route(&[1.0, 1.0, 1.0], /* token_idx ignored */ 42);
        assert_eq!(dec.experts.len(), 2);
        assert_eq!(dec.experts[0], 3);
        assert_eq!(dec.experts[1], 2);
        let sum: f32 = dec.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
