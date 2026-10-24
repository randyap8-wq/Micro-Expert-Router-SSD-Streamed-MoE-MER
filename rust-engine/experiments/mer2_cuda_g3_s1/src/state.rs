//! One ownership machine for every injected adapter. Fake receipts prove no physical residency.
use crate::abi::*;
use crate::host_token::{q4_matvec, silu, Route, Ticket};
use crate::identity::{contract_hash, source_hash, Model};
use crate::{fresh_identity, require, Result, Stop, D, EXPERTS, LAYERS, MAX_CONTEXT};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Cold,
    Ready,
    Active,
    Poisoned,
    Released,
    Quarantined,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub attempted: u64,
    pub submitted: u64,
    pub completed: u64,
    pub consumed_ffns: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub label: &'static str,
    pub phase: Phase,
    pub counts: Counts,
    pub first_error: Option<Stop>,
    pub secondary_errors: Vec<Stop>,
    pub drain_attempts: u8,
    pub cleanup: &'static str,
    pub context_identity: u64,
    pub initialization_attempted: bool,
    pub initialization_confirmed: bool,
}

pub struct Runner<A: Adapter> {
    adapter: Option<A>,
    weights: Vec<ExpertWeights>,
    residents: Vec<Resident>,
    model_sha256: String,
    source_sha256: String,
    contract_sha256: String,
    active: Option<Ticket>,
    cursor: usize,
    layer_route: Option<(Route, String)>,
    progress: BTreeMap<(u64, String, u64), (usize, u64)>,
    report: Report,
    settled: bool,
}
impl<A: Adapter> Runner<A> {
    /// No backend method is called here; initialize happens under the guarded owner.
    pub fn new(model: &Model, adapter: A) -> Result<Self> {
        let mut weights = Vec::new();
        let mut residents = Vec::new();
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                let w = ExpertWeights::from_model(model, layer, expert as u32)?;
                residents.push(Resident {
                    key: w.key().clone(),
                    slot: weights.len(),
                    generation: fresh_identity()?,
                });
                weights.push(w);
            }
        }
        Ok(Self {
            adapter: Some(adapter),
            weights,
            residents,
            model_sha256: model.hash().into(),
            source_sha256: source_hash()?,
            contract_sha256: contract_hash(),
            active: None,
            cursor: 0,
            layer_route: None,
            progress: BTreeMap::new(),
            settled: false,
            report: Report {
                label: crate::KIND,
                phase: Phase::Cold,
                counts: Counts::default(),
                first_error: None,
                secondary_errors: Vec::new(),
                drain_attempts: 0,
                cleanup: "NOT_ATTEMPTED",
                context_identity: fresh_identity()?,
                initialization_attempted: false,
                initialization_confirmed: false,
            },
        })
    }
    pub fn report(&self) -> &Report {
        &self.report
    }
    pub fn residents(&self) -> &[Resident] {
        &self.residents
    }
    fn adapter(&mut self) -> Result<&mut A> {
        self.adapter
            .as_mut()
            .ok_or_else(|| Stop::new("STOP_OWNERSHIP", "adapter no longer accessible"))
    }
    fn live(&self) -> Result<()> {
        if let Some(e) = &self.report.first_error {
            return Err(e.clone());
        }
        require(!self.settled, "STOP_OWNERSHIP", "retired owner")
    }
    fn fail(&mut self, e: Stop) {
        if self.report.first_error.is_none() {
            self.report.first_error = Some(e);
        } else {
            self.report.secondary_errors.push(e);
        }
        self.report.phase = Phase::Poisoned;
    }
    fn guarded<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.live()?;
        let result = catch_unwind(AssertUnwindSafe(|| f(self))).unwrap_or_else(|_| {
            Err(Stop::new(
                "STOP_PANIC",
                "adapter/ownership operation panicked",
            ))
        });
        if let Err(e) = &result {
            self.fail(e.clone());
            self.finish();
        }
        result
    }
    fn initialize(&mut self) -> Result<()> {
        if self.report.phase != Phase::Cold {
            return Ok(());
        }
        let ctx = self.report.context_identity;
        // All initialization failures remain inside this owner until checked cleanup.
        self.report.initialization_attempted = true;
        let receipt = self
            .adapter
            .as_mut()
            .ok_or_else(|| Stop::new("STOP_OWNERSHIP", "adapter"))?
            .initialize(ctx, &self.residents, &self.weights)?;
        self.adapter()?.check()?;
        require(
            receipt.context_identity == ctx
                && receipt.confirmed
                && receipt.residents == self.residents,
            "STOP_OWNERSHIP",
            "resident initialization not confirmed",
        )?;
        require(
            self.residents.len() == 8 && self.residents.iter().all(|r| r.generation != 0),
            "STOP_OWNERSHIP",
            "eight nonzero generations",
        )?;
        self.report.initialization_confirmed = true;
        self.report.phase = Phase::Ready;
        Ok(())
    }
    fn operation(&mut self, op: &Operation, input: &Vector) -> Result<Completion> {
        self.report.counts.attempted = self
            .report
            .counts
            .attempted
            .checked_add(1)
            .ok_or_else(|| Stop::new("STOP_OWNERSHIP", "attempt counter exhausted"))?;
        let submission = self.adapter()?.submit(op, input)?;
        self.report.counts.submitted = self
            .report
            .counts
            .submitted
            .checked_add(1)
            .ok_or_else(|| Stop::new("STOP_OWNERSHIP", "submission counter exhausted"))?;
        require(
            submission.operation == *op,
            "STOP_RECEIPT",
            "forged/stale submission",
        )?;
        self.adapter()?.check()?;
        let completion = self.adapter()?.complete(op)?;
        self.adapter()?.check()?;
        require(
            completion.operation == *op && completion.fence_confirmed,
            "STOP_RECEIPT",
            "wrong context/request/rank/expert/generation/event or unconfirmed fence",
        )?;
        match op.kind {
            OperationKind::Readback => {
                let stages = completion
                    .stages
                    .as_ref()
                    .ok_or_else(|| Stop::new("STOP_OUTPUT", "missing readback"))?;
                require(
                    completion.output_sha256.as_deref() == Some(stages.digest().as_str()),
                    "STOP_OUTPUT",
                    "raw stage commitment",
                )?;
            }
            _ => require(
                completion.stages.is_none() && completion.output_sha256.is_none(),
                "STOP_OUTPUT",
                "output before final readback",
            )?,
        }
        self.report.counts.completed = self
            .report
            .counts
            .completed
            .checked_add(1)
            .ok_or_else(|| Stop::new("STOP_OWNERSHIP", "completion counter exhausted"))?;
        Ok(completion)
    }
    /// Exactly one checked drain and, only with confirmed ownership, one checked release.
    /// Unconfirmed adapters are intentionally retained via forget, never automatically dropped.
    pub fn finish(&mut self) {
        if self.settled {
            return;
        }
        self.settled = true;
        if self.active.is_some() && self.report.first_error.is_none() {
            self.fail(Stop::new("STOP_CANCELLED", "owner retired during token"));
        }
        let ctx = self.report.context_identity;
        self.report.drain_attempts += 1;
        let drained = catch_unwind(AssertUnwindSafe(|| {
            let r = self.adapter()?.drain(ctx)?;
            self.adapter()?.check()?;
            require(
                r.context_identity == ctx && r.all_work_complete && r.ownership_confirmed,
                "STOP_DRAIN_UNCONFIRMED",
                "work or dependency ownership unconfirmed",
            )
        }))
        .unwrap_or_else(|_| Err(Stop::new("STOP_DRAIN_UNCONFIRMED", "drain panic")));
        if let Err(e) = drained {
            self.quarantine(e, "STOP_DRAIN_OR_DEPENDENCY_UNCONFIRMED");
            return;
        }
        let released = catch_unwind(AssertUnwindSafe(|| {
            let r = self.adapter()?.release(ctx)?;
            self.adapter()?.check()?;
            require(
                r.context_identity == ctx && r.all_work_complete && r.ownership_confirmed,
                "STOP_RELEASE_UNCONFIRMED",
                "release unconfirmed",
            )
        }))
        .unwrap_or_else(|_| Err(Stop::new("STOP_RELEASE_UNCONFIRMED", "release panic")));
        if let Err(e) = released {
            self.quarantine(e, "STOP_RELEASE_UNCONFIRMED");
            return;
        }
        self.report.cleanup = "CHECKED_DRAIN_AND_RELEASE_SIMULATED";
        if self.report.first_error.is_none() {
            self.report.phase = Phase::Released;
        }
        // Adapter contract requires no owned asynchronous resources after confirmed release.
        drop(self.adapter.take());
    }
    fn quarantine(&mut self, e: Stop, why: &'static str) {
        self.fail(e);
        self.report.phase = Phase::Quarantined;
        self.report.cleanup = why;
        if let Some(owner) = self.adapter.take() {
            std::mem::forget(owner);
        }
    }
}
impl<A: Adapter> Drop for Runner<A> {
    fn drop(&mut self) {
        self.finish();
    }
}
impl<A: Adapter> sealed::Sealed for Runner<A> {}
impl<A: Adapter> FfnProvider for Runner<A> {
    fn begin_token(&mut self, ticket: &Ticket) -> Result<()> {
        self.guarded(|s| {
            require(
                s.active.is_none(),
                "STOP_OWNERSHIP",
                "one outstanding token",
            )?;
            require(
                ticket.session != 0
                    && ticket.nonce != 0
                    && ticket.position < MAX_CONTEXT
                    && ticket.source_sha256 == s.source_sha256
                    && ticket.model_sha256 == s.model_sha256
                    && ticket.contract_sha256 == s.contract_sha256,
                "STOP_TICKET",
                "provider token identity",
            )?;
            let key = (ticket.session, ticket.request.clone(), ticket.nonce);
            let expected = s.progress.get(&key).copied().unwrap_or((0, 1));
            require(
                (ticket.position, ticket.attempt) == expected,
                "STOP_TICKET",
                "stale/foreign attempt",
            )?;
            s.initialize()?;
            s.active = Some(ticket.clone());
            s.cursor = 0;
            s.layer_route = None;
            s.report.phase = Phase::Active;
            Ok(())
        })
    }
    fn execute(
        &mut self,
        ticket: &Ticket,
        layer: usize,
        rank: usize,
        route: &Route,
        input: &Vector,
    ) -> Result<ConfirmedFfn> {
        self.guarded(|s| {
            require(
                s.active.as_ref() == Some(ticket)
                    && s.report.phase == Phase::Active
                    && layer < LAYERS
                    && rank < 2
                    && s.cursor == layer * 2 + rank,
                "STOP_OWNERSHIP",
                "token/layer/rank order",
            )?;
            validate_route(route)?;
            let input_hash = input.digest();
            if rank == 0 {
                s.layer_route = Some((route.clone(), input_hash.clone()));
            } else {
                require(
                    s.layer_route.as_ref() == Some(&(route.clone(), input_hash.clone())),
                    "STOP_ROUTE",
                    "rank input/route changed",
                )?;
            }
            let expert = route.ranked[rank];
            let slot = layer * EXPERTS + expert as usize;
            let resident = s
                .residents
                .get(slot)
                .ok_or_else(|| Stop::new("STOP_EXPERT", "slot"))?;
            let weight = s
                .weights
                .get(slot)
                .ok_or_else(|| Stop::new("STOP_EXPERT", "weight"))?;
            require(
                resident.key == *weight.key()
                    && resident.key.layer == layer
                    && resident.key.expert == expert
                    && resident.slot == slot
                    && resident.generation != 0,
                "STOP_EXPERT",
                "resident identity",
            )?;
            let lease = Lease {
                ticket: ticket.clone(),
                layer,
                rank,
                expert,
                slot,
                generation: resident.generation,
                weight_digest: resident.key.weight_digest.clone(),
                weight_sha256: weight.matrix_digests(),
                context_identity: s.report.context_identity,
                event_identity: fresh_identity()?,
                input_sha256: input_hash,
                selected_weight_bits: route.selected_weight_bits[rank],
            };
            let mut completions = Vec::new();
            for (sequence, kind) in OPERATIONS.into_iter().enumerate() {
                completions.push(s.operation(
                    &Operation {
                        lease: lease.clone(),
                        sequence,
                        kind,
                    },
                    input,
                )?);
            }
            let stages = completions
                .last()
                .and_then(|c| c.stages.clone())
                .ok_or_else(|| Stop::new("STOP_OUTPUT", "no fenced stages"))?;
            s.report.counts.consumed_ffns = s
                .report
                .counts
                .consumed_ffns
                .checked_add(1)
                .ok_or_else(|| Stop::new("STOP_OWNERSHIP", "FFN counter exhausted"))?;
            s.cursor += 1;
            Ok(ConfirmedFfn {
                lease,
                stages,
                completions,
            })
        })
    }
    fn end_token(&mut self, ticket: &Ticket) -> Result<()> {
        self.guarded(|s| {
            require(
                s.active.as_ref() == Some(ticket)
                    && s.cursor == 4
                    && s.report.counts.attempted == s.report.counts.submitted
                    && s.report.counts.submitted == s.report.counts.completed,
                "STOP_OWNERSHIP",
                "token ended before four confirmed experts",
            )?;
            let next = ticket
                .attempt
                .checked_add(1)
                .ok_or_else(|| Stop::new("STOP_TICKET", "attempt overflow"))?;
            s.progress.insert(
                (ticket.session, ticket.request.clone(), ticket.nonce),
                (ticket.position + 1, next),
            );
            s.active = None;
            s.layer_route = None;
            s.report.phase = Phase::Ready;
            Ok(())
        })
    }
    fn poison(&mut self, error: Stop) {
        if self.report.first_error.as_ref() != Some(&error) {
            self.fail(error);
        }
        self.finish();
    }
}

pub fn validate_route(r: &Route) -> Result<()> {
    require(
        r.observed,
        "STOP_ROUTE",
        "prediction is not observed routing",
    )?;
    let p = r.probability_bits.map(f32::from_bits);
    require(
        p.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0),
        "STOP_ROUTE",
        "probabilities",
    )?;
    let mut order = [0u32, 1, 2, 3];
    order.sort_by(|&a, &b| {
        p[b as usize]
            .partial_cmp(&p[a as usize])
            .unwrap()
            .then(a.cmp(&b))
    });
    require(
        order == r.ranked,
        "STOP_ROUTE",
        "descending probability/ascending ID",
    )?;
    let sum = p[order[0] as usize] + p[order[1] as usize];
    require(
        sum > 0.0 && sum.is_finite(),
        "STOP_ROUTE",
        "top2 denominator",
    )?;
    require(
        r.selected_weight_bits
            == [
                (p[order[0] as usize] / sum).to_bits(),
                (p[order[1] as usize] / sum).to_bits(),
            ],
        "STOP_ROUTE",
        "rank renormalization",
    )
}

/// Pure CPU implementation, using private owned buffers for every stage.
/// No S0 observations or precomputed golden outputs are used for execution.
#[derive(Default)]
pub struct FakeCpu {
    context: u64,
    residents: Vec<Resident>,
    weights: Vec<ExpertWeights>,
    pending: Option<Operation>,
    active: Option<Lease>,
    next_sequence: usize,
    input: Option<Vector>,
    stages: Vec<Vector>,
    released: bool,
}
impl Adapter for FakeCpu {
    fn initialize(
        &mut self,
        context: u64,
        residents: &[Resident],
        weights: &[ExpertWeights],
    ) -> Result<InitReceipt> {
        require(
            self.context == 0 && context != 0 && residents.len() == 8 && weights.len() == 8,
            "STOP_OWNERSHIP",
            "fake initialization",
        )?;
        self.context = context;
        self.residents = residents.to_vec();
        self.weights = weights.to_vec();
        Ok(InitReceipt {
            context_identity: context,
            residents: residents.to_vec(),
            confirmed: true,
        })
    }
    fn submit(&mut self, op: &Operation, input: &Vector) -> Result<Submission> {
        require(
            !self.released && self.pending.is_none() && op.lease.context_identity == self.context,
            "STOP_OWNERSHIP",
            "fake pending/context",
        )?;
        let l = &op.lease;
        let r = self
            .residents
            .get(l.slot)
            .ok_or_else(|| Stop::new("STOP_EXPERT", "fake slot"))?;
        require(
            r.generation == l.generation
                && r.key.layer == l.layer
                && r.key.expert == l.expert
                && r.key.weight_digest == l.weight_digest
                && l.event_identity != 0
                && input.digest() == l.input_sha256,
            "STOP_OWNERSHIP",
            "fake lease",
        )?;
        if op.sequence == 0 {
            require(
                self.active.is_none() && op.kind == OperationKind::UploadInput,
                "STOP_OWNERSHIP",
                "scratch reuse before completion",
            )?;
            self.active = Some(l.clone());
            self.next_sequence = 0;
            self.input = Some(input.clone());
            self.stages.clear();
        }
        require(
            self.active.as_ref() == Some(l)
                && op.sequence == self.next_sequence
                && OPERATIONS.get(op.sequence) == Some(&op.kind),
            "STOP_OWNERSHIP",
            "fake operation order",
        )?;
        let x = self
            .input
            .as_ref()
            .ok_or_else(|| Stop::new("STOP_INPUT", "fake input"))?
            .values();
        let weights = &self.weights[l.slot];
        let values = match op.kind {
            OperationKind::UploadInput | OperationKind::Readback => None,
            OperationKind::Gate => Some(q4_matvec(weights.matrices()[0].bytes(), &x)?),
            OperationKind::Up => Some(q4_matvec(weights.matrices()[1].bytes(), &x)?),
            OperationKind::Silu => Some(silu(&self.stages[0].values())?),
            OperationKind::Multiply => {
                let a = self.stages[2].values();
                let u = self.stages[1].values();
                Some((0..D).map(|i| a[i] * u[i]).collect())
            }
            OperationKind::Down => Some(q4_matvec(
                weights.matrices()[2].bytes(),
                &self.stages[3].values(),
            )?),
        };
        if let Some(values) = values {
            self.stages.push(Vector::new(&values)?);
        }
        self.pending = Some(op.clone());
        Ok(Submission {
            operation: op.clone(),
        })
    }
    fn complete(&mut self, op: &Operation) -> Result<Completion> {
        require(
            self.pending.as_ref() == Some(op),
            "STOP_RECEIPT",
            "fake missing/duplicate completion",
        )?;
        self.pending = None;
        self.next_sequence += 1;
        let stages = if op.kind == OperationKind::Readback {
            let values: [Vector; 5] = self
                .stages
                .clone()
                .try_into()
                .map_err(|_| Stop::new("STOP_OUTPUT", "five stages"))?;
            self.active = None;
            self.input = None;
            self.stages.clear();
            Some(FfnStages(values))
        } else {
            None
        };
        let output_sha256 = stages.as_ref().map(FfnStages::digest);
        Ok(Completion {
            operation: op.clone(),
            fence_confirmed: true,
            stages,
            output_sha256,
        })
    }
    fn check(&mut self) -> Result<()> {
        Ok(())
    }
    fn drain(&mut self, context: u64) -> Result<CleanupReceipt> {
        require(
            self.context == 0 || context == self.context,
            "STOP_DRAIN_UNCONFIRMED",
            "fake context",
        )?;
        self.pending = None;
        self.active = None;
        Ok(CleanupReceipt {
            context_identity: context,
            all_work_complete: true,
            ownership_confirmed: true,
        })
    }
    fn release(&mut self, context: u64) -> Result<CleanupReceipt> {
        require(
            self.pending.is_none() && self.active.is_none(),
            "STOP_RELEASE_UNCONFIRMED",
            "fake pending work",
        )?;
        self.weights.clear();
        self.residents.clear();
        self.stages.clear();
        self.input = None;
        self.released = true;
        Ok(CleanupReceipt {
            context_identity: context,
            all_work_complete: true,
            ownership_confirmed: true,
        })
    }
}
