//! UNEXECUTED definitions for a separately authorized locked/offline P2 gate.
//! No test grants permission to execute itself or to mutate the sealed S0 source.
use mer2_cuda_g3_s0 as s0;
use mer2_cuda_g3_s1 as s1;
use s1::abi::*;
use s1::host_token::{route, Session};
use s1::identity::{contract, source_files, Model, Variant, VerifiedSources};
use s1::state::{FakeCpu, Phase, Runner};
use s1::{Result, Stop};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap()
        .into()
}
fn sources() -> VerifiedSources {
    VerifiedSources::verify(&root(), &source_files()).unwrap()
}
fn setup() -> (Session, Runner<FakeCpu>) {
    let model = Model::generate(Variant::Primary).unwrap();
    let runner = Runner::new(&model, FakeCpu::default()).unwrap();
    (Session::new(model, sources()).unwrap(), runner)
}
fn error() -> Stop {
    Stop::new("INJECTED_FIRST_ERROR", "unexecuted negative witness")
}

#[derive(Clone, Debug)]
enum Fault {
    None,
    InitBefore,
    InitAfter,
    Submit(usize),
    Complete(usize),
    Deferred(usize),
    Cancel(usize),
    Panic(usize),
    Forge(usize, &'static str),
    Replay(usize),
    Early(usize),
    NonFinite(usize),
    Drain,
    Release,
    UnknownOwnership,
    ZeroDown,
}
#[derive(Default, Debug)]
struct Audit {
    initialized: usize,
    submitted: usize,
    completed: usize,
    drains: usize,
    releases: usize,
    drops: usize,
}
struct FaultAdapter {
    inner: FakeCpu,
    fault: Fault,
    audit: Arc<Mutex<Audit>>,
    index: usize,
    active_index: usize,
    deferred: bool,
    previous: Option<Completion>,
}
impl FaultAdapter {
    fn new(fault: Fault) -> (Self, Arc<Mutex<Audit>>) {
        let audit = Arc::new(Mutex::new(Audit::default()));
        (
            Self {
                inner: FakeCpu::default(),
                fault,
                audit: audit.clone(),
                index: 0,
                active_index: 0,
                deferred: false,
                previous: None,
            },
            audit,
        )
    }
}
impl Drop for FaultAdapter {
    fn drop(&mut self) {
        self.audit.lock().unwrap().drops += 1;
    }
}
impl Adapter for FaultAdapter {
    fn initialize(&mut self, c: u64, r: &[Resident], w: &[ExpertWeights]) -> Result<InitReceipt> {
        self.audit.lock().unwrap().initialized += 1;
        if matches!(self.fault, Fault::InitBefore) {
            return Err(error());
        }
        let receipt = self.inner.initialize(c, r, w)?;
        if matches!(self.fault, Fault::InitAfter) {
            return Err(error());
        }
        Ok(receipt)
    }
    fn submit(&mut self, op: &Operation, input: &Vector) -> Result<Submission> {
        let i = self.index;
        self.active_index = i;
        self.index += 1;
        self.audit.lock().unwrap().submitted += 1;
        match self.fault {
            Fault::Submit(n) if n == i => return Err(error()),
            Fault::Cancel(n) if n == i => {
                return Err(Stop::new("STOP_CANCELLED", "resource boundary"))
            }
            Fault::Panic(n) if n == i => panic!("intentional adapter fault"),
            _ => {}
        }
        let receipt = self.inner.submit(op, input)?;
        if matches!(self.fault,Fault::Deferred(n) if n==i) {
            self.deferred = true;
        }
        Ok(receipt)
    }
    fn complete(&mut self, op: &Operation) -> Result<Completion> {
        let i = self.active_index;
        self.audit.lock().unwrap().completed += 1;
        if matches!(self.fault,Fault::Complete(n) if n==i) {
            return Err(error());
        }
        let mut c = self.inner.complete(op)?;
        match self.fault {
            Fault::Replay(n) if n == i => return Ok(self.previous.clone().unwrap()),
            Fault::Forge(n, field) if n == i => match field {
                "session" => c.operation.lease.ticket.session += 1,
                "request" => c.operation.lease.ticket.request.push('X'),
                "nonce" => c.operation.lease.ticket.nonce += 1,
                "attempt" => c.operation.lease.ticket.attempt += 1,
                "position" => c.operation.lease.ticket.position += 1,
                "source" => c.operation.lease.ticket.source_sha256.push('X'),
                "contract" => c.operation.lease.ticket.contract_sha256.push('X'),
                "layer" => c.operation.lease.layer ^= 1,
                "rank" => c.operation.lease.rank ^= 1,
                "expert" => c.operation.lease.expert ^= 1,
                "slot" => c.operation.lease.slot ^= 1,
                "generation" => c.operation.lease.generation = 0,
                "weight" => c.operation.lease.weight_digest.push('X'),
                "context" => c.operation.lease.context_identity += 1,
                "event" => c.operation.lease.event_identity += 1,
                "input" => c.operation.lease.input_sha256.push('X'),
                "selected" => c.operation.lease.selected_weight_bits ^= 1,
                "sequence" => c.operation.sequence += 1,
                "kind" => c.operation.kind = OperationKind::Gate,
                "fence" => c.fence_confirmed = false,
                "missing" => c.stages = None,
                "digest" => c.output_sha256 = Some("forged".into()),
                _ => panic!("unknown witness"),
            },
            Fault::Early(n) if n == i => {
                let zero = Vector::new(&[0.0; 64])?;
                let stages = FfnStages(std::array::from_fn(|_| zero.clone()));
                c.output_sha256 = Some(stages.digest());
                c.stages = Some(stages);
            }
            Fault::NonFinite(n) if n == i => {
                let _ = Vector::new(&[f32::NAN; 64])?;
            }
            Fault::ZeroDown if op.kind == OperationKind::Readback => {
                let stages = c.stages.as_mut().unwrap();
                stages.0[4] = Vector::new(&[0.0; 64])?;
                c.output_sha256 = Some(stages.digest());
            }
            _ => {}
        }
        self.previous = Some(c.clone());
        Ok(c)
    }
    fn check(&mut self) -> Result<()> {
        if self.deferred {
            self.deferred = false;
            return Err(error());
        }
        self.inner.check()
    }
    fn drain(&mut self, c: u64) -> Result<CleanupReceipt> {
        self.audit.lock().unwrap().drains += 1;
        if matches!(self.fault, Fault::Drain) {
            return Err(Stop::new("CLEANUP_ERROR", "drain"));
        }
        let mut receipt = self.inner.drain(c)?;
        if matches!(self.fault, Fault::UnknownOwnership) {
            receipt.ownership_confirmed = false;
        }
        Ok(receipt)
    }
    fn release(&mut self, c: u64) -> Result<CleanupReceipt> {
        self.audit.lock().unwrap().releases += 1;
        if matches!(self.fault, Fault::Release) {
            return Err(Stop::new("CLEANUP_ERROR", "release"));
        }
        self.inner.release(c)
    }
}

#[test]
fn physical_sources_and_external_manifest_fail_closed() {
    let mut wrong = source_files();
    wrong.insert("src/state.rs".into(), "00".repeat(32));
    assert!(s1::identity::verify_source(&wrong).is_err());
    assert_eq!(
        s1::identity::verify_physical_s0(&root()).unwrap(),
        s1::identity::sealed_source_files()
    );
    assert!(s1::identity::verify_physical_s0(&root().join("absent-source-root")).is_err());
    sources().recheck().unwrap();
    assert_eq!(contract().unwrap()["s0"]["raw_logit_u32_count"], 192);
}
#[test]
fn fixture_bytes_and_all_41_tensors_match_separate_sealed_source() {
    for (v, old_v) in [
        (Variant::Primary, s0::identity::Variant::Primary),
        (Variant::TieV1, s0::identity::Variant::TieV1),
    ] {
        let a = Model::generate(v).unwrap();
        let b = s0::identity::Model::generate(old_v).unwrap();
        assert_eq!(a.hash(), b.hash());
        assert_eq!(a.tensors().len(), 41);
        for (name, t) in a.tensors() {
            let old = b.tensor(name).unwrap();
            assert_eq!(t.raw, old.raw);
            assert_eq!(t.hash().unwrap(), old.hash().unwrap());
        }
    }
}
#[test]
fn shape_q4_corruption_and_nonfinite_inputs_rejected() {
    assert!(Vector::new(&[0.0; 63]).is_err());
    assert!(Vector::new(&[f32::INFINITY; 64]).is_err());
    let model = Model::generate(Variant::Primary).unwrap();
    let t = model.tensor("L0.E0.gate").unwrap();
    let digest = s1::identity::hex(&t.hash().unwrap());
    assert!(Q4Matrix::checked(t, "L1.E0.gate", &digest).is_err());
    let mut bad = t.clone();
    bad.raw.pop();
    assert!(Q4Matrix::checked(&bad, &bad.name, &digest).is_err());
    let mut corrupt = t.clone();
    corrupt.raw[2] ^= 1;
    assert!(Q4Matrix::checked(&corrupt, &corrupt.name, &digest).is_err());
    let mut scale = t.clone();
    scale.raw[0] = 0;
    scale.raw[1] = 0x7c;
    assert!(Q4Matrix::checked(&scale, &scale.name, &digest).is_err());
    let mut tensors = model.tensors().clone();
    tensors.get_mut("embedding").unwrap().raw[0] ^= 1;
    assert!(Model::from_tensors(Variant::Primary, tensors).is_err());
    assert!(ExpertWeights::from_model(&model, 2, 0).is_err());
    assert!(ExpertWeights::from_model(&model, 0, 4).is_err());
}
#[test]
fn observed_ties_and_greedy_use_lowest_id_and_renormalized_rank() {
    let r = route(&[0.0; 4]).unwrap();
    assert_eq!(r.ranked, [0, 1, 2, 3]);
    assert_eq!(r.selected_weight_bits, [0.5f32.to_bits(); 2]);
    assert_eq!(s1::host_token::greedy(&[0.0; 32]).unwrap(), 0);
    let mut predicted = r.clone();
    predicted.observed = false;
    assert!(s1::state::validate_route(&predicted).is_err());
    let mut reordered = r;
    reordered.ranked.swap(0, 1);
    assert!(s1::state::validate_route(&reordered).is_err());
}
#[test]
fn both_three_token_chains_all_66_stages_and_historical_bits() {
    let (mut session, mut runner) = setup();
    let mut a = session.request("A", 3).unwrap();
    let mut b = session.request("B", 17).unwrap();
    let mut old = s0::reference::Session::new(
        s0::identity::Model::generate(s0::identity::Variant::Primary).unwrap(),
    );
    let mut oa = old.request("A", 3).unwrap();
    let mut ob = old.request("B", 17).unwrap();
    let contract = contract().unwrap();
    let historical = contract["s0"]["arrays"].as_array().unwrap();
    let mut events = std::collections::BTreeSet::new();
    for position in 0..3 {
        for (index, (r, o)) in [(&mut a, &mut oa), (&mut b, &mut ob)]
            .into_iter()
            .enumerate()
        {
            let t = r.ticket();
            assert_eq!(t.position, position);
            let out = r.step_token(session.model(), t, &mut runner).unwrap();
            let ot = o.ticket();
            let reference = o.step_token(old.model(), ot).unwrap();
            assert_eq!(out.stages.len(), 66);
            assert_eq!(out.completions.len(), 4);
            assert_eq!(out.logits_bits, reference.logits_bits);
            assert_eq!(out.next_id, reference.next_id);
            let accepted = &historical[index * 3 + position];
            let expected: Vec<u32> = accepted["logits_f32_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(out.logits_bits, expected);
            assert_eq!(
                out.next_id as u64,
                accepted["greedy_next_token"].as_u64().unwrap()
            );
            for (s, witness) in out.stages.iter().zip(&reference.stages) {
                assert_eq!(s.stage, witness.stage);
                assert_eq!(s.shape, witness.shape);
                assert_eq!(s.f32_bits, witness.f32_bits);
                assert_eq!(s.sequence, witness.sequence);
                assert_eq!(s.layer, witness.layer);
                if let (Some(x), Some(y)) = (&s.route, &witness.route) {
                    assert_eq!(x.ranked, y.ranked);
                    assert_eq!(x.probability_bits, y.probability_bits);
                    assert_eq!(x.selected_weight_bits, y.selected_weight_bits);
                }
            }
            for c in &out.completions {
                let l = c.lease();
                assert_eq!(l.ticket, out.ticket);
                assert!(l.generation > 0);
                assert!(events.insert(l.event_identity));
                assert_eq!(c.completions().len(), 7);
                for (i, p) in c.completions().iter().enumerate() {
                    assert_eq!(p.operation.lease, *l);
                    assert_eq!(p.operation.sequence, i);
                    assert!(p.fence_confirmed);
                }
            }
            for l in 0..2 {
                for p in 0..=position {
                    let (k, v) = r.kv(l).unwrap().at(p).unwrap();
                    let (ok, ov) = o.kv(l).unwrap().at(p).unwrap();
                    assert_eq!(s1::bits(&k), s0::bits(&ok));
                    assert_eq!(s1::bits(&v), s0::bits(&ov));
                }
            }
            assert_eq!(r.len(), position + 1);
            assert_eq!(r.ticket().position, position + 1);
            r.verify_output(&out).unwrap();
        }
    }
    assert_eq!(runner.report().counts.submitted, 168);
    assert_eq!(runner.report().counts.completed, 168);
    runner.finish();
    assert_eq!(runner.report().drain_attempts, 1);
    assert_eq!(runner.report().phase, Phase::Released);
}
#[test]
fn provider_result_is_consumed_by_host_aggregation() {
    let model = Model::generate(Variant::Primary).unwrap();
    let (adapter, _) = FaultAdapter::new(Fault::ZeroDown);
    let mut runner = Runner::new(&model, adapter).unwrap();
    let mut session = Session::new(model, sources()).unwrap();
    let mut r = session.request("A", 3).unwrap();
    let t = r.ticket();
    let out = r.step_token(session.model(), t, &mut runner).unwrap();
    for layer in 0..2 {
        let att = out
            .stages
            .iter()
            .find(|s| s.layer == Some(layer) && s.stage == "h_att")
            .unwrap();
        let end = out
            .stages
            .iter()
            .find(|s| s.layer == Some(layer) && s.stage == "h_out")
            .unwrap();
        // Zero down projections force the provider-specific aggregate to zero.
        for (a, b) in att.f32_bits.iter().zip(&end.f32_bits) {
            assert_eq!(f32::from_bits(*a), f32::from_bits(*b));
        }
    }
}
#[test]
fn every_operation_boundary_faults_cancel_and_panics_rollback() {
    for i in 0..28 {
        for fault in [
            Fault::Submit(i),
            Fault::Complete(i),
            Fault::Deferred(i),
            Fault::Cancel(i),
            Fault::Panic(i),
        ] {
            let model = Model::generate(Variant::Primary).unwrap();
            let (adapter, audit) = FaultAdapter::new(fault);
            let mut runner = Runner::new(&model, adapter).unwrap();
            let mut session = Session::new(model, sources()).unwrap();
            let mut r = session.request("A", 3).unwrap();
            let before = r.committed_bytes();
            let t = r.ticket();
            let first = r.step_token(session.model(), t, &mut runner).unwrap_err();
            assert_eq!(r.committed_bytes(), before);
            assert_eq!(r.len(), 0);
            let t = r.ticket();
            assert_eq!(
                r.step_token(session.model(), t, &mut runner).unwrap_err(),
                first
            );
            assert_eq!(runner.report().first_error.as_ref(), Some(&first));
            assert_eq!(runner.report().drain_attempts, 1);
            assert_eq!(audit.lock().unwrap().drains, 1);
            assert!(runner.report().counts.completed <= runner.report().counts.submitted);
        }
    }
}
#[test]
fn every_stage_cancellation_preserves_previously_committed_two_layer_kv() {
    for target in 0..66 {
        let (mut session, mut runner) = setup();
        let mut r = session.request("A", 3).unwrap();
        let t = r.ticket();
        r.step_token(session.model(), t, &mut runner).unwrap();
        let before = r.committed_bytes();
        let t = r.ticket();
        let e = r
            .step_with_hook(session.model(), t, &mut runner, &mut |s| {
                if s.sequence == target {
                    Err(Stop::new("STOP_CANCELLED", "token boundary"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(e.code, "STOP_CANCELLED");
        assert_eq!(r.committed_bytes(), before);
        assert_eq!(r.kv(0).unwrap().len(), 1);
        assert_eq!(r.kv(1).unwrap().len(), 1);
        assert_eq!(runner.report().drain_attempts, 1);
    }
}
#[test]
fn forged_stale_foreign_and_unconfirmed_completions_never_commit() {
    for field in [
        "session",
        "request",
        "nonce",
        "attempt",
        "position",
        "source",
        "contract",
        "layer",
        "rank",
        "expert",
        "slot",
        "generation",
        "weight",
        "context",
        "event",
        "input",
        "selected",
        "sequence",
        "kind",
        "fence",
        "missing",
        "digest",
    ] {
        let model = Model::generate(Variant::Primary).unwrap();
        let (adapter, _) = FaultAdapter::new(Fault::Forge(6, field));
        let mut runner = Runner::new(&model, adapter).unwrap();
        let mut session = Session::new(model, sources()).unwrap();
        let mut r = session.request("A", 3).unwrap();
        let before = r.committed_bytes();
        let t = r.ticket();
        assert!(r.step_token(session.model(), t, &mut runner).is_err());
        assert_eq!(r.committed_bytes(), before);
        assert_eq!(runner.report().counts.consumed_ffns, 0);
    }
}
#[test]
fn replay_early_output_and_nonfinite_readback_rejected() {
    for fault in [Fault::Replay(7), Fault::Early(0), Fault::NonFinite(6)] {
        let model = Model::generate(Variant::Primary).unwrap();
        let (adapter, _) = FaultAdapter::new(fault);
        let mut runner = Runner::new(&model, adapter).unwrap();
        let mut session = Session::new(model, sources()).unwrap();
        let mut r = session.request("A", 3).unwrap();
        let t = r.ticket();
        assert!(r.step_token(session.model(), t, &mut runner).is_err());
        assert!(r.is_empty());
    }
}
#[test]
fn initialization_faults_are_drained_once_without_releasing_early() {
    for fault in [Fault::InitBefore, Fault::InitAfter] {
        let model = Model::generate(Variant::Primary).unwrap();
        let (adapter, audit) = FaultAdapter::new(fault);
        let mut runner = Runner::new(&model, adapter).unwrap();
        let mut session = Session::new(model, sources()).unwrap();
        let mut r = session.request("A", 3).unwrap();
        let t = r.ticket();
        assert_eq!(
            r.step_token(session.model(), t, &mut runner).unwrap_err(),
            error()
        );
        runner.finish();
        assert_eq!(runner.report().drain_attempts, 1);
        let a = audit.lock().unwrap();
        assert_eq!(a.drains, 1);
        assert_eq!(a.releases, 1);
        assert_eq!(a.drops, 1);
    }
}
#[test]
fn unconfirmed_cleanup_quarantines_preserving_first_error() {
    for fault in [Fault::Drain, Fault::Release, Fault::UnknownOwnership] {
        let model = Model::generate(Variant::Primary).unwrap();
        let (adapter, audit) = FaultAdapter::new(fault.clone());
        let mut runner = Runner::new(&model, adapter).unwrap();
        let mut session = Session::new(model, sources()).unwrap();
        let mut r = session.request("A", 3).unwrap();
        let t = r.ticket();
        assert_eq!(
            r.step_with_hook(session.model(), t, &mut runner, &mut |_| Err(error()))
                .unwrap_err(),
            error()
        );
        assert_eq!(runner.report().first_error.as_ref(), Some(&error()));
        assert_eq!(runner.report().phase, Phase::Quarantined);
        assert!(!runner.report().secondary_errors.is_empty());
        runner.finish();
        drop(runner);
        let a = audit.lock().unwrap();
        assert_eq!(a.drains, 1);
        assert_eq!(a.drops, 0);
        assert_eq!(a.releases, usize::from(matches!(fault, Fault::Release)));
    }
}
#[test]
fn second_layer_failure_retains_prior_commit_and_other_request() {
    let model = Model::generate(Variant::Primary).unwrap();
    let (adapter, _) = FaultAdapter::new(Fault::Complete(42));
    let mut runner = Runner::new(&model, adapter).unwrap();
    let mut session = Session::new(model, sources()).unwrap();
    let mut a = session.request("A", 3).unwrap();
    let b = session.request("B", 17).unwrap();
    let t = a.ticket();
    a.step_token(session.model(), t, &mut runner).unwrap();
    let before = a.committed_bytes();
    let other = b.committed_bytes();
    let t = a.ticket();
    assert!(a.step_token(session.model(), t, &mut runner).is_err());
    assert_eq!(a.committed_bytes(), before);
    assert_eq!(b.committed_bytes(), other);
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 0);
}
#[test]
fn stale_ticket_foreign_request_and_duplicate_ids_fail_closed() {
    let (mut session, mut runner) = setup();
    let mut a = session.request("A", 3).unwrap();
    let b = session.request("B", 17).unwrap();
    assert!(session.request("A", 3).is_err());
    let foreign = b.ticket();
    let before = a.committed_bytes();
    assert_eq!(
        a.step_token(session.model(), foreign, &mut runner)
            .unwrap_err()
            .code,
        "STOP_TICKET"
    );
    assert_eq!(a.committed_bytes(), before);
    let (mut session, mut runner) = setup();
    let mut a = session.request("A", 3).unwrap();
    let old = a.ticket();
    a.step_token(session.model(), old.clone(), &mut runner)
        .unwrap();
    let before = a.committed_bytes();
    assert!(a.step_token(session.model(), old, &mut runner).is_err());
    assert_eq!(a.committed_bytes(), before);
}
#[test]
fn wrong_rank_prediction_and_two_outstanding_tokens_are_rejected() {
    for mode in 0..3 {
        let (mut session, mut runner) = setup();
        let a = session.request("A", 3).unwrap();
        let b = session.request("B", 17).unwrap();
        let t = a.ticket();
        runner.begin_token(&t).unwrap();
        if mode == 2 {
            assert!(runner.begin_token(&b.ticket()).is_err());
            continue;
        }
        let mut r = route(&[0.0; 4]).unwrap();
        if mode == 1 {
            r.observed = false;
        }
        assert!(runner
            .execute(
                &t,
                0,
                usize::from(mode == 0),
                &r,
                &Vector::new(&[0.0; 64]).unwrap()
            )
            .is_err());
        assert_eq!(runner.report().counts.submitted, 0);
    }
}
#[test]
fn generations_do_not_alias_between_owners() {
    let model = Model::generate(Variant::Primary).unwrap();
    let a = Runner::new(&model, FakeCpu::default()).unwrap();
    let b = Runner::new(&model, FakeCpu::default()).unwrap();
    assert_ne!(a.report().context_identity, b.report().context_identity);
    for (x, y) in a.residents().iter().zip(b.residents()) {
        assert!(y.generation > x.generation);
        assert_eq!(x.key, y.key);
    }
}
#[test]
fn receipt_observer_cannot_replace_execution_or_forge_hash() {
    let (mut session, mut runner) = setup();
    let mut a = session.request("A", 3).unwrap();
    let t = a.ticket();
    let e = a
        .step_with_hook(session.model(), t, &mut runner, &mut |s| {
            if s.stage == "expert_down" {
                s.f32_bits[0] ^= 1;
                s.sha256 = s.commitment();
            }
            Ok(())
        })
        .unwrap_err();
    assert_eq!(e.code, "STOP_RECEIPT");
    assert!(a.is_empty());
}
#[test]
fn fourth_position_legal_fifth_rejected_and_stale_output_rejected() {
    let (mut session, mut runner) = setup();
    let mut a = session.request("A", 3).unwrap();
    let t = a.ticket();
    let first = a.step_token(session.model(), t, &mut runner).unwrap();
    for _ in 1..4 {
        let t = a.ticket();
        a.step_token(session.model(), t, &mut runner).unwrap();
    }
    assert!(a.verify_output(&first).is_err());
    let before = a.committed_bytes();
    let t = a.ticket();
    assert_eq!(
        a.step_token(session.model(), t, &mut runner)
            .unwrap_err()
            .code,
        "STOP_INPUT"
    );
    assert_eq!(a.committed_bytes(), before);
}
#[test]
fn clean_release_is_once_and_cuda_always_refuses() {
    let model = Model::generate(Variant::Primary).unwrap();
    let (adapter, audit) = FaultAdapter::new(Fault::None);
    let mut runner = Runner::new(&model, adapter).unwrap();
    runner.finish();
    runner.finish();
    drop(runner);
    let a = audit.lock().unwrap();
    assert_eq!(a.drains, 1);
    assert_eq!(a.releases, 1);
    assert_eq!(a.drops, 1);
    assert_eq!(
        s1::cuda_unavailable().unwrap_err().code,
        "STOP_CUDA_UNAVAILABLE"
    );
}
