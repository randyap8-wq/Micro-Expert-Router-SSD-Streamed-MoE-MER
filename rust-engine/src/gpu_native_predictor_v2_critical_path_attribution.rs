//! P1R diagnostic-only host attribution. No device-completion or speedup claim.
//! Recording is bounded, request-local, scalar-only and never gates execution.
use crate::predictor_v2::{p1e, P1jIdentity};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub(crate) const POSITIONS: usize = 143;
pub(crate) const ATTEMPTS: usize = 98;
const WIDTH: usize = 8;
const WORDS: usize = 12;
const POSITION_SLOTS: usize = 24;
const SET_SLOTS: usize = 8;
const MEMBER_SLOTS: usize = 6;
const PASS_SLOTS: usize = SET_SLOTS + WIDTH * MEMBER_SLOTS;
const ATTEMPT_SLOTS: usize = 4 + 2 * PASS_SLOTS;
const STRIDE: usize = POSITION_SLOTS + ATTEMPTS * ATTEMPT_SLOTS;
const SLOTS: usize = (POSITIONS + 1) * STRIDE; // final censored target retained

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Event {
    Observation,
    Freeze,
    Launch,
    Lease,
    Claim,
    Spawn,
    WriterStart,
    Payload,
    Close,
    LeaseRelease,
    DEntry,
    DP1e,
    Publication,
    Credit,
    Retire,
    DeferredRetire,
    CleanRoute,
    RequestStart,
    RequestEnd,
    P0Install,
    P0Reservation,
    Segment = POSITION_SLOTS,
    Boundary,
    Binding,
    Service,
    Selected = POSITION_SLOTS + 4,
    Probe,
    SourceSet,
    SourceBranch,
    LogicalSet,
    InstallSet,
    CopySet,
    LockedSet,
    SourceExpert = POSITION_SLOTS + 4 + SET_SLOTS,
    LogicalSource,
    Reservation,
    Stage,
    Commit,
    Installed,
}

struct Slot {
    state: AtomicU64,
    words: [AtomicU64; WORDS],
}
impl Default for Slot {
    fn default() -> Self {
        Self {
            state: AtomicU64::new(0),
            words: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

pub(crate) struct Recorder {
    origin: Instant,
    slots: Box<[Slot]>,
    invalid: AtomicBool,
}
#[derive(Clone)]
pub(crate) struct Context {
    recorder: Arc<Recorder>,
    pub(crate) position: usize,
    pub(crate) attempt: usize,
    pub(crate) pass: usize,
    pub(crate) member: usize,
    ids: [u32; WIDTH],
    count: usize,
    pub(crate) logical: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Record {
    pub(crate) slot: usize,
    pub(crate) words: [u64; WORDS],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Snapshot {
    pub(crate) invalid: bool,
    pub(crate) capacity: usize,
    pub(crate) storage_bytes: usize,
    pub(crate) records: Vec<Record>,
    pub(crate) layout: Option<RecorderLayout>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RecorderLayout {
    clock: String,
    words: usize,
    position_stride: usize,
    position_slots: usize,
    attempt_stride: usize,
    pass_stride: usize,
    member_stride: usize,
    timestamp_words: [usize; 2],
    event_offsets: Vec<(Event, usize)>,
    availability: String,
}
impl RecorderLayout {
    fn current() -> Self {
        Self { clock:"checked u64 ns from the same request-local P1E monotonic origin".into(), words:WORDS,
        position_stride:STRIDE,position_slots:POSITION_SLOTS,attempt_stride:ATTEMPT_SLOTS,pass_stride:PASS_SLOTS,member_stride:MEMBER_SLOTS,
        timestamp_words:[0,1], event_offsets:[Event::Observation,Event::Freeze,Event::Launch,Event::Lease,Event::Claim,Event::Spawn,Event::WriterStart,Event::Payload,Event::Close,Event::LeaseRelease,Event::DEntry,Event::DP1e,Event::Publication,Event::Credit,Event::Retire,Event::DeferredRetire,Event::CleanRoute,Event::RequestStart,Event::RequestEnd,Event::P0Install,Event::P0Reservation,Event::Segment,Event::Boundary,Event::Binding,Event::Service,Event::Selected,Event::Probe,Event::SourceSet,Event::SourceBranch,Event::LogicalSet,Event::InstallSet,Event::CopySet,Event::LockedSet,Event::SourceExpert,Event::LogicalSource,Event::Reservation,Event::Stage,Event::Commit,Event::Installed].into_iter().map(|e|(e,e as usize)).collect(),
        availability:"absent slot is unmeasured/not-reached; zero timestamp is valid; spans are inclusive and must not be summed with nested spans; set records carry result/count/IDs, identity records carry result/sequence/expert/logical-generation/epoch/writer/runtime/request/context/arena".into() }
    }
}
impl Recorder {
    pub(crate) fn new(origin: Instant) -> Arc<Self> {
        // Allocation is outside stepping. A recorder allocation failure marks
        // evidence invalid; it cannot cancel or change ordinary execution.
        let mut slots = Vec::new();
        let invalid = slots.try_reserve_exact(SLOTS).is_err();
        if !invalid {
            slots.resize_with(SLOTS, Slot::default);
        }
        Arc::new(Self {
            origin,
            slots: slots.into_boxed_slice(),
            invalid: AtomicBool::new(invalid),
        })
    }
    pub(crate) fn context(self: &Arc<Self>, position: usize, attempt: usize) -> Context {
        Context {
            recorder: self.clone(),
            position,
            attempt,
            pass: 0,
            member: 0,
            ids: [0; WIDTH],
            count: 0,
            logical: false,
        }
    }
    pub(crate) fn snapshot(&self) -> Snapshot {
        let mut invalid = self.invalid.load(Ordering::Acquire);
        let records = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| match entry.state.load(Ordering::Acquire) {
                0 => None,
                2 => Some(Record {
                    slot,
                    words: std::array::from_fn(|i| entry.words[i].load(Ordering::Relaxed)),
                }),
                _ => {
                    invalid = true;
                    None
                }
            })
            .collect();
        Snapshot {
            invalid,
            capacity: self.slots.len(),
            storage_bytes: self.slots.len() * std::mem::size_of::<Slot>(),
            records,
            layout: Some(RecorderLayout::current()),
        }
    }
}

fn slot_index(
    position: usize,
    attempt: usize,
    pass: usize,
    member: usize,
    event: Event,
) -> Option<usize> {
    if position > POSITIONS || attempt >= ATTEMPTS || pass >= 2 || member >= WIDTH {
        return None;
    }
    let kind = event as usize;
    let offset = if kind < POSITION_SLOTS {
        kind
    } else if kind < POSITION_SLOTS + 4 {
        POSITION_SLOTS + attempt * ATTEMPT_SLOTS + kind - POSITION_SLOTS
    } else if kind < POSITION_SLOTS + 4 + SET_SLOTS {
        POSITION_SLOTS + attempt * ATTEMPT_SLOTS + 4 + pass * PASS_SLOTS + kind - POSITION_SLOTS - 4
    } else {
        POSITION_SLOTS
            + attempt * ATTEMPT_SLOTS
            + 4
            + pass * PASS_SLOTS
            + SET_SLOTS
            + member * MEMBER_SLOTS
            + kind
            - POSITION_SLOTS
            - 4
            - SET_SLOTS
    };
    position.checked_mul(STRIDE)?.checked_add(offset)
}
impl Context {
    pub(crate) fn selected(&self, ids: &[u32]) -> Self {
        let mut c = self.clone();
        if ids.len() > WIDTH {
            c.invalidate();
        } else {
            c.ids[..ids.len()].copy_from_slice(ids);
            c.count = ids.len();
        }
        c
    }
    pub(crate) fn for_id(&self, id: u32) -> Option<Self> {
        match self.ids[..self.count].iter().position(|v| *v == id) {
            Some(i) => Some(self.member(i)),
            None => {
                self.invalidate();
                None
            }
        }
    }
    pub(crate) fn logical(&self) -> Self {
        Self {
            logical: true,
            ..self.clone()
        }
    }
    pub(crate) fn target(&self, position: usize) -> Self {
        Self {
            position,
            attempt: 0,
            pass: 0,
            member: 0,
            ..self.clone()
        }
    }
    pub(crate) fn pass(&self, pass: usize) -> Self {
        Self {
            pass,
            ..self.clone()
        }
    }
    pub(crate) fn member(&self, member: usize) -> Self {
        Self {
            member,
            ..self.clone()
        }
    }
    pub(crate) fn invalidate(&self) {
        self.recorder.invalid.store(true, Ordering::Release);
    }
    pub(crate) fn now(&self) -> Option<u64> {
        let ns = checked_ns(self.recorder.origin.elapsed().as_nanos());
        if ns.is_none() {
            self.invalidate();
        }
        ns
    }
    pub(crate) fn record(&self, event: Event, start: Option<u64>, end: Option<u64>, data: &[u64]) {
        let Some(index) = slot_index(self.position, self.attempt, self.pass, self.member, event)
        else {
            self.invalidate();
            return;
        };
        let Some(entry) = self.recorder.slots.get(index) else {
            self.invalidate();
            return;
        };
        if event == Event::Retire && entry.state.load(Ordering::Acquire) == 2 {
            return;
        }
        let (Some(start), Some(end)) = (start, end) else {
            self.invalidate();
            return;
        };
        if end < start
            || data.len() > WORDS - 2
            || entry
                .state
                .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
        {
            self.invalidate();
            return;
        }
        entry.words[0].store(start, Ordering::Relaxed);
        entry.words[1].store(end, Ordering::Relaxed);
        for (i, value) in data.iter().enumerate() {
            entry.words[i + 2].store(*value, Ordering::Relaxed);
        }
        entry.state.store(2, Ordering::Release);
    }
    pub(crate) fn stamp(&self, event: Event, data: &[u64]) {
        let ns = self.now();
        self.record(event, ns, ns, data);
    }
    pub(crate) fn set(&self, event: Event, start: Option<u64>, ids: &[u32], code: u64) {
        if ids.len() > WIDTH {
            self.invalidate();
            return;
        }
        let mut data = [0; 10];
        data[0] = code;
        data[1] = ids.len() as u64;
        for (i, id) in ids.iter().enumerate() {
            data[i + 2] = u64::from(*id);
        }
        self.record(event, start, self.now(), &data);
    }
    pub(crate) fn identity(&self, event: Event, start: Option<u64>, id: P1jIdentity, code: u64) {
        self.record(event, start, self.now(), &identity_words(id, code));
    }
    pub(crate) fn span_identity(&self, event: Event, id: P1jIdentity, code: u64) -> Span {
        self.span(event, &identity_words(id, code))
    }
}
fn identity_words(id: P1jIdentity, code: u64) -> [u64; 10] {
    [
        code,
        id.candidate.sequence,
        id.candidate.expert as u64,
        id.logical_generation,
        id.epoch as u64,
        id.writer_sequence,
        id.candidate.request.runtime_namespace,
        id.candidate.request.request_sequence,
        id.candidate.namespace.context,
        id.candidate.namespace.arena as u64,
    ]
}
pub(crate) fn checked_ns(ns: u128) -> Option<u64> {
    u64::try_from(ns).ok()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
enum Readiness {
    BEFORE_D,
    AFTER_D,
    D_WINDOW_AMBIGUOUS,
    READY_DURING_D_WINDOW,
    UNAVAILABLE,
}
fn readiness(close: Option<(u64, u64)>, d: Option<u64>, consumed: bool) -> Readiness {
    match (close, d) {
        (Some((lo, hi)), Some(d)) if lo <= hi && hi < d => Readiness::BEFORE_D,
        (Some((lo, hi)), Some(d)) if lo <= hi && lo > d && !consumed => Readiness::AFTER_D,
        (Some((lo, hi)), Some(d)) if lo <= hi && lo > d && consumed => {
            Readiness::READY_DURING_D_WINDOW
        }
        (Some((lo, hi)), Some(_)) if lo <= hi => Readiness::D_WINDOW_AMBIGUOUS,
        _ => Readiness::UNAVAILABLE,
    }
}

fn semantic_candidate(mut c: p1e::Candidate) -> p1e::Candidate {
    c.request.runtime_namespace = 0;
    c.request.request_sequence = 0;
    c.namespace.runtime = 0;
    c.namespace.context = 0;
    c.namespace.arena = 0;
    c
}

// An unfinished span records failure/unwind without changing the operation's result.
pub(crate) struct Span {
    context: Context,
    event: Event,
    start: Option<u64>,
    data: [u64; 10],
}
impl Context {
    pub(crate) fn span(&self, event: Event, data: &[u64]) -> Span {
        let mut words = [0; 10];
        if data.len() > words.len() {
            self.invalidate();
        } else {
            words[..data.len()].copy_from_slice(data);
        }
        Span {
            context: self.clone(),
            event,
            start: self.now(),
            data: words,
        }
    }
}
impl Span {
    pub(crate) fn value(&mut self, index: usize, value: u64) {
        if let Some(word) = self.data.get_mut(index) {
            *word = value;
        } else {
            self.context.invalidate();
        }
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        self.context
            .record(self.event, self.start, self.context.now(), &self.data);
    }
}

#[cfg(test)]
pub(crate) fn historical_source(path: &str) -> &'static str {
    // Test-only immutable Git-object read. The bounded leak supplies include_str's
    // lifetime to historical witnesses; no checkout/index/ref mutation occurs.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let object = format!("a6b9cbcd6e6b576f90a1d875c9351c7276c950e2:rust-engine/src/{path}");
    let output = std::process::Command::new("git")
        .args(["show", &object])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "historical object unavailable: {path}"
    );
    String::from_utf8(output.stdout).unwrap().leak()
}

#[cfg(test)]
impl Recorder {
    pub(crate) fn fixture(origin: Instant, positions: usize) -> Arc<Self> {
        Arc::new(Self {
            origin,
            slots: (0..positions * STRIDE).map(|_| Slot::default()).collect(),
            invalid: AtomicBool::new(false),
        })
    }
}
#[cfg(test)]
impl Snapshot {
    pub(crate) fn event(&self, position: usize, event: Event) -> Option<[u64; WORDS]> {
        let slot = slot_index(position, 0, 0, 0, event)?;
        self.records
            .iter()
            .find(|r| r.slot == slot)
            .map(|r| r.words)
    }
}

// Separate P1R command/transport. P1Q3 production and its six-pair runner remain frozen.
pub(crate) use driver::{run_command, CommandArgs};
mod driver {
    use super::*;
    use crate::gpu_native_real_benchmark as evidence;
    use crate::gpu_native_token_loop::{P1jLaunchSnapshot, P1jRequestMode, P1qInitialSnapshot};
    use crate::predictor_v2::{ObservationConfig, ReconciliationSnapshot, RequestPhase};
    use serde_json::{json, Value};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const SCHEMA: &str = "mer.predictor-v2-p1r-critical-path-attribution.v1";
    const CHILD_PROTOCOL: &str = "mer.predictor-v2-p1r-arm.v1";
    const MAX_POSITIONS: usize = 4096;
    const OUTPUT_TOKENS: usize = 128;
    const PLANNED_POSITIONS: usize = 143;
    #[derive(Clone, Debug, clap::Args, Serialize)]
    pub(crate) struct CommandArgs {
        #[arg(long)]
        pub(crate) config: PathBuf,
        #[arg(long)]
        request_json: PathBuf,
        #[arg(long)]
        expected_adapter_name: String,
        #[arg(long)]
        report_out: PathBuf,
        #[arg(long)]
        diagnostic_id: String,
        #[arg(long, hide = true, requires = "p1r_child_arm")]
        p1r_child_protocol: Option<String>,
        #[arg(long, hide = true, value_enum, requires = "p1r_child_protocol")]
        p1r_child_arm: Option<Arm>,
        #[arg(long, hide = true)]
        control_artifact: Option<PathBuf>,
    }
    // Reject unsupported request semantics instead of silently discarding them.
    // Sampling must explicitly select greedy; neutral optional fields are accepted.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RequestInput {
        prompt: Option<String>,
        messages: Option<Vec<Message>>,
        max_tokens: usize,
        temperature: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        n: Option<usize>,
        stream: Option<bool>,
        frequency_penalty: Option<f64>,
        presence_penalty: Option<f64>,
        repetition_penalty: Option<f64>,
        model: Option<String>,
    }

    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Message {
        role: String,
        content: String,
    }

    fn parse_request(bytes: &[u8]) -> Result<(String, usize)> {
        let input: RequestInput = serde_json::from_slice(bytes)?;
        if input.max_tokens != OUTPUT_TOKENS
            || !(input.temperature == Some(0.0) || input.top_k == Some(1))
            || input.temperature.is_some_and(|v| v != 0.0)
            || input.top_k.is_some_and(|v| v != 1)
            || input.top_p.is_some_and(|v| v != 1.0)
            || input.n.is_some_and(|v| v != 1)
            || input.stream == Some(true)
            || input.frequency_penalty.is_some_and(|v| v != 0.0)
            || input.presence_penalty.is_some_and(|v| v != 0.0)
            || input.repetition_penalty.is_some_and(|v| v != 1.0)
        {
            return Err("requires max_tokens=128 and explicit unmodified greedy sampling (temperature=0 or top_k=1)".into());
        }
        // A model name is descriptive only; strict artifact identity is validated
        // against the resolved checkpoint below, as in the existing CLI helpers.
        let _ = input.model;
        let prompt = match (input.prompt, input.messages) {
            (Some(prompt), None) => prompt,
            (None, Some(messages)) if !messages.is_empty() => {
                if messages.iter().any(|m| {
                    !matches!(m.role.as_str(), "system" | "user" | "assistant")
                        || m.content.trim().is_empty()
                }) {
                    return Err(
                        "messages require explicit text content and system/user/assistant roles"
                            .into(),
                    );
                }
                let values = messages
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                crate::flatten_bench_messages(&values)
            }
            _ => {
                return Err(
                    "requires exactly one nonempty string prompt or text messages array".into(),
                )
            }
        };
        if prompt.trim().is_empty() {
            return Err("prompt must contain non-whitespace text".into());
        }
        Ok((prompt, input.max_tokens))
    }

    fn add(a: usize, b: usize) -> Result<usize> {
        a.checked_add(b)
            .ok_or_else(|| "observation count overflow".into())
    }
    fn mul(a: usize, b: usize) -> Result<usize> {
        a.checked_mul(b)
            .ok_or_else(|| "observation count overflow".into())
    }
    fn increment(count: &mut usize) -> Result<()> {
        *count = add(*count, 1)?;
        Ok(())
    }

    fn observation_config(prompt_length: usize, output_tokens: usize) -> Result<ObservationConfig> {
        let decode = output_tokens.checked_sub(1).ok_or("zero output tokens")?;
        let positions = add(prompt_length, decode)?;
        if prompt_length == 0 || !(3..=MAX_POSITIONS).contains(&positions) {
            return Err("requires 3..=4096 completed positions including a nonempty prompt".into());
        }
        // P0 retains 48 route records per completed position, and at most 64 new
        // cells per adjacent top-8 transition. Each (layer pair, position-kind pair)
        // has only 128*128 possible cells. Count prompt and decode buckets separately;
        // the single prompt->decode transition is a distinct 64-cell bucket.
        // This is the smallest common capacity implied by these worst-case bounds,
        // with no padding. It also dominates P1E's same-layer table and one report
        // record per completed position (including the final censored nomination).
        // Recovery does not retrain P0; P1E keeps its frozen 512 events/record ceiling.
        let cells =
            |transitions: usize| -> Result<usize> { Ok(mul(transitions, 64)?.min(128 * 128)) };
        let within = mul(47, add(cells(prompt_length)?, cells(decode)?)?)?;
        let prompt_edges = prompt_length.checked_sub(1).ok_or("empty prompt")?;
        let decode_edges = if decode == 0 {
            0
        } else {
            decode.checked_sub(1).ok_or("decode edge overflow")?
        };
        let across = add(cells(prompt_edges)?, cells(decode_edges)?)?;
        let transitions = add(add(within, across)?, if decode > 0 { 64 } else { 0 })?;
        Ok(ObservationConfig {
            phase: RequestPhase::Measured,
            phase_run_index: 0,
            prompt_length,
            capacity_per_collection: transitions.max(mul(positions, 48)?),
        })
    }

    fn ensure_output_absent(path: &Path) -> Result<()> {
        match std::fs::symlink_metadata(path) {
            Ok(_) => Err(format!(
                "refusing to overwrite qualification evidence {}",
                path.display()
            )
            .into()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn write_report(path: &Path, report: &impl Serialize) -> Result<()> {
        ensure_output_absent(path)?;
        let bytes = serde_json::to_vec_pretty(report)?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let name = path
            .file_name()
            .ok_or("report path has no file name")?
            .to_string_lossy();
        let temp = parent.join(format!(".{name}.{}.tmp", std::process::id()));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let result = (|| -> Result<()> {
            file.write_all(&bytes)?;
            file.write_all(b"\n")?;
            file.flush()?;
            file.sync_all()?;
            // Atomic no-clobber publication in the same directory. Unlike rename,
            // hard_link cannot replace evidence created after the preflight check.
            std::fs::hard_link(&temp, path)?;
            std::fs::remove_file(&temp)?;
            std::fs::File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
    enum Arm {
        Control,
        Treatment,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ArmProvenance {
        #[serde(deserialize_with = "Deserialize::deserialize")]
        provenance: evidence::BenchmarkProvenance,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        config_path: String,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        config_sha256: String,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        model_identity: crate::greedy_parity::ModelIdentityEvidence,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        production_configuration: evidence::ProductionConfiguration,
    }

    fn provenance_identity(provenance: &ArmProvenance) -> Result<Value> {
        // Compare typed provenance directly in every process. A JSON-text
        // roundtrip can reparse a shortest f32 decimal as a different f64 Value.
        Ok(serde_json::to_value(provenance)?)
    }

    fn bind_provenance_identity(expected: &mut Option<Value>, identity: Value) -> Result<()> {
        if expected.as_ref().is_some_and(|p| p != &identity) {
            return Err("input/config/model/executable provenance drift".into());
        }
        *expected = Some(identity);
        Ok(())
    }

    struct PreparedArm {
        spec: crate::ResolvedRealCliSpec,
        provenance: ArmProvenance,
    }
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ArmReport {
        #[serde(deserialize_with = "Deserialize::deserialize")]
        arm: Arm,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        provenance: Option<ArmProvenance>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        runtime_build_attempted: bool,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        runtime_constructed: bool,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        runtime_resolved_config_sha256: Option<String>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        model_load: Option<crate::greedy_parity::ModelLoadEvidence>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        adapter: Option<crate::backend::GpuDeviceIdentity>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        runtime_contract: Option<evidence::RuntimeContractEvidence>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        generated_token_ids: Vec<u32>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        generated_token_ids_sha256: String,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        completed_positions: usize,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        observation_capacity_per_collection: usize,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        raw_p1e_report: Option<p1e::Report>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        predictor_v2_snapshot: Option<ReconciliationSnapshot>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        p1j_launch_before: Option<P1jLaunchSnapshot>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        p1j_launch_after: Option<P1jLaunchSnapshot>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        p1j_launch_delta: Option<P1jLaunchSnapshot>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        runtime_counters: Option<evidence::RequestSnapshots>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        runtime_shutdown: Option<crate::greedy_parity::BackgroundShutdownEvidence>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        errors: Vec<String>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        ordinary_invariants_pass: bool,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        mode: P1jRequestMode,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        resources_before_opt_in: Option<crate::gpu_native_residency::P1qResourceSnapshot>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        initial: Option<P1qInitialSnapshot>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        resource_identity: Option<Value>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        sidecar_freshly_initialized: bool,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        initial_state_pass: bool,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        request_wall_ns: Option<u64>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        generated_tps: Option<f64>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        planned_position_tps: Option<f64>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        production_install_after: Option<Value>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        production_install_delta: Option<Value>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        diagnostics: Option<Snapshot>,
        #[serde(skip)]
        recorder: Option<Arc<Recorder>>,
    }
    impl ArmReport {
        fn new(arm: Arm) -> Self {
            Self {
                arm,
                provenance: None,
                runtime_build_attempted: false,
                runtime_constructed: false,
                runtime_resolved_config_sha256: None,
                model_load: None,
                adapter: None,
                runtime_contract: None,
                generated_token_ids: Vec::new(),
                generated_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&[]),
                completed_positions: 0,
                observation_capacity_per_collection: 0,
                raw_p1e_report: None,
                predictor_v2_snapshot: None,
                p1j_launch_before: None,
                p1j_launch_after: None,
                p1j_launch_delta: None,
                runtime_counters: None,
                runtime_shutdown: None,
                errors: Vec::new(),
                ordinary_invariants_pass: false,
                mode: P1jRequestMode::InertResourceOnly,
                resources_before_opt_in: None,
                initial: None,
                resource_identity: None,
                sidecar_freshly_initialized: false,
                initial_state_pass: false,
                request_wall_ns: None,
                generated_tps: None,
                planned_position_tps: None,
                production_install_after: None,
                production_install_delta: None,
                diagnostics: None,
                recorder: None,
            }
        }
    }
    fn control_gate(p: &ReconciliationSnapshot) -> Vec<String> {
        let mut errors = Vec::new();
        if p.incomplete.is_some() {
            errors.push(format!("control accounting incomplete: {:?}", p.incomplete));
        }
        // No P0 movement lifecycle stage is permitted in the control arm.
        let mut zero = p.clone();
        zero.incomplete = None;
        if zero != ReconciliationSnapshot::default() {
            errors.push(format!(
                "control P0 movement lifecycle must be entirely zero: {p:?}"
            ));
        }
        errors
    }
    // Reconcile retained launch before/after/delta evidence independently.
    fn launch_diagnostics_gate(report: &ArmReport) -> Vec<String> {
        let mut errors = Vec::new();
        match (
            report.p1j_launch_before,
            report.p1j_launch_after,
            report.p1j_launch_delta,
        ) {
            (Some(before), Some(after), Some(delta)) => {
                if after.checked_delta(before) != Some(delta)
                    || !before.reconciled()
                    || !after.reconciled()
                    || !delta.reconciled()
                {
                    errors.push(format!(
                        "{:?} P1J launch diagnostics incomplete or unreconciled",
                        report.arm
                    ));
                }
                if report.mode == P1jRequestMode::InertResourceOnly
                    && [before, after, delta]
                        .iter()
                        .any(|s| *s != P1jLaunchSnapshot::default())
                {
                    errors.push("control P1J launch diagnostics must be entirely zero".into());
                }
            }
            _ => errors.push(format!(
                "{:?} P1J launch diagnostics unavailable",
                report.arm
            )),
        }
        errors
    }

    fn finish_launch_diagnostics(report: &mut ArmReport, after: P1jLaunchSnapshot) {
        report.p1j_launch_after = Some(after);
        report.p1j_launch_delta = report
            .p1j_launch_before
            .and_then(|before| after.checked_delta(before));
    }

    // Both arms must exclude every optional legacy predictive mode. Accept the
    // user's configuration only when it already meets this contract; never rewrite it.
    fn validate_no_other_predictive_modes(cfg: &crate::config::Config) -> Result<()> {
        let p = &cfg.predictive;
        if cfg.storage.predict_fanout != 0
            || p.locality_enabled
            || p.speculator_enabled
            || p.affinity_enabled
            || p.prefetch_governor
            || p.cost_aware_eviction
            || p.pregate_enabled
            || p.static_residency_fraction != 0.0
            || p.static_residency_warmup_tokens != 0
            || p.static_residency_profile.is_some()
        {
            return Err(
                "P1Q requires predict_fanout=0 and all optional predictive modes disabled".into(),
            );
        }
        Ok(())
    }

    // Revalidate each arm from source bytes and retain its independent provenance.
    // The second arm also has to match the first arm's resolved input identity.
    fn prepare_arm(args: &CommandArgs) -> Result<PreparedArm> {
        let build = crate::qualification::BuildProvenance::embedded();
        evidence::validate_preflight_provenance(&build)?;
        let bytes = std::fs::read(&args.config)?;
        let config_sha256 = crate::greedy_parity::sha256_hex(&bytes);
        let cfg: crate::config::Config = toml::from_str(std::str::from_utf8(&bytes)?)?;
        cfg.validate()?;
        evidence::validate_source_config(&cfg)?;
        validate_no_other_predictive_modes(&cfg)?;
        let (artifacts, errors) = crate::qualification_artifacts(&args.config, &cfg);
        evidence::validate_artifacts(&artifacts, &errors)?;
        if artifacts.config.as_ref().map(|v| v.sha256.as_str()) != Some(config_sha256.as_str()) {
            return Err("config changed between parsed snapshot and artifact hashing".into());
        }
        let metadata =
            crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
                .map_err(|e| format!("expert metadata unavailable: {e}"))?;
        evidence::validate_expert_metadata(&metadata)?;
        let spec = crate::resolve_real_cli_spec_from_config(
            cfg,
            crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
        )?;
        let model_identity = crate::greedy_parity_model_identity(&spec);
        if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
            return Err("strict Qwen3-MoE Q4_0 model identity required".into());
        }
        let (executable, executable_sha256) = crate::current_executable_identity()?;
        let provenance = ArmProvenance {
            provenance: evidence::BenchmarkProvenance {
                build,
                executable_canonical_path: std::fs::canonicalize(executable)?.display().to_string(),
                executable_sha256,
                resolved_config_sha256: crate::resolved_real_cli_spec_sha256(&spec)?,
                artifacts,
                expert_metadata: metadata.clone(),
            },
            config_path: std::fs::canonicalize(&args.config)?.display().to_string(),
            config_sha256,
            model_identity,
            production_configuration: evidence::ProductionConfiguration::from_config(
                &spec.cfg, &metadata,
            ),
        };
        Ok(PreparedArm { spec, provenance })
    }
    fn validate_runtime(
        runtime: &crate::BenchRealRuntime,
        expected_adapter: &str,
        report: &mut ArmReport,
    ) -> Result<()> {
        let hash = crate::resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        report.runtime_resolved_config_sha256 = Some(hash.clone());
        if report
            .provenance
            .as_ref()
            .map(|p| p.provenance.resolved_config_sha256.as_str())
            != Some(hash.as_str())
        {
            return Err("runtime configuration identity drift".into());
        }
        let model_load = crate::greedy_parity_model_load(runtime);
        let input = evidence::RuntimeContractInput {
            real_transformer_enabled: runtime.cfg.real_transformer.enabled,
            real_transformer_gpu_native: runtime.cfg.real_transformer.gpu_native,
            compute_offload: runtime.cfg.real_transformer.compute_offload,
            legacy_execution_plan: runtime.engine.execution_context().plan().into(),
            token_loop_geometry: runtime
                .gpu_native_token_loop
                .as_ref()
                .map(|t| t.model_geometry()),
            authoritative_device: runtime.engine.gpu_device_identity(),
            model_load: model_load.clone(),
            routed_failure_policy: runtime.engine.routed_expert_gpu_failure_policy(),
        };
        report.model_load = Some(model_load);
        report.adapter = input.authoritative_device.clone();
        if report
            .adapter
            .as_ref()
            .is_none_or(|a| !a.driver_info.contains("580.178.04"))
        {
            return Err("P1R requires frozen GCP-L4-580.178.04 driver profile".into());
        }
        report.runtime_contract =
            Some(evidence::validate_runtime_contract(&input, expected_adapter)?.0);
        let t = runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or("missing token loop")?;
        if t.snapshot() != Default::default()
            || t.recovery_snapshot() != Default::default()
            || runtime.engine.routed_expert_execution_snapshot() != Default::default()
        {
            return Err("isolated runtime has nonzero initial counters".into());
        }
        Ok(())
    }

    fn finish_snapshots(
        snapshots: evidence::RequestSnapshotStart,
        runtime: &crate::BenchRealRuntime,
        report: &mut ArmReport,
    ) {
        if let Some(token_loop) = runtime.gpu_native_token_loop.as_ref() {
            finish_launch_diagnostics(report, token_loop.p1j_launch_snapshot());
        }
        match snapshots.finish(runtime) {
            Ok(s) => report.runtime_counters = Some(s),
            Err(e) => report.errors.push(e.to_string()),
        }
    }
    fn finish_request(
        request: &mut crate::gpu_native_token_loop::GpuNativeRequestState,
        report: &mut ArmReport,
        cancelled: bool,
    ) {
        request.finish_predictor_v2_observation(cancelled);
        report.raw_p1e_report = request.predictor_v2_p1e_report();
        report.generated_token_ids_sha256 =
            crate::greedy_parity::token_ids_sha256(&report.generated_token_ids);
        match request.predictor_v2_snapshot() {
            Some(Ok(s)) => report.predictor_v2_snapshot = Some(s),
            Some(Err(e)) => report
                .errors
                .push(format!("P0 reconciliation failed: {e:?}")),
            None => report.errors.push("P0 reconciliation unavailable".into()),
        }
    }
    fn arm_invariant_errors(
        report: &ArmReport,
        prompt_tokens: usize,
        output_tokens: usize,
        positions: usize,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if report.completed_positions != positions {
            errors.push("completed positions differ from planned positions".into());
        }
        if report.generated_token_ids.len() != output_tokens {
            errors.push("generated output count differs from requested output count".into());
        }
        match &report.raw_p1e_report {
            None => errors.push("P1E report unavailable".into()),
            Some(raw) => {
                if raw.incomplete.is_some() {
                    errors.push(format!("global P1E incomplete: {:?}", raw.incomplete));
                }
                if raw.observations.len().checked_add(raw.no_emissions.len()) != Some(positions) {
                    errors.push(
                        "P1E observations/no-emissions do not cover completed positions".into(),
                    );
                }
                let mut partitions = p1e::Partitions {
                    emitted: raw.observations.len(),
                    ..Default::default()
                };
                for (i, r) in raw.observations.iter().enumerate() {
                    let c = r.freeze.candidate;
                    if c.request != raw.request
                        || c.namespace.runtime != raw.request.runtime_namespace
                        || c.source_layer != p1e::LAYER
                        || c.target_layer != p1e::LAYER
                        || c.signal_revision != p1e::REVISION
                        || c.position_distance != 1
                        || c.source_position.absolute_position.checked_add(1)
                            != Some(c.target_position.absolute_position)
                        || r.deadline.is_some_and(|d| {
                            d.request != c.request || d.position != c.target_position
                        })
                    {
                        errors.push(format!("P1E record {i} identity/causal contract mismatch"));
                    }
                    match r.outcome {
                        p1e::Outcome::Pending => partitions.pending += 1,
                        p1e::Outcome::Censored => partitions.censored += 1,
                        p1e::Outcome::Resolved { prediction_hit } => {
                            partitions.resolved += 1;
                            if prediction_hit {
                                partitions.prediction_hits += 1;
                            } else {
                                partitions.route_misses += 1;
                            }
                        }
                    }
                    match r.freeze.current {
                        Some(true) => {
                            partitions.valid_f += 1;
                            partitions.current_at_f += 1;
                        }
                        Some(false) => {
                            partitions.valid_f += 1;
                            partitions.absent_at_f += 1;
                        }
                        None => {}
                    }
                    partitions.useful += usize::from(r.opportunities().useful == Some(true));
                    partitions.target_confirmed_useful +=
                        usize::from(r.opportunities().target_confirmed_useful == Some(true));
                    if r.incomplete.is_some()
                        || r.freeze.incomplete.is_some()
                        || r.deadline.is_some_and(|d| d.incomplete.is_some())
                    {
                        errors.push(format!("P1E record {i} incomplete"));
                    }
                    if r.outcome == p1e::Outcome::Pending {
                        errors.push(format!("P1E record {i} remains pending"));
                    }
                }
                if partitions != raw.partitions {
                    errors.push("P1E partitions do not reconcile with raw records".into());
                }
                if raw
                    .no_emissions
                    .iter()
                    .any(|r| r.reason == p1e::NoEmissionReason::Incomplete)
                {
                    errors.push("incomplete P1E no-emission record".into());
                }
                if raw.partitions.pending != 0 {
                    errors.push("P1E pending partition nonzero".into());
                }
                let observed = raw
                    .observations
                    .iter()
                    .map(|r| (r.freeze.candidate.sequence, r.opportunities()))
                    .collect::<BTreeMap<_, _>>();
                let retained = raw
                    .opportunities
                    .iter()
                    .copied()
                    .collect::<BTreeMap<_, _>>();
                if observed.len() != raw.observations.len()
                    || retained.len() != raw.opportunities.len()
                    || retained != observed
                {
                    errors.push("P1E opportunities disagree with raw observations".into());
                }
            }
        }
        if let Some(c) = &report.runtime_counters {
            if let Err(e) = evidence::validate_request_postconditions(
                prompt_tokens,
                output_tokens,
                report.generated_token_ids.len(),
                c.token_loop_delta,
                c.recovery_delta,
                c.routed_execution_delta,
            ) {
                errors.push(e.to_string());
            }
            let t = c.token_loop_delta;
            if t.queue_submissions != t.token_attempts
                || t.boundary_maps != t.token_attempts
                || t.boundary_readbacks != t.token_attempts
            {
                errors.push("ordinary attempt/submission/map/readback accounting mismatch".into());
            }
        } else {
            errors.push("ordinary runtime snapshots unavailable".into());
        }
        if !shutdown_complete(report) {
            errors.push("normal isolated shutdown evidence missing or incomplete".into());
        }
        errors
    }
    fn check_arm(
        report: &mut ArmReport,
        prompt_tokens: usize,
        output_tokens: usize,
        positions: usize,
    ) {
        report.errors.extend(arm_invariant_errors(
            report,
            prompt_tokens,
            output_tokens,
            positions,
        ));
        report.ordinary_invariants_pass = report.errors.is_empty();
    }
    fn shutdown_complete(report: &ArmReport) -> bool {
        report
            .runtime_shutdown
            .is_some_and(|s| s.controlled_shutdown_requested && s.all_runtime_resources_released)
    }
    async fn shutdown(runtime: crate::BenchRealRuntime, report: &mut ArmReport) {
        match runtime.shutdown_isolated().await {
            Ok(s) => report.runtime_shutdown = Some(s),
            Err(e) => report.errors.push(e.to_string()),
        }
    }

    fn structural_initial(initial: &P1qInitialSnapshot) -> Result<Value> {
        let mut value = serde_json::to_value(initial)?;
        value
            .as_object_mut()
            .ok_or("invalid initial snapshot")?
            .remove("mode");
        for key in ["runtime", "context", "arena"] {
            let id = value
                .pointer_mut(&format!("/resources/namespace/{key}"))
                .ok_or("missing sidecar namespace")?;
            *id = json!(0);
        }
        Ok(value)
    }

    // Fresh runtimes have distinct legacy context incarnations. Normalize only
    // that ID for resource/calibration equality; retain the raw ArmReport evidence.
    fn structural_runtime_contract(raw: &evidence::RuntimeContractEvidence) -> Result<Value> {
        let mut value = serde_json::to_value(raw)?;
        *value
            .pointer_mut("/legacy_execution_plan/context_id")
            .ok_or("missing legacy execution context ID")? = json!("runtime-local");
        Ok(value)
    }

    fn initial_state_valid(
        before: &crate::gpu_native_residency::P1qResourceSnapshot,
        s: &P1qInitialSnapshot,
    ) -> bool {
        let r = &s.resources;
        !before.sidecar_initialized
            && !before.sidecar_allocated
            && before.namespace.is_none()
            && r.sidecar_initialized
            && r.sidecar_allocated
            && r.namespace.is_some()
            && r.sidecar_stride_bytes == 2_654_212
            && r.sidecar_bank == 1
            && r.sidecar_slot == 0
            && r.ordinary_total_expert_budget_bytes == 2 * 1024 * 1024 * 1024
            && r.ordinary_layer_capacities.len() == 48
            && r.ordinary_layer_capacities.get(47) == Some(&8)
            && r.ordinary_layer_resident_counts == vec![0; 48]
            && before.ordinary_layer_resident_counts == r.ordinary_layer_resident_counts
            && before.ordinary_layer_capacities == r.ordinary_layer_capacities
            && before.ordinary_total_expert_budget_bytes == r.ordinary_total_expert_budget_bytes
            && before.ordinary_arena_allocation_bytes == r.ordinary_arena_allocation_bytes
            && r.p1e_shadow_present
            && !before.p1e_shadow_present
            && before.activity_counters == [0; 14]
            && r.activity_counters == [0; 14]
            && s.token_loop == Default::default()
            && s.recovery == Default::default()
            && s.production_install == Default::default()
            && s.launch == Default::default()
            && s.p0 == Default::default()
            && s.committed_position == 0
            && !s.pending_sidecar
            && s.observation_capacity
                == observation_config(16, OUTPUT_TOKENS)
                    .map(|c| c.capacity_per_collection)
                    .unwrap_or(0)
    }
    fn counter_delta(before: &Value, after: &Value) -> Result<Value> {
        let b = before.as_object().ok_or("counter object unavailable")?;
        let a = after.as_object().ok_or("counter object unavailable")?;
        if b.len() != a.len() {
            return Err("counter schema changed".into());
        }
        let mut delta = serde_json::Map::new();
        for (name, value) in b {
            let count = a
                .get(name)
                .and_then(Value::as_u64)
                .ok_or("counter missing")?
                .checked_sub(value.as_u64().ok_or("invalid counter")?)
                .ok_or("counter regressed")?;
            delta.insert(name.clone(), json!(count));
        }
        Ok(Value::Object(delta))
    }

    async fn execute_arm(
        runtime: &crate::BenchRealRuntime,
        prompt: &[u32],
        report: &mut ArmReport,
        expected_resources: &mut Option<Value>,
    ) -> Result<()> {
        let token_loop = runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or("missing token loop")?;
        let config = observation_config(prompt.len(), OUTPUT_TOKENS)?;
        report.observation_capacity_per_collection = config.capacity_per_collection;
        report.p1j_launch_before = Some(token_loop.p1j_launch_snapshot());
        let snapshots = evidence::RequestSnapshotStart::capture(runtime)?;
        let mut request = token_loop.create_request_state()?;
        let execution: Result<()> = async {
        report.resources_before_opt_in = Some(token_loop.p1q_resource_snapshot()?);
        request.enable_predictor_v2_p1e_observation(token_loop, config)
            .map_err(|e| format!("P1E enablement failed: {e:?}"))?;
        match report.mode {
            P1jRequestMode::InertResourceOnly => request.enable_predictor_v2_p1j_sidecar_inert(token_loop)?,
            P1jRequestMode::Active => request.enable_predictor_v2_p1j_sidecar(token_loop)?,
        }
        report.recorder = Some(request.enable_p1r()?);
        let initial = request.p1q_initial_snapshot(token_loop)?;
        report.initial_state_pass = initial_state_valid(report.resources_before_opt_in.as_ref().ok_or("missing allocation pre-state")?, &initial);
        report.sidecar_freshly_initialized = report.initial_state_pass;
        report.resource_identity = Some(json!({
            "initial": structural_initial(&initial)?,
            "runtime_config_identity": report.runtime_resolved_config_sha256,
            "model_identity": report.provenance.as_ref().ok_or("missing provenance")?.model_identity,
            "production_configuration": report.provenance.as_ref().ok_or("missing provenance")?.production_configuration,
            "adapter": report.adapter,
            "model_load": report.model_load,
            "runtime_contract": structural_runtime_contract(report.runtime_contract.as_ref().ok_or("missing runtime contract")?)?,
            "prompt_token_count": prompt.len(),
            "prompt_token_ids_sha256": crate::greedy_parity::token_ids_sha256(prompt),
            "output_token_count": OUTPUT_TOKENS,
            "planned_positions": PLANNED_POSITIONS,
            "diagnostic_capacity": SLOTS,
            "diagnostic_storage_bytes": SLOTS * std::mem::size_of::<Slot>(),
            "max_seq_len": token_loop.max_seq_len(),
        }));
        report.initial = Some(initial);
        if !report.initial_state_pass { return Err("P1Q initial resource/counter state invalid".into()); }
        let identity = report.resource_identity.as_ref().ok_or("missing resource identity")?;
        if expected_resources.as_ref().is_some_and(|p| p != identity)
 {
            return Err("pre-request resource/config/model/adapter binding mismatch".into());
        }
        if expected_resources.is_none() { *expected_resources = Some(identity.clone()); }
        step_request(runtime, &mut request, prompt, report).await
    }.await;
        // Timing has already stopped. Finalize even failed arms and retain all evidence.
        finish_request(&mut request, report, execution.is_err());
        finish_snapshots(snapshots, runtime, report);
        let after = serde_json::to_value(token_loop.p1q_install_snapshot())?;
        if let Some(initial) = &report.initial {
            report.production_install_delta = Some(counter_delta(
                &serde_json::to_value(initial.production_install)?,
                &after,
            )?);
        }
        report.production_install_after = Some(after);
        execution
    }

    async fn step_request(
        runtime: &crate::BenchRealRuntime,
        request: &mut crate::gpu_native_token_loop::GpuNativeRequestState,
        prompt: &[u32],
        report: &mut ArmReport,
    ) -> Result<()> {
        let token_loop = runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or("missing token loop")?;
        if prompt.len() != 16 || PLANNED_POSITIONS > token_loop.max_seq_len() {
            return Err("frozen P1Q requires 16 prompt tokens and 143 planned positions".into());
        }
        report.generated_token_ids.reserve_exact(OUTPUT_TOKENS);
        let mut started = None;
        let mut stopped = None;
        let execution: Result<()> = async {
            for position in 0..PLANNED_POSITIONS {
                let (token, sample) = if position < prompt.len() {
                    (prompt[position], position + 1 == prompt.len())
                } else {
                    (
                        *report
                            .generated_token_ids
                            .last()
                            .ok_or("missing previous generated token")?,
                        true,
                    )
                };
                if position == 0 {
                    started = Some(Instant::now());
                    if let Some(r) = &report.recorder {
                        r.context(0, 0).stamp(Event::RequestStart, &[]);
                    }
                }
                let sampled = token_loop
                    .step_token(&runtime.engine, request, token, position, sample)
                    .await?;
                if position + 1 == PLANNED_POSITIONS {
                    stopped = Some(Instant::now());
                    if let Some(r) = &report.recorder {
                        r.context(0, 0).stamp(Event::RequestEnd, &[]);
                    }
                }
                increment(&mut report.completed_positions)?;
                match (sample, sampled) {
                    (true, Some(id)) => report.generated_token_ids.push(id),
                    (false, None) => {}
                    _ => return Err("ordinary step returned unexpected sampling result".into()),
                }
            }
            Ok(())
        }
        .await;
        if execution.is_ok() {
            let ns: u64 = stopped
                .ok_or("missing request stop")?
                .duration_since(started.ok_or("missing request start")?)
                .as_nanos()
                .try_into()?;
            report.request_wall_ns = Some(ns);
            // Timing is descriptive; P1R computes no throughput verdict.
        }
        execution
    }

    async fn run_arm(
        args: &CommandArgs,
        bytes: &[u8],
        arm: Arm,
        expected_provenance: &mut Option<Value>,
        expected_resources: &mut Option<Value>,
    ) -> ArmReport {
        let mut report = ArmReport::new(arm);
        report.mode = if arm == Arm::Treatment {
            P1jRequestMode::Active
        } else {
            P1jRequestMode::InertResourceOnly
        };
        let preparation: Result<_> = (|| {
            let (prompt, _) = parse_request(bytes)?;
            let prepared = prepare_arm(args)?;
            let identity = provenance_identity(&prepared.provenance)?;
            report.provenance = Some(prepared.provenance);
            bind_provenance_identity(expected_provenance, identity)?;
            let mode = crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark;
            let tokenizer = crate::load_real_cli_tokenizer(&prepared.spec.cfg, mode)?;
            Ok((prepared.spec, tokenizer, prompt))
        })();
        match preparation {
            Err(e) => report.errors.push(e.to_string()),
            Ok((spec, tokenizer, prompt)) => {
                report.runtime_build_attempted = true;
                match crate::build_isolated_greedy_runtime(
                    &spec,
                    crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
                    tokenizer.clone(),
                )
                .await
                {
                    Err(e) => report.errors.push(e.to_string()),
                    Ok(runtime) => {
                        report.runtime_constructed = true;
                        let execution: Result<()> = async {
                            validate_runtime(&runtime, &args.expected_adapter_name, &mut report)?;
                            // Exactly one deterministic tokenization per independently built arm.
                            let prompt_ids = tokenizer.encode(&prompt)?;
                            if prompt_ids.len() != 16 {
                                return Err("frozen P1Q requires exactly 16 prompt tokens".into());
                            }
                            execute_arm(&runtime, &prompt_ids, &mut report, expected_resources)
                                .await
                        }
                        .await;
                        if let Err(e) = execution {
                            report.errors.push(e.to_string());
                        }
                        // Always shut down before the next arm, including setup/request failure.
                        shutdown(runtime, &mut report).await;
                        report.diagnostics = report.recorder.take().map(|r| r.snapshot());
                    }
                }
            }
        }
        if let Some(identity) = &report.resource_identity {
            if expected_resources.as_ref().is_some_and(|p| p != identity) {
                report.errors.push(
                    "resource/config/model/adapter calibration or experiment identity drift".into(),
                );
            }
            if expected_resources.is_none() {
                *expected_resources = Some(identity.clone());
            }
        }
        check_arm(&mut report, 16, OUTPUT_TOKENS, PLANNED_POSITIONS);
        if report.diagnostics.as_ref().is_none_or(|d| {
            d.invalid
                || d.capacity != SLOTS
                || d.storage_bytes != SLOTS * std::mem::size_of::<Slot>()
        }) {
            report
                .errors
                .push("P1R diagnostic evidence incomplete".into());
            report.ordinary_invariants_pass = false;
        }
        report
    }

    fn read_child_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
        serde_json::from_slice::<serde::de::IgnoredAny>(bytes)?;
        let mut json = ChildJson {
            remaining: std::str::from_utf8(bytes)?,
        };
        let result = T::deserialize(&mut json)?;
        if !json.remaining.trim().is_empty() {
            return Err("trailing child JSON".into());
        }
        Ok(result)
    }

    struct ChildJson<'de> {
        remaining: &'de str,
    }
    type JsonError = serde::de::value::Error;
    impl ChildJson<'_> {
        fn error(message: impl std::fmt::Display) -> JsonError {
            serde::de::Error::custom(message)
        }
        fn peek(&mut self) -> Option<u8> {
            self.remaining = self.remaining.trim_start();
            self.remaining.as_bytes().first().copied()
        }
        fn take(&mut self, byte: u8) -> std::result::Result<(), JsonError> {
            if self.peek() != Some(byte) {
                return Err(Self::error("unexpected child JSON token"));
            }
            self.remaining = &self.remaining[1..];
            Ok(())
        }
        fn string(&mut self) -> std::result::Result<String, JsonError> {
            if self.peek() != Some(b'"') {
                return Err(Self::error("expected child JSON string"));
            }
            let bytes = self.remaining.as_bytes();
            let mut end = 1;
            while end < bytes.len() {
                match bytes[end] {
                    b'\\' => end += 2,
                    b'"' => {
                        let value =
                            serde_json::from_str(&self.remaining[..=end]).map_err(Self::error)?;
                        self.remaining = &self.remaining[end + 1..];
                        return Ok(value);
                    }
                    _ => end += 1,
                }
            }
            Err(Self::error("unterminated child JSON string"))
        }
        fn number(&mut self) -> std::result::Result<&str, JsonError> {
            self.peek();
            let end = self
                .remaining
                .bytes()
                .take_while(|b| matches!(b, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'))
                .count();
            if end == 0 {
                return Err(Self::error("expected child JSON number"));
            }
            let (number, rest) = self.remaining.split_at(end);
            self.remaining = rest;
            Ok(number)
        }
    }

    struct ChildJsonAccess<'a, 'de> {
        json: &'a mut ChildJson<'de>,
        first: bool,
    }
    impl ChildJsonAccess<'_, '_> {
        fn next(&mut self, end: u8) -> std::result::Result<bool, JsonError> {
            if self.json.peek() == Some(end) {
                return Ok(false);
            }
            if !self.first {
                self.json.take(b',')?;
            }
            self.first = false;
            Ok(true)
        }
    }
    impl<'de> serde::de::MapAccess<'de> for ChildJsonAccess<'_, 'de> {
        type Error = JsonError;
        fn next_key_seed<K: serde::de::DeserializeSeed<'de>>(
            &mut self,
            seed: K,
        ) -> std::result::Result<Option<K::Value>, JsonError> {
            if !self.next(b'}')? {
                return Ok(None);
            }
            let key = seed.deserialize(serde::de::value::StringDeserializer::<JsonError>::new(
                self.json.string()?,
            ))?;
            self.json.take(b':')?;
            Ok(Some(key))
        }
        fn next_value_seed<V: serde::de::DeserializeSeed<'de>>(
            &mut self,
            seed: V,
        ) -> std::result::Result<V::Value, JsonError> {
            seed.deserialize(&mut *self.json)
        }
    }
    impl<'de> serde::de::SeqAccess<'de> for ChildJsonAccess<'_, 'de> {
        type Error = JsonError;
        fn next_element_seed<T: serde::de::DeserializeSeed<'de>>(
            &mut self,
            seed: T,
        ) -> std::result::Result<Option<T::Value>, JsonError> {
            if !self.next(b']')? {
                return Ok(None);
            }
            seed.deserialize(&mut *self.json).map(Some)
        }
    }
    impl<'de, 'a> serde::de::EnumAccess<'de> for &'a mut ChildJson<'de> {
        type Error = JsonError;
        type Variant = Self;
        fn variant_seed<V: serde::de::DeserializeSeed<'de>>(
            self,
            seed: V,
        ) -> std::result::Result<(V::Value, Self), JsonError> {
            let variant = seed.deserialize(
                serde::de::value::StringDeserializer::<JsonError>::new(self.string()?),
            )?;
            self.take(b':')?;
            Ok((variant, self))
        }
    }
    impl<'de> serde::de::VariantAccess<'de> for &mut ChildJson<'de> {
        type Error = JsonError;
        fn unit_variant(self) -> std::result::Result<(), JsonError> {
            Deserialize::deserialize(self)
        }
        fn newtype_variant_seed<T: serde::de::DeserializeSeed<'de>>(
            self,
            seed: T,
        ) -> std::result::Result<T::Value, JsonError> {
            seed.deserialize(self)
        }
        fn tuple_variant<V: serde::de::Visitor<'de>>(
            self,
            _: usize,
            visitor: V,
        ) -> std::result::Result<V::Value, JsonError> {
            serde::Deserializer::deserialize_seq(self, visitor)
        }
        fn struct_variant<V: serde::de::Visitor<'de>>(
            self,
            _: &'static [&'static str],
            visitor: V,
        ) -> std::result::Result<V::Value, JsonError> {
            serde::Deserializer::deserialize_map(self, visitor)
        }
    }
    impl<'de> serde::Deserializer<'de> for &mut ChildJson<'de> {
        type Error = JsonError;
        fn deserialize_any<V: serde::de::Visitor<'de>>(
            self,
            visitor: V,
        ) -> std::result::Result<V::Value, JsonError> {
            match self.peek() {
                Some(b'{') => {
                    self.take(b'{')?;
                    let value = visitor.visit_map(ChildJsonAccess {
                        json: &mut *self,
                        first: true,
                    })?;
                    self.take(b'}')?;
                    Ok(value)
                }
                Some(b'[') => {
                    self.take(b'[')?;
                    let value = visitor.visit_seq(ChildJsonAccess {
                        json: &mut *self,
                        first: true,
                    })?;
                    self.take(b']')?;
                    Ok(value)
                }
                Some(b'"') => visitor.visit_string(self.string()?),
                Some(b'n') => {
                    self.remaining = &self.remaining[4..];
                    visitor.visit_unit()
                }
                Some(b't') => {
                    self.remaining = &self.remaining[4..];
                    visitor.visit_bool(true)
                }
                Some(b'f') => {
                    self.remaining = &self.remaining[5..];
                    visitor.visit_bool(false)
                }
                Some(b'-' | b'0'..=b'9') => {
                    let number = self.number()?;
                    if number.contains(['.', 'e', 'E']) || number == "-0" {
                        let value: f64 = number.parse().map_err(ChildJson::error)?;
                        if !value.is_finite() {
                            return Err(ChildJson::error("non-finite child JSON number"));
                        }
                        visitor.visit_f64(value)
                    } else if number.starts_with('-') {
                        visitor.visit_i64(number.parse().map_err(ChildJson::error)?)
                    } else {
                        visitor.visit_u64(number.parse().map_err(ChildJson::error)?)
                    }
                }
                _ => Err(ChildJson::error("invalid child JSON token")),
            }
        }
        fn deserialize_f32<V: serde::de::Visitor<'de>>(
            self,
            visitor: V,
        ) -> std::result::Result<V::Value, JsonError> {
            let value: f32 = self.number()?.parse().map_err(ChildJson::error)?;
            if !value.is_finite() {
                return Err(ChildJson::error("non-finite child f32"));
            }
            visitor.visit_f32(value)
        }
        fn deserialize_f64<V: serde::de::Visitor<'de>>(
            self,
            visitor: V,
        ) -> std::result::Result<V::Value, JsonError> {
            let value: f64 = self.number()?.parse().map_err(ChildJson::error)?;
            if !value.is_finite() {
                return Err(ChildJson::error("non-finite child f64"));
            }
            visitor.visit_f64(value)
        }
        fn deserialize_option<V: serde::de::Visitor<'de>>(
            self,
            visitor: V,
        ) -> std::result::Result<V::Value, JsonError> {
            if self.peek() == Some(b'n') {
                self.remaining = &self.remaining[4..];
                visitor.visit_none()
            } else {
                visitor.visit_some(self)
            }
        }
        fn deserialize_newtype_struct<V: serde::de::Visitor<'de>>(
            self,
            _: &'static str,
            visitor: V,
        ) -> std::result::Result<V::Value, JsonError> {
            visitor.visit_newtype_struct(self)
        }
        fn deserialize_enum<V: serde::de::Visitor<'de>>(
            self,
            _: &'static str,
            _: &'static [&'static str],
            visitor: V,
        ) -> std::result::Result<V::Value, JsonError> {
            if self.peek() == Some(b'"') {
                visitor.visit_enum(serde::de::value::StringDeserializer::<JsonError>::new(
                    self.string()?,
                ))
            } else {
                self.take(b'{')?;
                let value = visitor.visit_enum(&mut *self)?;
                self.take(b'}')?;
                Ok(value)
            }
        }
        serde::forward_to_deserialize_any! { bool i8 i16 i32 i64 u8 u16 u32 u64 char str string bytes byte_buf unit unit_struct seq tuple tuple_struct map struct identifier ignored_any }
    }

    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ChildArtifact {
        schema: String,
        protocol: String,
        diagnostic_id: String,
        pair_index: u8,
        arm: Arm,
        pid: u32,
        binary_sha256: String,
        source_identity: Value,
        request_sha256: String,
        report: ArmReport,
    }
    #[derive(Serialize)]
    struct ProcessEvidence {
        arm: Arm,
        pid: Option<u32>,
        exit_code: Option<i32>,
        artifact_path: PathBuf,
        artifact_sha256: Option<String>,
        error: Option<String>,
    }
    #[derive(Serialize)]
    struct DiagnosticReport {
        schema: &'static str,
        protocol: &'static str,
        diagnostic_id: String,
        diagnostic_only: bool,
        performance_comparison_authorized: bool,
        performance_verdict: &'static str,
        gpu_completion: &'static str,
        exact_candidate_saved_ns: Option<u64>,
        source_identity: Value,
        request_sha256: String,
        binary_sha256: String,
        profile: &'static str,
        pair_index: u8,
        execution_order: [Arm; 2],
        processes: Vec<ProcessEvidence>,
        arms: Vec<ChildArtifact>,
        attribution: Vec<Attribution>,
        coverage: Value,
        errors: Vec<String>,
        diagnostic_evidence_valid: bool,
    }
    fn source_identity() -> Result<Value> {
        let build = crate::qualification::BuildProvenance::embedded();
        evidence::validate_preflight_provenance(&build)?;
        let commit = build.git_sha.as_deref().ok_or("missing source commit")?;
        let output = std::process::Command::new("git")
            .args(["show", "-s", "--format=%H%n%T%n%P", commit])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()?;
        if !output.status.success() {
            return Err("source lineage unavailable".into());
        }
        let line = String::from_utf8(output.stdout)?;
        let fields: Vec<_> = line.lines().collect();
        if fields.len() != 3
            || fields[0] != commit
            || fields[2] != "a6b9cbcd6e6b576f90a1d875c9351c7276c950e2"
        {
            return Err("P1R must be exactly one child of the frozen P1Q3 base".into());
        }
        Ok(
            json!({"commit":fields[0],"tree":fields[1],"parent":fields[2],"files":{
                "gpu_native_predictor_v2_critical_path_attribution.rs":crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_predictor_v2_critical_path_attribution.rs")),
                "gpu_native_token_loop.rs":crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_token_loop.rs")),
                "gpu_native_residency.rs":crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_residency.rs")),
                "backend/gpu_native.rs":crate::greedy_parity::sha256_hex(include_bytes!("backend/gpu_native.rs")),
                "engine.rs":crate::greedy_parity::sha256_hex(include_bytes!("engine.rs")),
                "main.rs":crate::greedy_parity::sha256_hex(include_bytes!("main.rs")),
                "gpu_native_predictor_v2_observation.rs":crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_predictor_v2_observation.rs")),
                "gpu_native_q4_route_parallel.rs":crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_q4_route_parallel.rs")),
                "gpu_native_predictor_v2_sidecar_performance.rs":crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_predictor_v2_sidecar_performance.rs"))
            }}),
        )
    }
    fn valid_arm(report: &ArmReport) -> bool {
        report.errors.is_empty()
            && arm_invariant_errors(report, 16, OUTPUT_TOKENS, PLANNED_POSITIONS).is_empty()
            && report.ordinary_invariants_pass
            && report.runtime_build_attempted
            && report.runtime_constructed
            && report.generated_token_ids_sha256
                == crate::greedy_parity::token_ids_sha256(&report.generated_token_ids)
            && report.request_wall_ns.is_some()
            && report.generated_tps.is_none()
            && report.planned_position_tps.is_none()
            && report
                .resources_before_opt_in
                .as_ref()
                .zip(report.initial.as_ref())
                .is_some_and(|(b, i)| i.mode == report.mode && initial_state_valid(b, i))
            && report.resource_identity.is_some()
            && report.provenance.as_ref().is_some_and(|p| {
                evidence::validate_preflight_provenance(&p.provenance.build).is_ok()
                    && p.model_identity.is_qwen3_coder_30b_a3b_q4_0()
                    && report.runtime_resolved_config_sha256.as_ref()
                        == Some(&p.provenance.resolved_config_sha256)
            })
            && report.model_load.is_some()
            && report.runtime_contract.is_some()
            && report.adapter.as_ref().is_some_and(|a| {
                a.name == "NVIDIA L4" && a.driver_info.contains("580.178.04") && !a.software_adapter
            })
            && report
                .initial
                .as_ref()
                .zip(report.production_install_after.as_ref())
                .is_some_and(|(initial, after)| {
                    serde_json::to_value(initial.production_install)
                        .ok()
                        .and_then(|before| counter_delta(&before, after).ok())
                        .as_ref()
                        == report.production_install_delta.as_ref()
                        && report.production_install_delta.is_some()
                })
            && shutdown_complete(report)
            && report.initial_state_pass
            && report.sidecar_freshly_initialized
            && report.generated_token_ids.len() == OUTPUT_TOKENS
            && report.completed_positions == PLANNED_POSITIONS
            && launch_diagnostics_gate(report).is_empty()
            && report.predictor_v2_snapshot.as_ref().is_some_and(|p| {
                if report.mode == P1jRequestMode::InertResourceOnly {
                    control_gate(p).is_empty()
                } else {
                    p.incomplete.is_none()
                        && p.live_predictions == 0
                        && p.source_live == 0
                        && p.reservations_live == 0
                        && p.available_installs == 0
                        && p.terminal_predictions == p.emitted
                        && p.source_failed == 0
                }
            })
            && complete_trace(report).is_ok()
    }
    fn complete_trace(report: &ArmReport) -> Result<()> {
        let raw = report.raw_p1e_report.as_ref().ok_or("P1E absent")?;
        let trace = Trace::new(report.diagnostics.as_ref().ok_or("recorder absent")?)?;
        if raw.observations.len() + raw.no_emissions.len() != POSITIONS {
            return Err("nomination/no-emission coverage incomplete".into());
        }
        let mut sources = BTreeSet::new();
        for r in &raw.observations {
            if !sources.insert(r.freeze.candidate.source_position.absolute_position) {
                return Err("duplicate source opportunity".into());
            }
        }
        for r in &raw.no_emissions {
            if !sources.insert(r.source.absolute_position) {
                return Err("duplicate no-emission source".into());
            }
        }
        if sources != (0..POSITIONS as u64).collect() {
            return Err("source opportunity coverage incomplete".into());
        }
        for position in 0..POSITIONS {
            trace.clean_route(position)?;
        }
        if trace.position(0, Event::RequestStart).is_none()
            || trace.position(0, Event::RequestEnd).is_none()
        {
            return Err("request interval missing".into());
        }
        Ok(())
    }
    fn validate_artifact(
        a: &ChildArtifact,
        args: &CommandArgs,
        arm: Arm,
        pid: u32,
        binary: &str,
        source: &Value,
        request: &str,
    ) -> Result<()> {
        if a.schema != SCHEMA
            || a.protocol != CHILD_PROTOCOL
            || a.pair_index != 1
            || a.arm != arm
            || a.report.arm != arm
            || a.pid == 0
            || a.pid != pid
            || a.diagnostic_id != args.diagnostic_id
            || a.binary_sha256 != binary
            || &a.source_identity != source
            || a.request_sha256 != request
            || a.report
                .provenance
                .as_ref()
                .is_none_or(|p| p.provenance.executable_sha256 != binary)
            || a.report.mode
                != if arm == Arm::Control {
                    P1jRequestMode::InertResourceOnly
                } else {
                    P1jRequestMode::Active
                }
            || !valid_arm(&a.report)
        {
            return Err("P1R child identity/evidence/shutdown mismatch".into());
        }
        Ok(())
    }
    fn child_path(parent: &Path, arm: Arm) -> PathBuf {
        parent.with_file_name(format!(
            "{}.{}.json",
            parent.file_name().unwrap_or_default().to_string_lossy(),
            if arm == Arm::Control {
                "control"
            } else {
                "treatment"
            }
        ))
    }
    fn launch(
        args: &CommandArgs,
        executable: &Path,
        arm: Arm,
        binary: &str,
        source: &Value,
        request: &str,
    ) -> (ProcessEvidence, Option<ChildArtifact>) {
        let path = child_path(&args.report_out, arm);
        let mut process = ProcessEvidence {
            arm,
            pid: None,
            exit_code: None,
            artifact_path: path.clone(),
            artifact_sha256: None,
            error: None,
        };
        let outcome: Result<ChildArtifact> = (|| {
            ensure_output_absent(&path)?;
            let stdout = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path.with_extension("stdout.log"))?;
            let stderr = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path.with_extension("stderr.log"))?;
            let mut command = std::process::Command::new(executable);
            command
                .arg("diagnose-gpu-native-predictor-v2-critical-path-attribution")
                .arg("--config")
                .arg(&args.config)
                .arg("--request-json")
                .arg(&args.request_json)
                .arg("--expected-adapter-name")
                .arg(&args.expected_adapter_name)
                .arg("--diagnostic-id")
                .arg(&args.diagnostic_id)
                .arg("--report-out")
                .arg(&path)
                .arg("--p1r-child-protocol")
                .arg(CHILD_PROTOCOL)
                .arg("--p1r-child-arm")
                .arg(if arm == Arm::Control {
                    "control"
                } else {
                    "treatment"
                })
                .stdout(stdout)
                .stderr(stderr);
            if arm == Arm::Treatment {
                command
                    .arg("--control-artifact")
                    .arg(child_path(&args.report_out, Arm::Control));
            }
            let mut child = command.spawn()?;
            process.pid = Some(child.id());
            let status = child.wait()?;
            process.exit_code = status.code();
            let bytes = std::fs::read(&path)?;
            process.artifact_sha256 = Some(crate::greedy_parity::sha256_hex(&bytes));
            let artifact = read_child_json::<ChildArtifact>(&bytes)?;
            if !status.success() || status.code() != Some(0) {
                return Err("child failed or exit uncertain".into());
            }
            validate_artifact(&artifact, args, arm, child.id(), binary, source, request)?;
            Ok(artifact)
        })();
        match outcome {
            Ok(a) => (process, Some(a)),
            Err(e) => {
                process.error = Some(e.to_string());
                (process, None)
            }
        }
    }
    // One serial pair only. The same gate is exercised with process fixtures.
    fn one_pair<T>(
        mut run: impl FnMut(Arm, Option<&T>) -> Result<T>,
        valid_control: impl Fn(&T) -> bool,
    ) -> (Option<T>, Option<T>, Vec<String>) {
        match run(Arm::Control, None) {
            Ok(control) if valid_control(&control) => match run(Arm::Treatment, Some(&control)) {
                Ok(treatment) => (Some(control), Some(treatment), vec![]),
                Err(e) => (Some(control), None, vec![e.to_string()]),
            },
            Ok(control) => (
                Some(control),
                None,
                vec!["CONTROL failed or shutdown uncertain; TREATMENT forbidden".into()],
            ),
            Err(e) => (None, None, vec![e.to_string()]),
        }
    }
    pub(crate) async fn run_command(args: CommandArgs) -> Result<()> {
        ensure_output_absent(&args.report_out)?;
        if args.expected_adapter_name != "NVIDIA L4" || args.diagnostic_id.trim().is_empty() {
            return Err("P1R requires a diagnostic ID and frozen NVIDIA L4 profile".into());
        }
        let child_arm = match (&args.p1r_child_protocol, args.p1r_child_arm) {
            (None, None) if args.control_artifact.is_none() => None,
            (Some(protocol), Some(arm))
                if protocol == CHILD_PROTOCOL
                    && (arm == Arm::Treatment) == args.control_artifact.is_some() =>
            {
                Some(arm)
            }
            _ => return Err("invalid P1R child protocol".into()),
        };
        let request_bytes = std::fs::read(&args.request_json)?;
        parse_request(&request_bytes)?;
        let request_sha256 = crate::greedy_parity::sha256_hex(&request_bytes);
        let source = source_identity()?;
        let (executable, binary) = crate::current_executable_identity()?;
        if let Some(arm) = child_arm {
            let mut expected_provenance = None;
            let mut expected_resources = None;
            if let Some(path) = &args.control_artifact {
                let bytes = std::fs::read(path)?;
                let control: ChildArtifact = read_child_json(&bytes)?;
                validate_artifact(
                    &control,
                    &args,
                    Arm::Control,
                    control.pid,
                    &binary,
                    &source,
                    &request_sha256,
                )?;
                if control.pid == std::process::id() {
                    return Err("P1R requires distinct isolated processes".into());
                }
                expected_provenance = Some(provenance_identity(
                    control
                        .report
                        .provenance
                        .as_ref()
                        .ok_or("missing CONTROL provenance")?,
                )?);
                expected_resources = control.report.resource_identity;
            }
            let report = run_arm(
                &args,
                &request_bytes,
                arm,
                &mut expected_provenance,
                &mut expected_resources,
            )
            .await;
            let valid = valid_arm(&report);
            let artifact = ChildArtifact {
                schema: SCHEMA.into(),
                protocol: CHILD_PROTOCOL.into(),
                diagnostic_id: args.diagnostic_id.clone(),
                pair_index: 1,
                arm,
                pid: std::process::id(),
                binary_sha256: binary,
                source_identity: source,
                request_sha256,
                report,
            };
            write_report(&args.report_out, &artifact)?;
            return if valid {
                Ok(())
            } else {
                Err("P1R child diagnostic invalid; evidence retained".into())
            };
        }
        let mut report = DiagnosticReport {
            schema: SCHEMA,
            protocol: CHILD_PROTOCOL,
            diagnostic_id: args.diagnostic_id.clone(),
            diagnostic_only: true,
            performance_comparison_authorized: false,
            performance_verdict: "NOT_AUTHORIZED",
            gpu_completion: "UNMEASURABLE",
            exact_candidate_saved_ns: None,
            source_identity: source.clone(),
            request_sha256: request_sha256.clone(),
            binary_sha256: binary.clone(),
            profile: "GCP-L4-580.178.04",
            pair_index: 1,
            execution_order: [Arm::Control, Arm::Treatment],
            processes: vec![],
            arms: vec![],
            attribution: vec![],
            coverage: Value::Null,
            errors: vec![],
            diagnostic_evidence_valid: false,
        };
        let (control, treatment, errors) = one_pair(
            |arm, _| {
                let (process, artifact) =
                    launch(&args, &executable, arm, &binary, &source, &request_sha256);
                report.processes.push(process);
                artifact.ok_or_else(|| "P1R child failed; see retained child evidence".into())
            },
            |c| valid_arm(&c.report),
        );
        report.errors.extend(errors);
        if let (Some(c), Some(t)) = (&control, &treatment) {
            if c.pid == t.pid
                || c.report.resource_identity != t.report.resource_identity
                || c.report
                    .provenance
                    .as_ref()
                    .map(provenance_identity)
                    .transpose()?
                    != t.report
                        .provenance
                        .as_ref()
                        .map(provenance_identity)
                        .transpose()?
                || c.report.generated_token_ids != t.report.generated_token_ids
            {
                report
                    .errors
                    .push("process/resource/input/output identity divergence".into());
            }
            match attribute(&c.report, &t.report) {
                Ok(rows) => report.attribution = rows,
                Err(e) => report.errors.push(e.to_string()),
            }
        }
        report.arms.extend(control);
        report.arms.extend(treatment);
        report.diagnostic_evidence_valid = report.errors.is_empty() && report.arms.len() == 2;
        report.coverage = coverage(
            &report.arms,
            &report.attribution,
            report.diagnostic_evidence_valid,
        );
        write_report(&args.report_out, &report)?;
        if report.diagnostic_evidence_valid {
            Ok(())
        } else {
            Err("P1R attribution incomplete; evidence retained; no retry authorized".into())
        }
    }

    use std::collections::{BTreeMap, BTreeSet};
    struct Trace<'a> {
        records: BTreeMap<usize, &'a Record>,
    }
    impl<'a> Trace<'a> {
        fn new(snapshot: &'a Snapshot) -> Result<Self> {
            if snapshot.invalid
                || snapshot.capacity != SLOTS
                || snapshot.storage_bytes != SLOTS * std::mem::size_of::<Slot>()
                || snapshot.layout.as_ref() != Some(&RecorderLayout::current())
            {
                return Err("INVALID_EVIDENCE: recorder bounds/completeness".into());
            }
            let mut records = BTreeMap::new();
            for r in &snapshot.records {
                if r.slot >= SLOTS
                    || (21..POSITION_SLOTS).contains(&(r.slot % STRIDE))
                    || r.words[1] < r.words[0]
                    || records.insert(r.slot, r).is_some()
                {
                    return Err("INVALID_EVIDENCE: duplicate/overflow/chronology".into());
                }
            }
            Ok(Self { records })
        }
        fn get(
            &self,
            p: usize,
            e: Event,
            attempt: usize,
            pass: usize,
            member: usize,
        ) -> Option<&'a [u64; WORDS]> {
            self.records
                .get(&slot_index(p, attempt, pass, member, e)?)
                .map(|r| &r.words)
        }
        fn position(&self, p: usize, e: Event) -> Option<&'a [u64; WORDS]> {
            self.get(p, e, 0, 0, 0)
        }
        fn clean_route(&self, p: usize) -> Result<Vec<u32>> {
            let r = self
                .position(p, Event::CleanRoute)
                .ok_or("missing clean route")?;
            if r[3] != 8 || r[2] == 0 || r[2] > ATTEMPTS as u64 {
                return Err("invalid clean route count/attempt".into());
            }
            let ids = r[4..12]
                .iter()
                .map(|n| u32::try_from(*n))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if ids.iter().any(|id| *id >= 128) || ids.iter().collect::<BTreeSet<_>>().len() != 8 {
                return Err("invalid clean route IDs".into());
            }
            let attempt = r[2] as usize - 1;
            let boundary = self
                .get(p, Event::Boundary, attempt, 0, 0)
                .ok_or("clean boundary absent")?;
            if boundary[2] != 1 || boundary[3..] != r[3..] {
                return Err("invalid speculative tail cannot be clean route truth".into());
            }
            Ok(ids)
        }
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
    #[allow(non_camel_case_types)]
    enum OutcomeClass {
        READY_AND_CRITICAL,
        READY_BUT_NOT_CRITICAL,
        LATE,
        ORDINARY_WON_RACE,
        UNUSED,
        NOT_LAUNCHED_OR_INELIGIBLE,
        D_WINDOW_AMBIGUOUS,
        READY_DURING_D_WINDOW,
        PUBLICATION_BUSY,
        GENERATION_DIVERGENT,
        GENERATION_UNAVAILABLE,
        ATTRIBUTION_UNRESOLVED,
        CENSORED,
        INVALID_EVIDENCE,
    }
    #[derive(Clone, Debug, Serialize, PartialEq, Eq)]
    struct SignedNs {
        negative: bool,
        ns: u64,
    }
    fn difference(a: u64, b: u64) -> SignedNs {
        SignedNs {
            negative: a < b,
            ns: a.abs_diff(b),
        }
    }
    #[derive(Debug, Serialize)]
    struct DirectService {
        attempt: usize,
        re_probe: usize,
        selected_set: Vec<u64>,
        physical_missing_set: Vec<u64>,
        candidate_physically_missing: bool,
        foreground_set_ns: u64,
        source_set_ns: Option<u64>,
        source_branch: Option<&'static str>,
        sequential_candidate_source_ns: Option<u64>,
        sequential_candidate_source_branch: Option<&'static str>,
        logical_retry_source_ns: Option<u64>,
        stage_ns: Option<u64>,
        commit_ns: Option<u64>,
        installed_generation: Option<u64>,
        installed: bool,
        stage_overlap_ns: Vec<u64>,
        scope: &'static str,
        exact_candidate_saved_ns: Option<u64>,
    }
    fn duration(r: &[u64; WORDS]) -> u64 {
        r[1] - r[0]
    }
    fn overlap(a: (u64, u64), b: (u64, u64)) -> u64 {
        a.1.min(b.1).saturating_sub(a.0.max(b.0))
    }
    fn service(trace: &Trace<'_>, p: usize, expert: u32) -> Result<Vec<DirectService>> {
        let mut services = Vec::new();
        let global = u64::from(47 * 128 + expert);
        for attempt in 0..ATTEMPTS {
            let Some(total) = trace.get(p, Event::Service, attempt, 0, 0) else {
                continue;
            };
            if total[2] != 1 {
                return Err("failed foreground demand service".into());
            }
            for pass in 0..2 {
                let Some(selected) = trace.get(p, Event::Selected, attempt, pass, 0) else {
                    if pass == 0 {
                        return Err("missing complete selected demand set".into());
                    } else {
                        continue;
                    }
                };
                if selected[3] != 8 {
                    return Err("selected set width mismatch".into());
                }
                let selected_set = selected[4..12].to_vec();
                let probe = trace
                    .get(p, Event::Probe, attempt, pass, 0)
                    .ok_or("missing physical probe")?;
                if probe[3] > 8 {
                    return Err("physical missing set overflow".into());
                }
                let missing = probe[4..4 + probe[3] as usize].to_vec();
                if missing.iter().any(|id| !selected_set.contains(id)) {
                    return Err("physical missing set is not selected".into());
                }
                let member = selected_set.iter().position(|id| *id == global);
                let get = |e| member.and_then(|m| trace.get(p, e, attempt, pass, m));
                let stage = get(Event::Stage);
                let commit = get(Event::Commit);
                let installed = get(Event::Installed);
                let candidate_missing = missing.contains(&global);
                if !candidate_missing
                    && [
                        get(Event::SourceExpert),
                        get(Event::LogicalSource),
                        get(Event::Reservation),
                        stage,
                        commit,
                        installed,
                    ]
                    .iter()
                    .any(Option::is_some)
                {
                    return Err("physical hit acquired source/install attribution".into());
                }
                if let Some(i) = installed {
                    if i[2] != 1
                        || stage.is_none_or(|s| s[2] != 1 || s[3] != global || s[4] != i[4])
                        || commit.is_none_or(|s| s[2] != 1 || s[3] != global || s[4] != i[4])
                        || i[3] != global
                    {
                        return Err(
                            "installed identity lacks successful matching stage/commit".into()
                        );
                    }
                }
                let mut overlaps = Vec::new();
                if let Some(a) = stage {
                    for m in 0..8 {
                        if Some(m) != member {
                            if let Some(b) = trace.get(p, Event::Stage, attempt, pass, m) {
                                overlaps.push(overlap((a[0], a[1]), (b[0], b[1])));
                            }
                        }
                    }
                }
                let branch = trace
                    .get(p, Event::SourceBranch, attempt, pass, 0)
                    .map(|r| match r[2] {
                        1 => "SEQUENTIAL_SINGLETON_OR_LOCAL_REUSE",
                        2 => "SEQUENTIAL_MIXED_RAM",
                        3 => "SEQUENTIAL_SINGLEFLIGHT_FALLBACK",
                        4 => "SEQUENTIAL_RESERVATION_FALLBACK",
                        5 => "PRODUCTION_BATCH_SET",
                        6 => "FUSED_SOURCE_BATCH_SET",
                        _ => "UNAVAILABLE_OR_FAILED_SETUP",
                    });
                services.push(DirectService {
                    attempt,
                    re_probe: pass,
                    selected_set,
                    physical_missing_set: missing.clone(),
                    candidate_physically_missing: candidate_missing,
                    foreground_set_ns: duration(total),
                    source_set_ns: trace
                        .get(p, Event::SourceSet, attempt, pass, 0)
                        .map(duration),
                    source_branch: branch,
                    sequential_candidate_source_ns: get(Event::SourceExpert).map(duration),
                    sequential_candidate_source_branch: get(Event::SourceExpert).map(|r| {
                        match r[4] {
                            1 => "LOCAL_RESIDENT_REUSE",
                            2 => "RAM_HIT",
                            3 => "FETCH_WITH_EXISTING_RETRY",
                            _ => "UNAVAILABLE",
                        }
                    }),
                    logical_retry_source_ns: get(Event::LogicalSource).map(duration),
                    stage_ns: stage.map(duration),
                    commit_ns: commit.map(duration),
                    installed_generation: installed.map(|i| i[4]),
                    installed: installed.is_some(),
                    stage_overlap_ns: overlaps,
                    scope: if candidate_missing && missing.len() == 1 {
                        "SINGLETON_MISSING_SET"
                    } else {
                        "SET"
                    },
                    exact_candidate_saved_ns: None,
                });
            }
        }
        Ok(services)
    }
    #[derive(Debug, Serialize)]
    struct Attribution {
        semantic_candidate: p1e::Candidate,
        control_raw_candidate: p1e::Candidate,
        treatment_raw_candidate: p1e::Candidate,
        control_logical_generation: Option<u64>,
        treatment_logical_generation: Option<u64>,
        p0_install_words: Option<[u64; WORDS]>,
        p0_reservation_words: Option<[u64; WORDS]>,
        direct_measured_overhead_ns: BTreeMap<&'static str, u64>,
        dispatch_start_lag_estimate_ns: Option<u64>,
        class: OutcomeClass,
        statuses: Vec<OutcomeClass>,
        readiness: Readiness,
        readiness_margin_bounds_ns: Option<[SignedNs; 2]>,
        margin_to_p1e_d_bounds_ns: Option<[SignedNs; 2]>,
        published: Option<bool>,
        consumed: bool,
        control_service: Vec<DirectService>,
        treatment_service: Vec<DirectService>,
        matched_service_contrast_estimate_ns: SignedNs,
        candidate_marginal_attribution: &'static str,
        criticality_scope: &'static str,
        exact_candidate_saved_ns: Option<u64>,
    }
    // Coverage is descriptive, not a threshold or authorization to repeat.
    // Failed children remain at their retained artifact paths; unavailable counts
    // stay null, while valid zero-movement arms retain measured zero counts.
    fn coverage(arms: &[ChildArtifact], rows: &[Attribution], valid: bool) -> Value {
        let mut classes = BTreeMap::<String, usize>::new();
        for row in rows {
            *classes.entry(format!("{:?}", row.class)).or_default() += 1;
        }
        json!({
            "evidence_status": if valid { "VALID" } else { "INVALID_EVIDENCE" },
            "arms": arms.iter().map(|a| json!({
                "arm": a.arm,
                "nominations": a.report.raw_p1e_report.as_ref().map(|r| r.observations.len()),
                "no_emissions": a.report.raw_p1e_report.as_ref().map(|r| r.no_emissions.len()),
                "actual_launches": a.report.diagnostics.as_ref().map(|d| d.records.iter().filter(|r| r.slot % STRIDE == Event::Launch as usize && r.words[2] == 6).count()),
                "matching_credits": a.report.predictor_v2_snapshot.as_ref().map(|p|p.direct_matching_demand_credits),
                "recorder_invalid": a.report.diagnostics.as_ref().map(|d|d.invalid),
            })).collect::<Vec<_>>(),
            "joined_rows": rows.len(),
            "class_counts": classes,
            "strict_generation_matches": rows.iter().filter(|r| r.control_logical_generation.is_some() && r.control_logical_generation == r.treatment_logical_generation && !r.statuses.contains(&OutcomeClass::GENERATION_DIVERGENT)).count(),
            "generation_unavailable": rows.iter().filter(|r|r.statuses.contains(&OutcomeClass::GENERATION_UNAVAILABLE)).count(),
            "measured_singleton_control_service": rows.iter().filter(|r|r.control_service.iter().any(|s|s.candidate_physically_missing && s.scope == "SINGLETON_MISSING_SET")).count(),
            "measured_shared_control_service": rows.iter().filter(|r|r.control_service.iter().any(|s|s.candidate_physically_missing && s.scope == "SET")).count(),
            "unresolved_readiness": rows.iter().filter(|r|matches!(r.readiness, Readiness::D_WINDOW_AMBIGUOUS | Readiness::UNAVAILABLE)).count(),
            "unresolved_marginality": rows.iter().filter(|r|r.candidate_marginal_attribution.starts_with("UNRESOLVED")).count(),
            "failed_or_missing_arm_count": 2usize.saturating_sub(arms.len()),
            "repeat_authorized": false,
            "performance_comparison_authorized": false,
        })
    }
    fn indexed(raw: &p1e::Report) -> Result<BTreeMap<u64, &p1e::Observation>> {
        let mut map = BTreeMap::new();
        for r in &raw.observations {
            if r.freeze.candidate.sequence == 0
                || map.insert(r.freeze.candidate.sequence, r).is_some()
            {
                return Err("duplicate semantic candidate".into());
            }
        }
        Ok(map)
    }
    fn owner_words(
        candidate: p1e::Candidate,
        generation: u64,
        claim: &[u64; WORDS],
        event: &[u64; WORDS],
    ) -> bool {
        event[8] == candidate.request.runtime_namespace
            && event[9] == candidate.request.request_sequence
            && event[10] == candidate.namespace.context
            && event[11] == candidate.namespace.arena as u64
            && event[3] == candidate.sequence
            && event[4] == u64::from(candidate.expert)
            && event[5] == generation
            && event[6] != 0
            && event[7] != 0
            && event[3..8] == claim[3..8]
    }
    fn attribute(control: &ArmReport, treatment: &ArmReport) -> Result<Vec<Attribution>> {
        let c = control
            .raw_p1e_report
            .as_ref()
            .ok_or("CONTROL P1E absent")?;
        let t = treatment
            .raw_p1e_report
            .as_ref()
            .ok_or("TREATMENT P1E absent")?;
        let cs = Trace::new(
            control
                .diagnostics
                .as_ref()
                .ok_or("CONTROL recorder absent")?,
        )?;
        let ts = Trace::new(
            treatment
                .diagnostics
                .as_ref()
                .ok_or("TREATMENT recorder absent")?,
        )?;
        for position in 0..POSITIONS {
            if cs.clean_route(position)? != ts.clean_route(position)? {
                return Err("exact clean layer47 route mismatch".into());
            }
        }
        let mut cn = c.no_emissions.clone();
        let mut tn = t.no_emissions.clone();
        cn.sort_by_key(|r| r.source.absolute_position);
        tn.sort_by_key(|r| r.source.absolute_position);
        if cn != tn || cn.windows(2).any(|r| r[0].source == r[1].source) {
            return Err("no-emission divergence/duplicate".into());
        }
        let ci = indexed(c)?;
        let ti = indexed(t)?;
        if ci.len() != ti.len() {
            return Err("orphan semantic nomination".into());
        }
        let targets = ti
            .values()
            .map(|r| r.freeze.candidate.target_position.absolute_position as usize)
            .collect::<BTreeSet<_>>();
        if targets.len() != ti.len() {
            return Err("duplicate target identity".into());
        }
        for trace in [&cs, &ts] {
            for (&slot, _) in &trace.records {
                let p = slot / STRIDE;
                let offset = slot % STRIDE;
                if matches!(offset,3..=9|11..=15|19..=20) && !targets.contains(&p) {
                    return Err("orphan writer/lease/publication record".into());
                }
            }
        }
        let mut rows = Vec::new();
        let mut credits = 0u64;
        let mut claims = 0u64;
        for (sequence, tr) in ti {
            let cr = ci.get(&sequence).ok_or("orphan treatment nomination")?;
            let candidate = tr.freeze.candidate;
            if semantic_candidate(cr.freeze.candidate) != semantic_candidate(candidate)
                || cr.outcome != tr.outcome
                || cr.incomplete.is_some()
                || tr.incomplete.is_some()
            {
                return Err("full semantic identity/outcome divergence".into());
            }
            let p = usize::try_from(candidate.target_position.absolute_position)?;
            if p > POSITIONS || candidate.target_layer != 47 {
                return Err("invalid target".into());
            }
            for (trace, r) in [(&cs, *cr), (&ts, tr)] {
                let f = trace
                    .position(p, Event::Freeze)
                    .ok_or("orphan diagnostic freeze")?;
                if f[0] != r.freeze.timestamp_ns
                    || f[2] != sequence
                    || f[3] != candidate.expert as u64
                    || f[4] != r.freeze.source.logical_generation.unwrap_or(0)
                    || f[5] != candidate.generation
                {
                    return Err("F identity/clock mismatch".into());
                }
                if let Some(d) = r.deadline {
                    let record = trace.position(p, Event::DP1e).ok_or("missing original D")?;
                    if record[0] != d.timestamp_ns || record[2] != sequence {
                        return Err("P1E D identity/clock mismatch".into());
                    }
                }
            }
            let claim = ts.position(p, Event::Claim).filter(|r| r[2] == 1);
            if cs.position(p, Event::Claim).is_some()
                || cs.position(p, Event::Lease).is_some()
                || cs.position(p, Event::WriterStart).is_some()
            {
                return Err("inert CONTROL movement".into());
            }
            let publication = ts.position(p, Event::Publication);
            let credit = ts.position(p, Event::Credit);
            let consumed = credit.is_some_and(|r| r[2] == 1);
            let launch = ts
                .position(p, Event::Launch)
                .ok_or("missing launch decision")?;
            if let Some(id) = claim {
                claims += 1;
                let generation = tr
                    .freeze
                    .source
                    .logical_generation
                    .ok_or("claimed generation absent")?;
                for event in [
                    Event::WriterStart,
                    Event::Payload,
                    Event::Close,
                    Event::LeaseRelease,
                    Event::Publication,
                    Event::Credit,
                    Event::DeferredRetire,
                ] {
                    if let Some(r) = ts.position(p, event) {
                        if !owner_words(candidate, generation, id, r) {
                            return Err("old epoch/writer/generation attribution rejected".into());
                        }
                    }
                }
                let released = ts
                    .position(p, Event::LeaseRelease)
                    .ok_or("missing lease release")?;
                if released[2] != 1
                    || ts.position(p, Event::Close).is_none()
                    || ts.position(p, Event::DeferredRetire).is_none()
                {
                    return Err("writer/lease/retirement incomplete".into());
                }
                if launch[2] == 6 {
                    let spawn = ts.position(p, Event::Spawn).ok_or("missing dispatch")?;
                    let start = ts
                        .position(p, Event::WriterStart)
                        .ok_or("missing worker start")?;
                    let payload = ts
                        .position(p, Event::Payload)
                        .ok_or("missing payload enqueue")?;
                    let close = ts.position(p, Event::Close).unwrap();
                    if start[0] < spawn[0]
                        || payload[0] < start[0]
                        || close[0] < payload[1]
                        || payload[2] != 1
                    {
                        return Err("writer chronology/failure invalid".into());
                    }
                }
            } else if consumed {
                return Err("credit without claim".into());
            }
            if consumed {
                let published = publication.ok_or("credit without publication")?;
                let id = claim.unwrap();
                let route = ts.clean_route(p)?;
                let clean_attempt = ts.position(p, Event::CleanRoute).unwrap()[2] as usize - 1;
                let binding = ts
                    .get(p, Event::Binding, clean_attempt, 0, 0)
                    .is_some_and(|r| owner_words(candidate, id[5], id, r));
                if published[2] != 0 || !route.contains(&candidate.expert) || !binding {
                    return Err("exact binding/clean-credit mismatch".into());
                }
                credits += 1;
            }
            let close_record = ts.position(p, Event::Close);
            let close = close_record.filter(|r| r[2] == 1).map(|r| (r[0], r[1]));
            let d = ts.position(p, Event::DEntry).map(|r| r[0]);
            let ready = if close_record.is_some_and(|r| r[2] == 2 && d.is_some_and(|d| r[0] > d)) {
                Readiness::AFTER_D
            } else {
                readiness(close, d, consumed)
            };
            let control_service = service(&cs, p, candidate.expert)?;
            let treatment_service = service(&ts, p, candidate.expert)?;
            let cg = cr.freeze.source.logical_generation;
            let tg = tr.freeze.source.logical_generation;
            let divergent = cg.zip(tg).is_some_and(|(a, b)| a != b)
                || control_service
                    .iter()
                    .filter_map(|s| s.installed_generation)
                    .any(|g| Some(g) != tg);
            let available = cg.is_some() && tg.is_some();
            let mut statuses = Vec::new();
            if divergent {
                statuses.push(OutcomeClass::GENERATION_DIVERGENT);
            }
            if !available {
                statuses.push(OutcomeClass::GENERATION_UNAVAILABLE);
            }
            if publication.is_some_and(|r| r[2] == 1) {
                statuses.push(OutcomeClass::PUBLICATION_BUSY);
            }
            let censored = tr.outcome == p1e::Outcome::Censored;
            if censored {
                statuses.push(OutcomeClass::CENSORED);
            }
            let missing_paid = control_service.iter().any(|s| {
                s.candidate_physically_missing
                    && s.installed
                    && s.foreground_set_ns > 0
                    && s.source_set_ns.is_some()
            });
            let ordinary = tr.deadline.is_some_and(|d| d.current == Some(true))
                || publication.is_some_and(|r| r[2] == 5);
            let class = if launch[2] != 6 {
                OutcomeClass::NOT_LAUNCHED_OR_INELIGIBLE
            } else if censored
                || tr.outcome
                    == (p1e::Outcome::Resolved {
                        prediction_hit: false,
                    })
            {
                OutcomeClass::UNUSED
            } else if ordinary && !consumed {
                OutcomeClass::ORDINARY_WON_RACE
            } else if divergent {
                OutcomeClass::GENERATION_DIVERGENT
            } else if !available {
                OutcomeClass::GENERATION_UNAVAILABLE
            } else if ready == Readiness::D_WINDOW_AMBIGUOUS {
                OutcomeClass::D_WINDOW_AMBIGUOUS
            } else if ready == Readiness::READY_DURING_D_WINDOW {
                OutcomeClass::READY_DURING_D_WINDOW
            } else if ready == Readiness::AFTER_D && publication.is_some_and(|r| r[2] == 2) {
                OutcomeClass::LATE
            } else if publication.is_some_and(|r| r[2] == 1) {
                OutcomeClass::PUBLICATION_BUSY
            } else if consumed
                && ready == Readiness::BEFORE_D
                && cr.deadline.is_some_and(|d| d.current == Some(false))
                && missing_paid
            {
                OutcomeClass::READY_AND_CRITICAL
            } else if consumed
                && ready == Readiness::BEFORE_D
                && !control_service
                    .iter()
                    .any(|s| s.candidate_physically_missing)
            {
                OutcomeClass::READY_BUT_NOT_CRITICAL
            } else {
                OutcomeClass::ATTRIBUTION_UNRESOLVED
            };
            let singleton = missing_paid
                && control_service
                    .iter()
                    .filter(|s| s.candidate_physically_missing)
                    .all(|s| s.scope == "SINGLETON_MISSING_SET");
            if missing_paid && !singleton {
                statuses.push(OutcomeClass::ATTRIBUTION_UNRESOLVED);
            }
            let total = |services: &[DirectService]| -> Result<u64> {
                let mut seen = BTreeSet::new();
                services
                    .iter()
                    .filter(|s| seen.insert(s.attempt))
                    .try_fold(0u64, |n, s| {
                        n.checked_add(s.foreground_set_ns)
                            .ok_or_else(|| "service duration overflow".into())
                    })
            };
            rows.push(Attribution {
                semantic_candidate: semantic_candidate(candidate),
                control_raw_candidate: cr.freeze.candidate,
                treatment_raw_candidate: candidate,
                control_logical_generation: cg,
                treatment_logical_generation: tg,
                p0_install_words: ts.position(p, Event::P0Install).copied(),
                p0_reservation_words: ts.position(p, Event::P0Reservation).copied(),
                direct_measured_overhead_ns: [
                    ("inclusive_observation_preparation", Event::Observation),
                    ("inclusive_launch", Event::Launch),
                    ("host_lease", Event::Lease),
                    ("sidecar_claim", Event::Claim),
                    ("dispatch_call", Event::Spawn),
                    ("writer_staging_enqueue", Event::Payload),
                    ("publication_call", Event::Publication),
                ]
                .into_iter()
                .filter_map(|(name, event)| ts.position(p, event).map(|r| (name, duration(r))))
                .collect(),
                dispatch_start_lag_estimate_ns: ts
                    .position(p, Event::Spawn)
                    .zip(ts.position(p, Event::WriterStart))
                    .and_then(|(spawn, start)| start[0].checked_sub(spawn[0])),
                class,
                statuses,
                readiness: ready,
                readiness_margin_bounds_ns: close
                    .zip(d)
                    .map(|((lo, hi), d)| [difference(d, hi), difference(d, lo)]),
                margin_to_p1e_d_bounds_ns: close.zip(tr.deadline).map(|((lo, hi), d)| {
                    [
                        difference(d.timestamp_ns, hi),
                        difference(d.timestamp_ns, lo),
                    ]
                }),
                published: publication.map(|r| r[2] == 0),
                consumed,
                matched_service_contrast_estimate_ns: difference(
                    total(&control_service)?,
                    total(&treatment_service)?,
                ),
                control_service,
                treatment_service,
                candidate_marginal_attribution: if divergent || !available {
                    "UNRESOLVED_GENERATION"
                } else if singleton {
                    "SINGLETON_FOREGROUND_EXPOSURE"
                } else {
                    "UNRESOLVED"
                },
                criticality_scope: "FOREGROUND_DEMAND_SET",
                exact_candidate_saved_ns: None,
            });
        }
        let p0 = treatment
            .predictor_v2_snapshot
            .as_ref()
            .ok_or("missing P0")?;
        if credits != p0.direct_matching_demand_credits || claims != p0.reservations {
            return Err("P0/claim/credit reconciliation mismatch".into());
        }
        Ok(rows)
    }

    #[cfg(test)]
    mod p1r_tests {
        use super::*;
        use crate::predictor_v2::{ModelMetadata, PositionIdentity, RequestIdentity};
        fn identity() -> RequestIdentity {
            RequestIdentity {
                runtime_namespace: 1,
                request_sequence: 1,
                phase: RequestPhase::Measured,
                phase_run_index: 0,
            }
        }
        fn namespace() -> p1e::Namespace {
            p1e::Namespace {
                runtime: 1,
                context: 2,
                arena: 3,
                layer: 47,
                capacity: 8,
            }
        }
        fn model() -> ModelMetadata {
            ModelMetadata {
                num_layers: 48,
                num_experts: 128,
                top_k: 8,
            }
        }
        fn physical(ids: [u32; 8]) -> p1e::PhysicalEvidence {
            p1e::PhysicalEvidence {
                snapshot: p1e::PhysicalSnapshot {
                    namespace: namespace(),
                    residents: std::array::from_fn(|i| {
                        Some(p1e::Resident {
                            expert: ids[i],
                            generation: 1,
                            bank: 0,
                            slot: i as u32,
                            epoch: 1,
                        })
                    }),
                },
                event_cutoff: 3,
                committed_installs: 2,
                physical_victims: 1,
            }
        }
        fn record(sequence: u64, hit: bool, f: Option<bool>, d: Option<bool>) -> p1e::Observation {
            let source = PositionIdentity::from_prompt_length(sequence as usize + 2, 4).unwrap();
            let target = PositionIdentity::from_prompt_length(sequence as usize + 3, 4).unwrap();
            let candidate = p1e::Candidate {
                request: identity(),
                model: model(),
                namespace: namespace(),
                source_position: source,
                target_position: target,
                source_layer: 47,
                target_layer: 47,
                position_distance: 1,
                nominal_layer_lead: 0,
                source_set: [0, 1, 2, 3, 4, 5, 6, 7],
                expert: 8,
                score: 17,
                signal_revision: p1e::REVISION,
                generation: sequence + 3,
                sequence,
                committed_position_cutoff: sequence + 3,
                table_update_cutoff: sequence + 2,
            };
            // Deserialize the complete public report format to retain its private
            // chronology bits, without exposing or changing any P1E implementation.
            serde_json::from_value(json!({
            "freeze": p1e::Freeze { candidate, timestamp_ns: 100,
                physical: Some(physical([0,1,2,3,4,5,6,7])), current: f,
                source: p1e::HostSource { logical_generation: Some(9), logical_materialized: false,
                    ram_resident: false, permanence: p1e::Permanence::Unknown }, incomplete: None },
            "deadline": p1e::Deadline { request: identity(), position: target,
                timestamp_ns: 100 + (sequence + 1) * 10,
                physical: Some(physical([0,1,2,3,4,5,6,7])), current: d,
                host_lead_ns: Some((sequence + 1) * 10), incomplete: None },
            "outcome": p1e::Outcome::Resolved { prediction_hit: hit },
            "recovery": [p1e::RecoveryEvent { attempt: 1, attempted_start: 47, attempted_end: 48,
                first_failure_layer: Some(47), final_status: 1, layer47_demand_service_completed: true }],
            "completion_physical": physical([8,9,10,11,12,13,14,15]),
            "incomplete": null, "initial_attempt_seen": true, "deadline_eligible": false
        })).unwrap()
        }
        fn raw(records: Vec<p1e::Observation>) -> p1e::Report {
            let mut p = p1e::Partitions {
                emitted: records.len(),
                ..Default::default()
            };
            for r in &records {
                match r.outcome {
                    p1e::Outcome::Pending => p.pending += 1,
                    p1e::Outcome::Censored => p.censored += 1,
                    p1e::Outcome::Resolved { prediction_hit } => {
                        p.resolved += 1;
                        if prediction_hit {
                            p.prediction_hits += 1;
                        } else {
                            p.route_misses += 1;
                        }
                    }
                }
                match r.freeze.current {
                    Some(true) => {
                        p.valid_f += 1;
                        p.current_at_f += 1;
                    }
                    Some(false) => {
                        p.valid_f += 1;
                        p.absent_at_f += 1;
                    }
                    None => {}
                }
                let o = r.opportunities();
                p.useful += usize::from(o.useful == Some(true));
                p.target_confirmed_useful += usize::from(o.target_confirmed_useful == Some(true));
            }
            p1e::Report {
                request: identity(),
                opportunities: records
                    .iter()
                    .map(|r| (r.freeze.candidate.sequence, r.opportunities()))
                    .collect(),
                observations: records,
                no_emissions: Vec::new(),
                incomplete: None,
                partitions: p,
                readiness_measured: false,
            }
        }

        fn snapshot() -> Snapshot {
            Snapshot {
                invalid: false,
                capacity: SLOTS,
                storage_bytes: SLOTS * std::mem::size_of::<Slot>(),
                records: vec![],
                layout: Some(RecorderLayout::current()),
            }
        }
        fn put(
            s: &mut Snapshot,
            p: usize,
            e: Event,
            a: usize,
            pass: usize,
            m: usize,
            lo: u64,
            hi: u64,
            data: &[u64],
        ) {
            let mut words = [0; WORDS];
            words[0] = lo;
            words[1] = hi;
            words[2..2 + data.len()].copy_from_slice(data);
            s.records.push(Record {
                slot: slot_index(p, a, pass, m, e).unwrap(),
                words,
            });
        }
        fn item_mut(
            s: &mut Snapshot,
            p: usize,
            e: Event,
            a: usize,
            pass: usize,
            m: usize,
        ) -> &mut Record {
            let slot = slot_index(p, a, pass, m, e).unwrap();
            s.records.iter_mut().find(|r| r.slot == slot).unwrap()
        }
        fn fixture() -> (ArmReport, ArmReport) {
            let mut c = ArmReport::new(Arm::Control);
            let mut t = ArmReport::new(Arm::Treatment);
            let records = vec![
                record(1, true, Some(false), Some(false)),
                record(2, true, Some(false), Some(false)),
            ];
            c.raw_p1e_report = Some(raw(records.clone()));
            t.raw_p1e_report = Some(raw(records));
            c.diagnostics = Some(snapshot());
            t.diagnostics = Some(snapshot());
            for arm in [&mut c, &mut t] {
                let raw = arm.raw_p1e_report.as_mut().unwrap();
                for p in 0..POSITIONS {
                    if !raw
                        .observations
                        .iter()
                        .any(|r| r.freeze.candidate.source_position.absolute_position == p as u64)
                    {
                        raw.no_emissions.push(p1e::NoEmission {
                            source: PositionIdentity::from_prompt_length(p, 4).unwrap(),
                            target: PositionIdentity::from_prompt_length(p + 1, 4).unwrap(),
                            reason: p1e::NoEmissionReason::NoPositiveHistory,
                        });
                    }
                }
                let snap = arm.diagnostics.as_mut().unwrap();
                for p in 0..POSITIONS {
                    for e in [Event::CleanRoute, Event::Boundary] {
                        put(
                            snap,
                            p,
                            e,
                            0,
                            0,
                            0,
                            160,
                            160,
                            &[1, 8, 8, 9, 10, 11, 12, 13, 14, 15],
                        );
                    }
                }
                for r in &raw.observations {
                    let c = r.freeze.candidate;
                    let p = c.target_position.absolute_position as usize;
                    put(
                        snap,
                        p,
                        Event::Freeze,
                        0,
                        0,
                        0,
                        100,
                        100,
                        &[c.sequence, 8, 9, c.generation],
                    );
                    put(
                        snap,
                        p,
                        Event::DP1e,
                        0,
                        0,
                        0,
                        r.deadline.unwrap().timestamp_ns,
                        r.deadline.unwrap().timestamp_ns,
                        &[c.sequence],
                    );
                    put(snap, p, Event::DEntry, 0, 0, 0, 118, 118, &[]);
                    put(
                        snap,
                        p,
                        Event::Launch,
                        0,
                        0,
                        0,
                        101,
                        108,
                        &[if arm.arm == Arm::Control { 0 } else { 6 }],
                    );
                    if arm.arm == Arm::Treatment {
                        let id = P1jIdentity {
                            candidate: c,
                            logical_generation: 9,
                            epoch: c.sequence as u32,
                            writer_sequence: c.sequence,
                        };
                        for (event, lo, hi, code) in [
                            (Event::Claim, 102, 103, 1),
                            (Event::WriterStart, 105, 105, 1),
                            (Event::Payload, 106, 110, 1),
                            (Event::Close, 111, 115, 1),
                            (Event::LeaseRelease, 116, 116, 1),
                            (Event::Publication, 120, 121, 0),
                            (Event::Binding, 122, 122, 1),
                            (Event::Credit, 161, 161, 1),
                            (Event::DeferredRetire, 162, 162, 1),
                        ] {
                            put(snap, p, event, 0, 0, 0, lo, hi, &identity_words(id, code));
                        }
                        put(snap, p, Event::Spawn, 0, 0, 0, 104, 109, &[]); // worker actually starts before spawn returns
                    }
                }
            }
            t.predictor_v2_snapshot = Some(ReconciliationSnapshot {
                reservations: 2,
                direct_matching_demand_credits: 2,
                ..Default::default()
            });
            (c, t)
        }
        fn add_service(
            arm: &mut ArmReport,
            p: usize,
            missing: &[u64],
            candidate_missing: bool,
            batch: bool,
        ) {
            let s = arm.diagnostics.as_mut().unwrap();
            let ids = [1, 8, 6024, 6025, 6026, 6027, 6028, 6029, 6030, 6031];
            for e in [Event::Service, Event::Selected] {
                put(s, p, e, 0, 0, 0, 122, 160, &ids);
            }
            let mut probe = vec![1, missing.len() as u64];
            probe.extend_from_slice(missing);
            put(s, p, Event::Probe, 0, 0, 0, 123, 123, &probe);
            if !missing.is_empty() {
                put(s, p, Event::SourceSet, 0, 0, 0, 124, 130, &probe);
                put(
                    s,
                    p,
                    Event::SourceBranch,
                    0,
                    0,
                    0,
                    124,
                    130,
                    &[if batch { 5 } else { 1 }],
                );
            }
            if candidate_missing {
                if !batch {
                    put(s, p, Event::SourceExpert, 0, 0, 0, 125, 129, &[1, 6024, 3]);
                }
                put(s, p, Event::Stage, 0, 0, 0, 131, 140, &[1, 6024, 9]);
                put(s, p, Event::Commit, 0, 0, 0, 145, 147, &[1, 6024, 9]);
                put(
                    s,
                    p,
                    Event::Installed,
                    0,
                    0,
                    0,
                    147,
                    147,
                    &[1, 6024, 9, 0, 0, 1],
                );
            }
            if batch {
                put(s, p, Event::Stage, 0, 0, 1, 135, 146, &[1, 6025, 9]);
            }
        }
        #[test]
        fn p1r_shared_clock_and_checked_conversion() {
            assert_eq!(checked_ns(u64::MAX as u128), Some(u64::MAX));
            assert_eq!(checked_ns(u64::MAX as u128 + 1), None);
            let origin = Instant::now();
            let r = Recorder::fixture(origin, 1);
            let d = r.context(0, 0);
            let w = d.clone();
            assert_eq!(d.recorder.origin, w.recorder.origin);
            let lower = d.now().unwrap();
            let worker = std::thread::spawn(move || w.now().unwrap()).join().unwrap();
            let upper = d.now().unwrap();
            assert!(lower <= worker && worker <= upper);
        }
        #[test]
        fn p1r_close_before_after_overlap_equal_and_consumed_window() {
            assert_eq!(readiness(Some((2, 4)), Some(5), false), Readiness::BEFORE_D);
            assert_eq!(readiness(Some((6, 8)), Some(5), false), Readiness::AFTER_D);
            assert_eq!(
                readiness(Some((6, 8)), Some(5), true),
                Readiness::READY_DURING_D_WINDOW
            );
            for bracket in [(2, 5), (5, 8), (4, 6), (5, 5)] {
                assert_eq!(
                    readiness(Some(bracket), Some(5), false),
                    Readiness::D_WINDOW_AMBIGUOUS
                );
            }
            assert_eq!(readiness(None, Some(5), true), Readiness::UNAVAILABLE);
        }
        #[test]
        fn p1r_writer_start_before_spawn_return_is_valid() {
            let (c, t) = fixture();
            assert!(attribute(&c, &t).is_ok());
        }
        #[test]
        fn p1r_semantic_join_independent_of_record_order() {
            let (c, mut t) = fixture();
            t.raw_p1e_report.as_mut().unwrap().observations.reverse();
            t.raw_p1e_report.as_mut().unwrap().no_emissions.reverse();
            t.diagnostics.as_mut().unwrap().records.reverse();
            assert_eq!(attribute(&c, &t).unwrap().len(), 2);
        }
        #[test]
        fn p1r_namespace_only_normalization_retains_all_generations() {
            let c = record(1, true, Some(false), Some(false)).freeze.candidate;
            let mut t = c;
            t.request.runtime_namespace = 999;
            t.request.request_sequence = 77;
            t.namespace.runtime = 99;
            t.namespace.context = 88;
            t.namespace.arena = 77;
            assert_eq!(semantic_candidate(c), semantic_candidate(t));
            for field in 0..8 {
                let mut v = t;
                match field {
                    0 => v.generation += 1,
                    1 => v.signal_revision += 1,
                    2 => v.score += 1,
                    3 => v.committed_position_cutoff += 1,
                    4 => v.table_update_cutoff += 1,
                    5 => v.namespace.capacity += 1,
                    6 => v.source_set[0] += 1,
                    _ => v.target_position.absolute_position += 1,
                };
                assert_ne!(semantic_candidate(c), semantic_candidate(v));
            }
        }
        #[test]
        fn p1r_logical_generation_divergence_is_explicit() {
            let (mut c, t) = fixture();
            c.raw_p1e_report.as_mut().unwrap().observations[0]
                .freeze
                .source
                .logical_generation = Some(10);
            item_mut(c.diagnostics.as_mut().unwrap(), 4, Event::Freeze, 0, 0, 0).words[4] = 10;
            let rows = attribute(&c, &t).unwrap();
            assert_eq!(rows[0].class, OutcomeClass::GENERATION_DIVERGENT);
            assert_eq!(rows[0].exact_candidate_saved_ns, None);
        }
        #[test]
        fn p1r_duplicate_orphan_no_emission_and_diagnostic_record_rejected() {
            for case in 0..4 {
                let (c, mut t) = fixture();
                match case {
                    0 => {
                        let raw = t.raw_p1e_report.as_mut().unwrap();
                        raw.observations.push(raw.observations[0].clone());
                    }
                    1 => {
                        t.raw_p1e_report.as_mut().unwrap().observations.pop();
                    }
                    2 => {
                        t.raw_p1e_report.as_mut().unwrap().no_emissions.pop();
                    }
                    _ => {
                        let s = t.diagnostics.as_mut().unwrap();
                        s.records.push(s.records[0].clone());
                    }
                };
                assert!(attribute(&c, &t).is_err());
            }
        }
        #[test]
        fn p1r_physical_hit_and_selected_but_not_missing_have_no_install_attribution() {
            let (mut c, t) = fixture();
            add_service(&mut c, 4, &[6025], false, false);
            let rows = attribute(&c, &t).unwrap();
            assert_eq!(rows[0].class, OutcomeClass::READY_BUT_NOT_CRITICAL);
            assert!(!rows[0].control_service[0].candidate_physically_missing);
            assert_eq!(rows[0].control_service[0].stage_ns, None);
        }
        #[test]
        fn p1r_singleton_source_install_measures_exposure_not_saved_time() {
            let (mut c, t) = fixture();
            add_service(&mut c, 4, &[6024], true, false);
            let rows = attribute(&c, &t).unwrap();
            assert_eq!(rows[0].class, OutcomeClass::READY_AND_CRITICAL);
            assert_eq!(rows[0].control_service[0].scope, "SINGLETON_MISSING_SET");
            assert_eq!(
                rows[0].control_service[0].sequential_candidate_source_ns,
                Some(4)
            );
            assert_eq!(rows[0].exact_candidate_saved_ns, None);
        }
        #[test]
        fn p1r_shared_batch_stays_set_scoped_and_stage_overlap_is_nonadditive() {
            let (mut c, t) = fixture();
            add_service(&mut c, 4, &[6024, 6025], true, true);
            let rows = attribute(&c, &t).unwrap();
            let m = &rows[0].control_service[0];
            assert_eq!(m.source_set_ns, Some(6));
            assert_eq!(m.sequential_candidate_source_ns, None);
            assert_eq!(m.scope, "SET");
            assert_eq!(m.stage_overlap_ns, vec![5]);
            assert!(rows[0]
                .statuses
                .contains(&OutcomeClass::ATTRIBUTION_UNRESOLVED));
        }
        #[test]
        fn p1r_failed_commit_after_successful_stage_never_installed() {
            let (mut c, t) = fixture();
            add_service(&mut c, 4, &[6024], true, false);
            item_mut(c.diagnostics.as_mut().unwrap(), 4, Event::Commit, 0, 0, 0).words[2] = 0;
            assert!(attribute(&c, &t).is_err());
        }
        #[test]
        fn p1r_old_epoch_writer_stale_generation_and_namespace_rejected() {
            for index in [5, 6, 7, 8, 9, 10, 11] {
                let (c, mut t) = fixture();
                item_mut(t.diagnostics.as_mut().unwrap(), 4, Event::Close, 0, 0, 0).words[index] +=
                    1;
                assert!(attribute(&c, &t).is_err());
            }
        }
        #[test]
        fn p1r_exact_binding_credit_and_invalid_tail_reconciliation() {
            for case in 0..3 {
                let (c, mut t) = fixture();
                match case {
                    0 => {
                        item_mut(t.diagnostics.as_mut().unwrap(), 4, Event::Binding, 0, 0, 0)
                            .words[6] += 1
                    }
                    1 => {
                        item_mut(t.diagnostics.as_mut().unwrap(), 4, Event::Boundary, 0, 0, 0)
                            .words[2] = 0
                    }
                    _ => {
                        t.predictor_v2_snapshot
                            .as_mut()
                            .unwrap()
                            .direct_matching_demand_credits += 1
                    }
                };
                assert!(attribute(&c, &t).is_err());
            }
        }
        #[test]
        fn p1r_bounded_overflow_and_duplicate_preserve_caller_execution() {
            let r = Recorder::fixture(Instant::now(), 1);
            let d = r.context(0, 0);
            let mut demand = 0;
            d.stamp(Event::DEntry, &[]);
            d.stamp(Event::DEntry, &[]);
            demand += 1;
            r.context(500, 0).stamp(Event::CleanRoute, &[]);
            demand += 1;
            assert_eq!(demand, 2);
            assert!(r.snapshot().invalid);
            assert_eq!(r.snapshot().records.len(), 1);
        }
        #[test]
        fn p1r_first_fresh_d_once_and_replay_cannot_reopen_source_hook() {
            let source = include_str!("gpu_native_token_loop.rs")
                .split("#[cfg(test)]")
                .next()
                .unwrap();
            assert_eq!(source.matches("stamp(P1rEvent::DEntry").count(), 1);
            assert!(source.contains("if p1e_deadline_eligible && layer_idx == p1e::LAYER"));
            assert!(source.contains(
                "!full_token_replay && segment.attempt_start == GpuNativeAttemptStart::Fresh"
            ));
        }
        #[test]
        fn p1r_disabled_and_inert_source_paths_are_dormant() {
            let token = include_str!("gpu_native_token_loop.rs");
            let launch = token
                .split("fn launch_p1j_at_freeze(")
                .nth(1)
                .unwrap()
                .split("fn publish_p1j_at_deadline")
                .next()
                .unwrap();
            let guard = launch
                .find("movement.mode == P1jRequestMode::InertResourceOnly")
                .unwrap();
            let end = guard + launch[guard..].find("return;").unwrap();
            for action in [
                "p1m_launch(",
                ".try_prepare_p1r(",
                "writer.spawn()",
                "*pending = Some(",
            ] {
                assert!(end < launch.find(action).unwrap());
            }
            assert!(token.contains("p1r: None,"));
            assert_eq!(token.matches("p1r::Recorder::new(").count(), 1);
            for s in [include_str!("server.rs"), include_str!("config.rs")] {
                assert!(!s.contains("enable_p1r"));
            }
            let core = include_str!("gpu_native_predictor_v2_critical_path_attribution.rs")
                .split("// Separate P1R command/transport.")
                .next()
                .unwrap();
            for forbidden in [
                "thread_local!",
                "static CURRENT",
                ".lock()",
                ".await",
                ".send(",
                "std::fs::write",
                "serde_json::to_",
            ] {
                assert!(!core.contains(forbidden), "{forbidden}");
            }
        }
        #[test]
        fn p1r_no_new_gpu_or_foreground_wait_operations() {
            for (name, source) in [
                (
                    "gpu_native_token_loop.rs",
                    include_str!("gpu_native_token_loop.rs"),
                ),
                (
                    "gpu_native_residency.rs",
                    include_str!("gpu_native_residency.rs"),
                ),
                ("engine.rs", include_str!("engine.rs")),
                (
                    "backend/gpu_native.rs",
                    include_str!("backend/gpu_native.rs"),
                ),
            ] {
                let base = historical_source(name);
                let production = |s: &str| {
                    s.rfind("\n#[cfg(test)]\nmod tests")
                        .or_else(|| s.rfind("\n#[cfg(test)]\npub(crate) mod tests"))
                        .unwrap()
                };
                let old = &base[..production(base)];
                let current = &source[..production(source)];
                for op in [
                    "device.poll(",
                    "queue.submit(",
                    ".map_async(",
                    ".spawn_blocking(",
                    ".join(",
                    ".wait(",
                    ".lock(",
                ] {
                    assert_eq!(
                        old.matches(op).count(),
                        current.matches(op).count(),
                        "{name}: {op}"
                    );
                }
            }
        }
        #[test]
        fn p1r_three_witness_production_prefixes_and_expected_digests_unchanged() {
            for (name, source) in [
                (
                    "gpu_native_predictor_v2_observation.rs",
                    include_str!("gpu_native_predictor_v2_observation.rs"),
                ),
                (
                    "gpu_native_q4_route_parallel.rs",
                    include_str!("gpu_native_q4_route_parallel.rs"),
                ),
                (
                    "gpu_native_predictor_v2_sidecar_performance.rs",
                    include_str!("gpu_native_predictor_v2_sidecar_performance.rs"),
                ),
            ] {
                let base = historical_source(name);
                assert_eq!(
                    base.split("#[cfg(test)]").next(),
                    source.split("#[cfg(test)]").next(),
                    "{name}"
                );
                let hashes = |s: &str| {
                    s.split('"')
                        .filter(|v| v.len() == 64 && v.chars().all(|c| c.is_ascii_hexdigit()))
                        .map(str::to_owned)
                        .collect::<BTreeSet<_>>()
                };
                assert_eq!(hashes(base), hashes(source), "{name}");
            }
        }
        #[test]
        fn p1r_exact_nine_path_scope_and_frozen_policy_sources() {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
            let git = |args: &[&str]| {
                let o = std::process::Command::new("git")
                    .args(args)
                    .current_dir(root)
                    .output()
                    .unwrap();
                assert!(o.status.success());
                String::from_utf8(o.stdout).unwrap()
            };
            let mut names = git(&[
                "diff",
                "--name-only",
                "a6b9cbcd6e6b576f90a1d875c9351c7276c950e2",
                "--",
            ])
            .lines()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
            names.extend(
                git(&["ls-files", "--others", "--exclude-standard"])
                    .lines()
                    .map(str::to_owned),
            );
            let expected = [
                "gpu_native_predictor_v2_critical_path_attribution.rs",
                "main.rs",
                "gpu_native_token_loop.rs",
                "gpu_native_residency.rs",
                "backend/gpu_native.rs",
                "engine.rs",
                "gpu_native_predictor_v2_observation.rs",
                "gpu_native_q4_route_parallel.rs",
                "gpu_native_predictor_v2_sidecar_performance.rs",
            ]
            .into_iter()
            .map(|f| format!("rust-engine/src/{f}"))
            .collect();
            assert_eq!(names, expected);
            for (name, bytes) in [
                (
                    "predictor_v2.rs",
                    include_bytes!("predictor_v2.rs").as_slice(),
                ),
                (
                    "expert_cache.rs",
                    include_bytes!("expert_cache.rs").as_slice(),
                ),
                (
                    "gpu_native_source_upload.rs",
                    include_bytes!("gpu_native_source_upload.rs").as_slice(),
                ),
                ("config.rs", include_bytes!("config.rs").as_slice()),
                ("server.rs", include_bytes!("server.rs").as_slice()),
            ] {
                assert_eq!(historical_source(name).as_bytes(), bytes, "{name}");
            }
        }
        #[test]
        fn p1r_coverage_keeps_missing_evidence_and_marginality_explicit() {
            let (c, t) = fixture();
            let rows = attribute(&c, &t).unwrap();
            let result = coverage(&[], &rows, false);
            assert_eq!(result["evidence_status"], "INVALID_EVIDENCE");
            assert_eq!(result["failed_or_missing_arm_count"], 2);
            assert_eq!(result["joined_rows"], 2);
            assert_eq!(result["strict_generation_matches"], 2);
            assert_eq!(result["repeat_authorized"], false);
            assert_eq!(result["performance_comparison_authorized"], false);
            assert!(result["arms"].as_array().unwrap().is_empty());
        }
        #[test]
        fn p1r_artifact_invariants_are_recomputed_independent_of_record_order() {
            let (mut control, _) = fixture();
            let before = arm_invariant_errors(&control, 16, OUTPUT_TOKENS, PLANNED_POSITIONS);
            control
                .raw_p1e_report
                .as_mut()
                .unwrap()
                .observations
                .reverse();
            assert_eq!(
                before,
                arm_invariant_errors(&control, 16, OUTPUT_TOKENS, PLANNED_POSITIONS)
            );
            control.ordinary_invariants_pass = true;
            control.initial_state_pass = true;
            control.sidecar_freshly_initialized = true;
            assert!(
                !valid_arm(&control),
                "claimed booleans cannot replace raw evidence"
            );
            control.raw_p1e_report.as_mut().unwrap().partitions.emitted += 1;
            assert!(
                arm_invariant_errors(&control, 16, OUTPUT_TOKENS, PLANNED_POSITIONS)
                    .iter()
                    .any(|e| e.contains("partitions do not reconcile"))
            );
            let raw = control.raw_p1e_report.as_mut().unwrap();
            raw.opportunities.push(raw.opportunities[0]);
            assert!(
                arm_invariant_errors(&control, 16, OUTPUT_TOKENS, PLANNED_POSITIONS)
                    .iter()
                    .any(|e| e.contains("opportunities disagree"))
            );
        }
        #[test]
        fn p1r_unknown_layout_and_reserved_slot_invalidate_evidence() {
            let (mut c, t) = fixture();
            c.diagnostics
                .as_mut()
                .unwrap()
                .layout
                .as_mut()
                .unwrap()
                .position_stride += 1;
            assert!(attribute(&c, &t).is_err());
            c.diagnostics.as_mut().unwrap().layout = Some(RecorderLayout::current());
            c.diagnostics.as_mut().unwrap().records.push(Record {
                slot: 21,
                words: [0; WORDS],
            });
            assert!(attribute(&c, &t).is_err());
        }
        fn provenance_fixture() -> ArmProvenance {
            let mut production_configuration = evidence::ProductionConfiguration::default();
            production_configuration
                .cache_residency
                .gpu_vram_anchor_ratio = 0.1_f32;
            ArmProvenance {
                provenance: evidence::BenchmarkProvenance {
                    build: crate::qualification::BuildProvenance {
                        git_sha: Some("c390bd4607e9108d8c63dc2e96f0aca9061741fc".into()),
                        dirty: Some(false),
                        package_version: "0.1.0".into(),
                    },
                    executable_canonical_path: "/fixture/mer".into(),
                    executable_sha256: "binary-sha256".into(),
                    resolved_config_sha256: "resolved-config-sha256".into(),
                    artifacts: crate::qualification::QualificationArtifacts::default(),
                    expert_metadata: crate::qualification::ExpertMetadataEvidence {
                        dtype: Some("q4_0".into()),
                        q4_0_layout: Some("standard_v1".into()),
                        conversion_mode: None,
                        source: None,
                        explicitly_synthetic: false,
                    },
                },
                config_path: "/fixture/config.toml".into(),
                config_sha256: "config-sha256".into(),
                model_identity: crate::greedy_parity::ModelIdentityEvidence {
                    architecture: "qwen3_moe".into(),
                    num_layers: 48,
                    num_experts_per_layer: 128,
                    total_experts: 6_144,
                    top_k: 8,
                    d_model: 2_048,
                    d_ff: 768,
                    routed_expert_dtype: "q4_0".into(),
                },
                production_configuration,
            }
        }

        #[test]
        fn p1r_provenance_old_text_transport_changes_f32_identity() {
            let provenance = provenance_fixture();
            let direct = serde_json::to_value(&provenance).unwrap();
            let old: Value = read_child_json(&serde_json::to_vec(&provenance).unwrap()).unwrap();
            let leaf = "/production_configuration/cache_residency/gpu_vram_anchor_ratio";
            assert_eq!(direct.pointer(leaf).unwrap().as_f64(), Some(0.1_f32 as f64));
            assert_eq!(old.pointer(leaf).unwrap().as_f64(), Some(0.1_f64));
            assert_ne!(direct, old);
        }

        #[test]
        fn p1r_provenance_typed_identity_matches_independent_child_reconstruction() {
            let control = provenance_fixture();
            let imported: ArmProvenance =
                read_child_json(&serde_json::to_vec(&control).unwrap()).unwrap();
            let treatment = provenance_fixture();
            let expected = provenance_identity(&imported).unwrap();
            assert_eq!(expected, provenance_identity(&control).unwrap());
            assert_eq!(expected, provenance_identity(&treatment).unwrap());
            let mut bound = Some(expected.clone());
            bind_provenance_identity(&mut bound, provenance_identity(&treatment).unwrap()).unwrap();
            assert_eq!(bound, Some(expected));
        }

        #[test]
        fn p1r_provenance_true_drift_is_rejected_without_rebinding() {
            let mut expected = None;
            bind_provenance_identity(
                &mut expected,
                provenance_identity(&provenance_fixture()).unwrap(),
            )
            .unwrap();
            let original = expected.clone();
            for field in [
                "f32",
                "config",
                "resolved_config",
                "model",
                "executable",
                "source",
            ] {
                let mut changed = provenance_fixture();
                match field {
                    "f32" => {
                        changed
                            .production_configuration
                            .cache_residency
                            .gpu_vram_anchor_ratio = f32::from_bits(0.1_f32.to_bits() + 1);
                    }
                    "config" => changed.config_sha256.push('x'),
                    "resolved_config" => changed.provenance.resolved_config_sha256.push('x'),
                    "model" => changed.model_identity.d_model += 1,
                    "executable" => changed.provenance.executable_sha256.push('x'),
                    "source" => changed.provenance.build.git_sha = Some("different-commit".into()),
                    _ => unreachable!(),
                }
                let error =
                    bind_provenance_identity(&mut expected, provenance_identity(&changed).unwrap())
                        .unwrap_err();
                assert_eq!(
                    error.to_string(),
                    "input/config/model/executable provenance drift",
                    "{field}"
                );
                assert_eq!(expected, original, "{field}");
            }
        }

        #[test]
        fn p1r_u64_transport_and_duplicate_field_rejection() {
            let r = Record {
                slot: 1,
                words: [u64::MAX; WORDS],
            };
            let bytes = serde_json::to_vec(&r).unwrap();
            let parsed: Record = read_child_json(&bytes).unwrap();
            assert_eq!(parsed.words, r.words);
            assert!(read_child_json::<Record>(br#"{"slot":1,"slot":2,"words":[]}"#).is_err());
        }
        #[test]
        fn p1r_process_child_fixture() {
            let Some(path) = std::env::var_os("MER_P1R_FIXTURE_OUTPUT") else {
                return;
            };
            let mut file = std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(path)
                .unwrap();
            writeln!(file, "{}", std::process::id()).unwrap();
        }
        #[test]
        fn p1r_exact_one_pair_parent_child_fixture() {
            let dir = std::env::temp_dir().join(format!(
                "mer-p1r-pair-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir(&dir).unwrap();
            let mut calls = Vec::new();
            let (c, t, errors) = one_pair(
                |arm, _| {
                    calls.push(arm);
                    let path = dir.join(if arm == Arm::Control { "c" } else { "t" });
                    let output=std::process::Command::new(std::env::current_exe()?).args(["--exact","gpu_native_predictor_v2_critical_path_attribution::driver::p1r_tests::p1r_process_child_fixture","--nocapture"]).env("MER_P1R_FIXTURE_OUTPUT",&path).output()?;
                    if !output.status.success() {
                        return Err("fixture process failed".into());
                    }
                    Ok(std::fs::read_to_string(path)?.trim().parse::<u32>()?)
                },
                |pid| *pid != 0,
            );
            assert!(errors.is_empty());
            assert_ne!(c, t);
            assert_ne!(c, Some(std::process::id()));
            assert_eq!(calls, [Arm::Control, Arm::Treatment]);
            std::fs::remove_dir_all(dir).unwrap();
        }
        #[test]
        fn p1r_failed_or_uncertain_control_prevents_treatment() {
            for failed in [true, false] {
                let mut calls = 0;
                let (c, t, e) = one_pair(
                    |_, _| {
                        calls += 1;
                        if failed {
                            Err("child failed".into())
                        } else {
                            Ok(false)
                        }
                    },
                    |clean| *clean,
                );
                assert!(t.is_none());
                assert_eq!(calls, 1);
                assert!(!e.is_empty());
                assert_eq!(c.is_none(), failed);
            }
        }
        #[test]
        fn p1r_driver_cannot_enter_p1q3_schedule() {
            let source = include_str!("gpu_native_predictor_v2_critical_path_attribution.rs")
                .split("mod p1r_tests")
                .next()
                .unwrap();
            for forbidden in [
                "::run_pairs(",
                "sidecar_performance::run_command",
                "noise_statistics(",
                "median_six(",
                "PAIRS: usize = 6",
            ] {
                assert!(!source.contains(forbidden));
            }
            assert!(source.contains("performance_verdict: \"NOT_AUTHORIZED\""));
        }

        #[test]
        fn p1r_censored_record_retained_without_fabricated_d_or_service() {
            let (mut c, mut t) = fixture();
            for arm in [&mut c, &mut t] {
                let raw = arm.raw_p1e_report.as_mut().unwrap();
                let row = &mut raw.observations[1];
                row.freeze.candidate.source_position =
                    PositionIdentity::from_prompt_length(142, 4).unwrap();
                row.freeze.candidate.target_position =
                    PositionIdentity::from_prompt_length(143, 4).unwrap();
                row.deadline = None;
                row.outcome = p1e::Outcome::Censored;
                let snap = arm.diagnostics.as_mut().unwrap();
                snap.records.retain(|r| {
                    r.slot / STRIDE != 5
                        || ![
                            Event::DEntry,
                            Event::DP1e,
                            Event::Publication,
                            Event::Binding,
                            Event::Credit,
                        ]
                        .iter()
                        .any(|e| slot_index(5, 0, 0, 0, *e) == Some(r.slot))
                });
                for r in &mut snap.records {
                    if r.slot / STRIDE == 5
                        && r.slot % STRIDE < POSITION_SLOTS
                        && r.slot % STRIDE != Event::CleanRoute as usize
                    {
                        r.slot += (143 - 5) * STRIDE;
                    }
                }
                let c = row.freeze.candidate;
                if arm.arm == Arm::Treatment {
                    let id = P1jIdentity {
                        candidate: c,
                        logical_generation: 9,
                        epoch: 2,
                        writer_sequence: 2,
                    };
                    put(
                        snap,
                        143,
                        Event::Credit,
                        0,
                        0,
                        0,
                        170,
                        170,
                        &identity_words(
                            id,
                            1 + crate::predictor_v2::P1jTerminal::RequestEnded as u64,
                        ),
                    );
                }
            }
            t.predictor_v2_snapshot
                .as_mut()
                .unwrap()
                .direct_matching_demand_credits = 1;
            let rows = attribute(&c, &t).unwrap();
            assert_eq!(rows[1].class, OutcomeClass::UNUSED);
            assert!(rows[1].statuses.contains(&OutcomeClass::CENSORED));
            assert!(rows[1].control_service.is_empty());
            assert_eq!(rows[1].readiness_margin_bounds_ns, None);
        }
        #[test]
        fn p1r_late_cancelled_close_busy_and_consumed_d_window_are_distinct() {
            for (code, pubcode, consumed, expected) in [
                (2, 2, false, OutcomeClass::LATE),
                (1, 1, false, OutcomeClass::PUBLICATION_BUSY),
                (1, 0, true, OutcomeClass::READY_DURING_D_WINDOW),
            ] {
                let (c, mut t) = fixture();
                let snap = t.diagnostics.as_mut().unwrap();
                let close = item_mut(snap, 4, Event::Close, 0, 0, 0);
                close.words[0] = 119;
                close.words[1] = 120;
                close.words[2] = code;
                item_mut(snap, 4, Event::Publication, 0, 0, 0).words[2] = pubcode;
                if !consumed {
                    item_mut(snap, 4, Event::Credit, 0, 0, 0).words[2] = 5;
                    t.predictor_v2_snapshot
                        .as_mut()
                        .unwrap()
                        .direct_matching_demand_credits = 1;
                }
                let rows = attribute(&c, &t).unwrap();
                assert_eq!(rows[0].class, expected);
            }
        }
        #[test]
        fn p1r_orphan_writer_and_invalid_tail_binding_cannot_gain_credit() {
            let (c, mut t) = fixture();
            let mut rogue = item_mut(
                t.diagnostics.as_mut().unwrap(),
                4,
                Event::WriterStart,
                0,
                0,
                0,
            )
            .clone();
            rogue.slot = slot_index(22, 0, 0, 0, Event::WriterStart).unwrap();
            t.diagnostics.as_mut().unwrap().records.push(rogue);
            assert!(attribute(&c, &t).is_err());
            let (c, mut t) = fixture();
            let r = item_mut(t.diagnostics.as_mut().unwrap(), 4, Event::Binding, 0, 0, 0);
            r.slot = slot_index(4, 1, 0, 0, Event::Binding).unwrap();
            assert!(attribute(&c, &t).is_err());
        }
        #[test]
        fn p1r_recorder_footprint_is_frozen_and_equal_in_both_arms() {
            assert_eq!(SLOTS, 1640448);
            assert_eq!(SLOTS * std::mem::size_of::<Slot>(), 170606592);
            let (c, t) = fixture();
            assert_eq!(
                c.diagnostics.unwrap().storage_bytes,
                t.diagnostics.unwrap().storage_bytes
            );
        }
        #[test]
        fn p1r_cli_requires_distinct_command_and_rejects_p1q3_options() {
            use clap::Parser;
            #[derive(Parser)]
            struct Cli {
                #[command(flatten)]
                args: CommandArgs,
            }
            let args = [
                "p1r",
                "--config",
                "c",
                "--request-json",
                "r",
                "--expected-adapter-name",
                "NVIDIA L4",
                "--report-out",
                "o",
                "--diagnostic-id",
                "fixture",
            ];
            assert!(Cli::try_parse_from(args).is_ok());
            assert!(Cli::try_parse_from(
                args.into_iter().chain(["--experiment-mode", "ab-movement"])
            )
            .is_err());
            assert!(Cli::try_parse_from(
                args.into_iter()
                    .chain(["--p1q3-child-protocol", "anything"])
            )
            .is_err());
        }
    }
} // driver
