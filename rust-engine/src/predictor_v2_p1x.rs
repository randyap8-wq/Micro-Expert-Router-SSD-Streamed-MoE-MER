//! P1X: bounded request-owned scalar observation. This module owns no device,
//! cache, source, lease, task, destination or movement capability.
use super::{PositionIdentity, PositionKind, RequestIdentity};
use serde::{Deserialize, Serialize};

pub(crate) const FIRST_LAYER: usize = 3;
pub(crate) const LAYERS: usize = 45;
pub(crate) const EXPERTS: usize = 128;
pub(crate) const TOP_K: usize = 8;
pub(crate) const MAX_POSITIONS: usize = 4096;
pub(crate) const REVISION: u64 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum Error {
    Identity,
    Overflow,
    Capacity,
    Chronology,
    PhysicalIdentity,
    Closed,
    Incomplete,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "P1X {self:?}")
    }
}
impl std::error::Error for Error {}
pub(crate) type Result<T> = std::result::Result<T, Error>;
fn require(ok: bool, error: Error) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(error)
    }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(Error::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(Error::Overflow)
}
fn partition(a: PositionKind, b: PositionKind) -> Result<usize> {
    match (a, b) {
        (PositionKind::Prompt, PositionKind::Prompt) => Ok(0),
        (PositionKind::Prompt, PositionKind::Decode) => Ok(1),
        (PositionKind::Decode, PositionKind::Decode) => Ok(2),
        _ => Err(Error::Identity),
    }
}
fn index(layer: usize, kind: usize, source: usize, target: usize) -> usize {
    ((layer * 3 + kind) * EXPERTS + source) * EXPERTS + target
}
fn selected(ids: &[u32]) -> Result<[u32; TOP_K]> {
    require(ids.len() == TOP_K, Error::Identity)?;
    let mut out = [0; TOP_K];
    out.copy_from_slice(ids);
    out.sort_unstable();
    require(
        out.iter().all(|&e| e < EXPERTS as u32) && out.windows(2).all(|w| w[0] != w[1]),
        Error::Identity,
    )?;
    Ok(out)
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct PhysicalIdentity {
    pub context: u64,
    pub arena: usize,
    pub layer: usize,
    pub expert: u32,
    pub generation: u64,
    pub bank: u32,
    pub slot: u32,
    pub epoch: u32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum Physical {
    Unknown,
    Absent,
    Current(PhysicalIdentity),
}
impl Physical {
    pub fn current(self) -> Option<bool> {
        match self {
            Self::Unknown => None,
            Self::Absent => Some(false),
            Self::Current(_) => Some(true),
        }
    }
    fn validate(self, candidate: Candidate) -> Result<()> {
        if let Self::Current(p) = self {
            require(
                p.context != 0
                    && p.arena != 0
                    && p.layer == candidate.layer
                    && p.expert == candidate.local_expert
                    && p.generation != 0
                    && p.epoch != 0,
                Error::PhysicalIdentity,
            )?;
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum SourceClass {
    LogicalMaterializedAndRam,
    LogicalMaterialized,
    RamOnly,
    Absent,
    Unknown,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct HostSource {
    pub logical_generation: Option<u64>,
    pub logical_materialized: Option<bool>,
    pub ram_resident: Option<bool>,
}
impl HostSource {
    pub fn classification(self) -> SourceClass {
        match (self.logical_materialized, self.ram_resident) {
            (Some(true), Some(true)) => SourceClass::LogicalMaterializedAndRam,
            (Some(true), _) => SourceClass::LogicalMaterialized,
            (Some(false), Some(true)) => SourceClass::RamOnly,
            (Some(false), Some(false)) => SourceClass::Absent,
            _ => SourceClass::Unknown,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct Candidate {
    pub request: RequestIdentity,
    pub layer: usize,
    pub sequence: u64,
    pub source_position: PositionIdentity,
    pub target_position: PositionIdentity,
    pub source_set: [u32; TOP_K],
    pub local_expert: u32,
    pub global_expert: u32,
    pub score: u64,
    pub signal_revision: u64,
    pub committed_position_cutoff: u64,
    pub table_update_cutoff: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct Freeze {
    pub timestamp_ns: u64,
    pub physical: Physical,
    pub source: HostSource,
    pub source_classification: SourceClass,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct Deadline {
    pub timestamp_ns: u64,
    pub physical: Physical,
    pub host_lead_ns: u64,
    pub attempt: usize,
    pub initial_attempt: bool,
    pub recovery: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum Outcome {
    NoEmissionNoPositiveHistory,
    Pending,
    Resolved { prediction_hit: bool },
    Censored,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Row {
    pub layer: usize,
    pub source_position: PositionIdentity,
    pub target_position: PositionIdentity,
    pub candidate: Option<Candidate>,
    pub freeze: Option<Freeze>,
    pub deadline: Option<Deadline>,
    pub outcome: Outcome,
    pub target_attempts: usize,
    pub recovery_attempts: usize,
    pub candidate_seen_in_ordinary_missing_set: Option<bool>,
    pub candidate_physical_install_committed: Option<bool>,
    pub ordinary_set_service_ns: Option<u64>,
    pub ordinary_service_scope: &'static str,
    pub exact_candidate_saved_ns: Option<u64>,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Opportunities {
    pub prediction_hit: Option<bool>,
    pub physical_miss_at_f: Option<bool>,
    pub physical_miss_at_d: Option<bool>,
    pub useful: Option<bool>,
    pub target_confirmed_useful: Option<bool>,
    pub already_resident: Option<bool>,
    pub redundant_route_hit: Option<bool>,
    pub host_backed_at_f: Option<bool>,
    pub host_backed_target_confirmed_useful: Option<bool>,
    pub ram_resident_at_f: Option<bool>,
}
// Unknown remains unknown, including false AND unknown: derive only with known inputs.
fn and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a && b),
        _ => None,
    }
}
impl Row {
    pub fn opportunities(&self) -> Opportunities {
        let hit = match self.outcome {
            Outcome::Resolved { prediction_hit } => Some(prediction_hit),
            _ => None,
        };
        let current_f = self.freeze.and_then(|f| f.physical.current());
        let miss_f = current_f.map(|v| !v);
        let miss_d = self.deadline.and_then(|d| d.physical.current()).map(|v| !v);
        let useful = and(hit, miss_f);
        let target = and(useful, miss_d);
        let host = self.freeze.and_then(|f| f.source.logical_materialized);
        Opportunities {
            prediction_hit: hit,
            physical_miss_at_f: miss_f,
            physical_miss_at_d: miss_d,
            useful,
            target_confirmed_useful: target,
            already_resident: current_f,
            redundant_route_hit: and(hit, current_f),
            host_backed_at_f: host,
            host_backed_target_confirmed_useful: and(target, host),
            ram_resident_at_f: self.freeze.and_then(|f| f.source.ram_resident),
        }
    }
}

pub(crate) struct Observer {
    request: RequestIdentity,
    prompt_length: usize,
    max_positions: usize,
    max_rows: usize,
    table: Vec<u64>,
    previous: Option<[[u32; TOP_K]; LAYERS]>,
    completed: usize,
    sequence: u64,
    pending: [Option<usize>; LAYERS],
    rows: Vec<Row>,
    attempt_position: Option<usize>,
    attempt: usize,
    eligible: bool,
    closed: bool,
    pub error: Option<Error>,
}
impl Observer {
    pub fn new(request: RequestIdentity, prompt_length: usize, positions: usize) -> Result<Self> {
        require(
            request.runtime_namespace != 0
                && request.request_sequence != 0
                && prompt_length > 0
                && positions >= prompt_length
                && positions <= MAX_POSITIONS,
            Error::Capacity,
        )?;
        let cells = mul(mul(mul(LAYERS, 3)?, EXPERTS)?, EXPERTS)?;
        let _bytes = mul(cells, std::mem::size_of::<u64>())?;
        let max_rows = mul(LAYERS, positions)?;
        let _row_bytes = mul(max_rows, std::mem::size_of::<Row>())?;
        Ok(Self {
            request,
            prompt_length,
            max_positions: positions,
            max_rows,
            table: vec![0; cells],
            previous: None,
            completed: 0,
            sequence: 0,
            pending: [None; LAYERS],
            rows: Vec::with_capacity(max_rows),
            attempt_position: None,
            attempt: 0,
            eligible: false,
            closed: false,
            error: None,
        })
    }
    fn ready(&self) -> Result<()> {
        require(!self.closed, Error::Closed)?;
        self.error.map_or(Ok(()), Err)
    }
    pub fn invalidate(&mut self, error: Error) {
        self.error.get_or_insert(error);
    }
    fn position(&self, p: usize) -> Result<PositionIdentity> {
        PositionIdentity::from_prompt_length(p, self.prompt_length).map_err(|_| Error::Overflow)
    }
    /// Called only at the existing clean whole-position commit, never on an
    /// attempt report. No target-position truth is accepted by freeze().
    pub fn commit(
        &mut self,
        position: usize,
        routes: &[Vec<u32>],
        clean_committed: bool,
    ) -> Result<()> {
        self.ready()?;
        if !clean_committed {
            return Ok(());
        }
        require(
            position == self.completed && position < self.max_positions,
            Error::Chronology,
        )?;
        require(routes.len() == 48, Error::Identity)?;
        let mut sets = [[0; TOP_K]; LAYERS];
        for (i, set) in sets.iter_mut().enumerate() {
            *set = selected(&routes[i + FIRST_LAYER])?;
        }
        require(
            add(self.rows.len(), LAYERS)? <= self.max_rows,
            Error::Capacity,
        )?;
        let source = self.position(position)?;
        let target = self.position(add(position, 1)?)?;
        let score_kind = partition(source.position_kind, target.position_kind)?;
        let train_kind = if position > 0 {
            Some(partition(
                self.position(position - 1)?.position_kind,
                source.position_kind,
            )?)
        } else {
            None
        };
        // Validate every cell before updating any table.
        if let (Some(previous), Some(kind)) = (&self.previous, train_kind) {
            for i in 0..LAYERS {
                for &s in &previous[i] {
                    for &e in &sets[i] {
                        self.table[index(i, kind, s as usize, e as usize)]
                            .checked_add(1)
                            .ok_or(Error::Overflow)?;
                    }
                }
            }
        }
        if let (Some(previous), Some(kind)) = (&self.previous, train_kind) {
            for i in 0..LAYERS {
                for &s in &previous[i] {
                    for &e in &sets[i] {
                        self.table[index(i, kind, s as usize, e as usize)] += 1;
                    }
                }
            }
        }
        let mut nominations = [None; LAYERS];
        for i in 0..LAYERS {
            let mut best = None;
            for expert in 0..EXPERTS {
                let score = sets[i].iter().try_fold(0u64, |n, &s| {
                    n.checked_add(self.table[index(i, score_kind, s as usize, expert)])
                        .ok_or(Error::Overflow)
                })?;
                if score > best.map_or(0, |(_, score)| score) {
                    best = Some((expert, score));
                }
            }
            if let Some((expert, score)) = best {
                self.sequence = self.sequence.checked_add(1).ok_or(Error::Overflow)?;
                let global = add(mul(i + FIRST_LAYER, EXPERTS)?, expert)?;
                nominations[i] = Some(Candidate {
                    request: self.request,
                    layer: i + FIRST_LAYER,
                    sequence: self.sequence,
                    source_position: source,
                    target_position: target,
                    source_set: sets[i],
                    local_expert: expert as u32,
                    global_expert: u32::try_from(global).map_err(|_| Error::Overflow)?,
                    score,
                    signal_revision: REVISION,
                    committed_position_cutoff: u64::try_from(add(position, 1)?)
                        .map_err(|_| Error::Overflow)?,
                    table_update_cutoff: position as u64,
                });
            }
        }
        for i in 0..LAYERS {
            if let Some(old) = self.pending[i].take() {
                let row = &mut self.rows[old];
                let c = row.candidate.ok_or(Error::Identity)?;
                require(
                    c.target_position == source && row.freeze.is_some(),
                    Error::Chronology,
                )?;
                row.outcome = Outcome::Resolved {
                    prediction_hit: sets[i].contains(&c.local_expert),
                };
            }
            let candidate = nominations[i];
            if candidate.is_some() {
                self.pending[i] = Some(self.rows.len());
            }
            self.rows.push(Row {
                layer: i + FIRST_LAYER,
                source_position: source,
                target_position: target,
                candidate,
                freeze: None,
                deadline: None,
                outcome: if candidate.is_some() {
                    Outcome::Pending
                } else {
                    Outcome::NoEmissionNoPositiveHistory
                },
                target_attempts: 0,
                recovery_attempts: 0,
                candidate_seen_in_ordinary_missing_set: None,
                candidate_physical_install_committed: None,
                ordinary_set_service_ns: None,
                ordinary_service_scope: "SET_SCOPED / NOT_CANDIDATE_MARGINAL",
                exact_candidate_saved_ns: None,
            });
        }
        self.previous = Some(sets);
        self.completed = add(self.completed, 1)?;
        self.eligible = false;
        Ok(())
    }
    pub fn pending_candidate(&self, layer: usize, target: usize) -> Option<Candidate> {
        let i = layer.checked_sub(FIRST_LAYER).filter(|&i| i < LAYERS)?;
        let row = self.rows.get(self.pending[i]?)?;
        row.candidate.filter(|c| {
            c.target_position.absolute_position == target as u64 && row.outcome == Outcome::Pending
        })
    }
    pub fn freeze(
        &mut self,
        candidate: Candidate,
        timestamp_ns: u64,
        physical: Physical,
        source: HostSource,
    ) -> Result<()> {
        self.ready()?;
        require(
            candidate.request == self.request
                && candidate.layer >= FIRST_LAYER
                && candidate.layer < FIRST_LAYER + LAYERS,
            Error::Identity,
        )?;
        require(
            candidate.target_position.absolute_position == self.completed as u64
                && self.attempt_position != Some(self.completed),
            Error::Chronology,
        )?;
        physical.validate(candidate)?;
        let row =
            &mut self.rows[self.pending[candidate.layer - FIRST_LAYER].ok_or(Error::Identity)?];
        require(
            row.candidate == Some(candidate) && row.freeze.is_none(),
            Error::Chronology,
        )?;
        row.freeze = Some(Freeze {
            timestamp_ns,
            physical,
            source,
            source_classification: source.classification(),
        });
        Ok(())
    }
    pub fn begin_attempt(&mut self, position: usize, initial_fresh: bool) -> Result<()> {
        self.ready()?;
        require(position == self.completed, Error::Chronology)?;
        if self.attempt_position != Some(position) {
            self.attempt_position = Some(position);
            self.attempt = 0;
        }
        self.attempt = add(self.attempt, 1)?;
        self.eligible = self.attempt == 1 && initial_fresh;
        for i in self.pending.into_iter().flatten() {
            let row = &mut self.rows[i];
            row.target_attempts = add(row.target_attempts, 1)?;
            if !self.eligible {
                row.recovery_attempts = add(row.recovery_attempts, 1)?;
            }
        }
        Ok(())
    }
    pub fn deadline_candidate(&self, layer: usize, position: usize) -> Option<Candidate> {
        if !self.eligible || self.error.is_some() || self.closed {
            return None;
        }
        self.pending_candidate(layer, position)
    }
    pub fn deadline(
        &mut self,
        candidate: Candidate,
        timestamp_ns: u64,
        physical: Physical,
    ) -> Result<()> {
        self.ready()?;
        require(
            self.deadline_candidate(candidate.layer, self.completed) == Some(candidate),
            Error::Chronology,
        )?;
        physical.validate(candidate)?;
        let row =
            &mut self.rows[self.pending[candidate.layer - FIRST_LAYER].ok_or(Error::Identity)?];
        if row.deadline.is_some() {
            return Ok(());
        }
        let f = row.freeze.ok_or(Error::Chronology)?;
        let lead = timestamp_ns
            .checked_sub(f.timestamp_ns)
            .ok_or(Error::Chronology)?;
        row.deadline = Some(Deadline {
            timestamp_ns,
            physical,
            host_lead_ns: lead,
            attempt: self.attempt,
            initial_attempt: true,
            recovery: false,
        });
        Ok(())
    }
    pub fn demand_evidence(&mut self, position: usize, layer: usize, evidence: DemandEvidence) {
        if self.pending_candidate(layer, position).is_none() {
            return;
        }
        let row = &mut self.rows[self.pending[layer - FIRST_LAYER].unwrap()];
        row.candidate_seen_in_ordinary_missing_set = evidence.seen_missing;
        row.candidate_physical_install_committed = evidence.install_committed;
        row.ordinary_set_service_ns = evidence.service_ns;
    }
    pub fn finish(&mut self) {
        self.closed = true;
        self.previous = None;
        self.table.clear();
        self.table.shrink_to_fit();
        for i in self.pending.iter_mut().filter_map(Option::take) {
            self.rows[i].outcome = Outcome::Censored;
        }
    }
    pub fn report(&self) -> Result<Report> {
        require(self.closed, Error::Closed)?;
        let mut layers = Vec::with_capacity(LAYERS);
        let mut raw = Vec::with_capacity(self.rows.len());
        let mut sequence = 0;
        for (n, row) in self.rows.iter().enumerate() {
            require(
                row.layer == FIRST_LAYER + n % LAYERS
                    && row.source_position.absolute_position == (n / LAYERS) as u64,
                Error::Identity,
            )?;
            if let Some(c) = row.candidate {
                require(
                    c.request == self.request
                        && c.sequence == sequence + 1
                        && c.layer == row.layer
                        && c.source_position == row.source_position
                        && c.target_position == row.target_position,
                    Error::Identity,
                )?;
                sequence = c.sequence;
            }
            raw.push(ReportedRow {
                row: row.clone(),
                opportunities: row.opportunities(),
            });
        }
        require(
            self.rows.len() == mul(self.completed, LAYERS)?,
            Error::Incomplete,
        )?;
        for layer in FIRST_LAYER..FIRST_LAYER + LAYERS {
            layers.push(aggregate(
                layer,
                self.completed,
                self.rows.iter().filter(|r| r.layer == layer),
            )?);
        }
        Ok(Report {
            schema: "mer.predictor-v2-p1x.v1",
            request: self.request,
            completed_positions: self.completed,
            error: self.error,
            raw,
            layers,
            score_and_lead_percentiles: "nearest rank; null when empty",
            signal_cells: LAYERS * 3 * EXPERTS * EXPERTS,
            signal_bytes: LAYERS * 3 * EXPERTS * EXPERTS * 8,
            max_rows: self.max_rows,
        })
    }
}

/// Ordinary demand receives only one scalar candidate and a mutable scalar
/// receipt. This is never a source request, callback, lease or scheduling input.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DemandEvidence {
    pub global: u32,
    pub seen_missing: Option<bool>,
    pub install_committed: Option<bool>,
    pub service_ns: Option<u64>,
}
impl DemandEvidence {
    pub fn missing(&mut self, ids: &[u32]) {
        self.seen_missing = Some(self.seen_missing == Some(true) || ids.contains(&self.global));
    }
}
#[derive(Debug, Serialize)]
pub(crate) struct ReportedRow {
    #[serde(flatten)]
    pub row: Row,
    pub opportunities: Opportunities,
}
#[derive(Debug, Serialize)]
pub(crate) struct Report {
    pub schema: &'static str,
    pub request: RequestIdentity,
    pub completed_positions: usize,
    pub error: Option<Error>,
    pub raw: Vec<ReportedRow>,
    pub layers: Vec<Aggregate>,
    pub score_and_lead_percentiles: &'static str,
    pub signal_cells: usize,
    pub signal_bytes: usize,
    pub max_rows: usize,
}
#[derive(Debug, Default, Serialize)]
pub(crate) struct Distribution {
    pub min: Option<u64>,
    pub p50: Option<u64>,
    pub p95: Option<u64>,
    pub max: Option<u64>,
}
fn distribution(mut v: Vec<u64>) -> Result<Distribution> {
    v.sort_unstable();
    let percentile = |p| -> Result<Option<u64>> {
        if v.is_empty() {
            Ok(None)
        } else {
            Ok(Some(v[(add(mul(v.len(), p)?, 99)? / 100) - 1]))
        }
    };
    Ok(Distribution {
        min: v.first().copied(),
        p50: percentile(50)?,
        p95: percentile(95)?,
        max: v.last().copied(),
    })
}
#[derive(Debug, Default, Serialize)]
pub(crate) struct Aggregate {
    pub layer: usize,
    pub ordinary_slot_capacity: usize,
    pub completed_positions: usize,
    pub emitted: usize,
    pub no_positive_history: usize,
    pub resolved: usize,
    pub censored: usize,
    pub prediction_hits: usize,
    pub route_misses: usize,
    pub valid_f: usize,
    pub current_at_f: usize,
    pub absent_at_f: usize,
    pub unknown_at_f: usize,
    pub valid_d: usize,
    pub current_at_d: usize,
    pub absent_at_d: usize,
    pub unknown_at_d: usize,
    pub useful: usize,
    pub target_confirmed_useful: usize,
    pub host_backed_at_f: usize,
    pub host_backed_target_confirmed_useful: usize,
    pub ram_resident_at_f: usize,
    pub ordinary_missing_confirmed: usize,
    pub physical_install_committed: usize,
    pub score: Distribution,
    pub host_lead_ns: Distribution,
    pub prediction_hit_rate: Option<f64>,
    pub useful_density: Option<f64>,
    pub target_confirmed_useful_density: Option<f64>,
    pub host_backed_useful_density: Option<f64>,
}
fn inc(v: &mut usize) -> Result<()> {
    *v = add(*v, 1)?;
    Ok(())
}
fn ratio(n: usize, d: usize) -> Option<f64> {
    (d > 0).then(|| n as f64 / d as f64)
}
fn aggregate<'a>(
    layer: usize,
    completed: usize,
    rows: impl Iterator<Item = &'a Row>,
) -> Result<Aggregate> {
    let mut a = Aggregate {
        layer,
        ordinary_slot_capacity: if layer == 3 { 73 } else { 8 },
        completed_positions: completed,
        ..Default::default()
    };
    let mut scores = Vec::new();
    let mut leads = Vec::new();
    for row in rows {
        let Some(c) = row.candidate else {
            require(
                row.outcome == Outcome::NoEmissionNoPositiveHistory,
                Error::Identity,
            )?;
            inc(&mut a.no_positive_history)?;
            continue;
        };
        inc(&mut a.emitted)?;
        scores.push(c.score);
        match row.outcome {
            Outcome::Resolved { prediction_hit } => {
                inc(&mut a.resolved)?;
                inc(if prediction_hit {
                    &mut a.prediction_hits
                } else {
                    &mut a.route_misses
                })?;
            }
            Outcome::Censored => inc(&mut a.censored)?,
            _ => return Err(Error::Incomplete),
        }
        match row.freeze.and_then(|f| f.physical.current()) {
            Some(v) => {
                inc(&mut a.valid_f)?;
                inc(if v {
                    &mut a.current_at_f
                } else {
                    &mut a.absent_at_f
                })?;
            }
            None => inc(&mut a.unknown_at_f)?,
        }
        match row.deadline.and_then(|d| d.physical.current()) {
            Some(v) => {
                inc(&mut a.valid_d)?;
                inc(if v {
                    &mut a.current_at_d
                } else {
                    &mut a.absent_at_d
                })?;
            }
            None => inc(&mut a.unknown_at_d)?,
        }
        if let Some(d) = row.deadline {
            leads.push(d.host_lead_ns);
        }
        let o = row.opportunities();
        for (flag, count) in [
            (o.useful, &mut a.useful),
            (o.target_confirmed_useful, &mut a.target_confirmed_useful),
            (o.host_backed_at_f, &mut a.host_backed_at_f),
            (
                o.host_backed_target_confirmed_useful,
                &mut a.host_backed_target_confirmed_useful,
            ),
            (o.ram_resident_at_f, &mut a.ram_resident_at_f),
            (
                row.candidate_seen_in_ordinary_missing_set,
                &mut a.ordinary_missing_confirmed,
            ),
            (
                row.candidate_physical_install_committed,
                &mut a.physical_install_committed,
            ),
        ] {
            if flag == Some(true) {
                inc(count)?;
            }
        }
    }
    require(
        add(a.emitted, a.no_positive_history)? == completed
            && add(a.resolved, a.censored)? == a.emitted
            && add(a.prediction_hits, a.route_misses)? == a.resolved
            && add(a.valid_f, a.unknown_at_f)? == a.emitted
            && add(a.current_at_f, a.absent_at_f)? == a.valid_f
            && add(a.valid_d, a.unknown_at_d)? == a.emitted
            && add(a.current_at_d, a.absent_at_d)? == a.valid_d
            && a.target_confirmed_useful <= a.useful
            && a.useful <= a.prediction_hits
            && a.useful <= a.absent_at_f
            && a.host_backed_target_confirmed_useful <= a.target_confirmed_useful,
        Error::Incomplete,
    )?;
    a.score = distribution(scores)?;
    a.host_lead_ns = distribution(leads)?;
    a.prediction_hit_rate = ratio(a.prediction_hits, a.resolved);
    a.useful_density = ratio(a.useful, a.emitted);
    a.target_confirmed_useful_density = ratio(a.target_confirmed_useful, a.emitted);
    a.host_backed_useful_density = ratio(a.host_backed_target_confirmed_useful, a.emitted);
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor_v2::RequestPhase;
    fn observer(prompt: usize) -> Observer {
        Observer::new(
            RequestIdentity {
                runtime_namespace: 1,
                request_sequence: 1,
                phase: RequestPhase::Fixture,
                phase_run_index: 0,
            },
            prompt,
            16,
        )
        .unwrap()
    }
    fn routes(start: u32) -> Vec<Vec<u32>> {
        vec![(start..start + 8).collect(); 48]
    }
    fn source() -> HostSource {
        HostSource {
            logical_generation: Some(9),
            logical_materialized: Some(true),
            ram_resident: Some(true),
        }
    }
    fn freeze_all(o: &mut Observer, p: usize, physical: Physical) {
        for layer in 3..=47 {
            if let Some(c) = o.pending_candidate(layer, p + 1) {
                o.freeze(c, p as u64 * 100 + 1, physical, source()).unwrap();
            }
        }
    }
    fn step(o: &mut Observer, p: usize, truth: Vec<Vec<u32>>) {
        o.begin_attempt(p, true).unwrap();
        for layer in 3..=47 {
            if let Some(c) = o.deadline_candidate(layer, p) {
                o.deadline(c, p as u64 * 100, Physical::Absent).unwrap();
            }
        }
        o.commit(p, &truth, true).unwrap();
        freeze_all(o, p, Physical::Absent);
    }
    #[test]
    fn p1x_all_45_layers_reconcile_cold_hit_miss_censor_and_unique_sequences() {
        let mut o = observer(16);
        step(&mut o, 0, routes(0));
        step(&mut o, 1, routes(8));
        step(&mut o, 2, routes(0));
        let c = o.pending_candidate(3, 3).unwrap();
        assert_eq!((c.local_expert, c.score, c.sequence), (8, 8, 1));
        let mut truth = routes(8);
        truth[47] = (9..17).collect();
        step(&mut o, 3, truth);
        o.finish();
        let r = o.report().unwrap();
        assert_eq!(r.layers.len(), 45);
        assert_eq!(r.raw.len(), 180);
        for a in &r.layers {
            assert_eq!(
                (a.emitted, a.no_positive_history, a.resolved, a.censored),
                (2, 2, 1, 1)
            );
        }
        assert_eq!(r.layers[0].ordinary_slot_capacity, 73);
        assert_eq!(r.layers[44].ordinary_slot_capacity, 8);
        assert_eq!(r.layers[0].host_backed_target_confirmed_useful, 1);
        assert_eq!(r.layers[44].route_misses, 1);
        assert_eq!(r.layers[0].host_lead_ns.min, Some(99));
        assert!(o.table.is_empty());
        assert!(o.previous.is_none());
    }
    #[test]
    fn p1x_layer_isolation_ties_order_invariance_no_runner_up() {
        let mut o = observer(16);
        let mut a = routes(0);
        let mut b = routes(8);
        a[47] = (32..40).collect();
        b[47] = (40..48).collect();
        step(&mut o, 0, a.clone());
        step(&mut o, 1, b);
        step(&mut o, 2, a);
        assert_eq!(o.pending_candidate(3, 3).unwrap().local_expert, 8);
        assert_eq!(o.pending_candidate(47, 3).unwrap().local_expert, 40);
        let c = o.pending_candidate(3, 3).unwrap();
        let before = o.rows[90].candidate;
        o.rows[90].freeze = Some(Freeze {
            timestamp_ns: 201,
            physical: Physical::Current(PhysicalIdentity {
                context: 1,
                arena: 1,
                layer: 3,
                expert: 8,
                generation: 1,
                bank: 0,
                slot: 72,
                epoch: 1,
            }),
            source: HostSource::default(),
            source_classification: SourceClass::Unknown,
        });
        assert_eq!(o.rows[90].candidate, before);
        assert_eq!(c.score, 8);
        assert_eq!(
            selected(&[7, 6, 5, 4, 3, 2, 1, 0]).unwrap(),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
    }
    #[test]
    fn p1x_three_position_partitions_never_blend() {
        let mut o = observer(3);
        for p in 0..5 {
            step(&mut o, p, routes(0));
        }
        for p in [0, 2, 3] {
            assert!(o.rows[p * 45].candidate.is_none());
        }
        assert!(o.rows[45].candidate.is_some());
        assert!(o.rows[180].candidate.is_some());
        let sum = |kind| {
            (0..128)
                .flat_map(|s| (0..128).map(move |e| (s, e)))
                .map(|(s, e)| o.table[index(0, kind, s, e)])
                .sum::<u64>()
        };
        assert_eq!((sum(0), sum(1), sum(2)), (128, 64, 64));
        assert_eq!(
            partition(PositionKind::Decode, PositionKind::Prompt),
            Err(Error::Identity)
        );
    }
    #[test]
    fn p1x_checked_cell_score_sequence_and_capacity_overflow() {
        assert_eq!(mul(usize::MAX, 45), Err(Error::Overflow));
        let mut o = observer(16);
        step(&mut o, 0, routes(0));
        o.table[index(0, 0, 0, 8)] = u64::MAX;
        assert_eq!(o.commit(1, &routes(8), true), Err(Error::Overflow));
        let mut o = observer(16);
        step(&mut o, 0, routes(0));
        o.table[index(0, 0, 8, 64)] = u64::MAX;
        o.table[index(0, 0, 9, 64)] = 1;
        assert_eq!(o.commit(1, &routes(8), true), Err(Error::Overflow));
        let mut o = observer(16);
        step(&mut o, 0, routes(0));
        o.sequence = u64::MAX;
        assert_eq!(o.commit(1, &routes(0), true), Err(Error::Overflow));
        assert!(Observer::new(o.request, 1, MAX_POSITIONS + 1).is_err());
        assert_eq!(o.table.len(), 2_211_840);
        assert_eq!(o.table.len() * 8, 17_694_720);
        let mut small = Observer::new(o.request, 1, 1).unwrap();
        step(&mut small, 0, routes(0));
        assert!(small.commit(1, &routes(0), true).is_err());
    }
    #[test]
    fn p1x_no_future_truth_no_partial_recovery_training_and_first_deadline_only() {
        let mut o = observer(16);
        step(&mut o, 0, routes(0));
        step(&mut o, 1, routes(8));
        step(&mut o, 2, routes(0));
        let c = o.pending_candidate(3, 3).unwrap();
        let cells = o.table.clone();
        let frozen = o.rows[90].freeze;
        o.begin_attempt(3, true).unwrap();
        o.deadline(c, 300, Physical::Unknown).unwrap();
        o.commit(3, &routes(64), false).unwrap();
        assert_eq!(o.table, cells);
        assert_eq!(o.rows[90].freeze, frozen);
        o.begin_attempt(3, false).unwrap();
        assert!(o.deadline_candidate(3, 3).is_none());
        assert_eq!(o.deadline(c, 999, Physical::Absent), Err(Error::Chronology));
        o.commit(3, &routes(8), true).unwrap();
        freeze_all(&mut o, 3, Physical::Absent);
        assert_eq!(
            o.rows[90].outcome,
            Outcome::Resolved {
                prediction_hit: true
            }
        );
        assert_eq!(o.rows[90].deadline.unwrap().physical, Physical::Unknown);
        assert_eq!(o.rows[90].opportunities().target_confirmed_useful, None);
    }
    #[test]
    fn p1x_freeze_after_attempt_or_future_commit_is_rejected() {
        let mut o = observer(16);
        step(&mut o, 0, routes(0));
        o.commit(1, &routes(0), true).unwrap();
        let c = o.pending_candidate(3, 2).unwrap();
        o.begin_attempt(2, true).unwrap();
        assert_eq!(
            o.freeze(c, 99, Physical::Absent, source()),
            Err(Error::Chronology)
        );
        assert_eq!(o.commit(2, &routes(8), true), Err(Error::Chronology));
    }
    #[test]
    fn p1x_request_reset_and_zero_emission_layers_are_explicit() {
        let mut a = observer(16);
        step(&mut a, 0, routes(0));
        step(&mut a, 1, routes(0));
        a.finish();
        let mut b = observer(16);
        b.request.request_sequence = 2;
        step(&mut b, 0, routes(0));
        b.finish();
        let report = b.report().unwrap();
        assert_eq!(report.layers.len(), 45);
        for l in report.layers {
            assert_eq!((l.emitted, l.no_positive_history), (0, 1));
            assert_eq!(l.prediction_hit_rate, None);
        }
        assert_eq!(a.report().unwrap().layers[0].censored, 1);
        assert_ne!(a.request, b.request);
    }
    #[test]
    fn p1x_optional_and_requires_both_operands_known() {
        for (a, b, expected) in [
            (Some(false), None, None),
            (None, Some(false), None),
            (Some(true), None, None),
            (None, Some(true), None),
            (None, None, None),
            (Some(false), Some(false), Some(false)),
            (Some(false), Some(true), Some(false)),
            (Some(true), Some(false), Some(false)),
            (Some(true), Some(true), Some(true)),
        ] {
            assert_eq!(and(a, b), expected, "and({a:?}, {b:?})");
        }
    }
    #[test]
    fn p1x_report_opportunities_require_all_optional_evidence() {
        fn report(
            current_f: Option<bool>,
            current_d: Option<bool>,
            hit: Option<bool>,
            host: Option<bool>,
        ) -> Report {
            fn physical(c: Candidate, current: Option<bool>) -> Physical {
                match current {
                    None => Physical::Unknown,
                    Some(false) => Physical::Absent,
                    Some(true) => Physical::Current(PhysicalIdentity {
                        context: 1,
                        arena: 2,
                        layer: c.layer,
                        expert: c.local_expert,
                        generation: 1,
                        bank: 0,
                        slot: 0,
                        epoch: 1,
                    }),
                }
            }
            let mut o = observer(16);
            step(&mut o, 0, routes(0));
            o.commit(1, &routes(0), true).unwrap();
            for layer in 3..=47 {
                let c = o.pending_candidate(layer, 2).unwrap();
                o.freeze(
                    c,
                    101,
                    physical(c, current_f),
                    HostSource {
                        logical_materialized: host,
                        ..source()
                    },
                )
                .unwrap();
            }
            o.begin_attempt(2, true).unwrap();
            for layer in 3..=47 {
                let c = o.deadline_candidate(layer, 2).unwrap();
                o.deadline(c, 200, physical(c, current_d)).unwrap();
            }
            if let Some(hit) = hit {
                o.commit(2, &routes(if hit { 0 } else { 8 }), true).unwrap();
                freeze_all(&mut o, 2, Physical::Absent);
            }
            o.finish();
            o.report().unwrap()
        }

        // F unknown must propagate even when route truth is known false.
        for hit in [false, true] {
            let r = report(None, Some(false), Some(hit), Some(true));
            let o = r.raw[45].opportunities;
            assert_eq!(o.prediction_hit, Some(hit));
            assert_eq!(o.physical_miss_at_f, None);
            assert_eq!(o.already_resident, None);
            assert_eq!(o.useful, None);
            assert_eq!(o.redundant_route_hit, None);
            assert_eq!(o.target_confirmed_useful, None);
            assert_eq!(o.host_backed_target_confirmed_useful, None);
        }

        let r = report(Some(false), None, Some(false), Some(true));
        let o = r.raw[45].opportunities;
        assert_eq!(o.useful, Some(false));
        assert_eq!(o.physical_miss_at_d, None);
        assert_eq!(o.target_confirmed_useful, None);
        assert_eq!(o.host_backed_target_confirmed_useful, None);

        let r = report(Some(true), Some(true), None, Some(false));
        let o = r.raw[45].opportunities;
        assert_eq!(o.prediction_hit, None);
        assert_eq!(o.useful, None);
        assert_eq!(o.redundant_route_hit, None);
        assert_eq!(o.target_confirmed_useful, None);
        assert_eq!(o.host_backed_target_confirmed_useful, None);
        for layer in &r.layers {
            assert_eq!(
                (layer.resolved, layer.route_misses, layer.censored),
                (0, 0, 1)
            );
            assert_eq!(layer.prediction_hit_rate, None);
        }
        let json = serde_json::to_value(&r).unwrap();
        assert!(json["raw"][45]["opportunities"]["prediction_hit"].is_null());
        assert!(json["raw"][45]["opportunities"]["useful"].is_null());

        let r = report(Some(false), Some(false), Some(false), None);
        let o = r.raw[45].opportunities;
        assert_eq!(o.target_confirmed_useful, Some(false));
        assert_eq!(o.host_backed_at_f, None);
        assert_eq!(o.host_backed_target_confirmed_useful, None);

        let r = report(Some(false), Some(false), Some(false), Some(true));
        let o = r.raw[45].opportunities;
        assert_eq!(o.useful, Some(false));
        assert_eq!(o.target_confirmed_useful, Some(false));
        assert_eq!(o.redundant_route_hit, Some(false));
        assert_eq!(o.host_backed_target_confirmed_useful, Some(false));
        assert_eq!(r.layers[0].route_misses, 1);

        let r = report(Some(false), Some(false), Some(true), Some(true));
        let o = r.raw[45].opportunities;
        assert_eq!(o.useful, Some(true));
        assert_eq!(o.target_confirmed_useful, Some(true));
        assert_eq!(o.host_backed_target_confirmed_useful, Some(true));
        assert_eq!(r.layers[0].host_backed_target_confirmed_useful, 1);
    }
    #[test]
    fn p1x_unknown_source_physical_and_censored_truth_stay_unknown() {
        assert_eq!(and(Some(false), None), None);
        let mut o = observer(16);
        step(&mut o, 0, routes(0));
        o.commit(1, &routes(0), true).unwrap();
        for layer in 3..=47 {
            let c = o.pending_candidate(layer, 2).unwrap();
            o.freeze(c, 1, Physical::Unknown, HostSource::default())
                .unwrap();
        }
        o.finish();
        let r = o.report().unwrap();
        assert_eq!(r.layers[0].unknown_at_f, 1);
        assert_eq!(r.layers[0].censored, 1);
        let row = &r.raw[45];
        assert_eq!(row.opportunities.prediction_hit, None);
        assert_eq!(row.opportunities.host_backed_at_f, None);
        assert_eq!(row.row.exact_candidate_saved_ns, None);
    }
    #[test]
    fn p1x_report_rejects_missing_rows_or_identity_corruption() {
        let mut o = observer(16);
        step(&mut o, 0, routes(0));
        step(&mut o, 1, routes(0));
        o.finish();
        o.rows[45].candidate.as_mut().unwrap().sequence = 77;
        assert!(o.report().is_err());
        o.rows.pop();
        assert!(o.report().is_err());
    }
    #[test]
    fn p1x_physical_identity_fail_closed_and_demand_receipt_is_scalar() {
        let mut o = observer(16);
        step(&mut o, 0, routes(0));
        o.commit(1, &routes(0), true).unwrap();
        let c = o.pending_candidate(3, 2).unwrap();
        let p = Physical::Current(PhysicalIdentity {
            context: 1,
            arena: 2,
            layer: 47,
            expert: 0,
            generation: 1,
            bank: 0,
            slot: 0,
            epoch: 1,
        });
        assert_eq!(o.freeze(c, 1, p, source()), Err(Error::PhysicalIdentity));
        let mut receipt = DemandEvidence {
            global: c.global_expert,
            ..Default::default()
        };
        receipt.missing(&[c.global_expert]);
        receipt.missing(&[]);
        assert_eq!(receipt.seen_missing, Some(true));
        assert_eq!(receipt.install_committed, None);
        assert!(!std::mem::needs_drop::<DemandEvidence>());
        assert!(!std::mem::needs_drop::<Physical>());
        assert!(!std::mem::needs_drop::<HostSource>());
    }
}
