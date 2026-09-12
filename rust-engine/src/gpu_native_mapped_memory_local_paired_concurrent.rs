// HMA-1C-F: diagnostic-only local exact-source pairs. No inference call sites.
use super::*;

const F_SCHEMA: &str = "mer.gpu-native-mapped-memory-odirect-local-paired-concurrent.v1";
const F_SCHEDULE: &str = "Universe[i]=floor(i*6143/255), i=0..255. Measured c=0..15: g=c%4,r=c/4; p=0..6: w=(p+c)%7,K=w+2,family=7*g+w. Warmup c=0..3: g=c,r=0,family=28+7*g+w. ids[j]=universe[(17*family+64*r+13*j)%256]. Two passes in identical p order, HC first iff (g+r)%2==0. Global call indices include 56 warmup calls. Pair distance=7. Previous-two witnesses include phase boundary. Hash encoding: metadata=[global_index,measured(0/1),cycle,pass,position,family,role,g,width,hc_first(0/1)] as u64 LE; arm HC=0,MC=1 as u8; source=width u32 LE then ordered IDs u32 LE. Ordered-source hash concatenates source; execution hash concatenates metadata+arm; complete hash concatenates metadata+arm+source. Universe hash concatenates 256 u32 LE IDs.";
const F_TIMER: &str = "Prepare all seven distinct HC and all seven distinct MC destinations, maps, polls, views, alignment, slices and explicit fd evidence before the first source call in each cycle. The 14-call window only invokes the unchanged concurrent helper and stores duration/result metadata in a fixed array. Each timer contains exactly one awaited read_experts_batch_into_aligned_slices call, including its normal fd/proof-cache, scoped-thread scheduler, retries and breaker. No caller allocation/map/unmap/poll/hash/verification/GPU operation/fd probe occurs in this window. Exact byte comparison, hashes, GPU copy/readback, unmap and accounting follow all 14 calls.";
const F_CLASSIFICATION: &str = "CDE=sum(MC)-sum(HC), denominator=sum(HC). STRONG >=5%, MATERIAL >=3%, paired median>0, >56 positive pairs, >=5 positive widths, both arm-order aggregates positive. NEGATIVE <=-3%, median<0, >56 negative pairs, >=5 negative widths, both orders negative. AGAINST abs(CDE)<=1%, exact median MC/HC within inclusive [0.99,1.01], <5 positive widths. Otherwise AMBIGUOUS. Warmup never classifies. All decisions use exact integer/rational comparisons after authority gates.";
const F_RETRY: &str = "PENDING_EXTERNAL_RETRY_LOG_AUDIT: eventual FIRST wrapper must scan the complete external log. Any exact occurrence of 'transient I/O error; retrying', or an unavailable/incomplete log, makes performance non-authoritative. This report does not audit that log.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
enum FArm {
    HC,
    MC,
}
impl FArm {
    const ALL: [Self; 2] = [Self::HC, Self::MC];
    fn index(self) -> usize {
        if self == Self::HC {
            0
        } else {
            1
        }
    }
    fn cell(self) -> Cell {
        if self == Self::HC {
            Cell::HC
        } else {
            Cell::MC
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Previous {
    global_index: usize,
    shared_expert_ids: Vec<u32>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Call {
    global_index: usize,
    measured: bool,
    cycle: usize,
    pass: usize,
    position: usize,
    family: usize,
    role: usize,
    g: usize,
    width: usize,
    hc_first: bool,
    arm: FArm,
    ordered_expert_ids: Vec<u32>,
    previous_two: Vec<Previous>,
}
impl Call {
    fn source_set(&self) -> SourceSet {
        SourceSet {
            set_index: self.cycle * 7 + self.position,
            round: self.cycle,
            width: self.width,
            ordered_expert_ids: self.ordered_expert_ids.clone(),
            execution_order: if self.hc_first {
                ExecutionOrder::ControlFirst
            } else {
                ExecutionOrder::TreatmentFirst
            },
        }
    }
}
fn universe() -> Vec<u32> {
    (0..256).map(|i| i * 6143 / 255).collect()
}
fn schedule() -> Result<Vec<Call>> {
    let u = universe();
    let mut out: Vec<Call> = Vec::with_capacity(280);
    for (measured, cycles) in [(false, 4), (true, 16)] {
        for c in 0..cycles {
            let g = c % 4;
            let r = if measured { c / 4 } else { 0 };
            for pass in 0..2 {
                for p in 0..7 {
                    let w = (p + c) % 7;
                    let family = if measured { 0 } else { 28 } + 7 * g + w;
                    let ids: Vec<_> = (0..w + 2)
                        .map(|j| u[(17 * family + 64 * r + 13 * j) % 256])
                        .collect();
                    let mut previous_two = Vec::new();
                    for prior in out.iter().rev().take(2) {
                        let shared: Vec<_> = ids
                            .iter()
                            .filter(|id| prior.ordered_expert_ids.contains(id))
                            .copied()
                            .collect();
                        if !shared.is_empty() {
                            return Err(Failure::accounting("F previous-two overlap"));
                        }
                        previous_two.push(Previous {
                            global_index: prior.global_index,
                            shared_expert_ids: shared,
                        });
                    }
                    let hc_first = (g + r) % 2 == 0;
                    out.push(Call {
                        global_index: out.len(),
                        measured,
                        cycle: c,
                        pass,
                        position: p,
                        family,
                        role: r,
                        g,
                        width: w + 2,
                        hc_first,
                        arm: if (pass == 0) == hc_first {
                            FArm::HC
                        } else {
                            FArm::MC
                        },
                        ordered_expert_ids: ids,
                        previous_two,
                    });
                }
            }
        }
    }
    Ok(out)
}
fn phase_schedule(measured: bool) -> Result<Vec<Call>> {
    Ok(schedule()?
        .into_iter()
        .filter(|c| c.measured == measured)
        .collect())
}
fn validate_schedule(calls: &[Call], measured: bool) -> Result<()> {
    if calls != phase_schedule(measured)? {
        return Err(Failure::accounting(
            "F exact schedule reconstruction mismatch",
        ));
    }
    Ok(())
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ScheduleHashes {
    universe_sha256: String,
    ordered_source_ids_sha256: String,
    execution_order_sha256: String,
    complete_schedule_sha256: String,
}
fn schedule_hashes(calls: &[Call]) -> ScheduleHashes {
    let mut source = Sha256::new();
    let mut execution = Sha256::new();
    let mut complete = Sha256::new();
    for c in calls {
        let mut metadata = Vec::new();
        for n in [
            c.global_index,
            c.measured as usize,
            c.cycle,
            c.pass,
            c.position,
            c.family,
            c.role,
            c.g,
            c.width,
            c.hc_first as usize,
        ] {
            metadata.extend_from_slice(&(n as u64).to_le_bytes());
        }
        metadata.push(c.arm.index() as u8);
        let mut ids = (c.width as u32).to_le_bytes().to_vec();
        for id in &c.ordered_expert_ids {
            ids.extend_from_slice(&id.to_le_bytes());
        }
        source.update(&ids);
        execution.update(&metadata);
        complete.update(&metadata);
        complete.update(&ids);
    }
    ScheduleHashes {
        universe_sha256: sequence_sha(&universe()),
        ordered_source_ids_sha256: finish_sha(&source),
        execution_order_sha256: finish_sha(&execution),
        complete_schedule_sha256: finish_sha(&complete),
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Expected {
    cycles: u64,
    matched_pairs: u64,
    helper_calls: u64,
    calls_per_arm: u64,
    slots_per_arm: u64,
    full_source_bytes_per_arm: u64,
    payload_bytes_per_arm: u64,
    proof_requests: u64,
    proof_hits: u64,
    proof_misses: u64,
    proof_failures: u64,
}
fn expected(measured: bool) -> Expected {
    let factor = if measured { 4 } else { 1 };
    Expected {
        cycles: 4 * factor,
        matched_pairs: 28 * factor,
        helper_calls: 56 * factor,
        calls_per_arm: 28 * factor,
        slots_per_arm: 140 * factor,
        full_source_bytes_per_arm: 372_162_560 * factor,
        payload_bytes_per_arm: 140 * factor * PAYLOAD as u64,
        proof_requests: 280 * factor,
        proof_hits: 280 * factor,
        proof_misses: 0,
        proof_failures: 0,
    }
}
fn proof_hits_only(p: &SourceUploadFdProofSnapshot, measured: bool) -> bool {
    *p == SourceUploadFdProofSnapshot {
        source_upload_fd_proof_requests: expected(measured).proof_requests,
        source_upload_fd_proof_hits: expected(measured).proof_hits,
        ..Default::default()
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Raw {
    call: Call,
    ns: u64,
    returned_bytes: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Pair {
    family: usize,
    role: usize,
    g: usize,
    width: usize,
    cycle: usize,
    position: usize,
    hc_first: bool,
    ordered_expert_ids: Vec<u32>,
    hc_call_index: usize,
    mc_call_index: usize,
    helper_call_distance: usize,
    hc_ns: u64,
    mc_ns: u64,
    cde_ns: i128,
}
fn reconstruct(raw: &[Raw], measured: bool) -> Result<Vec<Pair>> {
    let calls: Vec<_> = raw.iter().map(|s| s.call.clone()).collect();
    validate_schedule(&calls, measured)?;
    let mut pairs = Vec::new();
    for cycle in raw.chunks_exact(14) {
        for p in 0..7 {
            let (hc, mc) = if cycle[p].call.hc_first {
                (&cycle[p], &cycle[p + 7])
            } else {
                (&cycle[p + 7], &cycle[p])
            };
            let c = &hc.call;
            if hc.ns == 0
                || mc.ns == 0
                || hc.returned_bytes != source_bytes(c.width as u64)?
                || mc.returned_bytes != hc.returned_bytes
                || c.ordered_expert_ids != mc.call.ordered_expert_ids
                || c.global_index.abs_diff(mc.call.global_index) != 7
            {
                return Err(Failure::accounting(
                    "F raw timing/source/pair distance/bytes mismatch",
                ));
            }
            pairs.push(Pair {
                family: c.family,
                role: c.role,
                g: c.g,
                width: c.width,
                cycle: c.cycle,
                position: c.position,
                hc_first: c.hc_first,
                ordered_expert_ids: c.ordered_expert_ids.clone(),
                hc_call_index: c.global_index,
                mc_call_index: mc.call.global_index,
                helper_call_distance: 7,
                hc_ns: hc.ns,
                mc_ns: mc.ns,
                cde_ns: i128::from(mc.ns) - i128::from(hc.ns),
            });
        }
    }
    Ok(pairs)
}
#[derive(Clone, Debug, PartialEq, Serialize)]
struct Stats {
    sample_count: u64,
    hc_total_ns: u64,
    mc_total_ns: u64,
    cde_ns: i128,
    cde_percent_exact: SignedFraction,
    cde_percent: f64,
    positive: u64,
    negative: u64,
    equal: u64,
    paired_median_mc_minus_hc: SignedFraction,
    median_mc_over_hc: RatioMedian,
}
fn stats(pairs: &[Pair]) -> Result<Stats> {
    if pairs.is_empty() {
        return Err(Failure::accounting("F empty statistics"));
    }
    let (mut hc, mut mc, mut positive, mut negative, mut equal) = (0, 0, 0, 0, 0);
    for p in pairs {
        add(&mut hc, p.hc_ns)?;
        add(&mut mc, p.mc_ns)?;
        add(
            if p.cde_ns > 0 {
                &mut positive
            } else if p.cde_ns < 0 {
                &mut negative
            } else {
                &mut equal
            },
            1,
        )?;
    }
    let delta = i128::from(mc) - i128::from(hc);
    Ok(Stats {
        sample_count: pairs.len() as u64,
        hc_total_ns: hc,
        mc_total_ns: mc,
        cde_ns: delta,
        cde_percent_exact: SignedFraction {
            numerator: delta
                .checked_mul(100)
                .ok_or_else(|| Failure::accounting("F percent overflow"))?,
            denominator: hc,
        },
        cde_percent: 100.0 * delta as f64 / hc as f64,
        positive,
        negative,
        equal,
        paired_median_mc_minus_hc: median_signed(pairs.iter().map(|p| p.cde_ns).collect())?
            .unwrap(),
        median_mc_over_hc: RatioMedian::new(
            pairs
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    (
                        i,
                        Ratio {
                            numerator: p.mc_ns.into(),
                            denominator: p.hc_ns.into(),
                        },
                    )
                })
                .collect(),
        )
        .unwrap(),
    })
}
#[derive(Clone, Debug, PartialEq, Serialize)]
struct Statistics {
    primary: Stats,
    width: BTreeMap<usize, Stats>,
    arm_order: BTreeMap<String, Stats>,
    source_role: BTreeMap<usize, Stats>,
    g: BTreeMap<usize, Stats>,
    cycle_position: BTreeMap<usize, Stats>,
}
fn partition(
    pairs: &[Pair],
    keys: std::ops::Range<usize>,
    key: impl Fn(&Pair) -> usize,
) -> Result<BTreeMap<usize, Stats>> {
    keys.map(|k| {
        Ok((
            k,
            stats(
                &pairs
                    .iter()
                    .filter(|p| key(p) == k)
                    .cloned()
                    .collect::<Vec<_>>(),
            )?,
        ))
    })
    .collect()
}
fn statistics(pairs: &[Pair], measured: bool) -> Result<Statistics> {
    let primary = stats(pairs)?;
    let width = partition(pairs, 2..9, |p| p.width)?;
    let source_role = partition(pairs, 0..if measured { 4 } else { 1 }, |p| p.role)?;
    let g = partition(pairs, 0..4, |p| p.g)?;
    let cycle_position = partition(pairs, 0..7, |p| p.position)?;
    let orders = partition(pairs, 0..2, |p| (!p.hc_first) as usize)?;
    for part in [&width, &source_role, &g, &cycle_position, &orders] {
        let (mut n, mut hc, mut mc) = (0, 0, 0);
        for s in part.values() {
            add(&mut n, s.sample_count)?;
            add(&mut hc, s.hc_total_ns)?;
            add(&mut mc, s.mc_total_ns)?;
        }
        if (n, hc, mc)
            != (
                primary.sample_count,
                primary.hc_total_ns,
                primary.mc_total_ns,
            )
        {
            return Err(Failure::accounting("F strata reconciliation"));
        }
    }
    Ok(Statistics {
        primary,
        width,
        source_role,
        g,
        cycle_position,
        arm_order: orders
            .into_iter()
            .map(|(k, v)| (if k == 0 { "HC-first" } else { "MC-first" }.into(), v))
            .collect(),
    })
}
fn interpretation(s: &Statistics) -> Result<&'static str> {
    let t = &s.primary;
    if t.sample_count != 112 || s.width.len() != 7 || s.arm_order.len() != 2 {
        return Err(Failure::accounting("F classification population"));
    }
    let pos = s.width.values().filter(|w| w.cde_ns > 0).count();
    let neg = s.width.values().filter(|w| w.cde_ns < 0).count();
    let pct = t.cde_percent_exact.numerator;
    let den = i128::from(t.hc_total_ns);
    let positive = t.paired_median_mc_minus_hc.numerator > 0
        && t.positive > 56
        && pos >= 5
        && s.arm_order.values().all(|o| o.cde_ns > 0);
    let negative = t.paired_median_mc_minus_hc.numerator < 0
        && t.negative > 56
        && neg >= 5
        && s.arm_order.values().all(|o| o.cde_ns < 0);
    Ok(if pct >= 5 * den && positive {
        "STRONG_POSITIVE_CDE"
    } else if pct >= 3 * den && positive {
        "MATERIAL_POSITIVE_CDE"
    } else if pct <= -3 * den && negative {
        "NEGATIVE_MATERIAL_CDE"
    } else if pct.abs() <= den
        && t.median_mc_over_hc.compare_hundredths(99)? != Ordering::Less
        && t.median_mc_over_hc.compare_hundredths(101)? != Ordering::Greater
        && pos < 5
    {
        "EVIDENCE_AGAINST_MATERIAL_POSITIVE_CDE"
    } else {
        "AMBIGUOUS"
    })
}

// All allocations and mutable slice construction precede timed_cycle. The
// unchanged helper's own scheduler/proof/retry allocations remain inside it.
struct PreparedCall<'a> {
    ids: &'a [u32],
    destinations: Vec<&'a mut [u8]>,
}
struct TimedRead {
    duration: Duration,
    read: io::Result<usize>,
}
async fn timed_cycle(storage: &NvmeStorage, calls: &mut [PreparedCall<'_>; 14]) -> [TimedRead; 14] {
    let mut results = std::array::from_fn(|_| TimedRead {
        duration: Duration::ZERO,
        read: Ok(0),
    });
    for (call, result) in calls.iter_mut().zip(results.iter_mut()) {
        let ids = call.ids;
        let destinations = call.destinations.as_mut_slice();
        let start = Instant::now();
        let read = storage
            .read_experts_batch_into_aligned_slices(ids, destinations)
            .await;
        let duration = start.elapsed();
        *result = TimedRead { duration, read };
    }
    results
}
// Views and prepared slices are declared after this owner and therefore dropped
// first on both error and unwind. No mapped allocation is reused within a cycle.
struct MappedCycle {
    buffers: Vec<wgpu::Buffer>,
    unmapped: bool,
}
impl Drop for MappedCycle {
    fn drop(&mut self) {
        if !self.unmapped {
            for b in &self.buffers {
                b.unmap();
            }
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct SourceHash {
    expert_id: u32,
    full_source_bytes: u64,
    payload_bytes: u64,
    source: String,
    payload: String,
    gpu: String,
    epoch: bool,
}
impl SourceHash {
    fn new(id: u32, h: Hashes) -> Self {
        Self {
            expert_id: id,
            full_source_bytes: FULL as u64,
            payload_bytes: PAYLOAD as u64,
            source: h.source,
            payload: h.payload,
            gpu: h.gpu,
            epoch: h.epoch,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct VerifiedPair {
    family: usize,
    role: usize,
    hc_call_index: usize,
    mc_call_index: usize,
    exact_bytes_equal: bool,
    hc: Vec<SourceHash>,
    mc: Vec<SourceHash>,
}
fn valid_hash(h: &str) -> bool {
    h.len() == 64
        && h.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn verified(p: &Pair, v: &VerifiedPair) -> bool {
    v.family == p.family
        && v.role == p.role
        && v.hc_call_index == p.hc_call_index
        && v.mc_call_index == p.mc_call_index
        && v.exact_bytes_equal
        && v.hc == v.mc
        && v.hc.len() == p.width
        && v.hc.iter().zip(&p.ordered_expert_ids).all(|(h, id)| {
            h.expert_id == *id
                && h.full_source_bytes == FULL as u64
                && h.payload_bytes == PAYLOAD as u64
                && valid_hash(&h.source)
                && valid_hash(&h.payload)
                && h.gpu == h.payload
                && h.epoch
        })
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct CycleIsolation {
    cycle: usize,
    first_call: usize,
    last_call: usize,
    prepared_host_sets: usize,
    prepared_mapped_sets: usize,
    prepared_fd_checks: usize,
    completed_helper_calls: usize,
    source_only_window_complete: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct DestinationEvidence {
    global_index: usize,
    arm: FArm,
    base_address: usize,
    capacity_bytes: usize,
    aligned_offset: usize,
    source_bytes: usize,
    gpu_payload_offsets: Vec<u64>,
}
fn destination_evidence(call: &Call, base: usize, capacity: usize) -> Result<DestinationEvidence> {
    let mapped = call.arm == FArm::MC;
    let offset = arena_offset(base, capacity, call.width, mapped)?;
    Ok(DestinationEvidence {
        global_index: call.global_index,
        arm: call.arm,
        base_address: base,
        capacity_bytes: capacity,
        aligned_offset: offset,
        source_bytes: call.width * FULL,
        gpu_payload_offsets: if mapped {
            (0..call.width)
                .map(|j| {
                    copy_offsets(offset + j * FULL, PREFIX, PAYLOAD, capacity)
                        .map_err(Failure::accounting)
                })
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        },
    })
}
fn valid_destinations(p: &FPhase) -> Result<bool> {
    if p.destinations.len() != p.schedule.len() {
        return Ok(false);
    }
    for (window, evidence) in p
        .schedule
        .chunks_exact(14)
        .zip(p.destinations.chunks_exact(14))
    {
        let mut ranges = Vec::new();
        for (call, e) in window.iter().zip(evidence) {
            if *e != destination_evidence(call, e.base_address, e.capacity_bytes)? {
                return Ok(false);
            }
            let end = e
                .base_address
                .checked_add(e.capacity_bytes)
                .ok_or_else(|| Failure::accounting("F mapped range overflow"))?;
            if ranges
                .iter()
                .any(|&(start, stop)| e.base_address < stop && start < end)
            {
                return Ok(false);
            }
            ranges.push((e.base_address, end));
        }
    }
    for a in FArm::ALL {
        let mut hist = BTreeMap::new();
        let mut offsets = Vec::new();
        for e in p.destinations.iter().filter(|e| e.arm == a) {
            let width = e.source_bytes / FULL;
            *hist.entry(e.base_address % ALIGN).or_insert(0u64) += width as u64;
            offsets.extend((0..width).map(|j| e.aligned_offset + j * FULL));
        }
        let Some(arm) = p.arms.get(&a) else {
            return Ok(false);
        };
        let v = &arm.evidence.pointers;
        if v.mapping_base_mod_4096_counts != hist
            || v.aligned_offset_min != offsets.iter().min().copied()
            || v.aligned_offset_max != offsets.iter().max().copied()
        {
            return Ok(false);
        }
        let mut fd_checks = 0;
        for count in arm.evidence.fd_evidence.flags_counts.values() {
            add(&mut fd_checks, *count)?;
        }
        if fd_checks != p.expected.slots_per_arm {
            return Ok(false);
        }
    }
    Ok(true)
}

#[derive(Debug, Serialize)]
struct FailedRead {
    call: Call,
    ns: u64,
    returned_bytes: Option<usize>,
    errno: Option<i32>,
    error: String,
}
#[derive(Debug, Serialize)]
struct FPhase {
    measured: bool,
    expected: Expected,
    schedule: Vec<Call>,
    schedule_hashes: ScheduleHashes,
    source_errors: Vec<FailedRead>,
    destinations: Vec<DestinationEvidence>,
    raw_samples: Vec<Raw>,
    matched_pairs: Vec<Pair>,
    verification: Vec<VerifiedPair>,
    cycles: Vec<CycleIsolation>,
    arms: BTreeMap<FArm, CellArm>,
    witnesses: BTreeMap<FArm, Witnesses>,
    fd_proof: SourceUploadFdProofSnapshot,
    statistics: Option<Statistics>,
}
impl FPhase {
    fn new(measured: bool) -> Result<Self> {
        let schedule = phase_schedule(measured)?;
        let hashes = schedule_hashes(&schedule);
        Ok(Self {
            measured,
            expected: expected(measured),
            schedule,
            schedule_hashes: hashes,
            source_errors: Vec::new(),
            destinations: Vec::new(),
            raw_samples: Vec::new(),
            matched_pairs: Vec::new(),
            verification: Vec::new(),
            cycles: Vec::new(),
            arms: FArm::ALL
                .into_iter()
                .map(|a| Ok((a, CellArm::new(a.cell())?)))
                .collect::<Result<_>>()?,
            witnesses: BTreeMap::new(),
            fd_proof: Default::default(),
            statistics: None,
        })
    }
    fn successful(&self) -> bool {
        let check = || -> Result<bool> {
            validate_schedule(&self.schedule, self.measured)?;
            let pairs = reconstruct(&self.raw_samples, self.measured)?;
            if !valid_destinations(self)?
                || !self.source_errors.is_empty()
                || self.expected != expected(self.measured)
                || self.schedule_hashes != schedule_hashes(&self.schedule)
                || self.matched_pairs != pairs
                || self.statistics != Some(statistics(&pairs, self.measured)?)
                || self.verification.len() != pairs.len()
                || !pairs
                    .iter()
                    .zip(&self.verification)
                    .all(|(p, v)| verified(p, v))
                || !proof_hits_only(&self.fd_proof, self.measured)
                || self.cycles.len() != self.expected.cycles as usize
                || self.arms.len() != 2
                || self.witnesses.len() != 2
            {
                return Ok(false);
            }
            for (cycle, window) in self.schedule.chunks_exact(14).enumerate() {
                let expected_cycle = CycleIsolation {
                    cycle,
                    first_call: window[0].global_index,
                    last_call: window[13].global_index,
                    prepared_host_sets: 7,
                    prepared_mapped_sets: 7,
                    prepared_fd_checks: 70,
                    completed_helper_calls: 14,
                    source_only_window_complete: true,
                };
                if self.cycles[cycle] != expected_cycle {
                    return Ok(false);
                }
            }
            for a in FArm::ALL {
                let sets: Vec<_> = self
                    .schedule
                    .iter()
                    .filter(|c| c.arm == a)
                    .map(Call::source_set)
                    .collect();
                let evidence = schedule_evidence(&sets)?;
                let Some(arm) = self.arms.get(&a) else {
                    return Ok(false);
                };
                let mut ns = 0;
                for r in self.raw_samples.iter().filter(|r| r.call.arm == a) {
                    add(&mut ns, r.ns)?;
                }
                if arm.cell != a.cell()
                    || arm.completed_source_sets != sets
                    || arm.evidence.pointers.gpu_offset_checks
                        != if a == FArm::MC {
                            self.expected.slots_per_arm
                        } else {
                            0
                        }
                    || !arm.successful(&evidence, a == FArm::MC)
                    || arm.evidence.times.source_direct_read_ns != ns
                {
                    return Ok(false);
                }
                let Some(w) = self.witnesses.get(&a) else {
                    return Ok(false);
                };
                if !valid_hash(&w.full_source_sha256)
                    || !valid_hash(&w.bare_payload_sha256)
                    || w.bare_payload_sha256 != w.gpu_destination_payload_sha256
                {
                    return Ok(false);
                }
            }
            let hc = &self.witnesses[&FArm::HC];
            let mc = &self.witnesses[&FArm::MC];
            Ok(hc.full_source_sha256 == mc.full_source_sha256
                && hc.bare_payload_sha256 == mc.bare_payload_sha256)
        };
        check().unwrap_or(false)
    }
}

async fn run_cycle(
    p: &mut FPhase,
    gpu: &Gpu,
    storage: &NvmeStorage,
    window: &[Call],
    streams: &mut [Streams; 2],
) -> Result<()> {
    if window.len() != 14 {
        return Err(Failure::accounting("F cycle length"));
    }
    // Own every destination before mapping, including distinct backing for every
    // source set. The mapped owner outlives its views and all borrowed slices.
    let mut host: Vec<_> = (0..7)
        .map(|_| AlignedBuffer::new(HOST_ARENA, ALIGN))
        .collect();
    let mut mapped = MappedCycle {
        buffers: (0..7)
            .map(|_| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("hma1cf-cycle-source"),
                    size: MAPPED_ARENA as u64,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect(),
        unmapped: false,
    };
    for buffer in &mapped.buffers {
        let arm = &mut p.arms.get_mut(&FArm::MC).unwrap().evidence;
        add(&mut arm.map_attempts, 1)?;
        gpu.map(buffer, wgpu::MapMode::Write)?;
        add(&mut arm.maps_completed, 1)?;
    }
    let mut views: Vec<_> = mapped
        .buffers
        .iter()
        .map(|b| b.slice(..).get_mapped_range_mut())
        .collect();
    let mut hc_slices = Vec::with_capacity(7);
    let mut mc_slices = Vec::with_capacity(7);
    let mut offsets = Vec::with_capacity(7);
    let mut bases = Vec::with_capacity(7);
    for (position, (h, m)) in host.iter_mut().zip(views.iter_mut()).enumerate() {
        let width = window[position].width;
        let hbase = h.as_slice().as_ptr() as usize;
        let mbase = m.as_ptr() as usize;
        bases.push((hbase, mbase));
        let (hoff, hs) = arena_slices(h.as_mut_slice(), width, false)?;
        let (moff, ms) = arena_slices(m, width, true)?;
        for (a, base, offset) in [(FArm::HC, hbase, hoff), (FArm::MC, mbase, moff)] {
            observe_slices(
                p.arms.get_mut(&a).unwrap(),
                base,
                offset,
                width,
                a == FArm::MC,
            )?;
        }
        let gpu_offsets = (0..width)
            .map(|j| {
                copy_offsets(moff + j * FULL, PREFIX, PAYLOAD, MAPPED_ARENA)
                    .map_err(Failure::accounting)
            })
            .collect::<Result<Vec<_>>>()?;
        hc_slices.push(hs);
        mc_slices.push(ms);
        offsets.push(gpu_offsets);
    }
    for call in window {
        let (base, capacity) = if call.arm == FArm::HC {
            (bases[call.position].0, HOST_ARENA)
        } else {
            (bases[call.position].1, MAPPED_ARENA)
        };
        p.destinations
            .push(destination_evidence(call, base, capacity)?);
        let arm = p.arms.get_mut(&call.arm).unwrap();
        begin_set(storage, &call.source_set(), arm)?;
        add(&mut arm.evidence.source_read_attempts, call.width as u64)?;
    }
    let mut prepared: [PreparedCall<'_>; 14] = std::array::from_fn(|i| {
        let slices = if window[i].arm == FArm::HC {
            &mut hc_slices
        } else {
            &mut mc_slices
        };
        PreparedCall {
            ids: &window[i].ordered_expert_ids,
            destinations: std::mem::take(&mut slices[i % 7]),
        }
    });
    // No caller setup or post-processing inside this entire 14-call window.
    let results = timed_cycle(storage, &mut prepared).await;
    // Only after the final helper returns do we allocate or inspect results.
    let mut failure = None;
    for (call, result) in window.iter().zip(results) {
        let ns = u64::try_from(result.duration.as_nanos()).map_err(Failure::accounting)?;
        let arm = p.arms.get_mut(&call.arm).unwrap();
        add(&mut arm.concurrent_helper_calls, 1)?;
        add(&mut arm.evidence.times.source_direct_read_ns, ns)?;
        match result.read {
            Ok(n) if n == call.width * FULL && ns > 0 => {
                add(&mut arm.evidence.source_read_ops, call.width as u64)?;
                add(&mut arm.evidence.full_source_bytes, n as u64)?;
                arm.completed_source_sets.push(call.source_set());
                p.raw_samples.push(Raw {
                    call: call.clone(),
                    ns,
                    returned_bytes: n as u64,
                });
            }
            other => {
                let detail = format!("F source call {}: {:?}, ns={ns}", call.global_index, other);
                p.source_errors.push(FailedRead {
                    call: call.clone(),
                    ns,
                    returned_bytes: other.as_ref().ok().copied(),
                    errno: other.as_ref().err().and_then(|e| e.raw_os_error()),
                    error: detail.clone(),
                });
                match &other {
                    Err(_) => add(&mut arm.evidence.source_failures, 1)?,
                    _ => add(&mut arm.evidence.exact_read_length_failures, 1)?,
                }
                failure.get_or_insert_with(|| Failure::runtime("source-failed", detail));
            }
        }
    }
    if let Some(failure) = failure {
        return Err(failure);
    }
    p.cycles.push(CycleIsolation {
        cycle: window[0].cycle,
        first_call: window[0].global_index,
        last_call: window[13].global_index,
        prepared_host_sets: 7,
        prepared_mapped_sets: 7,
        prepared_fd_checks: 70,
        completed_helper_calls: 14,
        source_only_window_complete: true,
    });
    let mut verified_cycle = Vec::with_capacity(7);
    let mut mapped_hashes = Vec::with_capacity(7);
    for position in 0..7 {
        let (hi, mi) = if window[0].hc_first {
            (position, position + 7)
        } else {
            (position + 7, position)
        };
        let hc = &prepared[hi];
        let mc = &prepared[mi];
        if hc.ids != mc.ids || hc.destinations.len() != mc.destinations.len() {
            return Err(Failure::accounting("F exact source identity mismatch"));
        }
        let mut vh = Vec::new();
        let mut mh = Vec::new();
        for ((&id, h), m) in hc.ids.iter().zip(&hc.destinations).zip(&mc.destinations) {
            if **h != **m {
                return Err(Failure::runtime(
                    "hash-parity-failed",
                    "F exact HC/MC source bytes differ",
                ));
            }
            let (_, mut hh) = streams[0].source(h).map_err(Failure::authority)?;
            let (_, mm) = streams[1].source(m).map_err(Failure::authority)?;
            if hh.source != mm.source || hh.payload != mm.payload {
                return Err(Failure::runtime(
                    "hash-parity-failed",
                    "F exact source/payload hashes differ",
                ));
            }
            note_payload(&mut p.arms.get_mut(&FArm::MC).unwrap().evidence)?;
            let arm = &mut p.arms.get_mut(&FArm::HC).unwrap().evidence;
            note_payload(arm)?;
            prepare_destination(gpu, arm)?;
            gpu.queue
                .write_buffer(&gpu.destination, 0, &EPOCH.to_le_bytes());
            let mut staging = gpu
                .queue
                .write_buffer_with(
                    &gpu.destination,
                    EPOCH_OFFSET as u64,
                    NonZeroU64::new(PAYLOAD as u64).unwrap(),
                )
                .ok_or_else(|| Failure::runtime("gpu-failed", "F host staging view unavailable"))?;
            staging.copy_from_slice(&h[PREFIX..]);
            drop(staging);
            add(&mut arm.cpu_payload_copy_bytes, PAYLOAD as u64)?;
            note_upload(arm, false)?;
            gpu.drain(None)?;
            add(&mut arm.gpu_completed_ops, 1)?;
            verify(gpu, arm, &mut hh, &mut streams[0])?;
            vh.push(SourceHash::new(id, hh));
            mh.push(mm);
        }
        let call = &window[hi];
        verified_cycle.push(VerifiedPair {
            family: call.family,
            role: call.role,
            hc_call_index: call.global_index,
            mc_call_index: window[mi].global_index,
            exact_bytes_equal: true,
            hc: vh,
            mc: Vec::new(),
        });
        mapped_hashes.push(mh);
    }
    drop(prepared);
    drop(views);
    // WGPU requires unmap before copy; both occur strictly after the source window.
    for b in &mapped.buffers {
        b.unmap();
    }
    mapped.unmapped = true;
    add(&mut p.arms.get_mut(&FArm::MC).unwrap().evidence.unmaps, 7)?;
    for (position, hashes) in mapped_hashes.into_iter().enumerate() {
        for (j, mut h) in hashes.into_iter().enumerate() {
            let arm = &mut p.arms.get_mut(&FArm::MC).unwrap().evidence;
            prepare_destination(gpu, arm)?;
            gpu.queue
                .write_buffer(&gpu.destination, 0, &EPOCH.to_le_bytes());
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("hma1cf-post-cycle-copy"),
                });
            encoder.copy_buffer_to_buffer(
                &mapped.buffers[position],
                offsets[position][j],
                &gpu.destination,
                EPOCH_OFFSET as u64,
                PAYLOAD as u64,
            );
            note_upload(arm, true)?;
            gpu.drain(Some(encoder.finish()))?;
            add(&mut arm.gpu_completed_ops, 1)?;
            verify(gpu, arm, &mut h, &mut streams[1])?;
            verified_cycle[position]
                .mc
                .push(SourceHash::new(window[position].ordered_expert_ids[j], h));
        }
    }
    p.verification.extend(verified_cycle);
    gpu.check()
}
async fn f_phase(p: &mut FPhase, gpu: &Gpu, storage: &NvmeStorage) -> Result<()> {
    validate_schedule(&p.schedule, p.measured)?;
    let mut streams = [Streams::default(), Streams::default()];
    let result = async {
        for window in p.schedule.clone().chunks_exact(14) {
            run_cycle(p, gpu, storage, window, &mut streams).await?;
        }
        Ok(())
    }
    .await;
    p.fd_proof = storage.source_upload_fd_proof_snapshot();
    for a in FArm::ALL {
        let arm = p.arms.get_mut(&a).unwrap();
        arm.source_schedule = schedule_evidence(&arm.completed_source_sets)?;
        p.witnesses.insert(a, streams[a.index()].snapshot());
    }
    result?;
    p.matched_pairs = reconstruct(&p.raw_samples, p.measured)?;
    p.statistics = Some(statistics(&p.matched_pairs, p.measured)?);
    if !p.successful() {
        return Err(Failure::authority(
            "F phase schedule/source/hash/accounting/proof/isolation reconciliation failed",
        ));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct FReport {
    schema: &'static str,
    args: Args,
    config_sha256: Option<String>,
    complete: bool,
    correctness_pass: bool,
    authoritative: bool,
    classification: String,
    failure: Option<String>,
    authority: CAuthority,
    warmup: FPhase,
    measured: FPhase,
    performance_required_for_correctness: bool,
    performance_authority: &'static str,
    concurrent_destination_interpretation_conditional_on_zero_retry_log: &'static str,
    primary_endpoint: &'static str,
    schedule_contract: &'static str,
    timing_contract: &'static str,
    interpretation_contract: &'static str,
    retry_evidence_contract: &'static str,
    source_byte_evidence_contract: &'static str,
}
impl FReport {
    fn new(args: Args) -> Result<Self> {
        let mut base = Report::new(args.clone()).authority;
        base.control_source_api = B_API;
        base.treatment_source_api = B_API;
        base.control_destination = "aligned-host-arena";
        base.treatment_destination = "wgpu-map-write-arena";
        base.upload_capacity_bytes = MAPPED_ARENA;
        Ok(Self {
            schema: F_SCHEMA,
            args,
            config_sha256: None,
            complete: false,
            correctness_pass: false,
            authoritative: false,
            classification: "not-run".into(),
            failure: None,
            authority: CAuthority {
                base,
                cell_source_apis: [Cell::HC, Cell::MC].into_iter().map(|c| (c, c.api())).collect(),
                host_arena_capacity_bytes: HOST_ARENA,
                max_width: MAX_WIDTH,
                fd_preproof_completed: false,
                fd_cache_capacity: 0,
                preproof_universe_size: 256,
                preproof_ordered_expert_ids: expert_sequence(256, NAMESPACE).map_err(Failure::accounting)?,
                preproof: SourceUploadFdProofSnapshot::default(),
                after_preproof_telemetry_reset: SourceUploadFdProofSnapshot::default(),
                after_warmup_telemetry_reset: SourceUploadFdProofSnapshot::default(),
            },
            warmup: FPhase::new(false)?,
            measured: FPhase::new(true)?,
            performance_required_for_correctness: false,
            performance_authority: "PENDING_EXTERNAL_RETRY_LOG_AUDIT",
            concurrent_destination_interpretation_conditional_on_zero_retry_log: "INCONCLUSIVE",
            primary_endpoint: "112 exact (family,source_role) matched units: CDE=sum(MC)-sum(HC); percent denominator=sum(HC)",
            schedule_contract: F_SCHEDULE,
            timing_contract: F_TIMER,
            interpretation_contract: F_CLASSIFICATION,
            retry_evidence_contract: F_RETRY,
            source_byte_evidence_contract: "Each helper returns exactly K*FULL. Every pair compares exact full HC and MC bytes after all 14 calls, then ordered source/payload/GPU hashes. Arm streams follow cycle/position/expert order and must match. Partial physical I/O on failed calls is unavailable. Unmap precedes mapped GPU copy as required by WGPU; all verification is post-cycle. CLI retains --iterations 128 --warmup-iterations 32; actual F layout is 16 measured cycles and 4 warmup cycles.",
        })
    }
    fn authority_valid(&self) -> bool {
        let a = &self.authority;
        let b = &a.base;
        self.schema == F_SCHEMA
            && !self.performance_required_for_correctness
            && self.args.iterations == 128
            && self.args.warmup_iterations == 32
            && !self.warmup.measured
            && self.measured.measured
            && a.cell_source_apis
                == [Cell::HC, Cell::MC]
                    .into_iter()
                    .map(|c| (c, c.api()))
                    .collect()
            && a.host_arena_capacity_bytes == HOST_ARENA
            && a.max_width == MAX_WIDTH
            && b.same_source_api
            && b.control_source_api == B_API
            && b.treatment_source_api == B_API
            && b.control_destination == "aligned-host-arena"
            && b.treatment_destination == "wgpu-map-write-arena"
            && b.upload_capacity_bytes == MAPPED_ARENA
            && b.full_source_bytes == FULL
            && b.block_alignment == ALIGN
            && b.uth_prefix_bytes == PREFIX
            && b.bare_payload_bytes == PAYLOAD
            && b.physical_slot_bytes == SLOT
            && b.source_timer_excludes_allocation
            && b.source_timer_excludes_map_async_device_poll
            && b.source_timer_excludes_alignment_setup
            && b.source_timer_excludes_hashes_readback_fd_evidence
            && b.source_timer_excludes_gpu_copy_unmap
            && b.linux
            && b.expected_adapter_name == "NVIDIA L4"
            && self.args.expected_adapter_name == "NVIDIA L4"
            && b.adapter_authoritative
            && b.adapter_name.as_deref() == Some("NVIDIA L4")
            && b.adapter_backend.as_deref() == Some("Vulkan")
            && b.adapter_device_type.as_deref() == Some("DiscreteGpu")
            && b.direct_io_requested
            && b.packed_storage == Some(false)
            && b.exact_geometry
            && a.fd_preproof_completed
            && a.fd_cache_capacity >= 256
            && a.preproof_universe_size == 256
            && expert_sequence(256, NAMESPACE).is_ok_and(|ids| ids == a.preproof_ordered_expert_ids)
            && a.preproof
                == SourceUploadFdProofSnapshot {
                    source_upload_fd_proof_requests: 256,
                    source_upload_fd_proof_misses: 256,
                    ..SourceUploadFdProofSnapshot::default()
                }
            && a.after_preproof_telemetry_reset == SourceUploadFdProofSnapshot::default()
            && a.after_warmup_telemetry_reset == SourceUploadFdProofSnapshot::default()
            && proof_hits_only(&self.warmup.fd_proof, false)
            && proof_hits_only(&self.measured.fd_proof, true)
    }

    fn classify(&mut self) -> Result<()> {
        self.complete = true;
        self.correctness_pass = false;
        self.authoritative = false;
        self.concurrent_destination_interpretation_conditional_on_zero_retry_log = "AMBIGUOUS";
        if !self.authority_valid() {
            self.classification = "authority-failed".into();
        } else if !self.warmup.successful() || !self.measured.successful() {
            self.classification = "evidence-reconciliation-failed".into();
        } else {
            self.correctness_pass = true;
            self.authoritative = true;
            self.classification = "local-paired-concurrent-complete".into();
            self.concurrent_destination_interpretation_conditional_on_zero_retry_log =
                interpretation(self.measured.statistics.as_ref().unwrap())?;
        }
        Ok(())
    }
    fn fail(&mut self, e: Failure) {
        self.complete = e.complete;
        self.correctness_pass = false;
        self.authoritative = false;
        self.classification = e.classification.into();
        self.failure = Some(e.detail);
        self.concurrent_destination_interpretation_conditional_on_zero_retry_log = "AMBIGUOUS";
    }
}
async fn execute(report: &mut FReport) -> Result<()> {
    if report.args.iterations != 128 || report.args.warmup_iterations != 32 {
        return Err(Failure::runtime("invalid-arguments", "F requires --iterations 128 --warmup-iterations 32 for CLI compatibility; actual F workload is 16 measured cycles and 4 non-classifying warmup cycles, 14 helper calls per cycle"));
    }
    validate_schedule(&report.warmup.schedule, false)?;
    validate_schedule(&report.measured.schedule, true)?;
    let bytes =
        std::fs::read(&report.args.config).map_err(|e| Failure::runtime("config-failed", e))?;
    report.config_sha256 = Some(sha(&bytes));
    let text = std::str::from_utf8(&bytes).map_err(|e| Failure::runtime("config-failed", e))?;
    let config: Config = toml::from_str(text).map_err(|e| Failure::runtime("config-failed", e))?;
    config.validate().map_err(Failure::authority)?;
    let a = &mut report.authority;
    a.base.direct_io_requested = !config.storage.no_direct;
    a.base.packed_storage =
        Some(config.storage.packed_blob.is_some() || config.storage.packed_manifest.is_some());
    a.base.source_data_dir = Some(config.model.data_dir.clone());
    validate_geometry(&config)?;
    a.base.exact_geometry = true;
    if !a.base.linux
        || report.args.expected_adapter_name != "NVIDIA L4"
        || !a.base.direct_io_requested
        || a.base.packed_storage != Some(false)
    {
        return Err(Failure::authority(
            "requires Linux, exact NVIDIA L4 Vulkan, O_DIRECT, unpacked full-file Qwen geometry",
        ));
    }
    let storage = NvmeStorage::new(StorageConfig {
        base_path: config.model.data_dir,
        expert_size: FULL,
        block_align: ALIGN,
        use_direct_io: true,
        num_experts_per_layer: Some(128),
    })
    .map_err(|e| Failure::runtime("source-failed", e))?;
    if storage.is_packed() {
        return Err(Failure::authority("packed storage forbidden"));
    }
    a.fd_cache_capacity = storage.max_open_files();
    if a.fd_cache_capacity < a.preproof_universe_size {
        return Err(Failure::authority(
            "fd cache cannot retain the complete deterministic universe",
        ));
    }
    let preproof = storage.preprove_source_upload_fds(&a.preproof_ordered_expert_ids);
    a.preproof = storage.source_upload_fd_proof_snapshot();
    preproof.map_err(|e| Failure::authority(format!("fd preproof: {e}")))?;
    if a.preproof.source_upload_fd_proof_requests != a.preproof_universe_size as u64
        || a.preproof.source_upload_fd_proof_misses != a.preproof_universe_size as u64
        || a.preproof.source_upload_fd_proof_hits != 0
        || a.preproof.source_upload_fd_proof_failures != 0
    {
        return Err(Failure::authority(
            "fresh universe preproof counters do not reconcile",
        ));
    }
    a.fd_preproof_completed = true;
    storage.reset_source_upload_fd_proof_telemetry();
    a.after_preproof_telemetry_reset = storage.source_upload_fd_proof_snapshot();
    if a.after_preproof_telemetry_reset != SourceUploadFdProofSnapshot::default() {
        return Err(Failure::authority("preproof telemetry reset failed"));
    }
    let gpu = Gpu::with_upload_capacity(&mut a.base, MAPPED_ARENA).await?;
    f_phase(&mut report.warmup, &gpu, &storage).await?;
    if !report.warmup.successful() {
        return Err(Failure::authority("warmup evidence did not reconcile"));
    }
    storage.reset_source_upload_fd_proof_telemetry();
    report.authority.after_warmup_telemetry_reset = storage.source_upload_fd_proof_snapshot();
    if report.authority.after_warmup_telemetry_reset != SourceUploadFdProofSnapshot::default() {
        return Err(Failure::authority("warmup telemetry reset failed"));
    }
    f_phase(&mut report.measured, &gpu, &storage).await?;
    gpu.check()?;
    report.classify()?;
    Ok(())
}
pub(crate) async fn run_command(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.report_out)?;
    let mut report = FReport::new(args).map_err(|e| io::Error::other(e.detail))?;
    match std::panic::AssertUnwindSafe(execute(&mut report))
        .catch_unwind()
        .await
    {
        Ok(Ok(())) => {}
        Ok(Err(e)) => report.fail(e),
        Err(p) => {
            let detail = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "non-string panic".into());
            report.fail(Failure::runtime(
                "runtime-failed",
                format!("F diagnostic panic: {detail}"),
            ));
        }
    }
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    if report.complete {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "{}: {}",
            report.classification,
            report.failure.as_deref().unwrap_or("incomplete")
        ))
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args() -> Args {
        Args {
            config: "unused.toml".into(),
            expected_adapter_name: "NVIDIA L4".into(),
            warmup_iterations: 32,
            iterations: 128,
            report_out: "unused.json".into(),
        }
    }
    fn raw_with(measured: bool, mut ns: impl FnMut(&Call) -> u64) -> Vec<Raw> {
        phase_schedule(measured)
            .unwrap()
            .into_iter()
            .map(|c| Raw {
                ns: ns(&c),
                returned_bytes: (c.width * FULL) as u64,
                call: c,
            })
            .collect()
    }
    fn fixture_phase(measured: bool) -> FPhase {
        let mut p = FPhase::new(measured).unwrap();
        let n = p.expected.slots_per_arm;
        let sets = p.expected.calls_per_arm;
        p.raw_samples = raw_with(measured, |c| {
            1_000_000
                + (c.cycle * 71 + c.position * 19) as u64
                + if c.arm == FArm::MC { 60_000 } else { 0 }
        });
        p.matched_pairs = reconstruct(&p.raw_samples, measured).unwrap();
        p.statistics = Some(statistics(&p.matched_pairs, measured).unwrap());
        for pair in &p.matched_pairs {
            let hashes: Vec<_> = pair
                .ordered_expert_ids
                .iter()
                .map(|&id| SourceHash {
                    expert_id: id,
                    full_source_bytes: FULL as u64,
                    payload_bytes: PAYLOAD as u64,
                    source: sha(&id.to_le_bytes()),
                    payload: sha(&id.to_le_bytes()),
                    gpu: sha(&id.to_le_bytes()),
                    epoch: true,
                })
                .collect();
            p.verification.push(VerifiedPair {
                family: pair.family,
                role: pair.role,
                hc_call_index: pair.hc_call_index,
                mc_call_index: pair.mc_call_index,
                exact_bytes_equal: true,
                hc: hashes.clone(),
                mc: hashes,
            });
        }
        for (cycle, window) in p.schedule.chunks_exact(14).enumerate() {
            p.cycles.push(CycleIsolation {
                cycle,
                first_call: window[0].global_index,
                last_call: window[13].global_index,
                prepared_host_sets: 7,
                prepared_mapped_sets: 7,
                prepared_fd_checks: 70,
                completed_helper_calls: 14,
                source_only_window_complete: true,
            });
        }
        p.destinations = p
            .schedule
            .iter()
            .map(|c| {
                destination_evidence(
                    c,
                    0x1000_0000 + (c.global_index % 14) * 0x0200_0000,
                    if c.arm == FArm::MC {
                        MAPPED_ARENA
                    } else {
                        HOST_ARENA
                    },
                )
                .unwrap()
            })
            .collect();
        for fa in FArm::ALL {
            let arm = p.arms.get_mut(&fa).unwrap();
            arm.source_sets_attempted = sets;
            arm.concurrent_helper_calls = sets;
            arm.completed_source_sets = p
                .schedule
                .iter()
                .filter(|c| c.arm == fa)
                .map(Call::source_set)
                .collect();
            arm.source_schedule = schedule_evidence(&arm.completed_source_sets).unwrap();
            let a = &mut arm.evidence;
            a.ops_attempted = n;
            a.source_read_attempts = n;
            a.source_read_ops = n;
            a.full_source_bytes = n * FULL as u64;
            a.payload_ops = n;
            a.payload_bytes = n * PAYLOAD as u64;
            a.upload_ops = n;
            a.gpu_copied_bytes = n * PAYLOAD as u64;
            a.epoch_bytes = n * 4;
            a.gpu_completed_ops = n;
            a.verified_ops = n;
            a.verification_readback_bytes = n * SLOT as u64;
            a.verification_destination_reset_ops = n;
            a.verification_destination_reset_bytes = n * SLOT as u64;
            a.pointers.observations = n;
            a.pointers.aligned = n;
            a.fd_evidence = FdEvidence {
                checks: n,
                direct_observed: n,
                full_file_length_observed: n,
                ..FdEvidence::default()
            };
            a.times.source_direct_read_ns = p
                .raw_samples
                .iter()
                .filter(|r| r.call.arm == fa)
                .map(|r| r.ns)
                .sum();
            if fa == FArm::MC {
                a.map_attempts = sets;
                a.maps_completed = sets;
                a.unmaps = sets;
                a.pointers.gpu_offset_checks = n;
                a.explicit_copy_buffer_bytes = n * PAYLOAD as u64;
            } else {
                a.cpu_payload_copy_bytes = n * PAYLOAD as u64;
            }

            a.fd_evidence.flags_counts.insert(0, n);
            let mut offsets = Vec::new();
            for e in p.destinations.iter().filter(|e| e.arm == fa) {
                let width = e.source_bytes / FULL;
                *a.pointers
                    .mapping_base_mod_4096_counts
                    .entry(e.base_address % ALIGN)
                    .or_default() += width as u64;
                offsets.extend((0..width).map(|j| e.aligned_offset + j * FULL));
            }
            a.pointers.aligned_offset_min = offsets.iter().min().copied();
            a.pointers.aligned_offset_max = offsets.iter().max().copied();
            p.witnesses.insert(
                fa,
                Witnesses {
                    full_source_sha256: sha(b"source"),
                    bare_payload_sha256: sha(b"payload"),
                    gpu_destination_payload_sha256: sha(b"payload"),
                },
            );
        }
        p.fd_proof = SourceUploadFdProofSnapshot {
            source_upload_fd_proof_requests: n * 2,
            source_upload_fd_proof_hits: n * 2,
            ..Default::default()
        };
        assert!(p.successful());
        p
    }
    fn fixture() -> FReport {
        let mut r = FReport::new(args()).unwrap();
        let a = &mut r.authority;
        a.base.linux = true;
        a.base.adapter_authoritative = true;
        a.base.direct_io_requested = true;
        a.base.packed_storage = Some(false);
        a.base.exact_geometry = true;
        a.base.adapter_name = Some("NVIDIA L4".into());
        a.base.adapter_backend = Some("Vulkan".into());
        a.base.adapter_device_type = Some("DiscreteGpu".into());
        a.fd_cache_capacity = 256;
        a.fd_preproof_completed = true;
        a.preproof = SourceUploadFdProofSnapshot {
            source_upload_fd_proof_requests: 256,
            source_upload_fd_proof_misses: 256,
            ..Default::default()
        };
        r.warmup = fixture_phase(false);
        r.measured = fixture_phase(true);
        r.classify().unwrap();
        assert!(r.authoritative);
        r
    }
    fn s_with(mut f: impl FnMut(&Call) -> u64) -> Statistics {
        statistics(
            &reconstruct(
                &raw_with(true, |c| if c.arm == FArm::HC { 1_000_000 } else { f(c) }),
                true,
            )
            .unwrap(),
            true,
        )
        .unwrap()
    }
    #[test]
    fn hma1c_f_exact_universe_source_formula_and_e_logical_units() {
        let u = universe();
        assert_eq!(u.len(), 256);
        assert_eq!(u[0], 0);
        assert_eq!(u[255], 6143);
        assert_eq!(u.iter().copied().collect::<BTreeSet<_>>().len(), 256);
        for (i, id) in u.iter().enumerate() {
            assert_eq!(*id as usize, i * 6143 / 255);
        }
        let all = schedule().unwrap();
        let mut logical = BTreeSet::new();
        for c in all {
            let r = if c.measured { c.cycle / 4 } else { 0 };
            let g = c.cycle % 4;
            let w = (c.position + c.cycle) % 7;
            assert_eq!(
                (c.role, c.g, c.width, c.family),
                (r, g, w + 2, if c.measured { 0 } else { 28 } + 7 * g + w)
            );
            let expected: Vec<_> = (0..w + 2)
                .map(|j| ((17 * c.family + 64 * r + 13 * j) % 256) * 6143 / 255)
                .map(|n| n as u32)
                .collect();
            assert_eq!(c.ordered_expert_ids, expected);
            assert_eq!(expected.iter().collect::<BTreeSet<_>>().len(), w + 2);
            if c.measured {
                logical.insert((c.family, c.role));
            }
        }
        assert_eq!(
            logical,
            (0..28).flat_map(|f| (0..4).map(move |r| (f, r))).collect()
        );
    }
    #[test]
    fn hma1c_f_measured_balances_all_frozen_partitions() {
        let calls = phase_schedule(true).unwrap();
        assert_eq!(calls.len(), 224);
        let pairs: Vec<_> = calls.iter().filter(|c| c.pass == 0).collect();
        assert_eq!(pairs.len(), 112);
        assert_eq!(pairs.iter().filter(|p| p.hc_first).count(), 56);
        for (count, key, expected) in [(7, 0, 8), (4, 1, 14), (4, 2, 14), (7, 3, 8)] {
            for k in 0..count {
                let rows:Vec<_>=pairs.iter().filter(|p|match key {0=>p.width-2,1=>p.role,2=>p.g,_=>p.position}==k).collect();
                assert_eq!(rows.len(), 2 * expected);
                assert_eq!(rows.iter().filter(|p| p.hc_first).count(), expected);
            }
        }
        for a in FArm::ALL {
            let rows: Vec<_> = calls.iter().filter(|c| c.arm == a).collect();
            assert_eq!(rows.len(), 112);
            assert_eq!(rows.iter().map(|c| c.width).sum::<usize>(), 560);
        }
    }
    #[test]
    fn hma1c_f_warmup_counts_and_exact_pair_distance() {
        for measured in [false, true] {
            let calls = phase_schedule(measured).unwrap();
            let e = expected(measured);
            assert_eq!(calls.len(), e.helper_calls as usize);
            assert_eq!(calls.len() / 2, e.matched_pairs as usize);
            for window in calls.chunks_exact(14) {
                for p in 0..7 {
                    let (a, b) = (&window[p], &window[p + 7]);
                    assert_eq!(b.global_index - a.global_index, 7);
                    assert_eq!(a.ordered_expert_ids, b.ordered_expert_ids);
                    assert_ne!(a.arm, b.arm);
                    assert_eq!(a.position, b.position);
                    assert_eq!(a.cycle, b.cycle);
                    assert_eq!(a.pass, 0);
                    assert_eq!(b.pass, 1);
                }
            }
        }
    }
    #[test]
    fn hma1c_f_previous_two_overlap_complete_stream_and_phase_boundary() {
        let all = schedule().unwrap();
        assert_eq!(all.len(), 280);
        let mut comparisons = 0;
        for (i, c) in all.iter().enumerate() {
            assert_eq!(c.previous_two.len(), i.min(2));
            for d in 1..=i.min(2) {
                assert!(c
                    .ordered_expert_ids
                    .iter()
                    .all(|id| !all[i - d].ordered_expert_ids.contains(id)));
                assert_eq!(c.previous_two[d - 1].global_index, i - d);
                assert!(c.previous_two[d - 1].shared_expert_ids.is_empty());
                comparisons += 1;
            }
        }
        assert_eq!(comparisons, 557);
        assert!(!all[55].measured);
        assert!(all[56].measured);
        assert_eq!(
            all[56]
                .previous_two
                .iter()
                .map(|p| p.global_index)
                .collect::<Vec<_>>(),
            vec![55, 54]
        );
        assert_eq!(all[57].previous_two[1].global_index, 55);
    }
    #[test]
    fn hma1c_f_exact_bytes_and_proof_totals() {
        for (measured, calls, slots, bytes, proof) in [
            (false, 28, 140, 372_162_560, 280),
            (true, 112, 560, 1_488_650_240, 1120),
        ] {
            let plan = phase_schedule(measured).unwrap();
            let e = expected(measured);
            for arm in FArm::ALL {
                let rows: Vec<_> = plan.iter().filter(|c| c.arm == arm).collect();
                assert_eq!(rows.len(), calls);
                assert_eq!(rows.iter().map(|c| c.width).sum::<usize>(), slots);
                assert_eq!(rows.iter().map(|c| c.width * FULL).sum::<usize>(), bytes);
            }
            assert_eq!(e.calls_per_arm, calls as u64);
            assert_eq!(e.slots_per_arm, slots as u64);
            assert_eq!(e.full_source_bytes_per_arm, bytes as u64);
            assert_eq!(e.proof_requests, proof);
            assert_eq!(e.proof_hits, proof);
            assert!(proof_hits_only(
                &SourceUploadFdProofSnapshot {
                    source_upload_fd_proof_requests: proof,
                    source_upload_fd_proof_hits: proof,
                    ..Default::default()
                },
                measured
            ));
        }
    }
    #[test]
    fn hma1c_f_schedule_hashes_match_independent_python_pins() {
        let h = schedule_hashes(&phase_schedule(false).unwrap());
        assert_eq!(
            h.universe_sha256,
            "7a993987f13a94c6b3cc3f75a38dde55919a820d68b57e28c5f9ac599c1565b3"
        );
        assert_eq!(
            h.ordered_source_ids_sha256,
            "990fbc595c6622e09b435d3bcf3cbaf0572589872f66e3c4bc626c39f675c8b7"
        );
        assert_eq!(
            h.execution_order_sha256,
            "471c95af5e72b33f736987e0c3fb78b162ada53b13bb1ecae02e44095f2bfa43"
        );
        assert_eq!(
            h.complete_schedule_sha256,
            "bf609ec4e927eb06e0d1c75f9f1f9aca2e6d1fc511fdd4a1368e392f5f6c4d94"
        );
        let h = schedule_hashes(&phase_schedule(true).unwrap());
        assert_eq!(
            h.universe_sha256,
            "7a993987f13a94c6b3cc3f75a38dde55919a820d68b57e28c5f9ac599c1565b3"
        );
        assert_eq!(
            h.ordered_source_ids_sha256,
            "fb0592379950947e39723bc23472ea1f61a2aefd5c597216f5de181eb20ead6b"
        );
        assert_eq!(
            h.execution_order_sha256,
            "c1abfbf7d7f571a7043740185fccb33d1caa62d4f911e8d42c504b7497677a31"
        );
        assert_eq!(
            h.complete_schedule_sha256,
            "fa4175dc220b9cba8d7888076667bde05820248116cb3a7346547d7674890501"
        );
    }
    #[test]
    fn hma1c_f_exact_primary_stats_reconstructed_from_raw_samples() {
        let raw = raw_with(true, |c| {
            1_000_000 + (c.global_index * 991 + c.family * 7 + c.role) as u64
        });
        let pairs = reconstruct(&raw, true).unwrap();
        let s = statistics(&pairs, true).unwrap();
        let hc: u64 = raw
            .iter()
            .filter(|r| r.call.arm == FArm::HC)
            .map(|r| r.ns)
            .sum();
        let mc: u64 = raw
            .iter()
            .filter(|r| r.call.arm == FArm::MC)
            .map(|r| r.ns)
            .sum();
        assert_eq!(
            (
                s.primary.sample_count,
                s.primary.hc_total_ns,
                s.primary.mc_total_ns
            ),
            (112, hc, mc)
        );
        assert_eq!(s.primary.cde_ns, mc as i128 - hc as i128);
        let mut deltas: Vec<_> = pairs
            .iter()
            .map(|p| p.mc_ns as i128 - p.hc_ns as i128)
            .collect();
        deltas.sort();
        assert_eq!(
            s.primary.paired_median_mc_minus_hc,
            SignedFraction {
                numerator: deltas[55] + deltas[56],
                denominator: 2
            }
        );
        let mut ratios: Vec<_> = pairs
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.mc_ns as u128, p.hc_ns as u128))
            .collect();
        ratios.sort_by(|a, b| (a.1 * b.2).cmp(&(b.1 * a.2)).then(a.0.cmp(&b.0)));
        assert_eq!(
            s.primary.median_mc_over_hc.lower,
            Ratio {
                numerator: ratios[55].1,
                denominator: ratios[55].2
            }
        );
        assert_eq!(
            s.primary.median_mc_over_hc.upper,
            Ratio {
                numerator: ratios[56].1,
                denominator: ratios[56].2
            }
        );
        assert_eq!(
            s.primary.positive,
            pairs.iter().filter(|p| p.cde_ns > 0).count() as u64
        );
        assert_eq!(
            s.primary.negative,
            pairs.iter().filter(|p| p.cde_ns < 0).count() as u64
        );
    }
    #[test]
    fn hma1c_f_classification_exact_thresholds_no_rounding() {
        for (mc, label) in [
            (1_050_000, "STRONG_POSITIVE_CDE"),
            (1_049_999, "MATERIAL_POSITIVE_CDE"),
            (1_030_000, "MATERIAL_POSITIVE_CDE"),
            (1_029_999, "AMBIGUOUS"),
            (970_000, "NEGATIVE_MATERIAL_CDE"),
            (970_001, "AMBIGUOUS"),
            (1_000_000, "EVIDENCE_AGAINST_MATERIAL_POSITIVE_CDE"),
            (990_000, "EVIDENCE_AGAINST_MATERIAL_POSITIVE_CDE"),
            (989_999, "AMBIGUOUS"),
            (1_010_000, "AMBIGUOUS"),
        ] {
            assert_eq!(interpretation(&s_with(|_| mc)).unwrap(), label, "{mc}");
        }
    }
    #[test]
    fn hma1c_f_both_order_majority_median_and_width_consistency() {
        for negative in [false, true] {
            for gate in 0..5 {
                let mut s = s_with(|_| if negative { 940_000 } else { 1_060_000 });
                match gate {
                    0 => s.primary.paired_median_mc_minus_hc.numerator = 0,
                    1 => {
                        if negative {
                            s.primary.negative = 56;
                        } else {
                            s.primary.positive = 56;
                        }
                    }
                    2 => {
                        for k in 2..5 {
                            s.width.get_mut(&k).unwrap().cde_ns = 0;
                        }
                    }
                    3 => s.arm_order.get_mut("HC-first").unwrap().cde_ns = 0,
                    _ => s.arm_order.get_mut("MC-first").unwrap().cde_ns = 0,
                }
                assert_eq!(interpretation(&s).unwrap(), "AMBIGUOUS");
            }
        }
        // Exactly 57 signs, five widths, and strictly signed medians/orders pass.
        let mut s = s_with(|_| 1_060_000);
        s.primary.positive = 57;
        for k in 2..4 {
            s.width.get_mut(&k).unwrap().cde_ns = 0;
        }
        assert_eq!(interpretation(&s).unwrap(), "STRONG_POSITIVE_CDE");
    }
    #[test]
    fn hma1c_f_against_inclusive_ratio_bounds_and_percent_bounds() {
        let baseline = s_with(|_| 1_000_000);
        for (num, expect) in [(9900, true), (10100, true), (9899, false), (10101, false)] {
            let mut s = baseline.clone();
            s.primary.median_mc_over_hc = RatioMedian {
                lower: Ratio {
                    numerator: num,
                    denominator: 10000,
                },
                upper: Ratio {
                    numerator: num,
                    denominator: 10000,
                },
            };
            assert_eq!(
                interpretation(&s).unwrap(),
                if expect {
                    "EVIDENCE_AGAINST_MATERIAL_POSITIVE_CDE"
                } else {
                    "AMBIGUOUS"
                }
            );
        }
        for (num, expect) in [
            (-112_000_000, true),
            (112_000_000, true),
            (-112_000_001, false),
            (112_000_001, false),
        ] {
            let mut s = baseline.clone();
            s.primary.cde_percent_exact.numerator = num;
            assert_eq!(
                interpretation(&s).unwrap(),
                if expect {
                    "EVIDENCE_AGAINST_MATERIAL_POSITIVE_CDE"
                } else {
                    "AMBIGUOUS"
                }
            );
        }
        let mut s = baseline;
        for k in 2..7 {
            s.width.get_mut(&k).unwrap().cde_ns = 1;
        }
        assert_eq!(interpretation(&s).unwrap(), "AMBIGUOUS");
    }
    #[test]
    fn hma1c_f_zero_missing_duplicate_bytes_and_overflow_rejected() {
        let mut raw = raw_with(true, |_| 100);
        raw[0].ns = 0;
        assert!(reconstruct(&raw, true).is_err());
        let mut raw = raw_with(true, |_| 100);
        raw.pop();
        assert!(reconstruct(&raw, true).is_err());
        let mut raw = raw_with(true, |_| 100);
        raw[7] = raw[0].clone();
        assert!(reconstruct(&raw, true).is_err());
        let mut raw = raw_with(true, |_| 100);
        raw[0].returned_bytes -= 1;
        assert!(reconstruct(&raw, true).is_err());
        let pairs = reconstruct(&raw_with(true, |_| u64::MAX), true).unwrap();
        assert!(statistics(&pairs, true).is_err());
    }
    #[test]
    fn hma1c_f_schedule_corruption_rejected_field_by_field() {
        let mutations: Vec<fn(&mut Call)> = vec![
            |c| c.global_index += 1,
            |c| c.cycle += 1,
            |c| c.pass += 1,
            |c| c.position += 1,
            |c| c.family += 1,
            |c| c.role += 1,
            |c| c.g += 1,
            |c| c.width += 1,
            |c| c.measured = false,
            |c| c.hc_first = !c.hc_first,
            |c| {
                c.arm = if c.arm == FArm::HC {
                    FArm::MC
                } else {
                    FArm::HC
                }
            },
            |c| c.ordered_expert_ids[0] ^= 1,
            |c| c.previous_two[0].global_index += 1,
            |c| c.previous_two[0].shared_expert_ids.push(0),
        ];
        for mutate in mutations {
            let mut calls = phase_schedule(true).unwrap();
            mutate(&mut calls[0]);
            assert!(validate_schedule(&calls, true).is_err());
        }
    }
    #[test]
    fn hma1c_f_all_accounting_fields_fail_closed() {
        let mutations: Vec<fn(&mut FPhase)> = vec![
            |p| p.expected.cycles += 1,
            |p| p.expected.matched_pairs += 1,
            |p| p.expected.helper_calls += 1,
            |p| p.expected.calls_per_arm += 1,
            |p| p.expected.slots_per_arm += 1,
            |p| p.expected.full_source_bytes_per_arm += 1,
            |p| p.expected.payload_bytes_per_arm += 1,
            |p| p.expected.proof_requests += 1,
            |p| p.expected.proof_hits += 1,
            |p| p.expected.proof_misses += 1,
            |p| p.expected.proof_failures += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.ops_attempted += 1,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .source_read_attempts += 1
            },
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.source_read_ops += 1,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .full_source_bytes += 1
            },
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.payload_ops += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.payload_bytes += 1,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .cpu_payload_copy_bytes += 1
            },
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.upload_ops += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.gpu_copied_bytes += 1,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .explicit_copy_buffer_bytes += 1
            },
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.epoch_bytes += 1,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .gpu_completed_ops += 1
            },
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.verified_ops += 1,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .verification_readback_bytes += 1
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .verification_destination_reset_ops += 1
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .verification_destination_reset_bytes += 1
            },
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.map_attempts += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.maps_completed += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.unmaps += 1,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .alignment_failures += 1
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .mapped_direct_io_rejections += 1
            },
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.map_failures += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.source_failures += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.gpu_failures += 1,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .accounting_failures += 1
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .exact_read_length_failures += 1
            },
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.fallback_reads += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().source_sets_attempted += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().serial_helper_calls += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().concurrent_helper_calls += 1,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .pointers
                    .observations += 1
            },
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.pointers.aligned += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().evidence.pointers.invalid += 1,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .pointers
                    .gpu_offset_checks += 1
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .pointers
                    .gpu_offset_failures += 1
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .fd_evidence
                    .checks += 1
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .fd_evidence
                    .direct_observed += 1
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .fd_evidence
                    .full_file_length_observed += 1
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .fd_evidence
                    .failures += 1
            },
            |p| p.fd_proof.source_upload_fd_proof_requests += 1,
            |p| p.fd_proof.source_upload_fd_proof_hits += 1,
            |p| p.fd_proof.source_upload_fd_proof_misses += 1,
            |p| p.fd_proof.source_upload_fd_proof_failures += 1,
            |p| p.cycles[0].cycle += 1,
            |p| p.cycles[0].first_call += 1,
            |p| p.cycles[0].last_call += 1,
            |p| p.cycles[0].prepared_host_sets += 1,
            |p| p.cycles[0].prepared_mapped_sets += 1,
            |p| p.cycles[0].prepared_fd_checks += 1,
            |p| p.cycles[0].completed_helper_calls += 1,
            |p| p.cycles[0].source_only_window_complete = false,
        ];
        for (i, mutate) in mutations.into_iter().enumerate() {
            let mut p = fixture_phase(true);
            mutate(&mut p);
            assert!(!p.successful(), "mutation {i}");
        }
    }
    #[test]
    fn hma1c_f_hash_stats_raw_pair_proof_and_authority_corruption() {
        let mutations: Vec<fn(&mut FReport)> = vec![
            |r| r.measured.raw_samples[0].ns += 1,
            |r| r.measured.matched_pairs[0].helper_call_distance = 6,
            |r| r.measured.matched_pairs[0].hc_call_index += 1,
            |r| r.measured.matched_pairs[0].ordered_expert_ids[0] ^= 1,
            |r| r.measured.matched_pairs[0].cde_ns += 1,
            |r| {
                r.measured.source_errors.push(FailedRead {
                    call: r.measured.schedule[0].clone(),
                    ns: 1,
                    returned_bytes: None,
                    errno: None,
                    error: "failure".into(),
                })
            },
            |r| r.measured.statistics.as_mut().unwrap().primary.cde_ns += 1,
            |r| {
                r.measured
                    .statistics
                    .as_mut()
                    .unwrap()
                    .width
                    .get_mut(&2)
                    .unwrap()
                    .hc_total_ns += 1
            },
            |r| {
                r.measured
                    .statistics
                    .as_mut()
                    .unwrap()
                    .source_role
                    .get_mut(&0)
                    .unwrap()
                    .mc_total_ns += 1
            },
            |r| {
                r.measured
                    .statistics
                    .as_mut()
                    .unwrap()
                    .g
                    .get_mut(&0)
                    .unwrap()
                    .positive += 1
            },
            |r| {
                r.measured
                    .statistics
                    .as_mut()
                    .unwrap()
                    .cycle_position
                    .get_mut(&0)
                    .unwrap()
                    .equal += 1
            },
            |r| {
                r.measured
                    .statistics
                    .as_mut()
                    .unwrap()
                    .arm_order
                    .get_mut("HC-first")
                    .unwrap()
                    .sample_count += 1
            },
            |r| r.measured.schedule_hashes.complete_schedule_sha256 = sha(b"bad"),
            |r| r.measured.verification[0].exact_bytes_equal = false,
            |r| r.measured.verification[0].mc[0].source = sha(b"bad"),
            |r| r.measured.verification[0].mc[0].payload = sha(b"bad"),
            |r| r.measured.verification[0].mc[0].gpu = sha(b"bad"),
            |r| r.measured.verification[0].mc[0].epoch = false,
            |r| r.measured.verification[0].mc[0].expert_id ^= 1,
            |r| r.measured.verification[0].mc[0].full_source_bytes += 1,
            |r| r.measured.verification[0].mc[0].payload_bytes += 1,
            |r| {
                r.measured
                    .witnesses
                    .get_mut(&FArm::HC)
                    .unwrap()
                    .bare_payload_sha256 = sha(b"bad")
            },
            |r| r.warmup.fd_proof.source_upload_fd_proof_misses = 1,
            |r| r.measured.fd_proof.source_upload_fd_proof_failures = 1,
            |r| r.authority.fd_cache_capacity = 255,
            |r| r.authority.preproof.source_upload_fd_proof_hits = 1,
            |r| {
                r.authority
                    .after_preproof_telemetry_reset
                    .source_upload_fd_proof_requests = 1
            },
            |r| {
                r.authority
                    .after_warmup_telemetry_reset
                    .source_upload_fd_proof_requests = 1
            },
            |r| r.authority.base.adapter_name = Some("bad".into()),
            |r| r.authority.base.direct_io_requested = false,
            |r| {
                r.warmup.schedule[55].previous_two[1]
                    .shared_expert_ids
                    .push(0)
            },
        ];
        for (i, mutate) in mutations.into_iter().enumerate() {
            let mut r = fixture();
            mutate(&mut r);
            r.classify().unwrap();
            assert!(!r.correctness_pass && !r.authoritative, "mutation {i}");
            assert_eq!(
                r.concurrent_destination_interpretation_conditional_on_zero_retry_log,
                "AMBIGUOUS"
            );
        }
    }
    #[test]
    fn hma1c_f_window_is_exactly_fourteen_helpers_with_no_caller_work() {
        let src = include_str!("gpu_native_mapped_memory_local_paired_concurrent.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let timed = src
            .split("async fn timed_cycle(")
            .nth(1)
            .unwrap()
            .split("// Views")
            .next()
            .unwrap();
        let body = timed
            .split("let start = Instant::now();")
            .nth(1)
            .unwrap()
            .split("let duration = start.elapsed();")
            .next()
            .unwrap()
            .split_whitespace()
            .collect::<String>();
        assert_eq!(
            body,
            "letread=storage.read_experts_batch_into_aligned_slices(ids,destinations).await;"
        );
        assert_eq!(
            src.matches(".read_experts_batch_into_aligned_slices(")
                .count(),
            1
        );
        assert!(!src.contains("read_experts_serial_into_aligned_slices"));
        for forbidden in [
            "Vec::",
            ".push(",
            ".collect(",
            "gpu.",
            "map_async",
            "unmap",
            "poll(",
            "fd_evidence",
            "source_upload_fd",
            "verify(",
            "Sha256",
            "format!(",
        ] {
            assert!(!timed.contains(forbidden), "{forbidden}");
        }
        let cycle = src
            .split("async fn run_cycle(")
            .nth(1)
            .unwrap()
            .split("async fn f_phase(")
            .next()
            .unwrap();
        let (before, after) = cycle
            .split_once("let results = timed_cycle(storage, &mut prepared).await;")
            .unwrap();
        for needle in [
            "AlignedBuffer::new",
            "create_buffer",
            "gpu.map(",
            "get_mapped_range_mut",
            "arena_slices(",
            "observe_slices(",
            "begin_set(",
            "PreparedCall",
        ] {
            assert!(before.contains(needle), "{needle}");
        }
        for needle in [
            "streams[0].source(",
            "prepare_destination(",
            ".write_buffer(",
            ".unmap()",
            "copy_buffer_to_buffer(",
            "verify(",
        ] {
            assert!(after.contains(needle), "{needle}");
            assert!(!before.contains(needle));
        }
    }
    #[test]
    fn hma1c_f_destination_ranges_alignment_distinctness_corruption() {
        let mutations: Vec<fn(&mut FPhase)> = vec![
            |p| p.destinations[0].global_index += 1,
            |p| p.destinations[0].arm = FArm::MC,
            |p| p.destinations[0].base_address += 1,
            |p| p.destinations[0].capacity_bytes -= 1,
            |p| p.destinations[0].aligned_offset += 1,
            |p| p.destinations[0].source_bytes -= 1,
            |p| p.destinations[7].gpu_payload_offsets[0] += 4,
            |p| p.destinations[1].base_address = p.destinations[0].base_address,
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .pointers
                    .aligned_offset_max = Some(0)
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .pointers
                    .mapping_base_mod_4096_counts
                    .clear()
            },
            |p| {
                p.arms
                    .get_mut(&FArm::MC)
                    .unwrap()
                    .evidence
                    .fd_evidence
                    .flags_counts
                    .clear()
            },
        ];
        for (i, mutate) in mutations.into_iter().enumerate() {
            let mut p = fixture_phase(true);
            mutate(&mut p);
            assert!(!p.successful(), "destination corruption {i}");
        }
    }
    #[test]
    fn hma1c_f_redundant_schedule_metadata_and_host_offset_counters_reject_corruption() {
        let mutations: Vec<fn(&mut FPhase)> = vec![
            |p| p.arms.get_mut(&FArm::HC).unwrap().completed_source_sets[0].round += 1,
            |p| p.arms.get_mut(&FArm::MC).unwrap().completed_source_sets[0].set_index += 1,
            |p| {
                p.arms.get_mut(&FArm::MC).unwrap().completed_source_sets[0].execution_order =
                    ExecutionOrder::TreatmentFirst
            },
            |p| {
                p.arms
                    .get_mut(&FArm::HC)
                    .unwrap()
                    .evidence
                    .pointers
                    .gpu_offset_checks = 1
            },
        ];
        for (i, mutate) in mutations.into_iter().enumerate() {
            let mut p = fixture_phase(true);
            mutate(&mut p);
            assert!(!p.successful(), "redundant field corruption {i}");
        }
    }
    #[test]
    fn hma1c_f_synthetic_report_and_retry_contract() {
        let r = fixture();
        assert_eq!(r.performance_authority, "PENDING_EXTERNAL_RETRY_LOG_AUDIT");
        assert!(r
            .retry_evidence_contract
            .contains("transient I/O error; retrying"));
        if let Some(path) = std::env::var_os("MER_HMA1CF_SYNTHETIC_JSON") {
            std::fs::write(
                path,
                serde_json::to_vec_pretty(
                    &serde_json::json!({"fixture":"portable-synthetic-not-hardware","report":r}),
                )
                .unwrap(),
            )
            .unwrap();
        }
        assert_eq!(
            "clean log".matches("transient I/O error; retrying").count(),
            0
        );
        assert_eq!(
            "transient I/O error; retrying\ntransient I/O error; retrying"
                .matches("transient I/O error; retrying")
                .count(),
            2
        );
    }
    #[tokio::test]
    async fn hma1c_f_invalid_cli_preserves_exclusive_report_without_gpu() {
        let dir = std::env::temp_dir().join(format!(
            "mer-hma1cf-cli-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let mut a = args();
        a.iterations = 16;
        a.report_out = dir.join("report.json");
        assert!(run_command(a.clone()).await.is_err());
        let bytes = std::fs::read(&a.report_out).unwrap();
        let j: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(j["schema"], F_SCHEMA);
        assert_eq!(j["classification"], "invalid-arguments");
        assert!(j["authority"]["adapter_name"].is_null());
        assert!(run_command(a).await.is_err());
        assert_eq!(std::fs::read(dir.join("report.json")).unwrap(), bytes);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
