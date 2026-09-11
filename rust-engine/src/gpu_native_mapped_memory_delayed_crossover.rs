// HMA-1C-E, issue #177 refinement 5628425605. Diagnostic only.
use super::*;

const E_SCHEMA: &str = "mer.gpu-native-mapped-memory-odirect-delayed-crossover.v1";
const MEASURED_COUNT: usize = 112;
const WARMUP_COUNT: usize = 28;
const E_UNIVERSE: usize = 256;
// Do not use Cell::index() for canonical source-role assignment: the retained
// C discriminants are HS,HC,MS,MC. These frozen E arrays are HS,MS,HC,MC.
const CANONICAL: [Cell; 4] = [Cell::HS, Cell::MS, Cell::HC, Cell::MC];
const E_WILLIAMS: [[Cell; 4]; 4] = [
    [Cell::HS, Cell::MS, Cell::MC, Cell::HC],
    [Cell::MS, Cell::HC, Cell::HS, Cell::MC],
    [Cell::HC, Cell::MC, Cell::MS, Cell::HS],
    [Cell::MC, Cell::HS, Cell::HC, Cell::MS],
];
const E_SCHEDULE_CONTRACT: &str = "Universe[i]=floor(i*6143/255). Measured q=0..3, family=0..27 ascending; w=family%7=K-2, g=family/7. Warmup families=28..55 ascending, q absent, g=(family/7)%4, assignment uses q=0. base=(family*17)%256; role_s[j]=universe[(base+64*s+13*j)%256]. phi=[0,2,3,1], x=w&3; exec=q XOR g XOR x; rot=q XOR phi[g] XOR phi[x]; class=4*rot+exec. Canonical cells HS,MS,HC,MC receive role (c+rot)%4. Williams sequences HS/MS/MC/HC; MS/HC/HS/MC; HC/MC/MS/HS; MC/HS/HC/MS. All 140 executed blocks must have zero expert overlap with their preceding two blocks. Exact measured family exposures are 28 executed blocks apart. CLI retains iterations=128/warmup_iterations=32; actual blocks=112/28. Hash metadata: [set_index,executed_block_index,family,q_or_4,width,g,exec,rot,class] u64 LE; source: width u32 LE then A/B/C/D IDs u32 LE; execution: four canonical cell codes u8 then four canonical-cell role codes u8. Ordered-source hash uses source; execution hash uses metadata+execution; complete hash uses metadata+source+execution. Overlap witnesses identify previous executed indices, nearest first. Predecessor descriptive strata use within-block START,HS,MS,HC,MC.";
const E_INTERPRETATION: &str = "Primary CDE=sum(MC)-sum(HC), denominator=sum(HC), from 112 exact (family,role) matched units. STRONG >=5%, MATERIAL >=3%, median MC-HC>0, >56 positive units, >=5 positive widths and all four raw-block exec, rotation and temporal strata positive. Negative material <=-3% with analogous negative consistency. AGAINST abs(CDE)<=1%, exact median MC/HC in inclusive [0.99,1.01], <5 positive widths, clean authority/correctness. Interaction is secondary with D thresholds and median/majority/reversal consistency over exact matched units, using block-level exec/rotation/temporal strata. Design classes, cell positions/predecessors, source family/role and warmup are descriptive. Never round thresholds; all classifications are conditional on complete external retry-log audit.";
const E_RETRY: &str = "Performance authority additionally requires zero exact occurrences of 'transient I/O error; retrying' in the complete external FIRST log. Any occurrence, missing/incomplete log, measured proof miss/failure, overlap, crossover, schedule/hash or accounting mismatch is non-authoritative and AMBIGUOUS. No diagnostic retry or fallback; this report does not audit its external log.";

fn canonical_index(cell: Cell) -> usize {
    match cell {
        Cell::HS => 0,
        Cell::MS => 1,
        Cell::HC => 2,
        Cell::MC => 3,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct OverlapWitness {
    executed_block_index: usize,
    shared_expert_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ESet {
    set_index: usize,
    executed_block_index: usize,
    family: usize,
    temporal_quartile: Option<usize>,
    width: usize,
    g: usize,
    execution_sequence_index: usize,
    source_rotation: usize,
    design_class: usize,
    role_ordered_expert_ids: [Vec<u32>; 4],
    execution_sequence: [Cell; 4],
    previous_two_blocks: Vec<OverlapWitness>,
}
impl ESet {
    fn role(&self, cell: Cell) -> usize {
        (canonical_index(cell) + self.source_rotation) % 4
    }
    fn position(&self, cell: Cell) -> usize {
        self.execution_sequence
            .iter()
            .position(|c| *c == cell)
            .unwrap()
    }
    fn b_set(&self, cell: Cell) -> SourceSet {
        let host = if cell.serial() { Cell::HS } else { Cell::HC };
        let mapped = if cell.serial() { Cell::MS } else { Cell::MC };
        SourceSet {
            set_index: self.set_index,
            round: self.temporal_quartile.unwrap_or(4),
            width: self.width,
            ordered_expert_ids: self.role_ordered_expert_ids[self.role(cell)].clone(),
            execution_order: if self.position(host) < self.position(mapped) {
                ExecutionOrder::ControlFirst
            } else {
                ExecutionOrder::TreatmentFirst
            },
        }
    }
    fn ids(&self) -> BTreeSet<u32> {
        self.role_ordered_expert_ids
            .iter()
            .flatten()
            .copied()
            .collect()
    }
    fn exposure(&self, cell: Cell) -> Exposure {
        let position = self.position(cell);
        Exposure {
            set_index: self.set_index,
            executed_block_index: self.executed_block_index,
            family: self.family,
            source_role: self.role(cell),
            width: self.width,
            temporal_quartile: self.temporal_quartile,
            execution_sequence_index: self.execution_sequence_index,
            source_rotation: self.source_rotation,
            design_class: self.design_class,
            cell,
            execution_position: position,
            predecessor_cell: position.checked_sub(1).map(|i| self.execution_sequence[i]),
            ordered_expert_ids: self.role_ordered_expert_ids[self.role(cell)].clone(),
        }
    }
}

fn all_schedule() -> Result<Vec<ESet>> {
    let universe = expert_sequence(E_UNIVERSE, NAMESPACE).map_err(Failure::accounting)?;
    let phi = [0, 2, 3, 1];
    let order = (28..56)
        .map(|family| (None, family))
        .chain((0..4).flat_map(|q| (0..28).map(move |family| (Some(q), family))));
    let mut sets: Vec<ESet> = Vec::with_capacity(140);
    for (executed_block_index, (quartile, family)) in order.enumerate() {
        let q = quartile.unwrap_or(0);
        let w = family % 7;
        let width = w + 2;
        let g = (family / 7) % 4;
        let x = w & 3;
        let exec = q ^ g ^ x;
        let rot = q ^ phi[g] ^ phi[x];
        let mut s = ESet {
            set_index: if quartile.is_some() {
                q * 28 + family
            } else {
                family - 28
            },
            executed_block_index,
            family,
            temporal_quartile: quartile,
            width,
            g,
            execution_sequence_index: exec,
            source_rotation: rot,
            design_class: 4 * rot + exec,
            role_ordered_expert_ids: std::array::from_fn(|role| {
                (0..width)
                    .map(|j| universe[(family * 17 + 64 * role + 13 * j) % 256])
                    .collect()
            }),
            execution_sequence: E_WILLIAMS[exec],
            previous_two_blocks: Vec::new(),
        };
        let ids = s.ids();
        if ids.len() != 4 * width {
            return Err(Failure::accounting("E within-block source overlap"));
        }
        for previous in sets.iter().rev().take(2) {
            let shared: Vec<_> = ids.intersection(&previous.ids()).copied().collect();
            if !shared.is_empty() {
                return Err(Failure::accounting("E previous-two-block overlap"));
            }
            s.previous_two_blocks.push(OverlapWitness {
                executed_block_index: previous.executed_block_index,
                shared_expert_ids: shared,
            });
        }
        sets.push(s);
    }
    prove_balance(&sets[28..])?;
    Ok(sets)
}

fn prove_balance(sets: &[ESet]) -> Result<()> {
    let fail = || Failure::accounting("E crossover/Williams balance failure");
    if sets.len() != 112 {
        return Err(fail());
    }
    for width in 2..=8 {
        let rows: Vec<_> = sets.iter().filter(|s| s.width == width).collect();
        if rows.len() != 16
            || rows.iter().map(|s| s.design_class).collect::<BTreeSet<_>>() != (0..16).collect()
        {
            return Err(fail());
        }
        for q in 0..4 {
            let group: Vec<_> = rows
                .iter()
                .filter(|s| s.temporal_quartile == Some(q))
                .collect();
            if group.len() != 4
                || group
                    .iter()
                    .map(|s| s.execution_sequence_index)
                    .collect::<BTreeSet<_>>()
                    != (0..4).collect()
                || group
                    .iter()
                    .map(|s| s.source_rotation)
                    .collect::<BTreeSet<_>>()
                    != (0..4).collect()
            {
                return Err(fail());
            }
        }
    }
    for family in 0..28 {
        let rows: Vec<_> = sets.iter().filter(|s| s.family == family).collect();
        if rows.len() != 4
            || rows
                .iter()
                .map(|s| s.execution_sequence_index)
                .collect::<BTreeSet<_>>()
                != (0..4).collect()
            || rows
                .iter()
                .map(|s| s.source_rotation)
                .collect::<BTreeSet<_>>()
                != (0..4).collect()
            || rows
                .windows(2)
                .any(|p| p[1].executed_block_index - p[0].executed_block_index != 28)
            || rows
                .iter()
                .any(|s| s.role_ordered_expert_ids != rows[0].role_ordered_expert_ids)
        {
            return Err(fail());
        }
        for cell in CANONICAL {
            if rows.iter().map(|s| s.role(cell)).collect::<BTreeSet<_>>() != (0..4).collect() {
                return Err(fail());
            }
        }
    }
    let mut positions = [[0u8; 4]; 4];
    let mut pairs = BTreeMap::new();
    for sequence in E_WILLIAMS {
        for (i, c) in sequence.iter().enumerate() {
            positions[canonical_index(*c)][i] += 1;
        }
        for p in sequence.windows(2) {
            *pairs.entry((p[0], p[1])).or_insert(0u8) += 1;
        }
    }
    if positions != [[1; 4]; 4]
        || pairs.len() != 12
        || pairs.iter().any(|((a, b), n)| a == b || *n != 1)
    {
        return Err(fail());
    }
    Ok(())
}

fn e_schedule(count: usize) -> Result<Vec<ESet>> {
    let all = all_schedule()?;
    match count {
        28 => Ok(all[..28].to_vec()),
        112 => Ok(all[28..].to_vec()),
        _ => Err(Failure::accounting(
            "E requires 28 warmup or 112 measured blocks",
        )),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct EPlan {
    block_count: u64,
    expert_slots_per_cell: u64,
    source_bytes_per_cell: u64,
    cell_source_schedules: BTreeMap<Cell, ScheduleEvidence>,
    universe_sha256: String,
    ordered_role_ids_sha256: String,
    execution_order_sha256: String,
    complete_schedule_sha256: String,
}
fn plan(sets: &[ESet]) -> Result<EPlan> {
    if e_schedule(sets.len())? != sets {
        return Err(Failure::accounting("E frozen schedule mismatch"));
    }
    let mut ids = Sha256::new();
    let mut order = Sha256::new();
    let mut complete = Sha256::new();
    for s in sets {
        let mut metadata = Vec::new();
        for n in [
            s.set_index,
            s.executed_block_index,
            s.family,
            s.temporal_quartile.unwrap_or(4),
            s.width,
            s.g,
            s.execution_sequence_index,
            s.source_rotation,
            s.design_class,
        ] {
            metadata
                .extend_from_slice(&u64::try_from(n).map_err(Failure::accounting)?.to_le_bytes());
        }
        let mut source = (s.width as u32).to_le_bytes().to_vec();
        for id in s.role_ordered_expert_ids.iter().flatten() {
            source.extend_from_slice(&id.to_le_bytes());
        }
        let execution: Vec<_> = s
            .execution_sequence
            .iter()
            .map(|c| canonical_index(*c) as u8)
            .chain(CANONICAL.iter().map(|c| s.role(*c) as u8))
            .collect();
        ids.update(&source);
        order.update(&metadata);
        order.update(&execution);
        complete.update(&metadata);
        complete.update(&source);
        complete.update(&execution);
    }
    let cell_source_schedules: BTreeMap<_, _> = CANONICAL
        .into_iter()
        .map(|cell| {
            Ok((
                cell,
                schedule_evidence(&sets.iter().map(|s| s.b_set(cell)).collect::<Vec<_>>())?,
            ))
        })
        .collect::<Result<_>>()?;
    let first = &cell_source_schedules[&Cell::HS];
    Ok(EPlan {
        block_count: sets.len() as u64,
        expert_slots_per_cell: first.expert_slot_count,
        source_bytes_per_cell: first.source_bytes,
        cell_source_schedules,
        universe_sha256: sequence_sha(
            &expert_sequence(256, NAMESPACE).map_err(Failure::accounting)?,
        ),
        ordered_role_ids_sha256: finish_sha(&ids),
        execution_order_sha256: finish_sha(&order),
        complete_schedule_sha256: finish_sha(&complete),
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Exposure {
    set_index: usize,
    executed_block_index: usize,
    family: usize,
    source_role: usize,
    width: usize,
    temporal_quartile: Option<usize>,
    execution_sequence_index: usize,
    source_rotation: usize,
    design_class: usize,
    cell: Cell,
    execution_position: usize,
    predecessor_cell: Option<Cell>,
    ordered_expert_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Timing {
    sample_index: usize,
    hs_ns: u64,
    ms_ns: u64,
    hc_ns: u64,
    mc_ns: u64,
    serial_destination_delta_ns: i128,
    concurrent_destination_delta_ns: i128,
    interaction_delta_ns: i128,
    mc_times_hs: u128,
    hc_times_ms: u128,
    multiplicative_interaction_direction: i8,
    serial_ratio: Ratio,
    concurrent_ratio: Ratio,
    interaction_ratio: Ratio,
}
impl Timing {
    fn new(sample_index: usize, hs_ns: u64, ms_ns: u64, hc_ns: u64, mc_ns: u64) -> Result<Self> {
        if [hs_ns, ms_ns, hc_ns, mc_ns].contains(&0) {
            return Err(Failure::accounting("zero E source duration"));
        }
        let serial_destination_delta_ns = signed_sub(ms_ns.into(), hs_ns.into())?;
        let concurrent_destination_delta_ns = signed_sub(mc_ns.into(), hc_ns.into())?;
        let interaction_delta_ns =
            signed_sub(concurrent_destination_delta_ns, serial_destination_delta_ns)?;
        let mc_times_hs = product(mc_ns.into(), hs_ns.into())?;
        let hc_times_ms = product(hc_ns.into(), ms_ns.into())?;
        let multiplicative_interaction_direction = match mc_times_hs.cmp(&hc_times_ms) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        };
        Ok(Self {
            sample_index,
            hs_ns,
            ms_ns,
            hc_ns,
            mc_ns,
            serial_destination_delta_ns,
            concurrent_destination_delta_ns,
            interaction_delta_ns,
            mc_times_hs,
            hc_times_ms,
            multiplicative_interaction_direction,
            serial_ratio: Ratio {
                numerator: ms_ns.into(),
                denominator: hs_ns.into(),
            },
            concurrent_ratio: Ratio {
                numerator: mc_ns.into(),
                denominator: hc_ns.into(),
            },
            interaction_ratio: Ratio {
                numerator: mc_times_hs,
                denominator: hc_times_ms,
            },
        })
    }
    fn valid(&self) -> bool {
        Self::new(
            self.sample_index,
            self.hs_ns,
            self.ms_ns,
            self.hc_ns,
            self.mc_ns,
        )
        .is_ok_and(|s| s == *self)
    }
    fn times(&self) -> [u64; 4] {
        [self.hs_ns, self.hc_ns, self.ms_ns, self.mc_ns]
    }
    fn ratio(&self) -> Ratio {
        Ratio {
            numerator: self.mc_times_hs,
            denominator: self.hc_times_ms,
        }
    }
}
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize)]
struct Totals {
    samples: u64,
    hs_ns: u64,
    ms_ns: u64,
    hc_ns: u64,
    mc_ns: u64,
    serial_destination_delta_ns: i128,
    concurrent_destination_delta_ns: i128,
    interaction_delta_ns: i128,
    interaction_samples: Directions,
    serial_destination_samples: Directions,
    concurrent_destination_samples: Directions,
    multiplicative_interaction_samples: Directions,
}
impl Totals {
    fn observe(&mut self, s: &Timing) -> Result<()> {
        if !s.valid() {
            return Err(Failure::accounting("E raw arithmetic does not reconcile"));
        }
        add(&mut self.samples, 1)?;
        add(&mut self.hs_ns, s.hs_ns)?;
        add(&mut self.ms_ns, s.ms_ns)?;
        add(&mut self.hc_ns, s.hc_ns)?;
        add(&mut self.mc_ns, s.mc_ns)?;
        self.serial_destination_delta_ns = signed_add(
            self.serial_destination_delta_ns,
            s.serial_destination_delta_ns,
        )?;
        self.concurrent_destination_delta_ns = signed_add(
            self.concurrent_destination_delta_ns,
            s.concurrent_destination_delta_ns,
        )?;
        self.interaction_delta_ns = signed_add(self.interaction_delta_ns, s.interaction_delta_ns)?;
        self.interaction_samples.observe(s.interaction_delta_ns)?;
        self.serial_destination_samples
            .observe(s.serial_destination_delta_ns)?;
        self.concurrent_destination_samples
            .observe(s.concurrent_destination_delta_ns)?;
        self.multiplicative_interaction_samples
            .observe(s.multiplicative_interaction_direction.into())?;
        self.reconcile()
    }
    fn reconcile(&self) -> Result<()> {
        if self.serial_destination_delta_ns != signed_sub(self.ms_ns.into(), self.hs_ns.into())?
            || self.concurrent_destination_delta_ns
                != signed_sub(self.mc_ns.into(), self.hc_ns.into())?
            || self.interaction_delta_ns
                != signed_sub(
                    self.concurrent_destination_delta_ns,
                    self.serial_destination_delta_ns,
                )?
            || [
                &self.interaction_samples,
                &self.serial_destination_samples,
                &self.concurrent_destination_samples,
                &self.multiplicative_interaction_samples,
            ]
            .into_iter()
            .any(|d| d.count().ok() != Some(self.samples))
        {
            return Err(Failure::accounting("C totals do not reconcile"));
        }
        Ok(())
    }
    fn times(&self) -> [u64; 4] {
        [self.hs_ns, self.hc_ns, self.ms_ns, self.mc_ns]
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ExactStats {
    #[serde(flatten)]
    totals: Totals,
    mean_signed_interaction_delta_ns: Option<SignedFraction>,
    median_signed_interaction_delta_ns: Option<SignedFraction>,
    median_ratio_of_ratios_exact: Option<RatioMedian>,
    serial_median_delta_ns: Option<SignedFraction>,
    concurrent_median_delta_ns: Option<SignedFraction>,
    serial_median_mapped_over_host_ratio_exact: Option<RatioMedian>,
    concurrent_median_mapped_over_host_ratio_exact: Option<RatioMedian>,
}
fn exact_stats(samples: &[Timing]) -> Result<ExactStats> {
    let mut totals = Totals::default();
    for s in samples {
        totals.observe(s)?;
    }
    let median = |f: fn(&Timing) -> i128| median_signed(samples.iter().map(f).collect());
    let ratios = |f: fn(&Timing) -> Ratio| {
        RatioMedian::new(samples.iter().map(|s| (s.sample_index, f(s))).collect())
    };
    Ok(ExactStats {
        mean_signed_interaction_delta_ns: (totals.samples != 0).then(|| SignedFraction {
            numerator: totals.interaction_delta_ns,
            denominator: totals.samples,
        }),
        median_signed_interaction_delta_ns: median(|s| s.interaction_delta_ns)?,
        median_ratio_of_ratios_exact: ratios(Timing::ratio),
        serial_median_delta_ns: median(|s| s.serial_destination_delta_ns)?,
        concurrent_median_delta_ns: median(|s| s.concurrent_destination_delta_ns)?,
        serial_median_mapped_over_host_ratio_exact: ratios(|s| Ratio {
            numerator: s.ms_ns.into(),
            denominator: s.hs_ns.into(),
        }),
        concurrent_median_mapped_over_host_ratio_exact: ratios(|s| Ratio {
            numerator: s.mc_ns.into(),
            denominator: s.hc_ns.into(),
        }),
        totals,
    })
}
#[derive(Debug, Serialize)]
struct DescriptiveStats {
    interaction_percent_of_hc: Option<f64>,
    serial_mapped_slowdown_percent: Option<f64>,
    concurrent_mapped_slowdown_percent: Option<f64>,
    slowdown_percentage_point_difference: Option<f64>,
    mean_signed_interaction_delta_ns: Option<f64>,
    median_signed_interaction_delta_ns: Option<f64>,
    median_ratio_of_ratios: Option<f64>,
    serial_median_delta_ns: Option<f64>,
    concurrent_median_delta_ns: Option<f64>,
    serial_median_mapped_over_host_ratio: Option<f64>,
    concurrent_median_mapped_over_host_ratio: Option<f64>,
}
fn descriptive(e: &ExactStats) -> DescriptiveStats {
    let t = &e.totals;
    let pct = |d: i128, n: u64| (n != 0).then(|| d as f64 / n as f64 * 100.0);
    let frac = |f: &Option<SignedFraction>| {
        f.as_ref()
            .map(|f| f.numerator as f64 / f.denominator as f64)
    };
    let serial = pct(t.serial_destination_delta_ns, t.hs_ns);
    let concurrent = pct(t.concurrent_destination_delta_ns, t.hc_ns);
    DescriptiveStats {
        interaction_percent_of_hc: pct(t.interaction_delta_ns, t.hc_ns),
        serial_mapped_slowdown_percent: serial,
        concurrent_mapped_slowdown_percent: concurrent,
        slowdown_percentage_point_difference: concurrent.zip(serial).map(|(c, s)| c - s),
        mean_signed_interaction_delta_ns: frac(&e.mean_signed_interaction_delta_ns),
        median_signed_interaction_delta_ns: frac(&e.median_signed_interaction_delta_ns),
        median_ratio_of_ratios: e
            .median_ratio_of_ratios_exact
            .as_ref()
            .map(RatioMedian::descriptive),
        serial_median_delta_ns: frac(&e.serial_median_delta_ns),
        concurrent_median_delta_ns: frac(&e.concurrent_median_delta_ns),
        serial_median_mapped_over_host_ratio: e
            .serial_median_mapped_over_host_ratio_exact
            .as_ref()
            .map(RatioMedian::descriptive),
        concurrent_median_mapped_over_host_ratio: e
            .concurrent_median_mapped_over_host_ratio_exact
            .as_ref()
            .map(RatioMedian::descriptive),
    }
}
#[derive(Debug, Serialize)]
struct EStats {
    exact: ExactStats,
    descriptive: DescriptiveStats,
}
impl EStats {
    fn new(samples: &[Timing]) -> Result<Self> {
        let exact = exact_stats(samples)?;
        Ok(Self {
            descriptive: descriptive(&exact),
            exact,
        })
    }
}
fn reconcile_partitions<'a>(
    parts: impl Iterator<Item = &'a ExactStats>,
    primary: &ExactStats,
) -> Result<()> {
    let mut totals = Totals::default();
    for e in parts {
        let t = &e.totals;
        add(&mut totals.samples, t.samples)?;
        add(&mut totals.hs_ns, t.hs_ns)?;
        add(&mut totals.ms_ns, t.ms_ns)?;
        add(&mut totals.hc_ns, t.hc_ns)?;
        add(&mut totals.mc_ns, t.mc_ns)?;
        totals.serial_destination_delta_ns = signed_add(
            totals.serial_destination_delta_ns,
            t.serial_destination_delta_ns,
        )?;
        totals.concurrent_destination_delta_ns = signed_add(
            totals.concurrent_destination_delta_ns,
            t.concurrent_destination_delta_ns,
        )?;
        totals.interaction_delta_ns =
            signed_add(totals.interaction_delta_ns, t.interaction_delta_ns)?;
        for (a, b) in [
            (&mut totals.interaction_samples, &t.interaction_samples),
            (
                &mut totals.serial_destination_samples,
                &t.serial_destination_samples,
            ),
            (
                &mut totals.concurrent_destination_samples,
                &t.concurrent_destination_samples,
            ),
            (
                &mut totals.multiplicative_interaction_samples,
                &t.multiplicative_interaction_samples,
            ),
        ] {
            add(&mut a.positive, b.positive)?;
            add(&mut a.negative, b.negative)?;
            add(&mut a.equal, b.equal)?;
        }
    }
    totals.reconcile()?;
    if totals != primary.totals {
        return Err(Failure::accounting("width/order partition mismatch"));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct RawBlock {
    source_set: ESet,
    timing: Timing,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct MatchedUnit {
    family: usize,
    source_role: usize,
    width: usize,
    ordered_expert_ids: Vec<u32>,
    exposures: BTreeMap<Cell, Exposure>,
    timing: Timing,
}

fn reconstruct(raw: &[RawBlock]) -> Result<Vec<MatchedUnit>> {
    let schedule = e_schedule(112)?;
    if raw.len() != 112
        || raw.iter().zip(&schedule).any(|(r, s)| {
            r.source_set != *s || !r.timing.valid() || r.timing.sample_index != s.set_index
        })
    {
        return Err(Failure::accounting(
            "E exact crossover requires all 112 valid blocks",
        ));
    }
    let mut units = Vec::with_capacity(112);
    let mut consumed = BTreeSet::new();
    for family in 0..28 {
        for role in 0..4 {
            let mut exposures = BTreeMap::new();
            let mut times = [0; 4];
            let ids = raw[family].source_set.role_ordered_expert_ids[role].clone();
            for q in 0..4 {
                let r = &raw[q * 28 + family];
                if r.source_set.role_ordered_expert_ids[role] != ids {
                    return Err(Failure::accounting(
                        "E exact source changed across exposures",
                    ));
                }
                let cell = CANONICAL
                    .into_iter()
                    .find(|c| r.source_set.role(*c) == role)
                    .ok_or_else(|| Failure::accounting("E missing role assignment"))?;
                if exposures
                    .insert(cell, r.source_set.exposure(cell))
                    .is_some()
                    || !consumed.insert((r.source_set.set_index, cell))
                {
                    return Err(Failure::accounting("E duplicate source-role/cell timing"));
                }
                times[cell.index()] = r.timing.times()[cell.index()];
            }
            if exposures.len() != 4 {
                return Err(Failure::accounting("E incomplete matched unit"));
            }
            units.push(MatchedUnit {
                family,
                source_role: role,
                width: ids.len(),
                ordered_expert_ids: ids,
                exposures,
                timing: Timing::new(family * 4 + role, times[0], times[2], times[1], times[3])?,
            });
        }
    }
    let raw_stats = exact_stats(&raw.iter().map(|r| r.timing.clone()).collect::<Vec<_>>())?;
    let matched = exact_stats(&units.iter().map(|u| u.timing.clone()).collect::<Vec<_>>())?;
    if consumed.len() != 448
        || raw_stats.totals.times() != matched.totals.times()
        || raw_stats.totals.serial_destination_delta_ns
            != matched.totals.serial_destination_delta_ns
        || raw_stats.totals.concurrent_destination_delta_ns
            != matched.totals.concurrent_destination_delta_ns
        || raw_stats.totals.interaction_delta_ns != matched.totals.interaction_delta_ns
    {
        return Err(Failure::accounting("E raw/matched totals do not reconcile"));
    }
    Ok(units)
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
struct CellStratum {
    helper_calls: u64,
    expert_slots: u64,
    source_ns: u64,
}
#[derive(Debug, Serialize)]
struct EStatistics {
    raw_block_descriptive: EStats,
    primary_exact_matched: Option<EStats>,
    per_width: BTreeMap<usize, EStats>,
    source_families: BTreeMap<usize, EStats>,
    source_roles: BTreeMap<usize, EStats>,
    block_execution_sequences: BTreeMap<usize, EStats>,
    block_source_rotations: BTreeMap<usize, EStats>,
    block_temporal_quartiles: BTreeMap<usize, EStats>,
    descriptive_design_classes: BTreeMap<usize, EStats>,
    cell_execution_positions: BTreeMap<Cell, BTreeMap<usize, CellStratum>>,
    cell_predecessors: BTreeMap<Cell, BTreeMap<String, CellStratum>>,
    concurrent_widths: Directions,
    interaction_widths: Directions,
}
fn e_statistics(raw: &[RawBlock], matched: &[MatchedUnit], count: usize) -> Result<EStatistics> {
    let schedule = e_schedule(count)?;
    if raw.len() > count
        || raw.iter().zip(&schedule).any(|(r, s)| {
            r.source_set != *s || !r.timing.valid() || r.timing.sample_index != s.set_index
        })
    {
        return Err(Failure::accounting("E invalid raw schedule/timing prefix"));
    }
    let expected_matched = if count == 112 && raw.len() == 112 {
        reconstruct(raw)?
    } else {
        Vec::new()
    };
    if matched != expected_matched {
        return Err(Failure::accounting("E matched-unit evidence mismatch"));
    }
    let raw_times: Vec<_> = raw.iter().map(|r| r.timing.clone()).collect();
    let raw_block_descriptive = EStats::new(&raw_times)?;
    let primary_exact_matched = if matched.is_empty() {
        None
    } else {
        Some(EStats::new(
            &matched.iter().map(|u| u.timing.clone()).collect::<Vec<_>>(),
        )?)
    };
    let block_partition = |n: usize, key: fn(&ESet) -> usize| -> Result<BTreeMap<usize, EStats>> {
        (0..n)
            .map(|i| {
                Ok((
                    i,
                    EStats::new(
                        &raw.iter()
                            .filter(|r| key(&r.source_set) == i)
                            .map(|r| r.timing.clone())
                            .collect::<Vec<_>>(),
                    )?,
                ))
            })
            .collect()
    };
    let unit_partition =
        |n: usize, key: fn(&MatchedUnit) -> usize| -> Result<BTreeMap<usize, EStats>> {
            (0..n)
                .map(|i| {
                    Ok((
                        i,
                        EStats::new(
                            &matched
                                .iter()
                                .filter(|u| key(u) == i)
                                .map(|u| u.timing.clone())
                                .collect::<Vec<_>>(),
                        )?,
                    ))
                })
                .collect()
        };
    let per_width: BTreeMap<_, _> = unit_partition(7, |u| u.width - 2)?
        .into_iter()
        .map(|(i, s)| (i + 2, s))
        .collect();
    let source_families = unit_partition(28, |u| u.family)?;
    let source_roles = unit_partition(4, |u| u.source_role)?;
    let block_execution_sequences = block_partition(4, |s| s.execution_sequence_index)?;
    let block_source_rotations = block_partition(4, |s| s.source_rotation)?;
    // Warmup has no measured temporal quartile and remains descriptive only.
    let block_temporal_quartiles = block_partition(if count == 112 { 4 } else { 1 }, |s| {
        s.temporal_quartile.unwrap_or(0)
    })?;
    let descriptive_design_classes = block_partition(16, |s| s.design_class)?;
    for part in [
        &block_execution_sequences,
        &block_source_rotations,
        &block_temporal_quartiles,
        &descriptive_design_classes,
    ] {
        reconcile_partitions(
            part.values().map(|s| &s.exact),
            &raw_block_descriptive.exact,
        )?;
    }
    if let Some(primary) = &primary_exact_matched {
        for part in [&per_width, &source_families, &source_roles] {
            reconcile_partitions(part.values().map(|s| &s.exact), &primary.exact)?;
        }
        if primary.exact.totals.times() != raw_block_descriptive.exact.totals.times() {
            return Err(Failure::accounting("E matched/raw reconciliation"));
        }
    }
    let mut cell_execution_positions = BTreeMap::new();
    let mut cell_predecessors = BTreeMap::new();
    for cell in CANONICAL {
        let mut positions: BTreeMap<usize, CellStratum> =
            (0..4).map(|i| (i, CellStratum::default())).collect();
        let mut predecessors: BTreeMap<String, CellStratum> = ["START", "HS", "MS", "HC", "MC"]
            .into_iter()
            .map(|s| (s.into(), CellStratum::default()))
            .collect();
        for r in raw {
            let e = r.source_set.exposure(cell);
            let key = e
                .predecessor_cell
                .map(|c| format!("{c:?}"))
                .unwrap_or("START".into());
            for bucket in [
                positions.get_mut(&e.execution_position).unwrap(),
                predecessors.get_mut(&key).unwrap(),
            ] {
                add(&mut bucket.helper_calls, 1)?;
                add(&mut bucket.expert_slots, r.source_set.width as u64)?;
                add(&mut bucket.source_ns, r.timing.times()[cell.index()])?;
            }
        }
        for buckets in [
            positions.values().collect::<Vec<_>>(),
            predecessors.values().collect(),
        ] {
            let mut n = 0;
            let mut ns = 0;
            let mut slots = 0;
            for b in buckets {
                add(&mut n, b.helper_calls)?;
                add(&mut ns, b.source_ns)?;
                add(&mut slots, b.expert_slots)?;
            }
            let expected_slots = raw.iter().try_fold(0u64, |mut n, r| {
                add(&mut n, r.source_set.width as u64)?;
                Ok(n)
            })?;
            if n != raw.len() as u64
                || ns != raw_block_descriptive.exact.totals.times()[cell.index()]
                || slots != expected_slots
            {
                return Err(Failure::accounting(
                    "E cell descriptive strata reconciliation",
                ));
            }
        }
        cell_execution_positions.insert(cell, positions);
        cell_predecessors.insert(cell, predecessors);
    }
    let mut concurrent_widths = Directions::default();
    let mut interaction_widths = Directions::default();
    for w in per_width.values().filter(|w| w.exact.totals.samples > 0) {
        concurrent_widths.observe(w.exact.totals.concurrent_destination_delta_ns)?;
        interaction_widths.observe(w.exact.totals.interaction_delta_ns)?;
    }
    Ok(EStatistics {
        raw_block_descriptive,
        primary_exact_matched,
        per_width,
        source_families,
        source_roles,
        block_execution_sequences,
        block_source_rotations,
        block_temporal_quartiles,
        descriptive_design_classes,
        cell_execution_positions,
        cell_predecessors,
        concurrent_widths,
        interaction_widths,
    })
}
impl EStatistics {
    fn exact_matches(&self, other: &Self) -> bool {
        let eq = |a: &BTreeMap<usize, EStats>, b: &BTreeMap<usize, EStats>| {
            a.keys().eq(b.keys()) && a.iter().all(|(k, s)| s.exact == b[k].exact)
        };
        self.raw_block_descriptive.exact == other.raw_block_descriptive.exact
            && self.primary_exact_matched.as_ref().map(|s| &s.exact)
                == other.primary_exact_matched.as_ref().map(|s| &s.exact)
            && self.concurrent_widths == other.concurrent_widths
            && self.interaction_widths == other.interaction_widths
            && self.cell_execution_positions == other.cell_execution_positions
            && self.cell_predecessors == other.cell_predecessors
            && [
                &self.per_width,
                &self.source_families,
                &self.source_roles,
                &self.block_execution_sequences,
                &self.block_source_rotations,
                &self.block_temporal_quartiles,
                &self.descriptive_design_classes,
            ]
            .into_iter()
            .zip([
                &other.per_width,
                &other.source_families,
                &other.source_roles,
                &other.block_execution_sequences,
                &other.block_source_rotations,
                &other.block_temporal_quartiles,
                &other.descriptive_design_classes,
            ])
            .all(|(a, b)| eq(a, b))
    }
}

// Endpoint=true selects the secondary interaction; false selects primary CDE.
fn interpretation(s: &EStatistics, interaction: bool) -> Result<&'static str> {
    let Some(primary) = &s.primary_exact_matched else {
        return Ok("INCONCLUSIVE");
    };
    let e = &primary.exact;
    let t = &e.totals;
    if t.samples != 112
        || t.hc_ns == 0
        || s.per_width.len() != 7
        || s.per_width.values().any(|w| w.exact.totals.samples != 16)
        || [
            &s.block_execution_sequences,
            &s.block_source_rotations,
            &s.block_temporal_quartiles,
        ]
        .into_iter()
        .any(|p| {
            p.len() != 4 || (0..4).any(|i| p.get(&i).is_none_or(|v| v.exact.totals.samples != 28))
        })
    {
        return Ok("INCONCLUSIVE");
    }
    let delta = |t: &Totals| {
        if interaction {
            t.interaction_delta_ns
        } else {
            t.concurrent_destination_delta_ns
        }
    };
    let directions = if interaction {
        &t.interaction_samples
    } else {
        &t.concurrent_destination_samples
    };
    let widths = if interaction {
        &s.interaction_widths
    } else {
        &s.concurrent_widths
    };
    let median = if interaction {
        &e.median_signed_interaction_delta_ns
    } else {
        &e.concurrent_median_delta_ns
    };
    let median = median
        .as_ref()
        .ok_or_else(|| Failure::accounting("E missing median"))?
        .numerator;
    let ratio = if interaction {
        &e.median_ratio_of_ratios_exact
    } else {
        &e.concurrent_median_mapped_over_host_ratio_exact
    };
    let ratio = ratio
        .as_ref()
        .ok_or_else(|| Failure::accounting("E missing ratio median"))?;
    let scaled = delta(t)
        .checked_mul(100)
        .ok_or_else(|| Failure::accounting("E threshold overflow"))?;
    let threshold = |percent: i128| {
        i128::from(t.hc_ns)
            .checked_mul(percent)
            .ok_or_else(|| Failure::accounting("E threshold product overflow"))
    };
    let strata: Vec<_> = [
        &s.block_execution_sequences,
        &s.block_source_rotations,
        &s.block_temporal_quartiles,
    ]
    .into_iter()
    .flat_map(|p| p.values())
    .map(|v| delta(&v.exact.totals))
    .collect();
    let positive = median > 0
        && directions.positive > 56
        && widths.positive >= 5
        && strata.iter().all(|v| *v > 0);
    let negative = median < 0
        && directions.negative > 56
        && widths.negative >= 5
        && strata.iter().all(|v| *v < 0);
    if positive && scaled >= threshold(5)? {
        return Ok(if interaction {
            "STRONG_POSITIVE_INTERACTION"
        } else {
            "STRONG_POSITIVE_CDE"
        });
    }
    if positive && scaled >= threshold(3)? {
        return Ok(if interaction {
            "MATERIAL_POSITIVE_INTERACTION"
        } else {
            "MATERIAL_POSITIVE_CDE"
        });
    }
    if negative && scaled <= threshold(-3)? {
        return Ok(if interaction {
            "NEGATIVE_MATERIAL_INTERACTION"
        } else {
            "NEGATIVE_MATERIAL_CDE"
        });
    }
    // The secondary endpoint retains D's additional near-zero consistency rules.
    if interaction {
        let direction = delta(t).signum();
        let disagreement = match direction {
            1 => directions.positive <= 56,
            -1 => directions.negative <= 56,
            _ => directions.positive > 56 || directions.negative > 56,
        };
        if median.signum() != direction
            || disagreement
            || strata.iter().any(|v| *v != 0 && v.signum() == -direction)
        {
            return Ok("AMBIGUOUS");
        }
    }
    if scaled
        .checked_abs()
        .ok_or_else(|| Failure::accounting("E absolute overflow"))?
        <= threshold(1)?
        && ratio.compare_hundredths(99)? != Ordering::Less
        && ratio.compare_hundredths(101)? != Ordering::Greater
        && widths.positive < 5
    {
        return Ok(if interaction {
            "EVIDENCE_AGAINST_MATERIAL_INTERACTION"
        } else {
            "EVIDENCE_AGAINST_MATERIAL_CDE"
        });
    }
    Ok("AMBIGUOUS")
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct CellHash {
    source: String,
    payload: String,
    gpu: String,
    epoch: bool,
}
impl From<Hashes> for CellHash {
    fn from(h: Hashes) -> Self {
        Self {
            source: h.source,
            payload: h.payload,
            gpu: h.gpu,
            epoch: h.epoch,
        }
    }
}
fn hash_valid(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct VerificationSet {
    source_set: ESet,
    cells: BTreeMap<Cell, Vec<CellHash>>,
}
impl VerificationSet {
    fn valid(&self) -> bool {
        self.cells.len() == 4
            && CANONICAL.into_iter().all(|c| {
                self.cells.get(&c).is_some_and(|hashes| {
                    hashes.len() == self.source_set.width
                        && hashes.iter().all(|h| {
                            h.epoch
                                && hash_valid(&h.source)
                                && hash_valid(&h.payload)
                                && h.payload == h.gpu
                        })
                })
            })
    }
}
fn crossover_hashes_valid(raw: &[VerificationSet], units: &[MatchedUnit]) -> bool {
    units.len() == 112
        && raw.len() == 112
        && units.iter().all(|u| {
            let Some(hs) = u
                .exposures
                .get(&Cell::HS)
                .and_then(|e| raw.get(e.set_index))
                .and_then(|r| r.cells.get(&Cell::HS))
            else {
                return false;
            };
            CANONICAL.into_iter().all(|cell| {
                u.exposures
                    .get(&cell)
                    .and_then(|e| raw.get(e.set_index))
                    .is_some_and(|r| {
                        r.source_set.role_ordered_expert_ids[u.source_role] == u.ordered_expert_ids
                            && r.cells.get(&cell) == Some(hs)
                    })
            })
        })
}
#[derive(Debug, Serialize)]
struct EPhase {
    name: &'static str,
    schedule_count: usize,
    schedule: Vec<ESet>,
    expected: EPlan,
    execution_trace: Vec<Exposure>,
    raw_blocks: Vec<RawBlock>,
    matched_units: Vec<MatchedUnit>,
    raw_verification: Vec<VerificationSet>,
    statistics: EStatistics,
    cells: BTreeMap<Cell, CellArm>,
    witnesses: BTreeMap<Cell, Witnesses>,
    fd_proof: SourceUploadFdProofSnapshot,
    fd_proof_hits_only: bool,
    mismatch_count: u64,
}
impl EPhase {
    fn new(name: &'static str, count: usize) -> Result<Self> {
        let schedule = e_schedule(count)?;
        Ok(Self {
            name,
            schedule_count: count,
            expected: plan(&schedule)?,
            schedule,
            execution_trace: Vec::new(),
            raw_blocks: Vec::new(),
            matched_units: Vec::new(),
            raw_verification: Vec::new(),
            statistics: e_statistics(&[], &[], count)?,
            cells: CANONICAL
                .into_iter()
                .map(|c| Ok((c, CellArm::new(c)?)))
                .collect::<Result<_>>()?,
            witnesses: CANONICAL
                .into_iter()
                .map(|c| (c, Witnesses::default()))
                .collect(),
            fd_proof: SourceUploadFdProofSnapshot::default(),
            fd_proof_hits_only: false,
            mismatch_count: 0,
        })
    }
    fn successful(&self) -> bool {
        if !e_schedule(self.schedule_count).is_ok_and(|s| s == self.schedule) {
            return false;
        }
        let expected_trace: Vec<_> = self
            .schedule
            .iter()
            .flat_map(|s| s.execution_sequence.into_iter().map(|c| s.exposure(c)))
            .collect();
        self.fd_proof_hits_only
            && c_proof_hits_only(&self.fd_proof, self.expected.expert_slots_per_cell)
            && e_schedule(self.schedule_count).is_ok_and(|s| s == self.schedule)
            && plan(&self.schedule).is_ok_and(|p| p == self.expected)
            && self.execution_trace == expected_trace
            && self.raw_blocks.len() == self.schedule_count
            && self.raw_verification.len() == self.schedule_count
            && self.cells.len() == 4
            && self.witnesses.len() == 4
            && self
                .raw_verification
                .iter()
                .zip(&self.schedule)
                .all(|(r, s)| r.source_set == *s && r.valid())
            && e_statistics(&self.raw_blocks, &self.matched_units, self.schedule_count)
                .is_ok_and(|s| s.exact_matches(&self.statistics))
            && (self.schedule_count != 112
                || crossover_hashes_valid(&self.raw_verification, &self.matched_units))
            && self.mismatch_count == 0
            && CANONICAL.into_iter().all(|cell| {
                self.cells.get(&cell).is_some_and(|arm| {
                    arm.cell == cell
                        && arm.completed_source_sets
                            == self
                                .schedule
                                .iter()
                                .map(|s| s.b_set(cell))
                                .collect::<Vec<_>>()
                        && arm
                            .successful(&self.expected.cell_source_schedules[&cell], cell.mapped())
                        && arm.evidence.times.source_direct_read_ns
                            == self.statistics.raw_block_descriptive.exact.totals.times()
                                [cell.index()]
                }) && self.witnesses.get(&cell).is_some_and(|w| {
                    hash_valid(&w.full_source_sha256)
                        && hash_valid(&w.bare_payload_sha256)
                        && w.bare_payload_sha256 == w.gpu_destination_payload_sha256
                })
            })
    }
}
async fn e_phase(
    p: &mut EPhase,
    gpu: &Gpu,
    storage: &NvmeStorage,
    host: &mut AlignedBuffer,
) -> Result<()> {
    let mut streams: [Streams; 4] = std::array::from_fn(|_| Streams::default());
    let result = async {
        for set in p.schedule.clone() {
            let mut times = [0; 4];
            let mut verification = VerificationSet {
                source_set: set.clone(),
                cells: BTreeMap::new(),
            };
            for cell in set.execution_sequence {
                let source = set.b_set(cell);
                p.execution_trace.push(set.exposure(cell));
                let arm = p
                    .cells
                    .get_mut(&cell)
                    .ok_or_else(|| Failure::accounting("E missing cell"))?;
                let outcome = if cell.mapped() {
                    mapped_cell(gpu, storage, &source, arm, &mut streams[cell.index()]).await
                } else {
                    host_cell(gpu, storage, host, &source, arm, &mut streams[cell.index()]).await
                };
                if let Err(e) = &outcome {
                    note_arm_error(arm, e)?;
                }
                let (ns, hashes) = outcome?;
                times[cell.index()] = ns;
                verification
                    .cells
                    .insert(cell, hashes.into_iter().map(CellHash::from).collect());
            }
            p.raw_blocks.push(RawBlock {
                timing: Timing::new(set.set_index, times[0], times[2], times[1], times[3])?,
                source_set: set,
            });
            let valid = verification.valid();
            p.raw_verification.push(verification);
            if !valid {
                add(&mut p.mismatch_count, 1)?;
                return Err(Failure::runtime(
                    "hash-parity-failed",
                    "E source/payload/GPU/epoch mismatch",
                ));
            }
        }
        Ok(())
    }
    .await;
    p.fd_proof = storage.source_upload_fd_proof_snapshot();
    p.fd_proof_hits_only = c_proof_hits_only(&p.fd_proof, p.expected.expert_slots_per_cell);
    for cell in CANONICAL {
        let arm = p
            .cells
            .get_mut(&cell)
            .ok_or_else(|| Failure::accounting("E missing cell"))?;
        arm.source_schedule = schedule_evidence(&arm.completed_source_sets)?;
        arm.evidence.rates();
        p.witnesses.insert(cell, streams[cell.index()].snapshot());
    }
    if p.schedule_count == 112 && p.raw_blocks.len() == 112 {
        p.matched_units = reconstruct(&p.raw_blocks)?;
    }
    p.statistics = e_statistics(&p.raw_blocks, &p.matched_units, p.schedule_count)?;
    result?;
    if !p.fd_proof_hits_only {
        return Err(Failure::authority(format!(
            "{} expected only proof hits: {:?}",
            p.name, p.fd_proof
        )));
    }
    if p.schedule_count == 112 && !crossover_hashes_valid(&p.raw_verification, &p.matched_units) {
        add(&mut p.mismatch_count, 1)?;
        return Err(Failure::runtime(
            "hash-parity-failed",
            "E delayed exact-source hash crossover mismatch",
        ));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct EReport {
    schema: &'static str,
    args: Args,
    config_sha256: Option<String>,
    complete: bool,
    correctness_pass: bool,
    authoritative: bool,
    classification: String,
    failure: Option<String>,
    authority: CAuthority,
    warmup: EPhase,
    measured: EPhase,
    performance_required_for_correctness: bool,
    performance_authority: &'static str,
    concurrent_destination_interpretation_conditional_on_zero_retry_log: &'static str,
    interaction_interpretation_conditional_on_zero_retry_log: &'static str,
    primary_endpoint: &'static str,
    schedule_contract: &'static str,
    timing_contract: &'static str,
    interpretation_contract: &'static str,
    secondary_destination_endpoints: &'static str,
    retry_evidence_contract: &'static str,
    source_byte_evidence_contract: &'static str,
}
impl EReport {
    fn new(args: Args) -> Result<Self> {
        let mut base = Report::new(args.clone()).authority;
        base.control_source_api = B_API;
        base.treatment_source_api = B_API;
        base.control_destination = "aligned-host-arena";
        base.treatment_destination = "wgpu-map-write-arena";
        base.upload_capacity_bytes = MAPPED_ARENA;
        Ok(Self {
            schema: E_SCHEMA,
            args,
            config_sha256: None,
            complete: false,
            correctness_pass: false,
            authoritative: false,
            classification: "not-run".into(),
            failure: None,
            authority: CAuthority {
                base,
                cell_source_apis: Cell::ALL.into_iter().map(|c| (c, c.api())).collect(),
                host_arena_capacity_bytes: HOST_ARENA,
                max_width: MAX_WIDTH,
                fd_preproof_completed: false,
                fd_cache_capacity: 0,
                preproof_universe_size: E_UNIVERSE,
                preproof_ordered_expert_ids: expert_sequence(E_UNIVERSE, NAMESPACE).map_err(Failure::accounting)?,
                preproof: SourceUploadFdProofSnapshot::default(),
                after_preproof_telemetry_reset: SourceUploadFdProofSnapshot::default(),
                after_warmup_telemetry_reset: SourceUploadFdProofSnapshot::default(),
            },
            warmup: EPhase::new("warmup", WARMUP_COUNT)?,
            measured: EPhase::new("measured", MEASURED_COUNT)?,
            performance_required_for_correctness: false,
            performance_authority: "PENDING_EXTERNAL_RETRY_LOG_AUDIT",
            concurrent_destination_interpretation_conditional_on_zero_retry_log: "INCONCLUSIVE",
            interaction_interpretation_conditional_on_zero_retry_log: "INCONCLUSIVE",
            primary_endpoint: "112 exact (family,source_role) matched units: CDE=sum(MC)-sum(HC); percent denominator=sum(HC)",
            schedule_contract: E_SCHEDULE_CONTRACT,
            timing_contract: C_TIMER_CONTRACT,
            interpretation_contract: E_INTERPRETATION,
            secondary_destination_endpoints: "Serial destination effect and separately classified destination-by-concurrency interaction; exact medians/ratios/signs and all frozen descriptive strata retained",
            retry_evidence_contract: E_RETRY,
            source_byte_evidence_contract: "Each successful helper must return exactly K*FULL bytes. Failed-helper partial physical I/O is unavailable, never inferred as zero. Each cell retains ordered per-set full-source/payload/verified-GPU hashes and concatenated byte-stream witnesses. Within a block all four sources are disjoint. Equality is required across the four delayed exposures of each exact (family,role); concatenated cell byte streams have different orders and are not compared across cells. Verification/reset/copy/readback are outside source timers.",
        })
    }
    fn authority_valid(&self) -> bool {
        let a = &self.authority;
        let b = &a.base;
        self.schema == E_SCHEMA
            && !self.performance_required_for_correctness
            && self.args.iterations == 128
            && self.args.warmup_iterations == 32
            && self.warmup.schedule_count == WARMUP_COUNT
            && self.measured.schedule_count == MEASURED_COUNT
            && a.cell_source_apis == Cell::ALL.into_iter().map(|c| (c, c.api())).collect()
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
            && c_proof_hits_only(&self.warmup.fd_proof, 140)
            && c_proof_hits_only(&self.measured.fd_proof, 560)
    }
    fn classify(&mut self) -> Result<()> {
        self.complete = true;
        self.correctness_pass = false;
        self.authoritative = false;
        self.concurrent_destination_interpretation_conditional_on_zero_retry_log = "AMBIGUOUS";
        self.interaction_interpretation_conditional_on_zero_retry_log = "AMBIGUOUS";
        if !self.authority_valid() {
            self.classification = "authority-failed".into();
        } else if !self.warmup.successful() || !self.measured.successful() {
            self.classification = "evidence-reconciliation-failed".into();
        } else {
            self.correctness_pass = true;
            self.authoritative = true;
            self.classification = "delayed-source-crossover-complete".into();
            self.concurrent_destination_interpretation_conditional_on_zero_retry_log =
                interpretation(&self.measured.statistics, false)?;
            self.interaction_interpretation_conditional_on_zero_retry_log =
                interpretation(&self.measured.statistics, true)?;
        }
        Ok(())
    }
    fn fail(&mut self, failure: Failure) {
        self.complete = failure.complete;
        self.correctness_pass = false;
        self.authoritative = false;
        self.classification = failure.classification.into();
        self.failure = Some(failure.detail);
        self.concurrent_destination_interpretation_conditional_on_zero_retry_log = "AMBIGUOUS";
        self.interaction_interpretation_conditional_on_zero_retry_log = "AMBIGUOUS";
    }
}
async fn execute(report: &mut EReport) -> Result<()> {
    if report.args.iterations != 128 || report.args.warmup_iterations != 32 {
        return Err(Failure::runtime("invalid-arguments", "E requires --iterations 128 --warmup-iterations 32 for CLI compatibility; E generates 112 measured blocks (q=0..3, families=0..27) and 28 warmup blocks (families=28..55)"));
    }
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
    let mut host = AlignedBuffer::new(HOST_ARENA, ALIGN);
    e_phase(&mut report.warmup, &gpu, &storage, &mut host).await?;
    if !report.warmup.successful() {
        return Err(Failure::authority("warmup evidence did not reconcile"));
    }
    storage.reset_source_upload_fd_proof_telemetry();
    report.authority.after_warmup_telemetry_reset = storage.source_upload_fd_proof_snapshot();
    if report.authority.after_warmup_telemetry_reset != SourceUploadFdProofSnapshot::default() {
        return Err(Failure::authority("warmup telemetry reset failed"));
    }
    e_phase(&mut report.measured, &gpu, &storage, &mut host).await?;
    gpu.check()?;
    report.classify()?;
    Ok(())
}
pub(crate) async fn run_command(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.report_out)?;
    let mut report = EReport::new(args).map_err(|e| io::Error::other(e.detail))?;
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
                format!("E diagnostic panic: {detail}"),
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
    fn fixture_phase(name: &'static str, count: usize, times: [u64; 4]) -> EPhase {
        let mut p = EPhase::new(name, count).unwrap();
        let n = p.expected.expert_slots_per_cell;
        let sets = p.expected.block_count;
        p.raw_blocks = p
            .schedule
            .iter()
            .map(|s| {
                let mut ns = times;
                for cell in CANONICAL {
                    ns[cell.index()] +=
                        (s.family * 31 + s.role(cell) * 7 + s.temporal_quartile.unwrap_or(4) * 17)
                            as u64;
                }
                RawBlock {
                    source_set: s.clone(),
                    timing: Timing::new(s.set_index, ns[0], ns[2], ns[1], ns[3]).unwrap(),
                }
            })
            .collect();
        if count == 112 {
            p.matched_units = reconstruct(&p.raw_blocks).unwrap();
        }
        p.statistics = e_statistics(&p.raw_blocks, &p.matched_units, count).unwrap();
        p.execution_trace = p
            .schedule
            .iter()
            .flat_map(|s| s.execution_sequence.into_iter().map(|c| s.exposure(c)))
            .collect();
        p.raw_verification = p
            .schedule
            .iter()
            .map(|s| VerificationSet {
                source_set: s.clone(),
                cells: CANONICAL
                    .into_iter()
                    .map(|c| {
                        (
                            c,
                            s.role_ordered_expert_ids[s.role(c)]
                                .iter()
                                .map(|id| {
                                    let bytes = id.to_le_bytes();
                                    CellHash {
                                        source: sha(&bytes),
                                        payload: sha(&bytes),
                                        gpu: sha(&bytes),
                                        epoch: true,
                                    }
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            })
            .collect();
        for cell in Cell::ALL {
            let arm = p.cells.get_mut(&cell).unwrap();
            arm.source_sets_attempted = sets;
            arm.serial_helper_calls = if cell.serial() { sets } else { 0 };
            arm.concurrent_helper_calls = if cell.serial() { 0 } else { sets };
            arm.completed_source_sets = p.schedule.iter().map(|s| s.b_set(cell)).collect();
            arm.source_schedule = p.expected.cell_source_schedules[&cell].clone();
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
            a.times.source_direct_read_ns =
                p.statistics.raw_block_descriptive.exact.totals.times()[cell.index()];
            if cell.mapped() {
                a.map_attempts = sets;
                a.maps_completed = sets;
                a.unmaps = sets;
                a.pointers.gpu_offset_checks = n;
                a.explicit_copy_buffer_bytes = n * PAYLOAD as u64;
            } else {
                a.cpu_payload_copy_bytes = n * PAYLOAD as u64;
            }
            p.witnesses.insert(
                cell,
                Witnesses {
                    full_source_sha256: sha(if cell.serial() {
                        b"serial"
                    } else {
                        b"concurrent"
                    }),
                    bare_payload_sha256: sha(b"payload"),
                    gpu_destination_payload_sha256: sha(b"payload"),
                },
            );
        }
        p.fd_proof = SourceUploadFdProofSnapshot {
            source_upload_fd_proof_requests: n * 4,
            source_upload_fd_proof_hits: n * 4,
            ..SourceUploadFdProofSnapshot::default()
        };
        p.fd_proof_hits_only = true;
        assert!(p.successful());
        p
    }
    fn fixture(times: [u64; 4]) -> EReport {
        let mut r = EReport::new(args()).unwrap();
        let a = &mut r.authority;
        a.base.linux = true;
        a.base.adapter_authoritative = true;
        a.base.direct_io_requested = true;
        a.base.packed_storage = Some(false);
        a.base.exact_geometry = true;
        a.fd_cache_capacity = 256;
        a.fd_preproof_completed = true;
        a.preproof = SourceUploadFdProofSnapshot {
            source_upload_fd_proof_requests: 256,
            source_upload_fd_proof_misses: 256,
            ..SourceUploadFdProofSnapshot::default()
        };
        r.warmup = fixture_phase("warmup", 28, times);
        r.measured = fixture_phase("measured", 112, times);
        r
    }

    fn raw_with(mut f: impl FnMut(&ESet) -> [u64; 4]) -> Vec<RawBlock> {
        e_schedule(112)
            .unwrap()
            .into_iter()
            .map(|s| {
                let t = f(&s);
                RawBlock {
                    timing: Timing::new(s.set_index, t[0], t[1], t[2], t[3]).unwrap(),
                    source_set: s,
                }
            })
            .collect()
    }
    fn stats(raw: &[RawBlock]) -> EStatistics {
        e_statistics(raw, &reconstruct(raw).unwrap(), 112).unwrap()
    }

    #[test]
    fn source_to_upload_copy_elision_e_independent_schedule_balance_overlap_and_spacing() {
        let all = all_schedule().unwrap();
        let mut history: Vec<BTreeSet<u32>> = Vec::new();
        let mut seen = BTreeMap::<(usize, usize, Cell), Vec<usize>>::new();
        for (i, s) in all.iter().enumerate() {
            let family = if i < 28 { i + 28 } else { (i - 28) % 28 };
            let q = if i < 28 { 0 } else { (i - 28) / 28 };
            let w = family % 7;
            let g = (family / 7) % 4;
            let phi = [0, 2, 3, 1];
            assert_eq!(s.family, family);
            assert_eq!(s.width, w + 2);
            assert_eq!(s.execution_sequence_index, q ^ g ^ (w & 3));
            assert_eq!(s.source_rotation, q ^ phi[g] ^ phi[w & 3]);
            let roles: [Vec<u32>; 4] = std::array::from_fn(|role| {
                (0..w + 2)
                    .map(|j| (((family * 17 + role * 64 + j * 13) % 256) * 6143 / 255) as u32)
                    .collect()
            });
            assert_eq!(s.role_ordered_expert_ids, roles);
            let ids: BTreeSet<_> = roles.into_iter().flatten().collect();
            assert_eq!(ids.len(), 4 * (w + 2));
            for previous in history.iter().rev().take(2) {
                assert!(ids.is_disjoint(previous));
            }
            assert_eq!(s.previous_two_blocks.len(), i.min(2));
            history.push(ids);
            if i >= 28 {
                for cell in CANONICAL {
                    seen.entry((family, s.role(cell), cell))
                        .or_default()
                        .push(i);
                }
            }
        }
        assert_eq!(seen.len(), 448);
        assert!(seen.values().all(|v| v.len() == 1));
        for family in 0..28 {
            for role in 0..4 {
                let mut indices: Vec<_> = CANONICAL
                    .into_iter()
                    .map(|c| seen[&(family, role, c)][0])
                    .collect();
                indices.sort();
                assert!(indices.windows(2).all(|p| p[1] - p[0] == 28));
            }
        }
        for (count, slots, bytes) in [(28, 140, 372_162_560), (112, 560, 1_488_650_240)] {
            let p = plan(&e_schedule(count).unwrap()).unwrap();
            assert_eq!(p.expert_slots_per_cell, slots);
            assert_eq!(p.source_bytes_per_cell, bytes);
        }
        assert!(e_schedule(111).is_err());
    }

    #[test]
    fn source_to_upload_copy_elision_e_exact_reconstruction_and_raw_reconciliation() {
        let r = fixture([1_000_000, 500_000, 1_010_000, 530_000]);
        assert!(r.measured.successful());
        let units = &r.measured.matched_units;
        assert_eq!(units.len(), 112);
        for u in units {
            for (cell, e) in &u.exposures {
                assert_eq!(
                    u.timing.times()[cell.index()],
                    r.measured.raw_blocks[e.set_index].timing.times()[cell.index()]
                );
                assert_eq!(e.ordered_expert_ids, u.ordered_expert_ids);
            }
        }
        let s = &r.measured.statistics;
        assert_eq!(
            s.primary_exact_matched
                .as_ref()
                .unwrap()
                .exact
                .totals
                .times(),
            s.raw_block_descriptive.exact.totals.times()
        );
        assert!(s.per_width.values().all(|v| v.exact.totals.samples == 16));
        // Block pseudo-medians and exact-source medians need not be equal.
        assert_ne!(
            s.primary_exact_matched.as_ref().unwrap().exact,
            s.raw_block_descriptive.exact
        );
    }

    #[test]
    fn source_to_upload_copy_elision_e_exact_thresholds_no_rounding() {
        for (mc, cde, interaction) in [
            (105, "STRONG_POSITIVE_CDE", "STRONG_POSITIVE_INTERACTION"),
            (
                103,
                "MATERIAL_POSITIVE_CDE",
                "MATERIAL_POSITIVE_INTERACTION",
            ),
            (102, "AMBIGUOUS", "AMBIGUOUS"),
            (
                100,
                "EVIDENCE_AGAINST_MATERIAL_CDE",
                "EVIDENCE_AGAINST_MATERIAL_INTERACTION",
            ),
            (97, "NEGATIVE_MATERIAL_CDE", "NEGATIVE_MATERIAL_INTERACTION"),
        ] {
            let s = stats(&raw_with(|_| [100, 100, 100, mc]));
            assert_eq!(interpretation(&s, false).unwrap(), cde);
            assert_eq!(interpretation(&s, true).unwrap(), interaction);
        }
        let hc = 100_000_000_000_000_000u64;
        let s = stats(&raw_with(|_| [hc, hc, hc, hc + 3_000_000_000_000_000 - 1]));
        assert_eq!(interpretation(&s, false).unwrap(), "AMBIGUOUS");
        assert_eq!(interpretation(&s, true).unwrap(), "AMBIGUOUS");
        let s = stats(&raw_with(|_| [100, 106, 100, 106]));
        assert_eq!(interpretation(&s, false).unwrap(), "STRONG_POSITIVE_CDE");
        assert_eq!(
            interpretation(&s, true).unwrap(),
            "EVIDENCE_AGAINST_MATERIAL_INTERACTION"
        );
    }

    #[test]
    fn source_to_upload_copy_elision_e_against_inclusive_ratio_bounds_and_widths() {
        for delta in [-1, 0, 1] {
            let s = stats(&raw_with(|s| {
                [
                    100,
                    100,
                    100,
                    (100 + if s.width <= 4 { delta } else { -delta }) as u64,
                ]
            }));
            assert_eq!(
                interpretation(&s, false).unwrap(),
                "EVIDENCE_AGAINST_MATERIAL_CDE"
            );
        }
        let s = stats(&raw_with(|_| [100, 100, 100, 101]));
        assert_eq!(s.concurrent_widths.positive, 7);
        assert_eq!(interpretation(&s, false).unwrap(), "AMBIGUOUS");
        let s = stats(&raw_with(|_| [100, 100, 100, 99]));
        assert_eq!(
            interpretation(&s, false).unwrap(),
            "EVIDENCE_AGAINST_MATERIAL_CDE"
        );
        let s = stats(&raw_with(|_| [10_000, 10_000, 10_000, 9899]));
        assert_eq!(interpretation(&s, false).unwrap(), "AMBIGUOUS");
    }

    #[test]
    fn source_to_upload_copy_elision_e_all_strata_and_matched_consistency_gate() {
        for interaction in [false, true] {
            for negative in [false, true] {
                for kind in 0..3 {
                    for index in 0..4 {
                        let mut s = stats(&raw_with(|_| {
                            [100, 100, 100, if negative { 94 } else { 106 }]
                        }));
                        assert_ne!(interpretation(&s, interaction).unwrap(), "AMBIGUOUS");
                        let p = match kind {
                            0 => &mut s.block_execution_sequences,
                            1 => &mut s.block_source_rotations,
                            _ => &mut s.block_temporal_quartiles,
                        };
                        let t = &mut p.get_mut(&index).unwrap().exact.totals;
                        if interaction {
                            t.interaction_delta_ns = if negative { 1 } else { -1 };
                        } else {
                            t.concurrent_destination_delta_ns = if negative { 1 } else { -1 };
                        }
                        assert_eq!(interpretation(&s, interaction).unwrap(), "AMBIGUOUS");
                    }
                }
            }
        }
        for interaction in [false, true] {
            for gate in 0..3 {
                let mut s = stats(&raw_with(|_| [100, 100, 100, 106]));
                let e = &mut s.primary_exact_matched.as_mut().unwrap().exact;
                match (interaction, gate) {
                    (false, 0) => e.concurrent_median_delta_ns.as_mut().unwrap().numerator = 0,
                    (true, 0) => {
                        e.median_signed_interaction_delta_ns
                            .as_mut()
                            .unwrap()
                            .numerator = 0
                    }
                    (false, 1) => e.totals.concurrent_destination_samples.positive = 56,
                    (true, 1) => e.totals.interaction_samples.positive = 56,
                    (false, _) => s.concurrent_widths.positive = 4,
                    (true, _) => s.interaction_widths.positive = 4,
                }
                assert_eq!(interpretation(&s, interaction).unwrap(), "AMBIGUOUS");
            }
        }
        let mut s = stats(&raw_with(|_| [100, 100, 100, 106]));
        for v in s.descriptive_design_classes.values_mut() {
            v.exact.totals.interaction_delta_ns = -1;
            v.exact.totals.concurrent_destination_delta_ns = -1;
        }
        assert_eq!(interpretation(&s, false).unwrap(), "STRONG_POSITIVE_CDE");
        assert_eq!(
            interpretation(&s, true).unwrap(),
            "STRONG_POSITIVE_INTERACTION"
        );
    }

    #[test]
    fn source_to_upload_copy_elision_e_correctness_and_authority_fail_closed() {
        let mutations: [fn(&mut EReport); 22] = [
            |r| r.measured.schedule[0].source_rotation ^= 1,
            |r| r.measured.schedule[28].role_ordered_expert_ids[0][0] ^= 1,
            |r| r.measured.execution_trace[0].source_role ^= 1,
            |r| r.measured.raw_blocks[0].timing.mc_ns += 1,
            |r| {
                r.measured.matched_units[0]
                    .exposures
                    .get_mut(&Cell::HS)
                    .unwrap()
                    .set_index += 1
            },
            |r| r.measured.matched_units[0].timing.interaction_delta_ns += 1,
            |r| {
                r.measured
                    .statistics
                    .per_width
                    .get_mut(&2)
                    .unwrap()
                    .exact
                    .totals
                    .mc_ns += 1
            },
            |r| {
                r.measured
                    .statistics
                    .block_execution_sequences
                    .get_mut(&0)
                    .unwrap()
                    .exact
                    .totals
                    .mc_ns += 1
            },
            |r| {
                r.measured
                    .statistics
                    .block_source_rotations
                    .get_mut(&0)
                    .unwrap()
                    .exact
                    .totals
                    .mc_ns += 1
            },
            |r| {
                r.measured
                    .statistics
                    .block_temporal_quartiles
                    .get_mut(&0)
                    .unwrap()
                    .exact
                    .totals
                    .mc_ns += 1
            },
            |r| r.measured.expected.complete_schedule_sha256 = sha(b"corrupt"),
            |r| {
                r.measured.raw_verification[0]
                    .cells
                    .get_mut(&Cell::HS)
                    .unwrap()[0]
                    .source = sha(b"corrupt")
            },
            |r| {
                r.measured.raw_verification[0]
                    .cells
                    .get_mut(&Cell::MC)
                    .unwrap()[0]
                    .epoch = false
            },
            |r| r.measured.fd_proof.source_upload_fd_proof_misses = 1,
            |r| r.measured.fd_proof.source_upload_fd_proof_failures = 1,
            |r| r.authority.fd_cache_capacity = 255,
            |r| r.authority.preproof.source_upload_fd_proof_hits = 1,
            |r| {
                r.authority
                    .after_warmup_telemetry_reset
                    .source_upload_fd_proof_requests = 1
            },
            |r| {
                r.measured
                    .cells
                    .get_mut(&Cell::HS)
                    .unwrap()
                    .serial_helper_calls += 1
            },
            |r| {
                r.measured
                    .cells
                    .get_mut(&Cell::MC)
                    .unwrap()
                    .evidence
                    .full_source_bytes += 1
            },
            |r| r.measured.cells.get_mut(&Cell::MC).unwrap().evidence.unmaps -= 1,
            |r| {
                r.warmup.schedule[27].previous_two_blocks[0]
                    .shared_expert_ids
                    .push(1)
            },
        ];
        for mutate in mutations {
            let mut r = fixture([1_000_000, 500_000, 1_010_000, 530_000]);
            mutate(&mut r);
            r.classify().unwrap();
            assert!(!r.authoritative);
            assert!(!r.correctness_pass);
            assert_eq!(
                r.concurrent_destination_interpretation_conditional_on_zero_retry_log,
                "AMBIGUOUS"
            );
            assert_eq!(
                r.interaction_interpretation_conditional_on_zero_retry_log,
                "AMBIGUOUS"
            );
        }
        for times in [
            [100, 100, 100, 100],
            [100, 100, 100, 106],
            [100, 100, 100, 94],
        ] {
            let mut r = fixture(times);
            r.classify().unwrap();
            assert!(r.correctness_pass);
            assert!(r.authoritative);
            assert_eq!(r.performance_authority, "PENDING_EXTERNAL_RETRY_LOG_AUDIT");
        }
    }

    #[test]
    fn source_to_upload_copy_elision_e_zero_overflow_duplicate_and_missing_exposures() {
        assert!(Timing::new(0, 0, 1, 1, 1).is_err());
        let raw = raw_with(|_| [u64::MAX, u64::MAX, u64::MAX, u64::MAX]);
        assert!(reconstruct(&raw).is_err());
        let mut raw = raw_with(|_| [100, 101, 100, 105]);
        raw[28] = raw[0].clone();
        assert!(reconstruct(&raw).is_err());
        let mut raw = raw_with(|_| [100, 101, 100, 105]);
        raw.pop();
        assert!(reconstruct(&raw).is_err());
    }

    #[test]
    fn source_to_upload_copy_elision_e_synthetic_json_for_independent_audit() {
        if let Some(path) = std::env::var_os("MER_HMA1CE_SYNTHETIC_JSON") {
            let mut r = fixture([1_000_000, 500_000, 1_010_000, 530_000]);
            r.classify().unwrap();
            assert!(r.authoritative);
            let envelope =
                serde_json::json!({"fixture":"portable-synthetic-not-hardware","report":r});
            std::fs::write(path, serde_json::to_vec_pretty(&envelope).unwrap()).unwrap();
        }
    }

    #[tokio::test]
    async fn source_to_upload_copy_elision_e_invalid_args_and_exclusive_report_without_gpu() {
        let mut a = args();
        let dir = std::env::temp_dir().join(format!(
            "mer-hma1ce-cli-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        a.report_out = dir.join("report.json");
        a.iterations = 112;
        assert!(run_command(a.clone()).await.is_err());
        let bytes = std::fs::read(&a.report_out).unwrap();
        let j: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(j["schema"], E_SCHEMA);
        assert_eq!(j["classification"], "invalid-arguments");
        assert!(j["authority"]["adapter_name"].is_null());
        assert!(run_command(a).await.is_err());
        assert_eq!(std::fs::read(dir.join("report.json")).unwrap(), bytes);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn source_to_upload_copy_elision_e_independent_schedule_hash_pins() {
        let p = plan(&e_schedule(28).unwrap()).unwrap();
        assert_eq!(
            p.ordered_role_ids_sha256,
            "2e0635b0964b7ab1a78e6ea8e40d8fe381a467b417cdd021f9569f6bbbce85a2"
        );
        assert_eq!(
            p.execution_order_sha256,
            "a24913c05dbaf99b1e355761d37eac8816403a8a843464c203266833d2e8a8a5"
        );
        assert_eq!(
            p.complete_schedule_sha256,
            "3b86637f11d8ece9a10117c995a327e7b61045ccf44e9e96089df057345dc274"
        );
        let p = plan(&e_schedule(112).unwrap()).unwrap();
        assert_eq!(
            p.ordered_role_ids_sha256,
            "1e674860e1dc602757d2f59270a5a802085ec8af1dba2dd1e5a0ef019de3fefd"
        );
        assert_eq!(
            p.execution_order_sha256,
            "03434fa07f8fa61cf5835cb3389376efe909d7a69dfacfec91a5b743a3b5fc90"
        );
        assert_eq!(
            p.complete_schedule_sha256,
            "f8cea5abcbb582095338da407c222cc84d253aae63d84ce5e7558c44fbf427df"
        );
    }
}
