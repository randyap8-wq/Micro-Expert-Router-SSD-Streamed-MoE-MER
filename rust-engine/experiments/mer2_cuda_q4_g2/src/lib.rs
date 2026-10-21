//! Isolated source-only resident-expert contract. CPU output is diagnostic only.
use mer2_cuda_q4_s2::{compare, fixtures, sha256, Fixture, ATOL, RTOL};
use serde_json::{json, Value};

#[cfg(feature = "cuda")]
mod cuda;
mod identity;
mod state;

pub const BASE: &str = "3d3aef41799b65b88f07feedc495707afaaad20c";
pub const SCHEMA: &str = "MER2-CUDA-G2-R0-v1";
pub const MIXED: &str = "1a774f651b0ab77f17d881a104c490841fa72c4c03faf85a162d15b95bf7b363";

pub fn contract() -> Result<Value, String> {
    serde_json::from_str(include_str!("../fixture-contract.json")).map_err(|e| e.to_string())
}

pub fn expert_digest(f: &Fixture) -> Result<String, String> {
    f.validate()?;
    let mut bytes = b"MER2-CUDA-G2-expert-v1\0".to_vec();
    for m in [&f.gate, &f.up, &f.down] {
        if m.rows != 128 || m.cols != 128 || m.packed.len() != 9216 {
            return Err("STOP_FIXTURE: exact 128x128 expert required".into());
        }
        bytes.extend((m.rows as u32).to_le_bytes());
        bytes.extend((m.cols as u32).to_le_bytes());
        bytes.extend(&m.packed);
    }
    Ok(sha256(&bytes))
}

pub fn input_digest(input: &[f32]) -> Result<String, String> {
    if input.len() != 128 || input.iter().any(|v| !v.is_finite()) {
        return Err("STOP_FIXTURE: invalid input".into());
    }
    let mut bytes = b"MER2-CUDA-G2-input-v1\0".to_vec();
    bytes.extend(128u32.to_le_bytes());
    for value in input {
        bytes.extend(value.to_le_bytes());
    }
    Ok(sha256(&bytes))
}

/// Only this data (never reference outputs) is supplied to the CUDA adapter.
pub(crate) struct Data {
    pub fixture: Fixture,
    pub inputs: [Vec<f32>; 2],
    pub expert: String,
    pub input_ids: [String; 2],
}

pub struct Prepared {
    pub(crate) data: Data,
    pub(crate) references: [Vec<f32>; 2],
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

pub fn prepare() -> Result<Prepared, String> {
    let frozen = contract()?;
    if frozen["schema"] != SCHEMA
        || frozen["base"] != BASE
        || frozen["atol"] != ATOL
        || frozen["rtol"] != RTOL
        || frozen["slot"] != 0
        || frozen["generation"] != 1
        || frozen["dimensions"] != json!([128, 128])
        || frozen["sentinel_bits"] != 0x7fc00000u32
        || frozen["s2_fixture_sha256"] != MIXED
    {
        return Err("STOP_FIXTURE: frozen contract metadata".into());
    }
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.name == "mixed")
        .ok_or("STOP_FIXTURE: missing mixed fixture")?;
    fixture.cuda_dimensions()?;
    if fixture.fingerprint() != MIXED {
        return Err("STOP_FIXTURE: mixed fingerprint".into());
    }
    let expert = expert_digest(&fixture)?;
    if frozen["expert_sha256"] != expert {
        return Err("STOP_FIXTURE: expert digest".into());
    }
    let inputs = [
        fixture.input.clone(),
        fixture.input.iter().map(|v| -*v).collect(),
    ];
    let input_ids = [input_digest(&inputs[0])?, input_digest(&inputs[1])?];
    if input_ids[0] == input_ids[1] {
        return Err("STOP_FIXTURE: identical requests".into());
    }
    let mut references = [Vec::new(), Vec::new()];
    for seq in 0..2 {
        let mut diagnostic = fixture.clone();
        diagnostic.input = inputs[seq].clone();
        references[seq] = diagnostic.reference()?;
        let record = &frozen["requests"][seq];
        if record["sequence"] != seq
            || record["input_sha256"] != input_ids[seq]
            || record["input_f32_bits"] != json!(bits(&inputs[seq]))
            || record["reference_f32_bits"] != json!(bits(&references[seq]))
        {
            return Err(format!("STOP_FIXTURE: frozen Rust oracle/input bits {seq}"));
        }
        let scalars: Vec<f32> = serde_json::from_value(record["reference"].clone())
            .map_err(|e| format!("STOP_FIXTURE: {e}"))?;
        if bits(&scalars) != bits(&references[seq]) {
            return Err("STOP_FIXTURE: reference scalar/bit mismatch".into());
        }
    }
    let k = frozen["witness"]["index"]
        .as_u64()
        .ok_or("STOP_FIXTURE: witness")? as usize;
    if k >= 128 {
        return Err("STOP_FIXTURE: witness index".into());
    }
    let a = f64::from(references[0][k]);
    let b = f64::from(references[1][k]);
    let gap = (a - b).abs();
    let tolerance = 2.0 * ATOL + RTOL * (a.abs() + b.abs());
    if !a.is_finite() || !b.is_finite() || !gap.is_finite() || !tolerance.is_finite() {
        return Err("STOP_FIXTURE: witness nonfinite arithmetic".into());
    }
    if gap <= tolerance {
        return Err("STOP_FIXTURE: witness lacks actual separation".into());
    }
    if frozen["witness"]["a_bits"] != references[0][k].to_bits() {
        return Err("STOP_FIXTURE: witness a_bits mismatch".into());
    }
    if frozen["witness"]["b_bits"] != references[1][k].to_bits() {
        return Err("STOP_FIXTURE: witness b_bits mismatch".into());
    }
    let parsed_gap = frozen["witness"]["absolute_gap"]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or("STOP_FIXTURE: witness absolute_gap missing/nonfinite")?;
    let gap_error = (parsed_gap - gap).abs();
    if !gap_error.is_finite() || gap_error > 1e-16 {
        return Err("STOP_FIXTURE: witness absolute_gap representation".into());
    }
    let parsed_sum = frozen["witness"]["tolerance_sum"]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or("STOP_FIXTURE: witness tolerance_sum missing/nonfinite")?;
    let sum_error = (parsed_sum - tolerance).abs();
    if !sum_error.is_finite() || sum_error > 1e-18 {
        return Err("STOP_FIXTURE: witness tolerance_sum representation".into());
    }
    if compare(&references[1], &references[0]).is_ok() {
        return Err("STOP_FIXTURE: witness stale A accepted as B".into());
    }
    if frozen["expected"] != state::expected() {
        return Err("STOP_FIXTURE: payload contract".into());
    }
    Ok(Prepared {
        data: Data {
            fixture,
            inputs,
            expert,
            input_ids,
        },
        references,
    })
}

pub fn verify_output(
    reference: &[f32],
    output: &[f32],
    output_bits: &[u32],
) -> Result<f64, String> {
    if output.len() != 128 || reference.len() != 128 || bits(output) != output_bits {
        return Err("STOP_FIXTURE: output shape/bits".into());
    }
    compare(reference, output)
}

pub fn preflight() -> Result<Value, String> {
    let prepared = prepare()?;
    let plan = state::plan(1);
    Ok(
        json!({"schema":SCHEMA,"source":identity::source_identity()?,
        "fixture":contract()?,"oracle":"CPU_REFERENCE_EXPECTED",
        "reference_output_count":prepared.references.iter().map(Vec::len).sum::<usize>(),
        "expected_schedule":plan.iter().map(|op| format!("{op:?}")).collect::<Vec<_>>(),
        "counters":{"kind":"EXPECTED","expected":state::expected(),"submitted":null,"completed":null},
        "timing":{"status":"NOT_MEASURED","batches":null},
        "vram":{"status":"NOT_MEASURED","free_bytes":null,"total_bytes":null},
        "CUDA_FEATURE_COMPILATION":"NOT_RUN","CUDA_GPU_TEST":"NOT_RUN","GPU_INVOCATIONS":0,
        "G2":"PARTIAL","G3_FULL_TOKEN":"NOT_IMPLEMENTED"}),
    )
}

pub fn run_cuda(report: &mut Value) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        cuda::run(prepare()?, report)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = report;
        Err("CUDA feature not compiled; no fallback".into())
    }
}
