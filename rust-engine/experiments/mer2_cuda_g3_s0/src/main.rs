//! Source candidate only: this CLI has not been compiled or run under D1.
use mer2_cuda_g3_s0::identity::{source_files, source_hash, Model, Variant};
use mer2_cuda_g3_s0::reference::{Session, StepOutput};
use mer2_cuda_g3_s0::{cuda_unavailable, require, Result, Stop, BASE, KIND};
use serde_json::{json, Value};

fn observed_routes(out: &StepOutput) -> Vec<[u32; 2]> {
    out.stages
        .iter()
        .filter(|s| s.stage == "router_probs")
        .map(|s| {
            let r = s
                .route
                .as_ref()
                .expect("internally generated router receipt");
            [r.ranked[0], r.ranked[1]]
        })
        .collect()
}
fn preflight() -> Result<Value> {
    let m = Model::generate(Variant::Primary)?;
    let mut session = Session::new(m.clone());
    let mut a = session.request("A", 3)?;
    let mut b = session.request("B", 17)?;
    let mut results = Vec::new();
    let mut logits = Vec::new();
    let mut routing_difference = false;
    for _ in 0..3 {
        let ta = a.ticket();
        let ao = a.step_token(&m, ta)?;
        let tb = b.ticket();
        let bo = b.step_token(&m, tb)?;
        a.verify_output(&ao)?;
        b.verify_output(&bo)?;
        routing_difference |= observed_routes(&ao) != observed_routes(&bo);
        logits.push(ao.logits_bits.clone());
        logits.push(bo.logits_bits.clone());
        results.push(ao.json());
        results.push(bo.json());
    }
    for i in 0..logits.len() {
        for j in 0..i {
            require(
                logits[i] != logits[j],
                "STOP_FIXTURE_NONDISCRIMINATING",
                "pairwise full-logit stale witness",
            )?;
        }
    }
    require(
        routing_difference,
        "STOP_FIXTURE_NONDISCRIMINATING",
        "A/B observed top2 route witness",
    )?;
    Ok(
        json!({"schema":"MER2-G3-S0-CPU-TRACE-v1","kind":KIND,"source_base":BASE,
        "source_files_sha256":source_files(),"source_receipt_sha256":source_hash()?,
        "model_fixture_sha256":m.hash(),"fixture_variant":m.variant().id(),"results":results,
        "status":"CPU_REFERENCE_PRODUCED_REQUIRES_INDEPENDENT_D2_REVIEW",
        "CPU_REPLAY_SCOPE":"SAME_BINARY_TOOLCHAIN_TARGET_ONLY",
        "G3_FULL_TOKEN":"NOT_QUALIFIED","G3_GPU_FULL_TOKEN":"NOT_IMPLEMENTED",
        "GPU_PROCESS_INVOCATIONS":0,"timing":null,"physical_pcie":null,"vram":null,"TPS":null}),
    )
}
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.as_slice() {
        [mode] if mode == "preflight-cpu" => preflight(),
        [mode] if mode == "cuda-unavailable" => cuda_unavailable().map(|_| Value::Null),
        _ => Err(Stop::new(
            "STOP_MODE",
            "exactly one mode required: preflight-cpu | cuda-unavailable",
        )),
    };
    match result {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!(
                "{}",
                json!({"schema":"MER2-G3-S0-CPU-TRACE-v1","kind":KIND,"result":error.code,
                "error":error.detail,"GPU_PROCESS_INVOCATIONS":0,"G3_FULL_TOKEN":"NOT_QUALIFIED"})
            );
            std::process::exit(2);
        }
    }
}
