//! Source-defined CLI only. Do not run under P1a authority.
use mer2_cuda_g3_s1::{
    cuda_unavailable, fake_cpu_preflight,
    host_token::Session,
    identity::{contract_hash, source_files, source_hash, Model, Variant},
    state::{FakeCpu, Runner},
    Result, Stop, KIND,
};
use serde_json::json;
use std::{collections::BTreeMap, path::Path};
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if cfg!(feature = "cuda") || args.get(1).map(String::as_str) == Some("cuda") {
        return cuda_unavailable();
    }
    if args.len() != 4 || !matches!(args[1].as_str(), "cpu-fake-preflight" | "cpu-fake-chain") {
        return Err(Stop::new("STOP_CLI","usage: cpu-fake-preflight|cpu-fake-chain REPO_ROOT EXTERNAL_SOURCE_SHA256_JSON; cuda always refuses"));
    }
    let bytes =
        std::fs::read(&args[3]).map_err(|e| Stop::new("STOP_SOURCE_IDENTITY", e.to_string()))?;
    let expected: BTreeMap<String, String> = serde_json::from_slice(&bytes)
        .map_err(|e| Stop::new("STOP_SOURCE_IDENTITY", e.to_string()))?;
    let sources = fake_cpu_preflight(Path::new(&args[2]), &expected)?;
    if args[1] == "cpu-fake-preflight" {
        println!(
            "{}",
            json!({"kind":KIND,"status":"SOURCE_IDENTITIES_VERIFIED_ONLY","source_sha256":source_hash()?,
            "contract_sha256":contract_hash(),"source_files":source_files(),"cuda":"UNAVAILABLE","computation":"NOT_RUN"})
        );
        return Ok(());
    }
    let session_model = Model::generate(Variant::Primary)?;
    let mut runner = Runner::new(&session_model, FakeCpu::default())?;
    let mut session = Session::new(session_model, sources)?;
    let mut requests = [session.request("A", 3)?, session.request("B", 17)?];
    let mut outputs = Vec::new();
    let result = (|| -> Result<()> {
        for _ in 0..3 {
            for r in &mut requests {
                let ticket = r.ticket();
                outputs.push(r.step_token(session.model(), ticket, &mut runner)?.json());
            }
        }
        Ok(())
    })();
    runner.finish();
    let report = runner.report();
    println!(
        "{}",
        json!({"kind":KIND,"status":"UNQUALIFIED_CPU_FAKE_RECEIPTS","outputs":outputs,
        "first_error":report.first_error.as_ref().map(ToString::to_string),"cleanup":report.cleanup,
        "drain_attempts":report.drain_attempts,"submitted":report.counts.submitted,"completed":report.counts.completed,
        "cuda":"UNAVAILABLE","gpu_results":null,"physical_vram":null,"performance":null})
    );
    result?;
    if let Some(e) = &report.first_error {
        return Err(e.clone());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
