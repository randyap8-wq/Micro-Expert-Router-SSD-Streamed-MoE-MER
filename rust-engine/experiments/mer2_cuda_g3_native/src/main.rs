//! Unexecuted A/B three-token native harness. No CPU expert fallback.
#![forbid(unsafe_code)]

#[cfg(feature = "cuda")]
fn run() -> mer2_cuda_g3_s1::Result<()> {
    use mer2_cuda_g3_native::{
        source_files, verify_native_sources, verify_s1_manifest, NativeCudaAdapter, KIND,
    };
    use mer2_cuda_g3_s1::{
        host_token::Session,
        identity::{Model, Variant, VerifiedSources},
        state::Runner,
        Result, Stop,
    };
    use serde_json::json;
    use std::{collections::BTreeMap, path::Path};

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 || args[1] != "cuda-chain" {
        return Err(Stop::new("STOP_CLI",
            "usage: cuda-chain REPO_ROOT S1_SOURCE_SHA256_JSON NATIVE_SOURCE_SHA256_JSON DEVICE_ORDINAL"));
    }
    let manifest = |path: &str| -> Result<BTreeMap<String, String>> {
        let bytes =
            std::fs::read(path).map_err(|e| Stop::new("STOP_SOURCE_IDENTITY", e.to_string()))?;
        serde_json::from_slice(&bytes).map_err(|e| Stop::new("STOP_SOURCE_IDENTITY", e.to_string()))
    };
    let root = Path::new(&args[2]);
    let expected_s1 = manifest(&args[3])?;
    let expected_native = manifest(&args[4])?;
    verify_s1_manifest(&expected_s1)?;
    verify_native_sources(root, &expected_native)?;
    let sources = VerifiedSources::verify(root, &expected_s1)?;
    let ordinal = args[5]
        .parse::<usize>()
        .map_err(|_| Stop::new("STOP_CLI", "nonnegative CUDA device ordinal"))?;
    let model = Model::generate(Variant::Primary)?;
    // new() does not call CUDA. Initialization occurs inside the guarded Runner.
    let (adapter, evidence) = NativeCudaAdapter::new(ordinal);
    let mut runner = Runner::new(&model, adapter)?;
    let mut session = Session::new(model, sources)?;
    let mut requests = [session.request("A", 3)?, session.request("B", 17)?];
    let mut outputs = Vec::new();
    let result = (|| -> Result<()> {
        for _ in 0..3 {
            for request in &mut requests {
                verify_native_sources(root, &expected_native)?;
                let ticket = request.ticket();
                let output = request.step_token(session.model(), ticket, &mut runner)?;
                evidence.verify_step(&output)?;
                // Keep the frozen serialization intact, explicitly scoped as inherited labels.
                outputs.push(json!({"native_kind":KIND,
                    "native_binding":"EXACT_OPERATIONS_AND_FENCED_STAGE_DIGESTS",
                    "inherited_s1_receipt":output.json()}));
            }
        }
        Ok(())
    })();
    runner.finish();
    let report = runner.report();
    let verification = result.and_then(|_| evidence.verify_chain(report));
    println!(
        "{}",
        json!({"kind":KIND,
        "status":if verification.is_ok() {"NATIVE_CHAIN_COMPLETED_REQUIRES_INDEPENDENT_REVIEW"} else {"STOP_NATIVE_CHAIN"},
        "qualification":"NONE", "native_sources":source_files(),
        "s1_labels_are_not_cuda_proof":true,
        "inherited_runner_label":report.label,"inherited_runner_cleanup":report.cleanup,
        "runner_first_error":report.first_error.as_ref().map(ToString::to_string),
        "error":verification.as_ref().err().map(ToString::to_string),
        "native":evidence.json(),"outputs":outputs,
        "timing":null,"physical_vram":null,"performance":null})
    );
    verification
}

#[cfg(not(feature = "cuda"))]
fn run() -> mer2_cuda_g3_s1::Result<()> {
    mer2_cuda_g3_native::cuda_unavailable()
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
