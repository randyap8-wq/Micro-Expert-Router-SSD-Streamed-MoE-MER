use mer2_cuda_q4_s1::preflight;
use serde_json::json;
use std::io::Write;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let gpu = match args.as_slice() {
        [] => false,
        [arg] if arg == "--cpu" => false,
        [arg] if arg == "--cuda" => true,
        _ => {
            println!(
                "{}",
                json!({"CUDA_GPU_TEST": "NOT_RUN", "error": "usage: mer2-cuda-q4-s1 [--cpu|--cuda]"})
            );
            return ExitCode::FAILURE;
        }
    };
    let mut report = match preflight() {
        Ok(report) => report,
        Err(error) => {
            println!("{}", json!({"CUDA_GPU_TEST": "NOT_RUN", "error": error}));
            return ExitCode::FAILURE;
        }
    };
    report["phase"] = json!("preflight");
    println!("{report}");
    if let Err(error) = std::io::stdout().flush() {
        eprintln!("cannot preserve preflight: {error}");
        return ExitCode::FAILURE;
    }
    if !gpu {
        return ExitCode::SUCCESS;
    }
    #[cfg(feature = "cuda")]
    let result = {
        // cudarc's dynamic library/symbol loader can panic on unavailable CUDA.
        // Convert unwinding failures into an explicit nonzero FAIL, never fallback.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mer2_cuda_q4_s1::cuda::run(&mut report)
        }))
        .unwrap_or_else(|_| Err("CUDA loader/runtime panicked; no fallback".into()))
    };
    #[cfg(not(feature = "cuda"))]
    let result: Result<(), String> = Err("CUDA feature not compiled; no fallback".into());

    report["phase"] = json!("result");
    report["CUDA_GPU_TEST"] = json!(if result.is_ok() { "PASS" } else { "FAIL" });
    if let Err(error) = &result {
        report["error"] = json!(error);
    }
    println!("{report}");
    if result.is_ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
