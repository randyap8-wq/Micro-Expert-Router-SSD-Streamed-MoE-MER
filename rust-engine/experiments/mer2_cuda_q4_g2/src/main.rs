use mer2_cuda_q4_g2::{preflight, run_cuda};
use serde_json::json;
use std::io::{self, Write};

fn emit(value: &serde_json::Value) -> Result<(), String> {
    let mut out = io::stdout().lock();
    writeln!(out, "{value}").map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 || !matches!(args[0].as_str(), "--cpu" | "--cuda") {
        eprintln!("usage: mer2-cuda-q4-g2 --cpu|--cuda");
        std::process::exit(2);
    }
    let mut report = match preflight() {
        Ok(v) => v,
        Err(e) => {
            let _ = emit(&json!({"error":e,"GPU_INVOCATIONS":0,"CUDA_GPU_TEST":"NOT_RUN"}));
            std::process::exit(1)
        }
    };
    if args[0] == "--cpu" {
        if emit(&report).is_err() {
            std::process::exit(1)
        };
        return;
    }
    if emit(&report).is_err() {
        std::process::exit(1)
    }
    match run_cuda(&mut report) {
        Ok(()) => {
            report["result"] = json!("PASS");
            if emit(&report).is_err() {
                std::process::exit(1)
            }
        }
        Err(e) => {
            report["error"] = json!(e);
            report["result"] = json!("STOP");
            let _ = emit(&report);
            std::process::exit(1)
        }
    }
}
