//! Prospective CLI. Authored only; never built or run in P1.
#![forbid(unsafe_code)]
use mer2_cuda_rtps_p1::{fixture::Fixture, native_once, reference, report};
use std::io::{self, Write};

fn run() -> io::Result<i32> {
    // Check arity and reject all paths/options before fixture allocation.
    let mut args = std::env::args_os().skip(1);
    let mode = args.next();
    if args.next().is_some() {
        writeln!(io::stdout().lock(), "terminal=STOP_CLI_ARITY")?;
        return Ok(64);
    }
    match mode.as_deref().and_then(|s| s.to_str()) {
        Some("native-once") => {
            native_once().write(&mut io::stdout().lock())?;
            Ok(78)
        }
        Some("cpu-reference") => {
            let fixture = Fixture::build();
            let result = fixture.validate().and_then(|()| reference::evaluate(fixture.bytes(), fixture.input()));
            match result {
                Ok(stages) => Ok(if report::write_reference(&mut io::stdout().lock(), &stages)? { 0 } else { 65 }),
                Err(reason) => {
                    let mut output = io::stdout().lock();
                    report::identity(&mut output)?;
                    writeln!(output, "terminal=STOP_CPU_REFERENCE\nfirst_failure={reason:?}\ngpu_output=ABSENT\nnative_initializations=0")?;
                    Ok(65)
                }
            }
        }
        _ => {
            writeln!(io::stdout().lock(), "terminal=STOP_CLI_MODE\nusage=mer2-rtps-p1 cpu-reference|native-once")?;
            Ok(64)
        }
    }
}
fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(_) => { let _ = writeln!(io::stderr().lock(), "terminal=STOP_REPORT_WRITE"); 74 }
    };
    // No device/resource owner exists. An OOM/abort cannot guarantee a terminal
    // record; an independent future parent must retain missing-terminal evidence.
    std::process::exit(code);
}
