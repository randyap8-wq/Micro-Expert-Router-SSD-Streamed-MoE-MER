//! Source-reviewed against cached cudarc 0.19.7; CUDA build/run NOT VERIFIED.
use crate::{compare, fixtures, sha256, KERNEL};
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use serde_json::{json, Value};

pub fn run(report: &mut Value) -> Result<(), String> {
    let ptx = candle_kernels::QUANTIZED.ptx();
    report["ptx_sha256"] = json!(sha256(ptx.as_bytes()));
    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e}"))?;
    report["device_ordinal"] = json!(ctx.ordinal());
    report["device_name"] = json!(ctx.name().map_err(|e| format!("device name: {e}"))?);
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_src(ptx))
        .map_err(|e| format!("load QUANTIZED: {e}"))?;
    let kernel = module
        .load_function(KERNEL)
        .map_err(|e| format!("load Q4_0 entrypoint: {e}"))?;
    report["results"] = json!([]);
    for fixture in fixtures() {
        let (ncols, nrows) = fixture.cuda_dimensions()?;
        let reference = fixture.reference()?;
        // These allocations contain only synthetic data. The CUDA allocator
        // supplies sufficient base alignment for block_q4_0's f16 field.
        let weights = stream
            .clone_htod(&fixture.weights)
            .map_err(|e| format!("weights H2D: {e}"))?;
        let input = stream
            .clone_htod(&fixture.input)
            .map_err(|e| format!("input H2D: {e}"))?;
        // NaN sentinel makes unwritten output fail, including zero-weight cases.
        let sentinel = vec![f32::NAN; fixture.rows];
        let mut output = stream
            .clone_htod(&sentinel)
            .map_err(|e| format!("output allocation: {e}"))?;
        stream.synchronize().map_err(|e| format!("H2D sync: {e}"))?;
        let config = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: reviewed quantized.cu ABI is (const void*, const float*,
        // float*, int, int). One full 32-lane warp per row; ncols % 64 == 0.
        // validate() bounds all C int arithmetic and exact allocation lengths.
        // Arrays are disjoint and remain alive through synchronization/D2H.
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&weights)
                .arg(&input)
                .arg(&mut output)
                .arg(&ncols)
                .arg(&nrows)
                .launch(config)
        }
        .map_err(|e| format!("kernel launch: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| format!("kernel sync: {e}"))?;
        let actual = stream
            .clone_dtoh(&output)
            .map_err(|e| format!("output D2H: {e}"))?;
        stream.synchronize().map_err(|e| format!("D2H sync: {e}"))?;
        // Capture exact device-returned f32 bits before checking even bad output.
        let comparison = compare(&reference, &actual);
        report["results"].as_array_mut().unwrap().push(json!({
            "name": fixture.name,
            "fixture_sha256": fixture.fingerprint(),
            "output": actual,
            "output_f32_bits": actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "comparison": match &comparison {
                Ok(max) => json!({"pass": true, "max_abs_error": max}),
                Err(error) => json!({"pass": false, "error": error}),
            },
        }));
        comparison?;
    }
    ctx.synchronize().map_err(|e| format!("final sync: {e}"))?;
    Ok(())
}
