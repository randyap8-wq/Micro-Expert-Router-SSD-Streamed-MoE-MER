//! Source/API reviewed only. Compilation and hardware execution are unqualified.
use crate::{compare, fixtures, sha256, Fixture, KERNEL, MUL, SILU};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use serde_json::{json, Value};
use std::sync::Arc;

fn sync(stream: &CudaStream, phase: &str) -> Result<(), String> {
    stream
        .synchronize()
        .map_err(|e| format!("{phase} sync: {e}"))
}

fn project(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    weights: &CudaSlice<u8>,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    phase: &str,
) -> Result<(), String> {
    if !matches!(input.len(), 64 | 128)
        || !matches!(output.len(), 64 | 128)
        || weights.len() != output.len() * (input.len() / 32) * 18
    {
        return Err(format!("{phase}: invalid device geometry"));
    }
    let (ncols, nrows) = (input.len() as i32, output.len() as i32);
    // SAFETY: exact vendor ABI (void*, float*, float*, int, int); disjoint
    // allocator-aligned buffers, 18-byte blocks aligned to half, no subviews.
    // Inner widths divisible by 64, one full warp per row, bounded int indexing.
    // Slices/context/module live through this explicit completion boundary.
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(weights)
            .arg(input)
            .arg(output)
            .arg(&ncols)
            .arg(&nrows)
            .launch(LaunchConfig {
                grid_dim: (nrows as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| format!("{phase} launch: {e}"))?;
    sync(stream, phase)
}

/// No CPU projection/activation or intermediate D2H in this computation path.
fn expert(
    stream: &Arc<CudaStream>,
    q4: &CudaFunction,
    silu: &CudaFunction,
    mul: &CudaFunction,
    fixture: &Fixture,
) -> Result<Vec<f32>, String> {
    let (dm, df) = fixture.cuda_dimensions()?;
    let numel = df as usize;
    let num_dims = 1usize;
    // Non-null, explicit contiguous metadata avoids null pointer arithmetic in
    // the vendor macros: dims[0], lhs_stride[0], rhs_stride[0]. Unary reads first 2.
    let host_layout = [numel, 1usize, 1usize];
    let host_hidden_sentinel = vec![f32::NAN; numel];
    let host_output_sentinel = vec![f32::NAN; dm as usize];
    let wg = stream
        .clone_htod(&fixture.gate.packed)
        .map_err(|e| format!("gate H2D: {e}"))?;
    let wu = stream
        .clone_htod(&fixture.up.packed)
        .map_err(|e| format!("up H2D: {e}"))?;
    let wd = stream
        .clone_htod(&fixture.down.packed)
        .map_err(|e| format!("down H2D: {e}"))?;
    let x = stream
        .clone_htod(&fixture.input)
        .map_err(|e| format!("input H2D: {e}"))?;
    let layout = stream
        .clone_htod(&host_layout)
        .map_err(|e| format!("layout H2D: {e}"))?;
    let mut gate = stream
        .clone_htod(&host_hidden_sentinel)
        .map_err(|e| format!("gate sentinel: {e}"))?;
    let mut up = stream
        .clone_htod(&host_hidden_sentinel)
        .map_err(|e| format!("up sentinel: {e}"))?;
    let mut activated = stream
        .clone_htod(&host_hidden_sentinel)
        .map_err(|e| format!("SiLU sentinel: {e}"))?;
    let mut hidden = stream
        .clone_htod(&host_hidden_sentinel)
        .map_err(|e| format!("hidden sentinel: {e}"))?;
    let mut y = stream
        .clone_htod(&host_output_sentinel)
        .map_err(|e| format!("output sentinel: {e}"))?;
    sync(stream, "H2D")?;
    project(stream, q4, &wg, &x, &mut gate, "gate")?;
    project(stream, q4, &wu, &x, &mut up, "up")?;
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: usilu_f32(size_t,size_t,const size_t*,const float*,float*).
    // 64-bit host checked; contiguous f32, exact lengths, distinct allocations.
    // The vendor grid-stride loop checks i<numel and writes every output once.
    unsafe {
        stream
            .launch_builder(silu)
            .arg(&numel)
            .arg(&num_dims)
            .arg(&layout)
            .arg(&gate)
            .arg(&mut activated)
            .launch(config)
    }
    .map_err(|e| format!("SiLU launch: {e}"))?;
    sync(stream, "SiLU")?;
    // SAFETY: bmul_f32 adds const float* rhs before output to the unary ABI.
    // Metadata [numel,1,1] selects both contiguous paths; no alias/broadcast.
    unsafe {
        stream
            .launch_builder(mul)
            .arg(&numel)
            .arg(&num_dims)
            .arg(&layout)
            .arg(&activated)
            .arg(&up)
            .arg(&mut hidden)
            .launch(config)
    }
    .map_err(|e| format!("multiply launch: {e}"))?;
    sync(stream, "multiply")?;
    project(stream, q4, &wd, &hidden, &mut y, "down")?;
    // Sole computation-path D2H. All host and device buffers remain live here.
    let actual = stream
        .clone_dtoh(&y)
        .map_err(|e| format!("final y D2H: {e}"))?;
    sync(stream, "D2H")?;
    Ok(actual)
}

pub fn run(report: &mut Value) -> Result<(), String> {
    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e}"))?;
    report["device_ordinal"] = json!(ctx.ordinal());
    report["device_name"] = json!(ctx.name().map_err(|e| format!("device name: {e}"))?);
    let stream = ctx.default_stream();
    let qptx = candle_kernels::QUANTIZED.ptx();
    let uptx = candle_kernels::UNARY.ptx();
    let bptx = candle_kernels::BINARY.ptx();
    report["ptx_sha256"] = json!({"QUANTIZED": sha256(qptx.as_bytes()),
        "UNARY": sha256(uptx.as_bytes()), "BINARY": sha256(bptx.as_bytes())});
    let qm = ctx
        .load_module(Ptx::from_src(qptx))
        .map_err(|e| format!("load QUANTIZED: {e}"))?;
    let um = ctx
        .load_module(Ptx::from_src(uptx))
        .map_err(|e| format!("load UNARY: {e}"))?;
    let bm = ctx
        .load_module(Ptx::from_src(bptx))
        .map_err(|e| format!("load BINARY: {e}"))?;
    let q4 = qm
        .load_function(KERNEL)
        .map_err(|e| format!("Q4 entry: {e}"))?;
    let silu = um
        .load_function(SILU)
        .map_err(|e| format!("SiLU entry: {e}"))?;
    let mul = bm
        .load_function(MUL)
        .map_err(|e| format!("multiply entry: {e}"))?;
    report["results"] = json!([]);
    for fixture in fixtures() {
        // Independent diagnostic oracle; never passed to expert().
        let reference = fixture.reference()?;
        let actual = expert(&stream, &q4, &silu, &mul, &fixture)?;
        let comparison = compare(&reference, &actual);
        report["results"].as_array_mut().ok_or("invalid result report")?.push(json!({
            "name": fixture.name, "fixture_sha256": fixture.fingerprint(),
            "output": actual, "output_f32_bits": actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "comparison": match &comparison {
                Ok(max) => json!({"pass": true, "max_abs_error": max}),
                Err(error) => json!({"pass": false, "error": error}),
            },
        }));
        comparison?;
    }
    // Unload functions/modules while the context can still report drop errors.
    drop((q4, silu, mul));
    drop((qm, um, bm));
    drop(stream);
    ctx.synchronize()
        .map_err(|e| format!("final context sync: {e}"))?;
    ctx.check_err()
        .map_err(|e| format!("CUDA deferred error: {e}"))?;
    Ok(())
}
