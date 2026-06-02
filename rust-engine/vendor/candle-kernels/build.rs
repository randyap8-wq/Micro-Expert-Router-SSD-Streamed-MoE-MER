use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

// Resolve the absolute path to the glibc<->CUDA compatibility header that
// must be force-included ahead of every translation unit. The header lives
// in the consuming engine crate at `rust-engine/cuda/glibc_cuda_compat.h`;
// this vendored crate sits at `rust-engine/vendor/candle-kernels`, so the
// header is two levels up under `cuda/`. An explicit override is honoured via
// the `MER_GLIBC_CUDA_COMPAT_HEADER` environment variable.
//
// See the header itself for why this shim is needed (glibc 2.41 C23 sinpi/
// cospi declarations clash with CUDA's crt/math_functions.h). The header is
// internally guarded by `__GLIBC_PREREQ(2, 41)`, so force-including it is a
// strict no-op on older glibc, macOS and Windows.
fn glibc_cuda_compat_header() -> Option<PathBuf> {
    if let Ok(explicit) = env::var("MER_GLIBC_CUDA_COMPAT_HEADER") {
        let path = PathBuf::from(explicit);
        return path.canonicalize().ok().or(Some(path));
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").ok()?);
    let candidate = manifest_dir
        .join("..")
        .join("..")
        .join("cuda")
        .join("glibc_cuda_compat.h");
    candidate.canonicalize().ok()
}

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-env-changed=MER_GLIBC_CUDA_COMPAT_HEADER");

    // glibc 2.41 + CUDA host-compiler compatibility shim. cudaforge ignores
    // NVCC_PREPEND_FLAGS, so the only way to force-include the header is to
    // push `-include <abs path>` through the KernelBuilder `.arg()` API.
    let compat_header = glibc_cuda_compat_header();
    let compat_header_arg = compat_header
        .as_ref()
        .map(|h| h.to_string_lossy().into_owned());
    if let Some(ref header) = compat_header {
        println!("cargo::rerun-if-changed={}", header.display());
    }

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let mut ptx_builder = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&["moe_*.cu"]) // Exclude moe kernels for ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    if let Some(ref header) = compat_header_arg {
        ptx_builder = ptx_builder.arg("-include").arg(header);
    }
    let bindings = ptx_builder.build_ptx()?;

    bindings.write(&ptx_path)?;

    let mut moe_builder = KernelBuilder::default()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    if let Some(ref header) = compat_header_arg {
        moe_builder = moe_builder.arg("-include").arg(header);
    }

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
