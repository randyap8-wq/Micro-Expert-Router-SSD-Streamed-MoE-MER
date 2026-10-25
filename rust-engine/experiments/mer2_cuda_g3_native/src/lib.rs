//! Native selected-expert source draft; no build or GPU qualification is implied.
#![deny(unsafe_code)]

#[cfg(any(feature = "cuda", test))]
#[cfg_attr(feature = "cuda", allow(unsafe_code))]
mod cuda;
#[cfg(test)]
mod ownership;
#[cfg(feature = "cuda")]
pub use cuda::{NativeCudaAdapter, NativeEvidence};

use mer2_cuda_g3_s1::{identity::sha256, require, Result, Stop};
use std::{collections::BTreeMap, path::Path};

pub const KIND: &str = "MER2_G3_NATIVE_CUDA_UNQUALIFIED";
pub const S1_MANIFEST_SHA256: &str =
    "b66d6bd2854299fe2379119ca557c301fef9291cf47679f299168b992235a34e";

pub fn cuda_unavailable() -> Result<()> {
    Err(Stop::new(
        "STOP_CUDA_UNAVAILABLE",
        "native CUDA is disabled; separate build and execution authorization is required",
    ))
}

pub fn source_files() -> BTreeMap<String, String> {
    [
        ("Cargo.toml", include_bytes!("../Cargo.toml").as_slice()),
        ("README.md", include_bytes!("../README.md").as_slice()),
        ("src/lib.rs", include_bytes!("lib.rs").as_slice()),
        ("src/cuda.rs", include_bytes!("cuda.rs").as_slice()),
        ("src/main.rs", include_bytes!("main.rs").as_slice()),
    ]
    .into_iter()
    .map(|(name, bytes)| (name.into(), sha256(bytes)))
    .collect()
}

/// Bind the future executable and its physical sources to a separately reviewed manifest.
pub fn verify_native_sources(root: &Path, expected: &BTreeMap<String, String>) -> Result<()> {
    require(
        *expected == source_files(),
        "STOP_SOURCE_IDENTITY",
        "native compiled sources differ from external manifest",
    )?;
    for (name, digest) in expected {
        let path = root
            .join("rust-engine/experiments/mer2_cuda_g3_native")
            .join(name);
        let metadata = std::fs::symlink_metadata(&path)
            .map_err(|e| Stop::new("STOP_SOURCE_IDENTITY", e.to_string()))?;
        require(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "STOP_SOURCE_IDENTITY",
            "native regular file",
        )?;
        let bytes =
            std::fs::read(path).map_err(|e| Stop::new("STOP_SOURCE_IDENTITY", e.to_string()))?;
        require(sha256(&bytes) == *digest, "STOP_SOURCE_IDENTITY", name)?;
    }
    Ok(())
}

pub fn verify_s1_manifest(expected: &BTreeMap<String, String>) -> Result<()> {
    let manifest: String = expected
        .iter()
        .map(|(name, hash)| format!("{hash}  {name}\n"))
        .collect();
    require(
        sha256(manifest.as_bytes()) == S1_MANIFEST_SHA256,
        "STOP_SOURCE_IDENTITY",
        "accepted P2-R3 S1 manifest",
    )
}
