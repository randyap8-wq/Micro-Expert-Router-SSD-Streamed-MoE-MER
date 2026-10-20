use crate::{sha256, BASE, KERNEL, MUL, SILU};
use serde_json::{json, Map, Value};

pub fn source_identity() -> Result<Value, String> {
    let mut pinned = Map::new();
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s1/Cargo.lock"
        ));
        if actual != "8eecd4303bd3149a5c72d72099e6e4c411cdc6dc7b7354685c23fe7dc84a7931" {
            return Err(
                "STOP_SOURCE_API: rust-engine/experiments/mer2_cuda_q4_s1/Cargo.lock changed"
                    .into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s1/Cargo.lock".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s1/Cargo.toml"
        ));
        if actual != "514338aa417a6aaf2f49cc6c825c47281f24e238d573b0f87ac4cf6ed32e6e36" {
            return Err(
                "STOP_SOURCE_API: rust-engine/experiments/mer2_cuda_q4_s1/Cargo.toml changed"
                    .into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s1/Cargo.toml".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s1/README.md"
        ));
        if actual != "0f46c2ba50e34ec8709f6300a734ba121723c471f4f4e0bd7b150cd465447a6e" {
            return Err(
                "STOP_SOURCE_API: rust-engine/experiments/mer2_cuda_q4_s1/README.md changed".into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s1/README.md".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s1/src/cuda.rs"
        ));
        if actual != "c353bd9c0fef53750a7ab908979e9c91d35d2e685df9394dbe1da87647338af2" {
            return Err(
                "STOP_SOURCE_API: rust-engine/experiments/mer2_cuda_q4_s1/src/cuda.rs changed"
                    .into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s1/src/cuda.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s1/src/lib.rs"
        ));
        if actual != "a0c38dcccfd8b461a116f0b14d1cd9810db24db65ebde6e25a9ffd84bb7c8c2f" {
            return Err(
                "STOP_SOURCE_API: rust-engine/experiments/mer2_cuda_q4_s1/src/lib.rs changed"
                    .into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s1/src/lib.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s1/src/main.rs"
        ));
        if actual != "80cdf054b23d2ed1175b8058b4e1b53fe31ee134442a25e1e333beb7413ed472" {
            return Err(
                "STOP_SOURCE_API: rust-engine/experiments/mer2_cuda_q4_s1/src/main.rs changed"
                    .into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s1/src/main.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s1/tests/cpu.rs"
        ));
        if actual != "dc9558ecc2edc70bea582c07225b32d47bd39174302f75abcba57e5f2c9e9241" {
            return Err(
                "STOP_SOURCE_API: rust-engine/experiments/mer2_cuda_q4_s1/tests/cpu.rs changed"
                    .into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s1/tests/cpu.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/quantized.cu"
        ));
        if actual != "510c665e914346cbf361c4c60463a9cbad28032c09760cd90a227d9d2ab92972" {
            return Err(
                "STOP_SOURCE_API: rust-engine/vendor/candle-kernels/src/quantized.cu changed"
                    .into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/quantized.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/unary.cu"
        ));
        if actual != "bea116bebb9b84106661dd425ed56ceddabbff337762edcee05f6e9134fb7e5b" {
            return Err(
                "STOP_SOURCE_API: rust-engine/vendor/candle-kernels/src/unary.cu changed".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/unary.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/binary.cu"
        ));
        if actual != "dc9961a0013fe24972adf27a37a986df5d78081912a6de21684a0433ba8d1b17" {
            return Err(
                "STOP_SOURCE_API: rust-engine/vendor/candle-kernels/src/binary.cu changed".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/binary.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/binary_op_macros.cuh"
        ));
        if actual != "a83f97301a9ee52790b2c39b80e4f24393cf60dee153dbc68c0f802ea3ad95c4" {
            return Err("STOP_SOURCE_API: rust-engine/vendor/candle-kernels/src/binary_op_macros.cuh changed".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/binary_op_macros.cuh".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/cuda_utils.cuh"
        ));
        if actual != "45d3624977ab23fc4eb2c1c2e42357108dce477c1272e36b9b21cd54be75ecc0" {
            return Err(
                "STOP_SOURCE_API: rust-engine/vendor/candle-kernels/src/cuda_utils.cuh changed"
                    .into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/cuda_utils.cuh".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/compatibility.cuh"
        ));
        if actual != "ebbc4317b3fc34f1de608ba51eaf379c07a99ad44aa2836175707ff6d17e9865" {
            return Err(
                "STOP_SOURCE_API: rust-engine/vendor/candle-kernels/src/compatibility.cuh changed"
                    .into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/compatibility.cuh".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/src/lib.rs"));
        if actual != "91992b48ff931befcaccc854f639a173de1d1f831c7f5b45185b6bd811a7d7ad" {
            return Err(
                "STOP_SOURCE_API: rust-engine/vendor/candle-kernels/src/lib.rs changed".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/lib.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/Cargo.toml"));
        if actual != "fc19f7428b8df705b699c386dac86f25f7b3ad02dfa624146b88d3b351c39805" {
            return Err(
                "STOP_SOURCE_API: rust-engine/vendor/candle-kernels/Cargo.toml changed".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/Cargo.toml".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/build.rs"));
        if actual != "d2683e67d2c82d7d3b2e03e937c625155e2218702fe21cf38a164a82b8bdeedf" {
            return Err(
                "STOP_SOURCE_API: rust-engine/vendor/candle-kernels/build.rs changed".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/build.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../cuda/glibc_cuda_compat.h"));
        if actual != "7f84797c9c440978534dd54d7ee476b83e37e6d591063582f48e8f8b1b964fee" {
            return Err("STOP_SOURCE_API: rust-engine/cuda/glibc_cuda_compat.h changed".into());
        }
        pinned.insert("rust-engine/cuda/glibc_cuda_compat.h".into(), json!(actual));
    }
    Ok(json!({
        "base": BASE, "base_tree": "4c46e997d8bd7ae298ed659c909b73f7fbb243fa",
        "candle_kernels_version": "0.10.2", "cudarc_version": "0.19.7",
        "entries": {"QUANTIZED": KERNEL, "UNARY": SILU, "BINARY": MUL},
        "activation": "existing vendored f32 unary and binary PTX; no local CUDA source",
        "pinned_source_sha256": pinned,
        "s2_source_sha256": {
            "Cargo.toml": sha256(include_bytes!("../Cargo.toml")),
            "Cargo.lock": sha256(include_bytes!("../Cargo.lock")),
            "src/lib.rs": sha256(include_bytes!("../src/lib.rs")),
            "src/main.rs": sha256(include_bytes!("../src/main.rs")),
            "src/cuda.rs": sha256(include_bytes!("../src/cuda.rs")),
            "src/identity.rs": sha256(include_bytes!("../src/identity.rs")),
            "tests/cpu.rs": sha256(include_bytes!("../tests/cpu.rs")),
            "fixture-contract.json": sha256(include_bytes!("../fixture-contract.json")),
        }
    }))
}
