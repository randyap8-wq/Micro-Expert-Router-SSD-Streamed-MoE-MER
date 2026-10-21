use mer2_cuda_q4_s2::sha256;
use serde_json::{json, Map, Value};
pub fn source_identity() -> Result<Value, String> {
    let mut pinned = Map::new();
    {
        let actual = sha256(include_bytes!("../../../Cargo.lock"));
        if actual != "857922a38b4317e1767890f9d9d1438581314ffc0af8191d2a671a91bf7df9c5" {
            return Err("STOP_IDENTITY: rust-engine/Cargo.lock".into());
        }
        pinned.insert("rust-engine/Cargo.lock".into(), json!(actual));
    }
    {
        let actual = sha256(include_bytes!("../../../Cargo.toml"));
        if actual != "3bbe81d80e83619608bcec21641ccfae8e419d9f83b52490fd9da97bcd30aaff" {
            return Err("STOP_IDENTITY: rust-engine/Cargo.toml".into());
        }
        pinned.insert("rust-engine/Cargo.toml".into(), json!(actual));
    }
    {
        let actual = sha256(include_bytes!("../../../cuda/glibc_cuda_compat.h"));
        if actual != "7f84797c9c440978534dd54d7ee476b83e37e6d591063582f48e8f8b1b964fee" {
            return Err("STOP_IDENTITY: rust-engine/cuda/glibc_cuda_compat.h".into());
        }
        pinned.insert("rust-engine/cuda/glibc_cuda_compat.h".into(), json!(actual));
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s1/Cargo.lock"
        ));
        if actual != "8eecd4303bd3149a5c72d72099e6e4c411cdc6dc7b7354685c23fe7dc84a7931" {
            return Err("STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s1/Cargo.lock".into());
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
            return Err("STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s1/Cargo.toml".into());
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
            return Err("STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s1/README.md".into());
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
                "STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s1/src/cuda.rs".into(),
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
            return Err("STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s1/src/lib.rs".into());
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
                "STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s1/src/main.rs".into(),
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
                "STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s1/tests/cpu.rs".into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s1/tests/cpu.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s2/Cargo.lock"
        ));
        if actual != "74a5667daddfcaa84da8587ef55a78a74cea601df0822ca8d4240e01560752a6" {
            return Err("STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s2/Cargo.lock".into());
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s2/Cargo.lock".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s2/Cargo.toml"
        ));
        if actual != "17e8bd29940eeaf59276460ed0effd3121c8575d31c624dfb4f670a9da8fd1f9" {
            return Err("STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s2/Cargo.toml".into());
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s2/Cargo.toml".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s2/README.md"
        ));
        if actual != "f4dba0116272b16fe451e21e7d5604dfbc3153003732eb6eae6844ba98fc1978" {
            return Err("STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s2/README.md".into());
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s2/README.md".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s2/fixture-contract.json"
        ));
        if actual != "ae0ab643121adf2186b8a8ce3244d4526bc213a5aab1be617c41730dd9c5debf" {
            return Err(
                "STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s2/fixture-contract.json"
                    .into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s2/fixture-contract.json".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s2/src/cuda.rs"
        ));
        if actual != "f3953695d55521a6be91da7b0d74bfa4974953090edd3e279cb5ee5510ab7e31" {
            return Err(
                "STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s2/src/cuda.rs".into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s2/src/cuda.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s2/src/identity.rs"
        ));
        if actual != "ee5390ff93b3470dc21cae4fff9cfe0a824f793c51c0f773d5c5bbb2729859d8" {
            return Err(
                "STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s2/src/identity.rs".into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s2/src/identity.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s2/src/lib.rs"
        ));
        if actual != "b2ad4e4502af2040c04358a541341aadc5ee0ac9c245c0053198b082ed66e22f" {
            return Err("STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s2/src/lib.rs".into());
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s2/src/lib.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s2/src/main.rs"
        ));
        if actual != "af2a6b094c805003ea30351c2e29706e4495cc22ad57dda57ff62168cf1540e8" {
            return Err(
                "STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s2/src/main.rs".into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s2/src/main.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../experiments/mer2_cuda_q4_s2/tests/cpu.rs"
        ));
        if actual != "00787f8edc24e47dc44b084d1360c968c610461135dab1fdf3c85ca0829d9a02" {
            return Err(
                "STOP_IDENTITY: rust-engine/experiments/mer2_cuda_q4_s2/tests/cpu.rs".into(),
            );
        }
        pinned.insert(
            "rust-engine/experiments/mer2_cuda_q4_s2/tests/cpu.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/Cargo.lock"));
        if actual != "33c63216c7b4a45d65b50ff45fd62fce54efb26741bb73fdb9a297e9602f4d97" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/Cargo.lock".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/Cargo.lock".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/Cargo.toml"));
        if actual != "fc19f7428b8df705b699c386dac86f25f7b3ad02dfa624146b88d3b351c39805" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/Cargo.toml".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/Cargo.toml".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/README.md"));
        if actual != "d7e2cbac730f88fc60a50d07b3974e1964fd84310c46b5ec8cdb335d25e5715e" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/README.md".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/README.md".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/build.rs"));
        if actual != "d2683e67d2c82d7d3b2e03e937c625155e2218702fe21cf38a164a82b8bdeedf" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/build.rs".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/build.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/affine.cu"
        ));
        if actual != "8b7865c585e2b870d42f815cb0a73c0738c8de98bc18b4901bf32c1248d1592d" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/affine.cu".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/affine.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/binary.cu"
        ));
        if actual != "dc9961a0013fe24972adf27a37a986df5d78081912a6de21684a0433ba8d1b17" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/binary.cu".into());
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
            return Err(
                "STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/binary_op_macros.cuh".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/binary_op_macros.cuh".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/src/cast.cu"));
        if actual != "4f63c7f66b0079ff02cec0d69bf70fd31673c78e32aec48f8928289bf1bb37aa" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/cast.cu".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/cast.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/compatibility.cuh"
        ));
        if actual != "ebbc4317b3fc34f1de608ba51eaf379c07a99ad44aa2836175707ff6d17e9865" {
            return Err(
                "STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/compatibility.cuh".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/compatibility.cuh".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/src/conv.cu"));
        if actual != "6d9187017ba53c9511f28962f836c409b326073aab1ecfa036758e0499f9ad4c" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/conv.cu".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/conv.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/cuda_utils.cuh"
        ));
        if actual != "45d3624977ab23fc4eb2c1c2e42357108dce477c1272e36b9b21cd54be75ecc0" {
            return Err(
                "STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/cuda_utils.cuh".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/cuda_utils.cuh".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/src/ffi.rs"));
        if actual != "b1ff962f7f9161b5cf065ed13907311de4e539e84554d31dabb9e63ec25e60bf" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/ffi.rs".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/ffi.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/src/fill.cu"));
        if actual != "4fa211565dd589c2a69c881ee65b4e4e87c5f69ecede4071b2276368f21ea8fe" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/fill.cu".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/fill.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/indexing.cu"
        ));
        if actual != "8604566870e58418693517974f6ffe2ebc306dff641ec2de5e5d901646939fe7" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/indexing.cu".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/indexing.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/src/lib.rs"));
        if actual != "91992b48ff931befcaccc854f639a173de1d1f831c7f5b45185b6bd811a7d7ad" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/lib.rs".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/lib.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/moe/gguf.cuh"
        ));
        if actual != "cc80a31e79d17169e9a604907979885ebd8b857be3737dd97d2f1e7953d4771b" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/moe/gguf.cuh".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/moe/gguf.cuh".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/moe/moe_gguf.cu"
        ));
        if actual != "71aa0a76621d222737d069d62c437f70061f2a5bab31a5c1b1dca218d13de06f" {
            return Err(
                "STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/moe/moe_gguf.cu".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/moe/moe_gguf.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/moe/moe_utils.cuh"
        ));
        if actual != "0ba0412b593f1db391d6c332d37b9dde1d896726d12cb6f0d0cfc7ecaf36bb22" {
            return Err(
                "STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/moe/moe_utils.cuh".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/moe/moe_utils.cuh".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/moe/moe_wmma.cu"
        ));
        if actual != "708d9bb8a945fcb195fb69cc8a85d8935a0ef23b925234ea96b7410fdd70c687" {
            return Err(
                "STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/moe/moe_wmma.cu".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/moe/moe_wmma.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/moe/moe_wmma_gguf.cu"
        ));
        if actual != "536205c9d568b5c67b5e5889d88ace36dadcc8517242fd569f43ba11ca236735" {
            return Err(
                "STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/moe/moe_wmma_gguf.cu".into(),
            );
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/moe/moe_wmma_gguf.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/src/ptx.rs"));
        if actual != "1815594dc23daf6354c343059621165cf1ee4bd3cfa71729d35668b22de6239f" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/ptx.rs".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/ptx.rs".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/quantized.cu"
        ));
        if actual != "510c665e914346cbf361c4c60463a9cbad28032c09760cd90a227d9d2ab92972" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/quantized.cu".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/quantized.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/reduce.cu"
        ));
        if actual != "0fe950f0e52c58e20d118174e33cf1fdb1f0ba066d155e3e1658831d0df0eda1" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/reduce.cu".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/reduce.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!("../../../vendor/candle-kernels/src/sort.cu"));
        if actual != "de957d82ecd715b4f43cafe64464e3f82a00ae98724093739b09bda7d7c10129" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/sort.cu".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/sort.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/ternary.cu"
        ));
        if actual != "4e31593afde2f6acd3971c81ed94f6b34140de4681966183db221ee26cd965c9" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/ternary.cu".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/ternary.cu".into(),
            json!(actual),
        );
    }
    {
        let actual = sha256(include_bytes!(
            "../../../vendor/candle-kernels/src/unary.cu"
        ));
        if actual != "bea116bebb9b84106661dd425ed56ceddabbff337762edcee05f6e9134fb7e5b" {
            return Err("STOP_IDENTITY: rust-engine/vendor/candle-kernels/src/unary.cu".into());
        }
        pinned.insert(
            "rust-engine/vendor/candle-kernels/src/unary.cu".into(),
            json!(actual),
        );
    }
    Ok(
        json!({"base":crate::BASE,"tree":"d3ab4c1120726970356ac5e2dc4c64583b3f3de8","sole_parent":"be19b4217ba5e6b5bb963663bf3c3cf88566006f","cudarc":"0.19.7","cudarc_archive_sha256":"1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc","vendor":"candle-kernels 0.10.2","pinned_source_sha256":pinned,"g2_source_sha256":{
        "Cargo.toml":sha256(include_bytes!("../Cargo.toml")),
        "Cargo.lock":sha256(include_bytes!("../Cargo.lock")),
        "README.md":sha256(include_bytes!("../README.md")),
        "fixture-contract.json":sha256(include_bytes!("../fixture-contract.json")),
        "src/lib.rs":sha256(include_bytes!("../src/lib.rs")),
        "src/state.rs":sha256(include_bytes!("../src/state.rs")),
        "src/cuda.rs":sha256(include_bytes!("../src/cuda.rs")),
        "src/identity.rs":sha256(include_bytes!("../src/identity.rs")),
        "src/main.rs":sha256(include_bytes!("../src/main.rs")),
        "tests/cpu.rs":sha256(include_bytes!("../tests/cpu.rs")),
        }}),
    )
}
