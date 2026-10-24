//! S1 instance of frozen S0 fixture serialization, plus independently checked physical source identities.
//! Derived algorithms are not an independent mathematical oracle.
use crate::{
    finite, require, Result, Stop, BASE, D, EXPERTS, LAYERS, Q4_BYTES, SCHEMA, SEED, VOCAB,
    VOCAB_ID,
};
use half::f16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
pub fn sha256(bytes: &[u8]) -> String {
    hex(&digest(bytes))
}

pub fn name16(out: &mut Vec<u8>, name: &str) -> Result<()> {
    require(
        name.is_ascii() && name.len() <= u16::MAX as usize,
        "STOP_IDENTITY",
        "non-ASCII/oversize name",
    )?;
    out.extend((name.len() as u16).to_le_bytes());
    out.extend(name.as_bytes());
    Ok(())
}

pub fn word(label: &str, index: u64) -> Result<u32> {
    let mut bytes = SCHEMA.as_bytes().to_vec();
    bytes.push(0);
    name16(&mut bytes, SEED)?;
    name16(&mut bytes, label)?;
    bytes.extend(index.to_le_bytes());
    let hash = digest(&bytes);
    Ok(u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Primary,
    TieV1,
}
impl Variant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Primary => "PRIMARY_V1",
            Self::TieV1 => "TIE_V1",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub dtype: u8,
    pub shape: Vec<u32>,
    pub raw: Vec<u8>,
}
impl Tensor {
    pub fn validate(&self) -> Result<()> {
        require(
            !self.name.is_empty() && self.name.is_ascii() && self.name.len() <= 128,
            "STOP_TENSOR",
            "tensor name",
        )?;
        require(
            !self.shape.is_empty()
                && self.shape.len() <= 2
                && self.shape.iter().all(|&n| n > 0 && n <= 64),
            "STOP_TENSOR",
            "bounded tensor shape",
        )?;
        let elements: usize = self.shape.iter().map(|&n| n as usize).product();
        match self.dtype {
            1 => {
                require(
                    self.raw.len() == elements * 4,
                    "STOP_TENSOR",
                    "f32 raw length",
                )?;
                finite(&self.decode_f32()?)?;
            }
            2 => {
                require(
                    self.shape == [64, 64] && self.raw.len() == Q4_BYTES,
                    "STOP_TENSOR",
                    "Q4 shape/length",
                )?;
                for block in self.raw.chunks_exact(18) {
                    require(
                        f16::from_bits(u16::from_le_bytes([block[0], block[1]])).is_finite(),
                        "STOP_NONFINITE",
                        "Q4 scale",
                    )?;
                }
            }
            _ => return Err(Stop::new("STOP_TENSOR", "unknown dtype")),
        }
        Ok(())
    }
    pub fn decode_f32(&self) -> Result<Vec<f32>> {
        require(
            self.dtype == 1 && self.raw.len() % 4 == 0,
            "STOP_TENSOR",
            "f32 dtype/bytes",
        )?;
        Ok(self
            .raw
            .chunks_exact(4)
            .map(|b| f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect())
    }
    pub fn metadata(&self, bytes: &mut Vec<u8>) -> Result<()> {
        name16(bytes, &self.name)?;
        bytes.push(self.dtype);
        bytes.push(self.shape.len() as u8);
        for dim in &self.shape {
            bytes.extend(dim.to_le_bytes());
        }
        Ok(())
    }
    pub fn hash(&self) -> Result<[u8; 32]> {
        self.validate()?;
        let mut bytes = b"MER2-G3-TENSOR-v1\0".to_vec();
        self.metadata(&mut bytes)?;
        bytes.extend((self.raw.len() as u64).to_le_bytes());
        bytes.extend(&self.raw);
        Ok(digest(&bytes))
    }
}

/// Exact 41 names in ASCII sorting order; no optional tensors.
pub fn specifications() -> BTreeMap<String, (u8, Vec<u32>)> {
    let mut specs = BTreeMap::new();
    specs.insert("embedding".into(), (1, vec![VOCAB as u32, D as u32]));
    specs.insert("final_gamma".into(), (1, vec![D as u32]));
    specs.insert("lm_head".into(), (1, vec![VOCAB as u32, D as u32]));
    for l in 0..LAYERS {
        for n in ["attn_gamma", "moe_gamma"] {
            specs.insert(format!("L{l}.{n}"), (1, vec![64]));
        }
        for n in ["Wq", "Wk", "Wv", "Wo"] {
            specs.insert(format!("L{l}.{n}"), (1, vec![64, 64]));
        }
        specs.insert(format!("L{l}.router"), (1, vec![4, 64]));
        for e in 0..EXPERTS {
            for n in ["gate", "up", "down"] {
                specs.insert(format!("L{l}.E{e}.{n}"), (2, vec![64, 64]));
            }
        }
    }
    specs
}

pub fn generate_tensors(variant: Variant) -> Result<BTreeMap<String, Tensor>> {
    let mut tensors = BTreeMap::new();
    for (name, (dtype, shape)) in specifications() {
        let mut raw = Vec::new();
        if dtype == 1 {
            let count: usize = shape.iter().map(|&d| d as usize).product();
            for i in 0..count {
                let w = word(&name, i as u64)?;
                let value = if variant == Variant::TieV1 && name == "L0.router" {
                    0.0f32
                } else if name.ends_with("_gamma") {
                    1.0f32 + ((w % 17) as i32 - 8) as f32 / 64.0f32
                } else {
                    ((w % 65) as i32 - 32) as f32 / 128.0f32
                };
                raw.extend(value.to_bits().to_le_bytes());
            }
        } else {
            for b in 0..128u64 {
                let scale = [0x3400u16, 0x3800, 0xb400, 0xb800]
                    [(word(&format!("{name}.scale"), b)? % 4) as usize];
                raw.extend(scale.to_le_bytes());
                for j in 0..16u64 {
                    let lo = (word(&format!("{name}.nibble"), b * 32 + j)? % 16) as u8;
                    let hi = (word(&format!("{name}.nibble"), b * 32 + j + 16)? % 16) as u8;
                    raw.push(lo | (hi << 4));
                }
            }
        }
        let tensor = Tensor {
            name: name.clone(),
            dtype,
            shape,
            raw,
        };
        tensor.validate()?;
        tensors.insert(name, tensor);
    }
    Ok(tensors)
}

pub fn model_hash(tensors: &BTreeMap<String, Tensor>) -> Result<String> {
    let mut bytes = b"MER2-G3-MODEL-v1\0".to_vec();
    name16(&mut bytes, SEED)?;
    bytes.extend((tensors.len() as u32).to_le_bytes());
    for (name, t) in tensors {
        require(name == &t.name, "STOP_IDENTITY", "map key/name mismatch")?;
        t.metadata(&mut bytes)?;
        bytes.extend(t.hash()?);
    }
    bytes.extend(VOCAB_ID.as_bytes());
    Ok(sha256(&bytes))
}

#[derive(Clone, Debug)]
pub struct Model {
    tensors: BTreeMap<String, Tensor>,
    variant: Variant,
    hash: String,
}
impl Model {
    pub fn generate(variant: Variant) -> Result<Self> {
        Self::from_tensors(variant, generate_tensors(variant)?)
    }
    /// Reject mutated payloads against the independently serialized manifest.
    pub fn from_tensors(variant: Variant, tensors: BTreeMap<String, Tensor>) -> Result<Self> {
        let outer = contract()?;
        let contract = &outer["s0_fixture"];
        require(
            contract["schema"] == SCHEMA
                && contract["seed_ascii"] == SEED
                && contract["source_base"] == BASE,
            "STOP_CONTRACT",
            "fixture metadata",
        )?;
        require(
            contract["generator"] == "SHA256_R1_WORD_LE_V1"
                && contract["source"] == "D0_ORIGINAL_PLUS_R1_ADDENDUM"
                && contract["topology"]
                    == json!({"layers":2,"d_model":64,"d_ff":64,"vocab":32,
                "heads":4,"kv_heads":4,"head_dim":16,"rope_dim":16,"rope_base":10000,
                "rms_eps_f32_bits":crate::RMS_EPS.to_bits(),"max_context":4,"experts_per_layer":4,
                "top_k":2,"vocabulary":VOCAB_ID})
                && contract["requests"]
                    == json!([{"id":"A","initial_input":3},{"id":"B","initial_input":17}])
                && contract["processed_positions"] == json!([0, 1, 2])
                && contract["fourth_legal_position"] == 3
                && contract["first_rejected_position"] == 4,
            "STOP_CONTRACT",
            "topology/generator/request policy",
        )?;
        let expected = &contract["fixtures"][variant.id()];
        let specs = specifications();
        require(
            tensors.len() == specs.len()
                && expected["tensors"].as_object().map(|m| m.len()) == Some(specs.len()),
            "STOP_IDENTITY",
            "tensor count",
        )?;
        for (name, (dtype, shape)) in specs {
            let t = tensors
                .get(&name)
                .ok_or_else(|| Stop::new("STOP_IDENTITY", format!("missing {name}")))?;
            require(
                t.name == name && t.dtype == dtype && t.shape == shape,
                "STOP_IDENTITY",
                "tensor layout",
            )?;
            require(
                expected["tensors"][&name]["sha256"] == hex(&t.hash()?),
                "STOP_IDENTITY",
                &name,
            )?;
            require(
                expected["tensors"][&name]["dtype"] == dtype
                    && expected["tensors"][&name]["shape"] == json!(shape)
                    && expected["tensors"][&name]["raw_bytes"] == t.raw.len(),
                "STOP_CONTRACT",
                "manifest layout",
            )?;
        }
        let hash = model_hash(&tensors)?;
        require(
            expected["model_fixture_sha256"] == hash,
            "STOP_IDENTITY",
            "model digest",
        )?;
        Ok(Self {
            tensors,
            variant,
            hash,
        })
    }
    pub fn hash(&self) -> &str {
        &self.hash
    }
    pub fn variant(&self) -> Variant {
        self.variant
    }
    pub fn tensors(&self) -> &BTreeMap<String, Tensor> {
        &self.tensors
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Stop::new("STOP_TENSOR", name))
    }
    pub fn floats(&self, name: &str) -> Result<Vec<f32>> {
        self.tensor(name)?.decode_f32()
    }
}

/// Self-report is never approval: an external manifest is required by VerifiedSources.
pub fn source_files() -> BTreeMap<String, String> {
    [
        ("Cargo.toml", include_bytes!("../Cargo.toml").as_slice()),
        ("README.md", include_bytes!("../README.md").as_slice()),
        (
            "fixture-contract.json",
            include_bytes!("../fixture-contract.json").as_slice(),
        ),
        ("src/abi.rs", include_bytes!("abi.rs").as_slice()),
        (
            "src/host_token.rs",
            include_bytes!("host_token.rs").as_slice(),
        ),
        ("src/state.rs", include_bytes!("state.rs").as_slice()),
        ("src/identity.rs", include_bytes!("identity.rs").as_slice()),
        ("src/lib.rs", include_bytes!("lib.rs").as_slice()),
        ("src/main.rs", include_bytes!("main.rs").as_slice()),
        ("tests/cpu.rs", include_bytes!("../tests/cpu.rs").as_slice()),
    ]
    .into_iter()
    .map(|(n, b)| (n.into(), sha256(b)))
    .collect()
}
pub fn verify_source(expected: &BTreeMap<String, String>) -> Result<()> {
    require(
        *expected == source_files(),
        "STOP_SOURCE_IDENTITY",
        "compiled files differ from supplied source manifest",
    )
}
pub fn source_hash() -> Result<String> {
    let mut raw = b"MER2-G3-S1-SOURCE-RECEIPT-v1\0".to_vec();
    name16(&mut raw, BASE)?;
    for (name, hash) in source_files() {
        name16(&mut raw, &name)?;
        name16(&mut raw, &hash)?;
    }
    Ok(sha256(&raw))
}

/// S1 receipt encoding; separate from the sealed S0 receipt commitments.
/// Length-prefix every context field, including fixed-width integer bytes.
pub fn field(bytes: &mut Vec<u8>, raw: &[u8]) {
    bytes.extend((raw.len() as u64).to_le_bytes());
    bytes.extend(raw);
}
pub fn tokens_hash(request: &str, nonce: u64, tokens: &[u32]) -> String {
    let mut raw = b"MER2-G3-S1-TOKENS-v1\0".to_vec();
    field(&mut raw, b"G3-S1-RECEIPTS-v1");
    field(&mut raw, request.as_bytes());
    field(&mut raw, &nonce.to_le_bytes());
    field(&mut raw, &(tokens.len() as u64).to_le_bytes());
    for t in tokens {
        field(&mut raw, &t.to_le_bytes());
    }
    sha256(&raw)
}

pub const S0_AGGREGATE: &str = "77858567feeaf0b03736bdd913c954946b6ba6a7c480a79d891a635a3e40193b";
pub const PRIMARY_FIXTURE: &str =
    "6927d4063b79e0c45ee711e35c3184edffd70088f97cf2b484cd396e028751a1";
pub const LOGIT_BYTES: &str = "7f44266eede1fc67c1a0f2b6f1a7e651f629ca7f9e9d87d2a85a009560ccaf5c";
pub fn sealed_source_files() -> BTreeMap<String, String> {
    [
        (
            "Cargo.lock",
            "a45cef1b1d5af7a584581a6d9040426d265cd71c823540372216551560c89bee",
        ),
        (
            "Cargo.toml",
            "39a56d0f5da07088a7074d3763afb91579470929c631643ed99ebca77fc1dcb4",
        ),
        (
            "README.md",
            "1a1d7514d8712a69da0b58dffb4aeec3c03b0a857c43a293fd6b92591c4cda10",
        ),
        (
            "fixture-contract.json",
            "8cab28d538421f1125093eabad8f3039da5a41d2eed0bda588b40e211d42e7ca",
        ),
        (
            "src/identity.rs",
            "a6541b2dfa091641e43bc772e46cc993a94f976175942f34e12b8c3ff09d62f2",
        ),
        (
            "src/lib.rs",
            "f461c780a62eb1eb822c224f3bee9bae7971b73cc823e7d67697d5c443751f15",
        ),
        (
            "src/main.rs",
            "2fa8da152f666a579ed30354de34ae52c0a30ad001a48485155d5fbdd74a8503",
        ),
        (
            "src/reference.rs",
            "d5aae7a9db877f6901be2e83cd53d5019cce462727927af9be653e637aaa47d1",
        ),
        (
            "tests/cpu.rs",
            "190b98a1e8cde9c935a33e176529e9d743bb2e62a6cb6aa39f38c9da7e7fe215",
        ),
    ]
    .into_iter()
    .map(|(n, h)| (n.into(), h.into()))
    .collect()
}

pub fn contract_hash() -> String {
    sha256(include_bytes!("../fixture-contract.json"))
}
pub fn contract() -> Result<Value> {
    let c: Value = serde_json::from_str(include_str!("../fixture-contract.json"))
        .map_err(|e| Stop::new("STOP_CONTRACT", e.to_string()))?;
    require(c["schema"] == "MER2-G3-S1-P1A-CONTRACT-v1"
        && c["source_base"] == BASE
        && c["s0"]["source_aggregate_sha256"] == S0_AGGREGATE
        && c["s0"]["model_fixture_sha256"] == PRIMARY_FIXTURE
        && c["s0"]["logit_bytes_sha256"] == LOGIT_BYTES
        && c["s0"]["nine_source_files_sha256"] == json!(sealed_source_files())
        && c["geometry"] == json!({"layers":2,"d_model":64,"d_ff":64,"heads":4,"head_dim":16,
            "experts_per_layer":4,"top_k":2,"vocab":32,"max_context":4,"q4_block_bytes":18,
            "q4_matrix_bytes":2304,"q4_expert_bytes":6912,"all_expert_bytes":55296})
        && c["reference_stages_per_token"] == 66 && c["atomic_kv_layers"] == 2
        && c["operation_order"] == json!(["UploadInput","Gate","Up","Silu","Multiply","Down","Readback"])
        && c["routing"] == "OBSERVED_STABLE_SOFTMAX_PROBABILITY_DESCENDING_EXPERT_ID_ASCENDING_SELECTED_RENORMALIZATION_RANK_ORDER"
        && c["residency"] == json!({"maximum_experts":8,"policy":"LOAD_ALL_ONCE_NO_EVICTION_OR_RELOAD",
            "scope":"SIMULATED_LOGICAL_ONLY","one_outstanding_token":true})
        && c["cuda"] == "UNAVAILABLE_P1A" && c["compiled_ptx_sha256"] == "UNRESOLVED"
        && c["device_results"] == "UNRESOLVED", "STOP_CONTRACT", "frozen host/ownership contract")?;
    let arrays = c["s0"]["arrays"]
        .as_array()
        .ok_or_else(|| Stop::new("STOP_CONTRACT", "reference arrays"))?;
    require(
        arrays.len() == 6,
        "STOP_CONTRACT",
        "six historical logit arrays",
    )?;
    let mut raw = Vec::new();
    for a in arrays {
        let bits = a["logits_f32_bits"]
            .as_array()
            .ok_or_else(|| Stop::new("STOP_CONTRACT", "bits"))?;
        require(bits.len() == 32, "STOP_CONTRACT", "32 logits")?;
        for b in bits {
            let n = b
                .as_u64()
                .filter(|n| *n <= u64::from(u32::MAX))
                .ok_or_else(|| Stop::new("STOP_CONTRACT", "u32 logit"))?;
            raw.extend((n as u32).to_le_bytes());
        }
    }
    require(
        raw.len() == 768 && sha256(&raw) == LOGIT_BYTES,
        "STOP_CONTRACT",
        "accepted 192 raw bits",
    )?;
    Ok(c)
}

/// Independently read physical S0 bytes. Embedded/reference packet copies cannot satisfy this check.
pub fn verify_physical_s0(root: &std::path::Path) -> Result<BTreeMap<String, String>> {
    let mut actual = BTreeMap::new();
    for (name, expected) in sealed_source_files() {
        let p = root
            .join("rust-engine/experiments/mer2_cuda_g3_s0")
            .join(&name);
        let meta = std::fs::symlink_metadata(&p)
            .map_err(|e| Stop::new("STOP_SOURCE_INPUT", format!("{}: {e}", p.display())))?;
        require(
            meta.is_file() && !meta.file_type().is_symlink(),
            "STOP_SOURCE_INPUT",
            "S0 regular physical file",
        )?;
        let bytes = std::fs::read(&p).map_err(|e| Stop::new("STOP_SOURCE_INPUT", e.to_string()))?;
        let hash = sha256(&bytes);
        require(hash == expected, "STOP_SOURCE_INPUT", &name)?;
        if name == "fixture-contract.json" {
            let physical: Value = serde_json::from_slice(&bytes)
                .map_err(|e| Stop::new("STOP_CONTRACT", e.to_string()))?;
            require(
                contract()?["s0_fixture"] == physical,
                "STOP_CONTRACT",
                "embedded S0 fixture equals sealed physical fixture",
            )?;
        }
        actual.insert(name, hash);
    }
    let mut raw = b"MER2-G3-SOURCE-RECEIPT-v1\0".to_vec();
    name16(&mut raw, BASE)?;
    for (n, h) in &actual {
        name16(&mut raw, n)?;
        name16(&mut raw, h)?;
    }
    require(
        sha256(&raw) == S0_AGGREGATE,
        "STOP_SOURCE_INPUT",
        "S0 aggregate",
    )?;
    Ok(actual)
}

/// Non-forgeable proof of checks, not approval or runtime qualification. Rechecked per token.
#[derive(Debug)]
pub struct VerifiedSources {
    root: std::path::PathBuf,
    expected: BTreeMap<String, String>,
}
impl VerifiedSources {
    pub fn verify(root: &std::path::Path, expected: &BTreeMap<String, String>) -> Result<Self> {
        require(
            cfg!(target_endian = "little") && usize::BITS == 64,
            "STOP_ABI",
            "little-endian 64-bit host",
        )?;
        verify_source(expected)?;
        verify_physical_s0(root)?;
        contract()?;
        for (name, expected_hash) in expected {
            let p = root
                .join("rust-engine/experiments/mer2_cuda_g3_s1")
                .join(name);
            let meta = std::fs::symlink_metadata(&p)
                .map_err(|e| Stop::new("STOP_SOURCE_IDENTITY", e.to_string()))?;
            require(
                meta.is_file() && !meta.file_type().is_symlink(),
                "STOP_SOURCE_IDENTITY",
                "S1 regular file",
            )?;
            let bytes =
                std::fs::read(p).map_err(|e| Stop::new("STOP_SOURCE_IDENTITY", e.to_string()))?;
            require(
                sha256(&bytes) == *expected_hash,
                "STOP_SOURCE_IDENTITY",
                name,
            )?;
        }
        Ok(Self {
            root: root.into(),
            expected: expected.clone(),
        })
    }
    pub fn recheck(&self) -> Result<()> {
        Self::verify(&self.root, &self.expected).map(|_| ())
    }
}
