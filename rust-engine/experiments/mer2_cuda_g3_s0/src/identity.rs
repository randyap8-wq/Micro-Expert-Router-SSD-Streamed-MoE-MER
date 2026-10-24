//! Canonical R1 tensor generation and identities. No model execution here.
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
        let contract: Value = serde_json::from_str(include_str!("../fixture-contract.json"))
            .map_err(|e| Stop::new("STOP_CONTRACT", e.to_string()))?;
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

/// Compiled source bytes are reported separately from model identity. D2 must
/// compare these to the external frozen D1 manifest; self-report is not approval.
pub fn source_files() -> BTreeMap<String, String> {
    [
        ("Cargo.toml", include_bytes!("../Cargo.toml").as_slice()),
        ("Cargo.lock", include_bytes!("../Cargo.lock").as_slice()),
        ("README.md", include_bytes!("../README.md").as_slice()),
        (
            "fixture-contract.json",
            include_bytes!("../fixture-contract.json").as_slice(),
        ),
        ("src/lib.rs", include_bytes!("lib.rs").as_slice()),
        (
            "src/reference.rs",
            include_bytes!("reference.rs").as_slice(),
        ),
        ("src/identity.rs", include_bytes!("identity.rs").as_slice()),
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
    let mut raw = b"MER2-G3-SOURCE-RECEIPT-v1\0".to_vec();
    name16(&mut raw, BASE)?;
    for (name, hash) in source_files() {
        name16(&mut raw, &name)?;
        name16(&mut raw, &hash)?;
    }
    Ok(sha256(&raw))
}

/// Receipt encoding choice G3-S0-RECEIPTS-v1 (not an approved golden hash).
/// Length-prefix every context field, including fixed-width integer bytes.
pub fn field(bytes: &mut Vec<u8>, raw: &[u8]) {
    bytes.extend((raw.len() as u64).to_le_bytes());
    bytes.extend(raw);
}
pub fn tokens_hash(request: &str, nonce: u64, tokens: &[u32]) -> String {
    let mut raw = b"MER2-G3-TOKENS-v1\0".to_vec();
    field(&mut raw, b"G3-S0-RECEIPTS-v1");
    field(&mut raw, request.as_bytes());
    field(&mut raw, &nonce.to_le_bytes());
    field(&mut raw, &(tokens.len() as u64).to_le_bytes());
    for t in tokens {
        field(&mut raw, &t.to_le_bytes());
    }
    sha256(&raw)
}
