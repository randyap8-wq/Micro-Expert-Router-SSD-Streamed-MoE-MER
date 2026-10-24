//! Immutable owned host buffers and scoped receipts. No native pointer or CUDA API.
use crate::host_token::Ticket;
use crate::identity::{field, hex, sha256, Model, Tensor};
use crate::{finite, require, Result, Stop, D, EXPERTS, LAYERS, Q4_BYTES};
use serde_json::{json, Value};

/// Raw f32 bits give exact receipt equality; only finite, exactly 64-wide input is constructible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vector([u32; D]);
impl Vector {
    pub fn new(values: &[f32]) -> Result<Self> {
        require(values.len() == D, "STOP_SHAPE", "64 contiguous f32 values")?;
        finite(values)?;
        Ok(Self(std::array::from_fn(|i| values[i].to_bits())))
    }
    pub fn values(&self) -> [f32; D] {
        self.0.map(f32::from_bits)
    }
    pub fn bits(&self) -> &[u32; D] {
        &self.0
    }
    pub fn digest(&self) -> String {
        sha256(
            &self
                .0
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
}

/// Owned row-major Q4_0 payload; no caller-supplied dimensions survive validation.
#[derive(Clone, Debug)]
pub struct Q4Matrix {
    bytes: Box<[u8; Q4_BYTES]>,
    digest: String,
}
impl Q4Matrix {
    pub fn checked(tensor: &Tensor, name: &str, expected_digest: &str) -> Result<Self> {
        tensor.validate()?;
        require(
            tensor.name == name && tensor.dtype == 2 && tensor.shape == [64, 64],
            "STOP_SHAPE",
            "named Q4_0 64x64 tensor",
        )?;
        let digest = hex(&tensor.hash()?);
        require(
            digest == expected_digest,
            "STOP_IDENTITY",
            "Q4 tensor digest",
        )?;
        let bytes: [u8; Q4_BYTES] = tensor
            .raw
            .as_slice()
            .try_into()
            .map_err(|_| Stop::new("STOP_SHAPE", "2304 Q4 bytes"))?;
        Ok(Self {
            bytes: Box::new(bytes),
            digest,
        })
    }
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExpertKey {
    pub layer: usize,
    pub expert: u32,
    pub weight_digest: String,
}
#[derive(Clone, Debug)]
pub struct ExpertWeights {
    key: ExpertKey,
    matrices: [Q4Matrix; 3],
}
impl ExpertWeights {
    pub fn from_model(model: &Model, layer: usize, expert: u32) -> Result<Self> {
        require(
            layer < LAYERS && expert < EXPERTS as u32,
            "STOP_EXPERT",
            "expert range",
        )?;
        let contract = crate::identity::contract()?;
        let expected = &contract["s0_fixture"]["fixtures"][model.variant().id()]["tensors"];
        let matrix = |suffix: &str| -> Result<Q4Matrix> {
            let name = format!("L{layer}.E{expert}.{suffix}");
            let digest = expected[&name]["sha256"]
                .as_str()
                .ok_or_else(|| Stop::new("STOP_CONTRACT", "matrix digest missing"))?;
            Q4Matrix::checked(model.tensor(&name)?, &name, digest)
        };
        let matrices = [matrix("gate")?, matrix("up")?, matrix("down")?];
        let mut raw = b"MER2-G3-S1-EXPERT-v1\0".to_vec();
        field(&mut raw, &(layer as u64).to_le_bytes());
        field(&mut raw, &expert.to_le_bytes());
        for m in &matrices {
            field(&mut raw, m.digest().as_bytes());
        }
        Ok(Self {
            key: ExpertKey {
                layer,
                expert,
                weight_digest: sha256(&raw),
            },
            matrices,
        })
    }
    pub fn key(&self) -> &ExpertKey {
        &self.key
    }
    pub fn matrices(&self) -> &[Q4Matrix; 3] {
        &self.matrices
    }
    pub fn matrix_digests(&self) -> [String; 3] {
        std::array::from_fn(|i| self.matrices[i].digest.clone())
    }
}

/// Frozen residency policy: all eight experts once, no eviction/reload in P1a.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resident {
    pub key: ExpertKey,
    pub slot: usize,
    pub generation: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lease {
    pub ticket: Ticket,
    pub layer: usize,
    pub rank: usize,
    pub expert: u32,
    pub slot: usize,
    pub generation: u64,
    pub weight_digest: String,
    pub weight_sha256: [String; 3],
    pub context_identity: u64,
    pub event_identity: u64,
    pub input_sha256: String,
    pub selected_weight_bits: u32,
}
impl Lease {
    pub fn commitment(&self) -> String {
        let mut raw = b"MER2-G3-S1-LEASE-v1\0".to_vec();
        self.ticket.encode(&mut raw);
        for n in [
            self.layer as u64,
            self.rank as u64,
            self.expert as u64,
            self.slot as u64,
            self.generation,
            self.context_identity,
            self.event_identity,
            u64::from(self.selected_weight_bits),
        ] {
            field(&mut raw, &n.to_le_bytes());
        }
        for h in [&self.weight_digest, &self.input_sha256]
            .into_iter()
            .chain(self.weight_sha256.iter())
        {
            field(&mut raw, h.as_bytes());
        }
        sha256(&raw)
    }
    pub fn json(&self) -> Value {
        json!({"session":self.ticket.session,"request":self.ticket.request,"nonce":self.ticket.nonce,
            "attempt":self.ticket.attempt,"position":self.ticket.position,"layer":self.layer,"rank":self.rank,
            "expert":self.expert,"slot":self.slot,"generation":self.generation,"weight_digest":self.weight_digest,
            "weight_sha256":self.weight_sha256,"context_identity":self.context_identity,"event_identity":self.event_identity,
            "input_sha256":self.input_sha256,"selected_weight_bits":self.selected_weight_bits,"sha256":self.commitment()})
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationKind {
    UploadInput,
    Gate,
    Up,
    Silu,
    Multiply,
    Down,
    Readback,
}
pub const OPERATIONS: [OperationKind; 7] = [
    OperationKind::UploadInput,
    OperationKind::Gate,
    OperationKind::Up,
    OperationKind::Silu,
    OperationKind::Multiply,
    OperationKind::Down,
    OperationKind::Readback,
];
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operation {
    pub lease: Lease,
    pub sequence: usize,
    pub kind: OperationKind,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Submission {
    pub operation: Operation,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FfnStages(pub [Vector; 5]);
impl FfnStages {
    pub fn digest(&self) -> String {
        let mut raw = b"MER2-G3-S1-FFN-STAGES-v1\0".to_vec();
        for v in &self.0 {
            field(&mut raw, v.digest().as_bytes());
        }
        sha256(&raw)
    }
}
/// Untrusted adapter reply. The Runner validates it against its own immutable operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    pub operation: Operation,
    pub fence_confirmed: bool,
    pub stages: Option<FfnStages>,
    pub output_sha256: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitReceipt {
    pub context_identity: u64,
    pub residents: Vec<Resident>,
    pub confirmed: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CleanupReceipt {
    pub context_identity: u64,
    pub all_work_complete: bool,
    pub ownership_confirmed: bool,
}

/// Only the shared Runner constructs this after every completion and readback fence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfirmedFfn {
    pub(crate) lease: Lease,
    pub(crate) stages: FfnStages,
    pub(crate) completions: Vec<Completion>,
}
impl ConfirmedFfn {
    pub fn lease(&self) -> &Lease {
        &self.lease
    }
    pub fn stages(&self) -> &FfnStages {
        &self.stages
    }
    pub fn completions(&self) -> &[Completion] {
        &self.completions
    }
    pub fn json(&self) -> Value {
        json!({"kind":crate::KIND,"lease":self.lease.json(),"sha256":self.commitment(),
            "stage_f32_bits":self.stages.0.iter().map(|v|v.bits().to_vec()).collect::<Vec<_>>(),
            "completions":self.completions.iter().map(|c|json!({"lease":c.operation.lease.json(),
                "sequence":c.operation.sequence,"operation":format!("{:?}",c.operation.kind),
                "fence_confirmed":c.fence_confirmed,"output_sha256":c.output_sha256})).collect::<Vec<_>>()})
    }
    pub fn commitment(&self) -> String {
        let mut raw = b"MER2-G3-S1-CONFIRMED-v1\0".to_vec();
        field(&mut raw, self.lease.commitment().as_bytes());
        field(&mut raw, self.stages.digest().as_bytes());
        for c in &self.completions {
            field(&mut raw, c.operation.lease.commitment().as_bytes());
            field(&mut raw, &(c.operation.sequence as u64).to_le_bytes());
            field(&mut raw, format!("{:?}", c.operation.kind).as_bytes());
            raw.push(u8::from(c.fence_confirmed));
        }
        sha256(&raw)
    }
}

/// A future native implementation must retain all asynchronous buffers inside itself
/// before calling a fallible API. `Err` does not transfer ownership to the caller.
/// No borrowed input may outlive submit. Drain/release must check deferred errors.
/// P1a implementations and receipts are SIMULATED / UNQUALIFIED only.
pub trait Adapter {
    fn initialize(
        &mut self,
        context: u64,
        residents: &[Resident],
        weights: &[ExpertWeights],
    ) -> Result<InitReceipt>;
    fn submit(&mut self, operation: &Operation, input: &Vector) -> Result<Submission>;
    fn complete(&mut self, operation: &Operation) -> Result<Completion>;
    fn check(&mut self) -> Result<()>;
    fn drain(&mut self, context: u64) -> Result<CleanupReceipt>;
    fn release(&mut self, context: u64) -> Result<CleanupReceipt>;
}

pub(crate) mod sealed {
    pub trait Sealed {}
}
/// Sealed to the shared Runner, while its Adapter remains injectable. No arbitrary
/// provider can manufacture a ConfirmedFfn and bypass the common ownership guard.
pub trait FfnProvider: sealed::Sealed {
    fn begin_token(&mut self, ticket: &Ticket) -> Result<()>;
    fn execute(
        &mut self,
        ticket: &Ticket,
        layer: usize,
        rank: usize,
        route: &crate::host_token::Route,
        input: &Vector,
    ) -> Result<ConfirmedFfn>;
    fn end_token(&mut self, ticket: &Ticket) -> Result<()>;
    fn poison(&mut self, error: Stop);
}
