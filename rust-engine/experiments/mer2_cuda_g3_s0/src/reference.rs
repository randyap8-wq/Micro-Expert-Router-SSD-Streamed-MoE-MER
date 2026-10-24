//! Single-threaded scalar f32 token path. No intentional FMA/reassociation.
use crate::identity::{field, hex, sha256, source_hash, tokens_hash, Model};
use crate::{
    bits, finite, require, Result, Stop, D, EXPERTS, HEADS, HEAD_DIM, KIND, LAYERS, MAX_CONTEXT,
    Q4_BYTES, RMS_EPS, VOCAB,
};
use half::f16;
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn scalar(v: f32) -> Result<f32> {
    finite(&[v])?;
    Ok(v)
}
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    require(!a.is_empty() && a.len() == b.len(), "STOP_SHAPE", "dot")?;
    finite(a)?;
    finite(b)?;
    let mut sum = 0.0f32;
    for j in 0..a.len() {
        let product = scalar(a[j] * b[j])?;
        sum = scalar(sum + product)?;
    }
    Ok(sum)
}
pub fn dense(weights: &[f32], rows: usize, x: &[f32]) -> Result<Vec<f32>> {
    require(
        rows > 0 && rows <= D && x.len() == D && weights.len() == rows * D,
        "STOP_SHAPE",
        "dense",
    )?;
    weights.chunks_exact(D).map(|row| dot(row, x)).collect()
}
pub fn rms_norm(x: &[f32], gamma: &[f32]) -> Result<Vec<f32>> {
    require(x.len() == D && gamma.len() == D, "STOP_SHAPE", "RMSNorm")?;
    finite(gamma)?;
    let sum = dot(x, x)?;
    let mean = scalar(sum / D as f32)?;
    let denom = scalar(scalar(mean + RMS_EPS)?.sqrt())?;
    x.iter()
        .zip(gamma)
        .map(|(&x, &g)| scalar(scalar(x / denom)? * g))
        .collect()
}
pub fn softmax(logits: &[f32]) -> Result<Vec<f32>> {
    require(!logits.is_empty(), "STOP_SHAPE", "softmax empty")?;
    finite(logits)?;
    let mut max = logits[0];
    for &x in &logits[1..] {
        if x > max {
            max = x;
        }
    }
    let mut sum = 0.0f32;
    let mut values = Vec::with_capacity(logits.len());
    for &x in logits {
        let value = scalar(scalar(x - max)?.exp())?;
        sum = scalar(sum + value)?;
        values.push(value);
    }
    require(sum > 0.0, "STOP_NONFINITE", "softmax denominator")?;
    values.into_iter().map(|x| scalar(x / sum)).collect()
}
pub fn rope(values: &[f32], position: usize) -> Result<Vec<f32>> {
    require(
        values.len() == D && position < MAX_CONTEXT,
        "STOP_SHAPE",
        "RoPE",
    )?;
    finite(values)?;
    let mut out = values.to_vec();
    for h in 0..HEADS {
        for j in 0..HEAD_DIM / 2 {
            let exponent = (2 * j) as f32 / HEAD_DIM as f32;
            let theta = scalar(position as f32 / scalar(10000.0f32.powf(exponent))?)?;
            let (s, c) = (scalar(theta.sin())?, scalar(theta.cos())?);
            let i = h * HEAD_DIM + 2 * j;
            let (a, b) = (values[i], values[i + 1]);
            out[i] = scalar(scalar(a * c)? - scalar(b * s)?)?;
            out[i + 1] = scalar(scalar(a * s)? + scalar(b * c)?)?;
        }
    }
    Ok(out)
}
pub fn dequantize(block: &[u8]) -> Result<[f32; 32]> {
    require(block.len() == 18, "STOP_SHAPE", "Q4 block")?;
    let scale = scalar(f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32())?;
    let mut out = [0.0f32; 32];
    for j in 0..16 {
        out[j] = scalar(scale * ((block[j + 2] & 15) as i32 - 8) as f32)?;
        out[j + 16] = scalar(scale * ((block[j + 2] >> 4) as i32 - 8) as f32)?;
    }
    Ok(out)
}
pub fn q4_matvec(packed: &[u8], x: &[f32]) -> Result<Vec<f32>> {
    require(
        packed.len() == Q4_BYTES && x.len() == D,
        "STOP_SHAPE",
        "Q4 matrix",
    )?;
    finite(x)?;
    let mut output = Vec::with_capacity(D);
    for row in packed.chunks_exact(36) {
        let mut sum = 0.0f32;
        for (b, block) in row.chunks_exact(18).enumerate() {
            let weights = dequantize(block)?;
            for j in 0..32 {
                let product = scalar(weights[j] * x[b * 32 + j])?;
                sum = scalar(sum + product)?;
            }
        }
        output.push(sum);
    }
    Ok(output)
}
pub fn silu(values: &[f32]) -> Result<Vec<f32>> {
    finite(values)?;
    values
        .iter()
        .map(|&x| {
            let exponential = scalar((-x).exp())?;
            scalar(x / scalar(1.0f32 + exponential)?)
        })
        .collect()
}
fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    require(a.len() == D && b.len() == D, "STOP_SHAPE", "residual")?;
    a.iter().zip(b).map(|(&a, &b)| scalar(a + b)).collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    pub observed: bool,
    pub ranked: [u32; 4],
    pub probability_bits: [u32; 4],
    pub selected_weight_bits: [u32; 2],
}
pub fn route(logits: &[f32]) -> Result<Route> {
    require(logits.len() == EXPERTS, "STOP_SHAPE", "router logits")?;
    let probabilities = softmax(logits)?;
    let mut ranked = [0u32, 1, 2, 3];
    ranked.sort_by(|&a, &b| {
        probabilities[b as usize]
            .partial_cmp(&probabilities[a as usize])
            .unwrap()
            .then(a.cmp(&b))
    });
    let sum = scalar(probabilities[ranked[0] as usize] + probabilities[ranked[1] as usize])?;
    require(sum > 0.0, "STOP_NONFINITE", "top2 denominator")?;
    Ok(Route {
        observed: true,
        ranked,
        probability_bits: std::array::from_fn(|i| probabilities[i].to_bits()),
        selected_weight_bits: [
            scalar(probabilities[ranked[0] as usize] / sum)?.to_bits(),
            scalar(probabilities[ranked[1] as usize] / sum)?.to_bits(),
        ],
    })
}
pub fn greedy(logits: &[f32]) -> Result<u32> {
    require(logits.len() == VOCAB, "STOP_SHAPE", "logits")?;
    finite(logits)?;
    let mut best = 0;
    for i in 1..VOCAB {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

/// CPU-owned cache, physically indexed [head][position][dimension].
/// Only a full-token commit installs a newly prepared copy of both layers.
#[derive(Clone, Debug)]
pub struct LayerKv {
    k: [[[f32; HEAD_DIM]; MAX_CONTEXT]; HEADS],
    v: [[[f32; HEAD_DIM]; MAX_CONTEXT]; HEADS],
    len: usize,
}
impl Default for LayerKv {
    fn default() -> Self {
        Self {
            k: [[[0.0; HEAD_DIM]; MAX_CONTEXT]; HEADS],
            v: [[[0.0; HEAD_DIM]; MAX_CONTEXT]; HEADS],
            len: 0,
        }
    }
}
impl LayerKv {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn at(&self, position: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        require(position < self.len, "STOP_KV", "future/uncommitted KV read")?;
        let k = (0..HEADS).flat_map(|h| self.k[h][position]).collect();
        let v = (0..HEADS).flat_map(|h| self.v[h][position]).collect();
        Ok((k, v))
    }
    /// Pure copy constructor for analytical attention tests; never imports state
    /// into a Request. Position must be the current committed length.
    pub fn appended(&self, position: usize, k: &[f32], v: &[f32]) -> Result<Self> {
        require(
            position == self.len && position < MAX_CONTEXT && k.len() == D && v.len() == D,
            "STOP_KV",
            "append identity/shape",
        )?;
        finite(k)?;
        finite(v)?;
        let mut copy = self.clone();
        for h in 0..HEADS {
            copy.k[h][position].copy_from_slice(&k[h * HEAD_DIM..(h + 1) * HEAD_DIM]);
            copy.v[h][position].copy_from_slice(&v[h * HEAD_DIM..(h + 1) * HEAD_DIM]);
        }
        copy.len += 1;
        Ok(copy)
    }
    fn raw(&self, bytes: &mut Vec<u8>) {
        bytes.extend((self.len as u64).to_le_bytes());
        for data in [&self.k, &self.v] {
            for head in data {
                for pos in head {
                    for x in pos {
                        bytes.extend(x.to_bits().to_le_bytes());
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Attention {
    pub scores: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub context: Vec<f32>,
}
pub fn attention(
    q: &[f32],
    kv: &LayerKv,
    position: usize,
    k: &[f32],
    v: &[f32],
) -> Result<Attention> {
    require(
        q.len() == D
            && k.len() == D
            && v.len() == D
            && position == kv.len
            && position < MAX_CONTEXT,
        "STOP_KV",
        "causal position/shape",
    )?;
    finite(q)?;
    finite(k)?;
    finite(v)?;
    let scale = scalar(1.0f32 / (HEAD_DIM as f32).sqrt())?;
    let mut scores = Vec::new();
    let mut probabilities = Vec::new();
    let mut context = vec![0.0f32; D];
    for h in 0..HEADS {
        let range = h * HEAD_DIM..(h + 1) * HEAD_DIM;
        let mut hs = Vec::new();
        for p in 0..=position {
            let key: &[f32] = if p == position {
                &k[range.clone()]
            } else {
                &kv.k[h][p]
            };
            hs.push(scalar(dot(&q[range.clone()], key)? * scale)?);
        }
        let hp = softmax(&hs)?;
        for d in 0..HEAD_DIM {
            let mut sum = 0.0f32;
            for p in 0..=position {
                let value = if p == position {
                    v[h * HEAD_DIM + d]
                } else {
                    kv.v[h][p][d]
                };
                let product = scalar(hp[p] * value)?;
                sum = scalar(sum + product)?;
            }
            context[h * HEAD_DIM + d] = sum;
        }
        scores.extend(hs);
        probabilities.extend(hp);
    }
    Ok(Attention {
        scores,
        probabilities,
        context,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ticket {
    pub request: String,
    pub nonce: u64,
    pub attempt: u64,
    pub position: usize,
    pub input_id: u32,
    pub kv_generation: u64,
    pub kv_sha256: String,
    pub source_sha256: String,
    pub model_sha256: String,
}
impl Ticket {
    fn encode(&self, raw: &mut Vec<u8>) {
        field(raw, self.request.as_bytes());
        for n in [
            self.nonce,
            self.attempt,
            self.position as u64,
            self.input_id as u64,
            self.kv_generation,
        ] {
            field(raw, &n.to_le_bytes());
        }
        for s in [&self.kv_sha256, &self.source_sha256, &self.model_sha256] {
            field(raw, s.as_bytes());
        }
    }
}

/// Logical immutable CPU expert handle; these fields describe no GPU slot/event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertLease {
    pub layer: usize,
    pub expert: u32,
    pub rank: usize,
    pub slot: usize,
    pub generation: u64,
    pub weight_sha256: [String; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageReceipt {
    pub kind: String,
    pub ticket: Ticket,
    pub sequence: usize,
    pub layer: Option<usize>,
    pub stage: String,
    pub shape: Vec<u32>,
    pub dtype: u8,
    pub f32_bits: Vec<u32>,
    pub route: Option<Route>,
    pub expert: Option<ExpertLease>,
    pub greedy_id: Option<u32>,
    pub sha256: String,
}
impl StageReceipt {
    pub fn commitment(&self) -> String {
        let mut raw = b"MER2-G3-STAGE-v1\0".to_vec();
        field(&mut raw, b"G3-S0-RECEIPTS-v1");
        field(&mut raw, self.kind.as_bytes());
        self.ticket.encode(&mut raw);
        field(&mut raw, &(self.sequence as u64).to_le_bytes());
        field(
            &mut raw,
            &self
                .layer
                .map(|n| n as u64)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        field(&mut raw, self.stage.as_bytes());
        field(&mut raw, &[self.dtype]);
        field(&mut raw, &(self.shape.len() as u64).to_le_bytes());
        for dim in &self.shape {
            field(&mut raw, &dim.to_le_bytes());
        }
        let data: Vec<u8> = self.f32_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        field(&mut raw, &data);
        raw.push(u8::from(self.route.is_some()));
        if let Some(r) = &self.route {
            raw.push(u8::from(r.observed));
            for n in r
                .ranked
                .into_iter()
                .chain(r.probability_bits)
                .chain(r.selected_weight_bits)
            {
                raw.extend(n.to_le_bytes());
            }
        }
        raw.push(u8::from(self.expert.is_some()));
        if let Some(e) = &self.expert {
            for n in [
                e.layer as u64,
                e.expert as u64,
                e.rank as u64,
                e.slot as u64,
                e.generation,
            ] {
                field(&mut raw, &n.to_le_bytes());
            }
            for hash in &e.weight_sha256 {
                field(&mut raw, hash.as_bytes());
            }
        }
        raw.push(u8::from(self.greedy_id.is_some()));
        if let Some(id) = self.greedy_id {
            raw.extend(id.to_le_bytes());
        }
        sha256(&raw)
    }
    pub fn json(&self) -> Value {
        json!({"kind":self.kind,"request":self.ticket.request,"nonce":self.ticket.nonce,
            "attempt":self.ticket.attempt,"position":self.ticket.position,"input_id":self.ticket.input_id,
            "kv_generation":self.ticket.kv_generation,"prior_kv_sha256":self.ticket.kv_sha256,
            "source_sha256":self.ticket.source_sha256,"model_fixture_sha256":self.ticket.model_sha256,
            "sequence":self.sequence,"layer":self.layer,"stage":self.stage,"dtype":self.dtype,
            "shape":self.shape,"strides_bytes":contiguous_strides(&self.shape),"byte_len":self.f32_bits.len()*4,
            "f32_bits":self.f32_bits,"sha256":self.sha256,"finite":true,
            "min":self.f32_bits.iter().map(|&b|f32::from_bits(b)).reduce(f32::min),
            "max":self.f32_bits.iter().map(|&b|f32::from_bits(b)).reduce(f32::max),
            "route":self.route.as_ref().map(|r|json!({"observed":r.observed,"ranked":r.ranked,
                "probability_bits":r.probability_bits,"selected":&r.ranked[..2],"selected_weight_bits":r.selected_weight_bits})),
            "cpu_expert_lease":self.expert.as_ref().map(|e|json!({"layer":e.layer,"expert":e.expert,
                "rank":e.rank,"slot":e.slot,"generation":e.generation,"weight_sha256":e.weight_sha256})),
            "greedy_id":self.greedy_id})
    }
}
fn contiguous_strides(shape: &[u32]) -> Vec<u64> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 4;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= u64::from(shape[i]);
    }
    strides
}

struct Trace<'a, F: FnMut(&mut StageReceipt) -> Result<()>> {
    ticket: &'a Ticket,
    receipts: Vec<StageReceipt>,
    hook: &'a mut F,
}
impl<F: FnMut(&mut StageReceipt) -> Result<()>> Trace<'_, F> {
    fn record(
        &mut self,
        layer: Option<usize>,
        stage: &str,
        shape: &[u32],
        values: &[f32],
        route: Option<Route>,
        expert: Option<ExpertLease>,
        greedy_id: Option<u32>,
    ) -> Result<()> {
        finite(values)?;
        require(
            shape.iter().map(|&v| v as usize).product::<usize>() == values.len(),
            "STOP_SHAPE",
            "receipt shape",
        )?;
        let mut r = StageReceipt {
            kind: KIND.into(),
            ticket: self.ticket.clone(),
            sequence: self.receipts.len(),
            layer,
            stage: stage.into(),
            shape: shape.into(),
            dtype: 1,
            f32_bits: bits(values),
            route,
            expert,
            greedy_id,
            sha256: String::new(),
        };
        r.sha256 = r.commitment();
        let expected = r.clone();
        // Hook is the only adversarial seam. Values are never read back from it.
        // Equality to a sealed pre-hook receipt rejects even rehashed forgeries.
        (self.hook)(&mut r)?;
        require(
            r == expected,
            "STOP_RECEIPT",
            "mutated/stale/misbound stage receipt",
        )?;
        self.receipts.push(r);
        Ok(())
    }
    fn vector(&mut self, l: Option<usize>, s: &str, v: &[f32]) -> Result<()> {
        self.record(l, s, &[v.len() as u32], v, None, None, None)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepOutput {
    pub ticket: Ticket,
    pub stages: Vec<StageReceipt>,
    pub logits_bits: Vec<u32>,
    pub next_id: u32,
    pub tokens_sha256: String,
    pub committed_kv_sha256: String,
}
impl StepOutput {
    pub fn commitment(&self) -> String {
        let mut raw = b"MER2-G3-TOKEN-OUTPUT-v1\0".to_vec();
        self.ticket.encode(&mut raw);
        raw.extend((self.stages.len() as u64).to_le_bytes());
        for r in &self.stages {
            field(&mut raw, r.commitment().as_bytes());
            field(&mut raw, r.sha256.as_bytes());
        }
        for b in &self.logits_bits {
            raw.extend(b.to_le_bytes());
        }
        raw.extend(self.next_id.to_le_bytes());
        field(&mut raw, self.tokens_sha256.as_bytes());
        field(&mut raw, self.committed_kv_sha256.as_bytes());
        sha256(&raw)
    }
    pub fn json(&self) -> Value {
        json!({"kind":KIND,"stages":self.stages.iter().map(StageReceipt::json).collect::<Vec<_>>(),
            "logits_f32_bits":self.logits_bits,"greedy_next_token":self.next_id,"tokens_sha256":self.tokens_sha256,
            "committed_kv_sha256":self.committed_kv_sha256,"output_sha256":self.commitment(),
            "reference_status":"UNQUALIFIED_D2_REVIEW_REQUIRED","timing":null,"vram":null,"pcie":null})
    }
}

/// Session prevents duplicate request IDs/nonces within one model session.
/// No Clone for Session or Request; each request owns disjoint KV storage.
pub struct Session {
    model: Model,
    ids: BTreeSet<String>,
    next_nonce: u64,
}
impl Session {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            ids: BTreeSet::new(),
            next_nonce: 1,
        }
    }
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn request(&mut self, id: &str, initial_id: u32) -> Result<Request> {
        require(
            !id.is_empty() && id.is_ascii() && id.len() <= 128 && !self.ids.contains(id),
            "STOP_REQUEST",
            "invalid/duplicate request identity",
        )?;
        require(initial_id < VOCAB as u32, "STOP_INPUT", "initial token")?;
        let next = self
            .next_nonce
            .checked_add(1)
            .ok_or_else(|| Stop::new("STOP_REQUEST", "nonce overflow"))?;
        let request = Request {
            id: id.into(),
            nonce: self.next_nonce,
            model: self.model.hash().into(),
            source: source_hash()?,
            generation: 0,
            attempt: 1,
            kv: std::array::from_fn(|_| LayerKv::default()),
            inputs: Vec::new(),
            next_id: initial_id,
            last_output: None,
            first_error: None,
        };
        self.next_nonce = next;
        self.ids.insert(id.into());
        Ok(request)
    }
}

pub struct Request {
    id: String,
    nonce: u64,
    model: String,
    source: String,
    generation: u64,
    attempt: u64,
    kv: [LayerKv; LAYERS],
    inputs: Vec<u32>,
    next_id: u32,
    last_output: Option<String>,
    first_error: Option<Stop>,
}
impl Request {
    pub fn len(&self) -> usize {
        self.inputs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
    pub fn next_id(&self) -> u32 {
        self.next_id
    }
    pub fn first_error(&self) -> Option<&Stop> {
        self.first_error.as_ref()
    }
    pub fn kv(&self, layer: usize) -> Result<&LayerKv> {
        self.kv
            .get(layer)
            .ok_or_else(|| Stop::new("STOP_KV", "layer"))
    }
    pub fn committed_bytes(&self) -> Vec<u8> {
        let mut raw = b"MER2-G3-CPU-KV-v1\0".to_vec();
        field(&mut raw, self.id.as_bytes());
        field(&mut raw, &self.nonce.to_le_bytes());
        field(&mut raw, self.model.as_bytes());
        field(&mut raw, self.source.as_bytes());
        raw.extend(self.generation.to_le_bytes());
        for layer in &self.kv {
            layer.raw(&mut raw);
        }
        raw.extend((self.inputs.len() as u64).to_le_bytes());
        for t in &self.inputs {
            raw.extend(t.to_le_bytes());
        }
        raw
    }
    pub fn ticket(&self) -> Ticket {
        Ticket {
            request: self.id.clone(),
            nonce: self.nonce,
            attempt: self.attempt,
            position: self.len(),
            input_id: self.next_id,
            kv_generation: self.generation,
            kv_sha256: sha256(&self.committed_bytes()),
            source_sha256: self.source.clone(),
            model_sha256: self.model.clone(),
        }
    }
    pub fn verify_output(&self, output: &StepOutput) -> Result<()> {
        require(
            self.first_error.is_none()
                && self.last_output.as_deref() == Some(output.commitment().as_str()),
            "STOP_OUTPUT",
            "stale/foreign/mutated output",
        )
    }
    pub fn step_token(&mut self, model: &Model, ticket: Ticket) -> Result<StepOutput> {
        self.step_with_hook(model, ticket, &mut |_| Ok(()))
    }
    pub fn step_with_hook<F: FnMut(&mut StageReceipt) -> Result<()>>(
        &mut self,
        model: &Model,
        ticket: Ticket,
        hook: &mut F,
    ) -> Result<StepOutput> {
        if let Some(error) = &self.first_error {
            return Err(error.clone());
        }
        // All fallible computation and hooks operate on immutable self and local
        // shadows. No committed bytes, generation, output or next ID changes on Err.
        let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.prepare(model, &ticket, hook)
        }))
        .unwrap_or_else(|_| {
            Err(Stop::new(
                "STOP_PANIC",
                "CPU token preparation/hook panicked; committed KV retained",
            ))
        });
        match prepared {
            Err(error) => {
                self.first_error = Some(error.clone());
                Err(error)
            }
            Ok((output, kv, inputs, generation, attempt)) => {
                let output_hash = output.commitment();
                self.kv = kv;
                self.inputs = inputs;
                self.generation = generation;
                self.attempt = attempt;
                self.next_id = output.next_id;
                self.last_output = Some(output_hash);
                Ok(output)
            }
        }
    }
    fn prepare<F: FnMut(&mut StageReceipt) -> Result<()>>(
        &self,
        model: &Model,
        ticket: &Ticket,
        hook: &mut F,
    ) -> Result<(StepOutput, [LayerKv; LAYERS], Vec<u32>, u64, u64)> {
        require(
            *ticket == self.ticket(),
            "STOP_TICKET",
            "input/request/source/fixture/attempt/KV binding",
        )?;
        require(
            ticket.position < MAX_CONTEXT && ticket.input_id < VOCAB as u32,
            "STOP_INPUT",
            "position/token bound",
        )?;
        require(
            model.hash() == self.model && source_hash()? == self.source,
            "STOP_IDENTITY",
            "model/source binding",
        )?;
        require(
            self.kv.iter().all(|l| l.len == ticket.position),
            "STOP_KV",
            "layer lengths",
        )?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Stop::new("STOP_KV", "generation overflow"))?;
        let attempt = self
            .attempt
            .checked_add(1)
            .ok_or_else(|| Stop::new("STOP_TICKET", "attempt overflow"))?;
        let mut trace = Trace {
            ticket,
            receipts: Vec::new(),
            hook,
        };
        trace.vector(None, "begin", &[])?;
        let embedding = model.floats("embedding")?;
        let start = ticket.input_id as usize * D;
        let mut x = embedding[start..start + D].to_vec();
        trace.vector(None, "embedding", &x)?;
        let mut provisional = self.kv.clone();
        for l in 0..LAYERS {
            let n = rms_norm(&x, &model.floats(&format!("L{l}.attn_gamma"))?)?;
            trace.vector(Some(l), "attention_pre_norm", &n)?;
            let q = dense(&model.floats(&format!("L{l}.Wq"))?, D, &n)?;
            trace.vector(Some(l), "q", &q)?;
            let k = dense(&model.floats(&format!("L{l}.Wk"))?, D, &n)?;
            trace.vector(Some(l), "k", &k)?;
            let v = dense(&model.floats(&format!("L{l}.Wv"))?, D, &n)?;
            trace.vector(Some(l), "v", &v)?;
            let q = rope(&q, ticket.position)?;
            trace.vector(Some(l), "q_rope", &q)?;
            let k = rope(&k, ticket.position)?;
            trace.vector(Some(l), "k_rope", &k)?;
            let mut kv_values = k.clone();
            kv_values.extend(&v);
            trace.record(
                Some(l),
                "provisional_kv",
                &[2, 4, 16],
                &kv_values,
                None,
                None,
                None,
            )?;
            let att = attention(&q, &self.kv[l], ticket.position, &k, &v)?;
            trace.record(
                Some(l),
                "attention_scores",
                &[4, (ticket.position + 1) as u32],
                &att.scores,
                None,
                None,
                None,
            )?;
            trace.record(
                Some(l),
                "attention_softmax",
                &[4, (ticket.position + 1) as u32],
                &att.probabilities,
                None,
                None,
                None,
            )?;
            trace.vector(Some(l), "context", &att.context)?;
            let projected = dense(&model.floats(&format!("L{l}.Wo"))?, D, &att.context)?;
            trace.vector(Some(l), "attention_projection", &projected)?;
            let h_att = add(&x, &projected)?;
            trace.vector(Some(l), "h_att", &h_att)?;
            let n = rms_norm(&h_att, &model.floats(&format!("L{l}.moe_gamma"))?)?;
            trace.vector(Some(l), "moe_pre_norm", &n)?;
            let scores = dense(&model.floats(&format!("L{l}.router"))?, 4, &n)?;
            trace.vector(Some(l), "router_logits", &scores)?;
            let r = route(&scores)?;
            let probs = r.probability_bits.map(f32::from_bits);
            trace.record(
                Some(l),
                "router_probs",
                &[4],
                &probs,
                Some(r.clone()),
                None,
                None,
            )?;
            let selected = r.selected_weight_bits.map(f32::from_bits);
            trace.record(
                Some(l),
                "topk_weights",
                &[2],
                &selected,
                Some(r.clone()),
                None,
                None,
            )?;
            let mut agg = vec![0.0f32; D];
            for rank in 0..2 {
                let e = r.ranked[rank];
                let name = format!("L{l}.E{e}");
                let wg = model.tensor(&format!("{name}.gate"))?;
                let wu = model.tensor(&format!("{name}.up"))?;
                let wd = model.tensor(&format!("{name}.down"))?;
                let lease = ExpertLease {
                    layer: l,
                    expert: e,
                    rank,
                    slot: l * EXPERTS + e as usize,
                    generation: 1,
                    weight_sha256: [hex(&wg.hash()?), hex(&wu.hash()?), hex(&wd.hash()?)],
                };
                let gate = q4_matvec(&wg.raw, &n)?;
                trace.record(
                    Some(l),
                    "expert_gate",
                    &[64],
                    &gate,
                    None,
                    Some(lease.clone()),
                    None,
                )?;
                let up = q4_matvec(&wu.raw, &n)?;
                trace.record(
                    Some(l),
                    "expert_up",
                    &[64],
                    &up,
                    None,
                    Some(lease.clone()),
                    None,
                )?;
                let activated = silu(&gate)?;
                trace.record(
                    Some(l),
                    "expert_silu",
                    &[64],
                    &activated,
                    None,
                    Some(lease.clone()),
                    None,
                )?;
                let hidden: Vec<f32> = activated
                    .iter()
                    .zip(&up)
                    .map(|(&a, &u)| scalar(a * u))
                    .collect::<Result<_>>()?;
                trace.record(
                    Some(l),
                    "expert_activated",
                    &[64],
                    &hidden,
                    None,
                    Some(lease.clone()),
                    None,
                )?;
                let down = q4_matvec(&wd.raw, &hidden)?;
                trace.record(
                    Some(l),
                    "expert_down",
                    &[64],
                    &down,
                    None,
                    Some(lease.clone()),
                    None,
                )?;
                let weighted: Vec<f32> = down
                    .iter()
                    .map(|&v| scalar(selected[rank] * v))
                    .collect::<Result<_>>()?;
                trace.record(
                    Some(l),
                    "expert_weighted",
                    &[64],
                    &weighted,
                    None,
                    Some(lease),
                    None,
                )?;
                for j in 0..D {
                    agg[j] = scalar(agg[j] + weighted[j])?;
                }
            }
            trace.vector(Some(l), "agg", &agg)?;
            x = add(&h_att, &agg)?;
            trace.vector(Some(l), "h_out", &x)?;
            provisional[l] = self.kv[l].appended(ticket.position, &k, &v)?;
        }
        let final_norm = rms_norm(&x, &model.floats("final_gamma")?)?;
        trace.vector(None, "final_norm", &final_norm)?;
        let logits = dense(&model.floats("lm_head")?, VOCAB, &final_norm)?;
        trace.vector(None, "head_logits", &logits)?;
        let next_id = greedy(&logits)?;
        trace.record(None, "greedy", &[0], &[], None, None, Some(next_id))?;
        trace.vector(None, "before_commit", &[])?;
        let mut inputs = self.inputs.clone();
        inputs.push(ticket.input_id);
        // Compute the post-commit digest from a shadow, before touching self.
        let shadow = Request {
            id: self.id.clone(),
            nonce: self.nonce,
            model: self.model.clone(),
            source: self.source.clone(),
            generation,
            attempt,
            kv: provisional.clone(),
            inputs: inputs.clone(),
            next_id,
            last_output: None,
            first_error: None,
        };
        let output = StepOutput {
            ticket: ticket.clone(),
            stages: trace.receipts,
            logits_bits: bits(&logits),
            next_id,
            tokens_sha256: tokens_hash(&self.id, self.nonce, &inputs),
            committed_kv_sha256: sha256(&shadow.committed_bytes()),
        };
        Ok((output, provisional, inputs, generation, attempt))
    }
}
