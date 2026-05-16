//! Sharded `RouteExperts` RPC scaffold (gist Part 4).
//!
//! The single-process server today owns every expert on local NVMe;
//! the gist's distributed plan in `docs/distributed.md` calls for
//! splitting the expert population across N worker nodes by hash and
//! issuing one zero-copy gRPC call per node per token.
//!
//! This module lays down the *transport-agnostic* primitives the
//! sharded path needs without pulling in a heavy gRPC runtime
//! (`tonic` + `prost` would inflate the dependency graph by ~150
//! crates and make the CPU-only build noticeably slower). Two things
//! live here:
//!
//! * [`shard_for_expert`] — the deterministic shard-routing function
//!   the request-receiving node uses to dispatch the top-K to the
//!   right worker. This is the **only** routing decision the
//!   transport layer needs to make at the request-receiving node;
//!   everything else (gating, combine) is unchanged.
//! * [`RouteExpertsRequest`] / [`RouteExpertsResponse`] — packed
//!   wire-format frames documenting the on-wire layout the gist
//!   prescribes ("`hidden_state` and `ffn_out` carried as packed
//!   `bytes` (f16)"). They serialise to a contiguous byte stream so
//!   a future tonic adapter can pass them through `tonic::Streaming`
//!   without a re-encode.
//!
//! When a follow-up PR wires in `tonic`, the `.proto` schema
//! mirroring the structs below lives at `proto/route_experts.proto`.
//! The Rust types in this file are intentionally *not* derived from
//! `prost`; they hand-roll a tiny length-prefixed format so the
//! engine can validate end-to-end on a unix socket before the
//! `tonic` dependency is committed to.

use std::convert::TryInto;
use std::io;

/// Deterministic shard assignment for an expert id (gist Part 4 /
/// `docs/distributed.md` "Sharding scheme"). Returns the **worker
/// index** in `[0, num_workers)` that owns the expert. `num_workers
/// == 0` returns `None` so the caller can fall through to the
/// single-process path.
///
/// Today we use `id % num_workers` because:
///
/// * the expert population is statically known at boot, so a hash
///   doesn't add anything;
/// * a contiguous-block partition (`id / shard_size`) would mean a
///   single node owns every adjacent expert id, which would hurt
///   spatial-prefetching efficiency on the local-NVMe drives;
/// * modulo distributes the top-K uniformly across shards by
///   construction — exactly what we want when the speculator's
///   lookahead window covers the whole expert population.
///
/// Stable: same `(id, num_workers)` always returns the same shard.
/// A follow-up rebalancer can swap the policy by changing this one
/// function — the transport types below carry the shard index, not
/// the policy.
#[inline]
pub fn shard_for_expert(id: u32, num_workers: u32) -> Option<u32> {
    if num_workers == 0 {
        return None;
    }
    Some(id % num_workers)
}

/// Group a flat top-K expert-id slice by shard ownership (gist Part
/// 4). Returns `Vec<(shard_idx, ids)>` so the caller can issue one
/// gRPC `RouteExperts` call per shard with the subset of ids that
/// live there. Ordering of `ids` within each shard preserves the
/// input order, which keeps the combiner's per-expert weight lookup
/// stable.
pub fn group_top_k_by_shard(top_k: &[u32], num_workers: u32) -> Vec<(u32, Vec<u32>)> {
    if num_workers == 0 {
        return Vec::new();
    }
    let mut buckets: Vec<Vec<u32>> = (0..num_workers).map(|_| Vec::new()).collect();
    for &id in top_k {
        let s = (id % num_workers) as usize;
        buckets[s].push(id);
    }
    buckets
        .into_iter()
        .enumerate()
        .filter(|(_, v)| !v.is_empty())
        .map(|(i, v)| (i as u32, v))
        .collect()
}

/// On-wire request frame for the sharded `RouteExperts` RPC (gist
/// Part 4 — mirrors `docs/distributed.md`'s gRPC sketch):
///
/// ```text
/// RouteExperts(request_id, layer_idx, expert_ids[], hidden_state)
///   -> (ffn_out: f16[d_model])
/// ```
///
/// The frame layout (little-endian):
///
/// | offset       | bytes | field           |
/// |--------------|-------|-----------------|
/// | 0            | 8     | request_id u64  |
/// | 8            | 4     | layer_idx u32   |
/// | 12           | 4     | d_model u32     |
/// | 16           | 4     | k u32 (#experts) |
/// | 20           | 4·k   | expert_ids[k]   |
/// | 20 + 4·k     | 2·d   | hidden_state f16[d_model] (raw bits) |
///
/// `hidden_state` is downcast to f16 on the wire (per the gist:
/// "f16 to halve wire size; the engine already loses no precision
/// against the f32 path because gate/up are followed by SwiGLU +
/// downcast"). The reference decoder upcasts back to f32 before
/// the local matmul.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteExpertsRequest {
    pub request_id: u64,
    pub layer_idx: u32,
    pub d_model: u32,
    pub expert_ids: Vec<u32>,
    /// Raw f16 bit pattern (length = `d_model`). Stored as `u16` to
    /// avoid pulling in a `half` crate dependency for the scaffold;
    /// the tonic integration will swap to `half::f16` once the
    /// dependency is committed to.
    pub hidden_state_f16: Vec<u16>,
}

impl RouteExpertsRequest {
    /// Encode the request to a contiguous byte stream using the
    /// packed layout documented on the struct. Returns the byte
    /// vector; the caller wraps it in whatever transport frame it's
    /// using (gRPC `bytes` payload, raw TCP, UDS).
    pub fn encode(&self) -> Vec<u8> {
        let k = self.expert_ids.len();
        let d = self.hidden_state_f16.len();
        let mut out = Vec::with_capacity(20 + 4 * k + 2 * d);
        out.extend_from_slice(&self.request_id.to_le_bytes());
        out.extend_from_slice(&self.layer_idx.to_le_bytes());
        out.extend_from_slice(&self.d_model.to_le_bytes());
        out.extend_from_slice(&(k as u32).to_le_bytes());
        for &id in &self.expert_ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        for &h in &self.hidden_state_f16 {
            out.extend_from_slice(&h.to_le_bytes());
        }
        out
    }

    /// Decode a wire frame produced by [`Self::encode`]. Validates
    /// every length prefix against the frame size so a corrupt
    /// packet cannot overrun the buffer.
    pub fn decode(buf: &[u8]) -> io::Result<Self> {
        if buf.len() < 20 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "RouteExpertsRequest: frame header is < 20 bytes",
            ));
        }
        let request_id = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let layer_idx = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let d_model = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let k = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
        let expert_off = 20usize;
        let hidden_off = expert_off + 4 * k;
        let hidden_end = hidden_off + 2 * d_model as usize;
        if buf.len() < hidden_end {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "RouteExpertsRequest: frame too short — need {hidden_end} bytes, got {}",
                    buf.len()
                ),
            ));
        }
        let mut expert_ids = Vec::with_capacity(k);
        for i in 0..k {
            let off = expert_off + 4 * i;
            expert_ids.push(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
        let mut hidden_state_f16 = Vec::with_capacity(d_model as usize);
        for i in 0..d_model as usize {
            let off = hidden_off + 2 * i;
            hidden_state_f16.push(u16::from_le_bytes(buf[off..off + 2].try_into().unwrap()));
        }
        Ok(Self {
            request_id,
            layer_idx,
            d_model,
            expert_ids,
            hidden_state_f16,
        })
    }
}

/// On-wire response frame: per-expert FFN outputs the shard
/// computed, ready to be folded back into the residual stream by the
/// combiner on the request-receiving node.
///
/// Layout (little-endian):
///
/// | offset       | bytes | field              |
/// |--------------|-------|--------------------|
/// | 0            | 8     | request_id u64     |
/// | 8            | 4     | d_model u32        |
/// | 12           | 4     | k u32              |
/// | 16           | 4·k   | expert_ids[k]      |
/// | 16 + 4·k     | 2·d·k | ffn_out f16[k][d]  |
///
/// Each `ffn_out[i]` corresponds to `expert_ids[i]` so the combiner
/// can join against the original top-K weights without reordering.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteExpertsResponse {
    pub request_id: u64,
    pub d_model: u32,
    pub expert_ids: Vec<u32>,
    /// Flat f16-bit array of length `k * d_model`. Row-major: row
    /// `i` is the FFN output for `expert_ids[i]`.
    pub ffn_out_f16: Vec<u16>,
}

impl RouteExpertsResponse {
    pub fn encode(&self) -> Vec<u8> {
        let k = self.expert_ids.len();
        let d = self.d_model as usize;
        let mut out = Vec::with_capacity(16 + 4 * k + 2 * d * k);
        out.extend_from_slice(&self.request_id.to_le_bytes());
        out.extend_from_slice(&self.d_model.to_le_bytes());
        out.extend_from_slice(&(k as u32).to_le_bytes());
        for &id in &self.expert_ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        debug_assert_eq!(self.ffn_out_f16.len(), k * d);
        for &h in &self.ffn_out_f16 {
            out.extend_from_slice(&h.to_le_bytes());
        }
        out
    }

    pub fn decode(buf: &[u8]) -> io::Result<Self> {
        if buf.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "RouteExpertsResponse: frame header is < 16 bytes",
            ));
        }
        let request_id = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let d_model = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let k = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
        let ids_off = 16usize;
        let ffn_off = ids_off + 4 * k;
        let ffn_end = ffn_off + 2 * (d_model as usize) * k;
        if buf.len() < ffn_end {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "RouteExpertsResponse: frame too short — need {ffn_end} bytes, got {}",
                    buf.len()
                ),
            ));
        }
        let mut expert_ids = Vec::with_capacity(k);
        for i in 0..k {
            let off = ids_off + 4 * i;
            expert_ids.push(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
        let mut ffn_out_f16 = Vec::with_capacity((d_model as usize) * k);
        for i in 0..(d_model as usize) * k {
            let off = ffn_off + 2 * i;
            ffn_out_f16.push(u16::from_le_bytes(buf[off..off + 2].try_into().unwrap()));
        }
        Ok(Self {
            request_id,
            d_model,
            expert_ids,
            ffn_out_f16,
        })
    }
}

/// IEEE-754 f32 → f16 narrowing without a `half` crate dependency.
/// Round-to-nearest-even on the mantissa, saturates to f16 ±∞ on
/// out-of-range exponents. Bit-identical to `half::f16::from_f32`
/// for finite, in-range inputs (the wire format only carries
/// finite activations after RMSNorm so subnormal handling matches
/// the reference path).
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0xFF {
        // NaN / Inf
        let mant16 = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | (0x1F << 10) | mant16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        // overflow → Inf
        return (sign << 15) | (0x1F << 10);
    }
    if new_exp <= 0 {
        // subnormal / underflow → 0 (good enough for activations)
        return sign << 15;
    }
    // Round-to-nearest-even on the 13 dropped mantissa bits.
    let round_bit = (mant >> 12) & 0x1;
    let sticky = (mant & 0xFFF) != 0;
    let mant16 = (mant >> 13) as u16;
    let mut out_mant = mant16;
    if round_bit == 1 && (sticky || (mant16 & 0x1) == 1) {
        out_mant = out_mant.wrapping_add(1);
        // Mantissa overflow → bump exponent
        if out_mant == 0x400 {
            return (sign << 15) | (((new_exp + 1) as u16) << 10);
        }
    }
    (sign << 15) | ((new_exp as u16) << 10) | (out_mant & 0x3FF)
}

/// IEEE-754 f16 bit pattern → f32. Subnormals are flushed-to-zero
/// (the wire format never carries them; see [`f32_to_f16_bits`]).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        // ±0 / subnormal (flushed to ±0)
        return f32::from_bits(sign << 31);
    }
    if exp == 0x1F {
        // ±Inf or NaN
        let mant32 = if mant != 0 { mant << 13 | 0x40_0000 } else { 0 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | mant32);
    }
    let new_exp = (exp - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (new_exp << 23) | (mant << 13))
}

/// Convenience: project an f32 hidden-state slice into the packed
/// f16 wire layout the request frame expects.
pub fn pack_hidden_state(hidden_f32: &[f32]) -> Vec<u16> {
    hidden_f32.iter().copied().map(f32_to_f16_bits).collect()
}

/// Convenience: invert [`pack_hidden_state`]. Used by the shard
/// worker before it runs the local FFN matmul.
pub fn unpack_hidden_state(hidden_f16: &[u16]) -> Vec<f32> {
    hidden_f16.iter().copied().map(f16_bits_to_f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_for_expert_distributes_modulo() {
        assert_eq!(shard_for_expert(0, 4), Some(0));
        assert_eq!(shard_for_expert(5, 4), Some(1));
        assert_eq!(shard_for_expert(7, 4), Some(3));
        // Zero workers => no shard
        assert_eq!(shard_for_expert(7, 0), None);
    }

    #[test]
    fn group_top_k_partitions_by_shard() {
        let groups = group_top_k_by_shard(&[1, 2, 5, 9, 6], 4);
        // shard 1: 1, 5, 9; shard 2: 2, 6
        let mut by_shard: std::collections::HashMap<u32, Vec<u32>> =
            std::collections::HashMap::new();
        for (s, ids) in groups {
            by_shard.insert(s, ids);
        }
        assert_eq!(by_shard.get(&1).unwrap(), &vec![1, 5, 9]);
        assert_eq!(by_shard.get(&2).unwrap(), &vec![2, 6]);
        assert!(by_shard.get(&0).is_none());
        assert!(by_shard.get(&3).is_none());
    }

    #[test]
    fn route_experts_request_roundtrip() {
        let req = RouteExpertsRequest {
            request_id: 0xdead_beef_cafe_babe,
            layer_idx: 7,
            d_model: 4,
            expert_ids: vec![3, 11, 25],
            hidden_state_f16: vec![
                f32_to_f16_bits(0.5),
                f32_to_f16_bits(-1.25),
                f32_to_f16_bits(0.0),
                f32_to_f16_bits(2.0),
            ],
        };
        let bytes = req.encode();
        let back = RouteExpertsRequest::decode(&bytes).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn route_experts_response_roundtrip() {
        let resp = RouteExpertsResponse {
            request_id: 42,
            d_model: 3,
            expert_ids: vec![5, 9],
            ffn_out_f16: vec![
                f32_to_f16_bits(0.1),
                f32_to_f16_bits(0.2),
                f32_to_f16_bits(0.3),
                f32_to_f16_bits(-0.1),
                f32_to_f16_bits(-0.2),
                f32_to_f16_bits(-0.3),
            ],
        };
        let bytes = resp.encode();
        let back = RouteExpertsResponse::decode(&bytes).unwrap();
        assert_eq!(resp, back);
    }

    #[test]
    fn f16_roundtrip_preserves_finite_values() {
        // f16 has ~3 decimal digits of precision; allow a small
        // relative envelope.
        for &x in &[0.0f32, 1.0, -1.0, 0.5, -2.5, 65504.0, -65504.0, 1e-3] {
            let bits = f32_to_f16_bits(x);
            let back = f16_bits_to_f32(bits);
            let err = if x == 0.0 {
                back.abs()
            } else {
                ((back - x) / x).abs()
            };
            assert!(err < 1e-3, "f16 roundtrip drifted: {x} -> {back}");
        }
    }

    #[test]
    fn truncated_frame_returns_unexpected_eof() {
        let req = RouteExpertsRequest {
            request_id: 1,
            layer_idx: 0,
            d_model: 2,
            expert_ids: vec![0],
            hidden_state_f16: vec![0, 0],
        };
        let bytes = req.encode();
        // Drop the trailing hidden-state bytes — decoder must reject.
        let truncated = &bytes[..bytes.len() - 3];
        let err = RouteExpertsRequest::decode(truncated).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }
}
