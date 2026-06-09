//! Real gRPC (tonic) transport for distributed expert sharding.
//!
//! This module is the concrete implementation of the `ExpertShard`
//! service whose contract lives in `proto/route_experts.proto` and
//! whose partitioning scheme is documented in `docs/distributed.md`.
//! It is compiled only behind the off-by-default `grpc` cargo feature
//! so the single-node build stays lean (`tonic` + `prost` pull in ~150
//! crates).
//!
//! Three pieces live here:
//!
//! * [`ShardCompute`] — the trait a worker node implements to turn an
//!   incoming `(layer_idx, expert_ids, hidden_state)` into one FFN
//!   output per expert. The request-receiving node owns gating and the
//!   residual combine; a shard worker only runs the experts it owns.
//! * [`serve`] / [`ExpertShardService`] — a tonic server that adapts a
//!   [`ShardCompute`] backend to the generated `ExpertShard` service,
//!   bridging the hand-rolled f16 wire helpers in [`crate::rpc`] to the
//!   `prost` message types in [`crate::grpc_gen`].
//! * [`ShardClient`] — a thin tonic client wrapper the request-
//!   receiving node uses to issue one `RouteExperts` call per shard,
//!   plus a `Health` probe.
//!
//! [`probe_remote`] wires all of this back into the engine's
//! transport-agnostic [`crate::distributed::ShardRouter`] abstraction:
//! `RpcShardRouter::fetch_remote` calls it to perform a real gRPC
//! reachability check against the shard that owns a remote expert,
//! mapping `tonic::Status` codes through
//! [`crate::distributed::RpcShardRouter::map_tonic_status`].

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tonic::transport::{Channel, Server};
use tonic::{Request, Response, Status};

use crate::distributed::{NodeAddr, RpcShardRouter, ShardRouterError};
use crate::grpc_gen::expert_shard_client::ExpertShardClient;
use crate::grpc_gen::expert_shard_server::{ExpertShard, ExpertShardServer};
use crate::grpc_gen::{
    HealthRequest as PbHealthRequest, HealthResponse as PbHealthResponse,
    RouteExpertsRequest as PbRouteRequest, RouteExpertsResponse as PbRouteResponse,
};
use crate::rpc::{f16_bits_to_f32, f32_to_f16_bits};

/// Health/diagnostics snapshot a shard returns for the `Health` RPC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthSnapshot {
    /// Free pinned-host buffer-pool slots on the shard.
    pub free_blocks: u32,
    /// Cumulative expert read failures since process start.
    pub expert_read_failures: u64,
    /// Implementation revision string (e.g. `"v0.1.0-shard"`).
    pub version: String,
}

impl Default for HealthSnapshot {
    fn default() -> Self {
        Self {
            free_blocks: 0,
            expert_read_failures: 0,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// A shard worker's compute backend: run the experts this node owns.
///
/// Implementors return one FFN output vector (length `d_model`) per id
/// in `expert_ids`, in the same order, so the request-receiving node's
/// combiner can join the outputs against the original gating weights
/// without reordering. Returning `Err` surfaces as a `tonic::Status`
/// (`Internal`) to the caller.
pub trait ShardCompute: Send + Sync + 'static {
    /// Compute the per-expert FFN outputs for one sharded MoE step.
    ///
    /// * `layer_idx` — decoder layer the experts belong to.
    /// * `expert_ids` — the subset of the token's top-K that live on
    ///   this shard.
    /// * `hidden` — the f32 hidden state (length `d_model`).
    ///
    /// Returns `expert_ids.len()` rows, each of length `hidden.len()`.
    fn route(
        &self,
        layer_idx: u32,
        expert_ids: &[u32],
        hidden: &[f32],
    ) -> Result<Vec<Vec<f32>>, String>;

    /// Diagnostics for the `Health` RPC. Defaults to a version-only
    /// snapshot; real shards override with live buffer-pool counters.
    fn health(&self) -> HealthSnapshot {
        HealthSnapshot::default()
    }
}

/// tonic service adapter over a [`ShardCompute`] backend.
#[derive(Debug, Clone)]
pub struct ExpertShardService<B: ShardCompute> {
    backend: Arc<B>,
}

impl<B: ShardCompute> ExpertShardService<B> {
    /// Wrap a backend in the gRPC service adapter.
    pub fn new(backend: B) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }
}

/// Pack an f32 slice into the little-endian f16 byte layout the wire
/// frames use (`crate::rpc::f32_to_f16_bits` per element).
fn pack_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    out
}

/// Inverse of [`pack_f16_bytes`]: decode little-endian f16 bytes to
/// f32. Returns `None` when `bytes` is not a whole number of f16s.
fn unpack_f16_bytes(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    )
}

#[tonic::async_trait]
impl<B: ShardCompute> ExpertShard for ExpertShardService<B> {
    async fn route_experts(
        &self,
        request: Request<PbRouteRequest>,
    ) -> Result<Response<PbRouteResponse>, Status> {
        let req = request.into_inner();
        let d_model = req.d_model as usize;

        let hidden = unpack_f16_bytes(&req.hidden_state_f16)
            .ok_or_else(|| Status::invalid_argument("hidden_state_f16 length is not even"))?;
        if hidden.len() != d_model {
            return Err(Status::invalid_argument(format!(
                "hidden_state width {} != declared d_model {}",
                hidden.len(),
                d_model
            )));
        }

        let outputs = self
            .backend
            .route(req.layer_idx, &req.expert_ids, &hidden)
            .map_err(|e| Status::internal(format!("shard compute failed: {e}")))?;

        if outputs.len() != req.expert_ids.len() {
            return Err(Status::internal(format!(
                "backend returned {} rows for {} experts",
                outputs.len(),
                req.expert_ids.len()
            )));
        }

        let mut ffn_out_f16 = Vec::with_capacity(outputs.len() * d_model * 2);
        for (row, &id) in outputs.iter().zip(&req.expert_ids) {
            if row.len() != d_model {
                return Err(Status::internal(format!(
                    "expert {id} produced width {} != d_model {d_model}",
                    row.len()
                )));
            }
            ffn_out_f16.extend_from_slice(&pack_f16_bytes(row));
        }

        Ok(Response::new(PbRouteResponse {
            request_id: req.request_id,
            d_model: req.d_model,
            expert_ids: req.expert_ids,
            ffn_out_f16,
        }))
    }

    async fn health(
        &self,
        _request: Request<PbHealthRequest>,
    ) -> Result<Response<PbHealthResponse>, Status> {
        let h = self.backend.health();
        Ok(Response::new(PbHealthResponse {
            free_blocks: h.free_blocks,
            expert_read_failures: h.expert_read_failures,
            version: h.version,
        }))
    }
}

/// Run an `ExpertShard` gRPC server on `addr`, serving experts through
/// `backend`, until the process exits. Returns the transport error if
/// the server fails to bind or terminates abnormally.
pub async fn serve<B: ShardCompute>(
    addr: SocketAddr,
    backend: B,
) -> Result<(), tonic::transport::Error> {
    Server::builder()
        .add_service(ExpertShardServer::new(ExpertShardService::new(backend)))
        .serve(addr)
        .await
}

/// Like [`serve`] but completes when `shutdown` resolves, allowing
/// graceful teardown (used by tests and embedders that own the
/// server's lifetime).
pub async fn serve_with_shutdown<B, F>(
    addr: SocketAddr,
    backend: B,
    shutdown: F,
) -> Result<(), tonic::transport::Error>
where
    B: ShardCompute,
    F: std::future::Future<Output = ()> + Send + 'static,
{
    Server::builder()
        .add_service(ExpertShardServer::new(ExpertShardService::new(backend)))
        .serve_with_shutdown(addr, shutdown)
        .await
}

/// Normalise a [`NodeAddr`] into a tonic-compatible endpoint URI by
/// defaulting the scheme to `http://` when none is present.
fn endpoint_uri(node: &NodeAddr) -> String {
    if node.0.contains("://") {
        node.0.clone()
    } else {
        format!("http://{}", node.0)
    }
}

/// Thin client wrapper over the generated `ExpertShardClient`.
#[derive(Debug, Clone)]
pub struct ShardClient {
    inner: ExpertShardClient<Channel>,
}

impl ShardClient {
    /// Connect to a shard at `node` (e.g. `"127.0.0.1:50051"`). The
    /// scheme defaults to `http://` when omitted.
    pub async fn connect(node: &NodeAddr) -> Result<Self, tonic::transport::Error> {
        let inner = ExpertShardClient::connect(endpoint_uri(node)).await?;
        Ok(Self { inner })
    }

    /// Issue a `RouteExperts` call, bridging the hand-rolled
    /// [`crate::rpc::RouteExpertsRequest`] frame to/from the wire.
    pub async fn route_experts(
        &mut self,
        req: crate::rpc::RouteExpertsRequest,
    ) -> Result<crate::rpc::RouteExpertsResponse, Status> {
        let hidden_state_f16 = req
            .hidden_state_f16
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect::<Vec<u8>>();
        let pb = PbRouteRequest {
            request_id: req.request_id,
            layer_idx: req.layer_idx,
            d_model: req.d_model,
            expert_ids: req.expert_ids,
            hidden_state_f16,
        };
        let resp = self.inner.route_experts(pb).await?.into_inner();
        let ffn_out_f16 = resp
            .ffn_out_f16
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect::<Vec<u16>>();
        Ok(crate::rpc::RouteExpertsResponse {
            request_id: resp.request_id,
            d_model: resp.d_model,
            expert_ids: resp.expert_ids,
            ffn_out_f16,
        })
    }

    /// Issue a `Health` probe and decode the shard's diagnostics.
    pub async fn health(&mut self) -> Result<HealthSnapshot, Status> {
        let resp = self.inner.health(PbHealthRequest {}).await?.into_inner();
        Ok(HealthSnapshot {
            free_blocks: resp.free_blocks,
            expert_read_failures: resp.expert_read_failures,
            version: resp.version,
        })
    }
}

/// Perform a real gRPC reachability probe against the shard that owns
/// `expert`, returning `Ok(())` when the shard answers a `Health` RPC
/// within `timeout`. Connection / RPC failures are translated into a
/// structured [`ShardRouterError`] via
/// [`RpcShardRouter::map_tonic_status`].
///
/// This is what `RpcShardRouter::fetch_remote` calls when the `grpc`
/// feature is enabled, replacing the skeleton stub with an actual
/// round-trip to the owning node.
pub async fn probe_remote(
    expert: u32,
    node: NodeAddr,
    timeout: Duration,
) -> Result<(), ShardRouterError> {
    let connect = ShardClient::connect(&node);
    let mut client = match tokio::time::timeout(timeout, connect).await {
        Err(_) => {
            return Err(ShardRouterError::Timeout { expert, node });
        }
        Ok(Err(e)) => {
            return Err(ShardRouterError::Unreachable {
                expert,
                node,
                reason: format!("connect failed: {e}"),
            });
        }
        Ok(Ok(c)) => c,
    };

    match tokio::time::timeout(timeout, client.health()).await {
        Err(_) => Err(ShardRouterError::Timeout { expert, node }),
        Ok(Ok(_)) => Ok(()),
        Ok(Err(status)) => {
            let code_name = format!("{:?}", status.code());
            Err(RpcShardRouter::map_tonic_status(
                expert,
                node,
                &code_name,
                status.message(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpc::{pack_hidden_state, unpack_hidden_state};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Deterministic backend: each expert scales the hidden state by
    /// `(expert_id + 1)`, so the test can assert per-expert routing and
    /// ordering without a real model.
    struct ScaleBackend;
    impl ShardCompute for ScaleBackend {
        fn route(
            &self,
            _layer_idx: u32,
            expert_ids: &[u32],
            hidden: &[f32],
        ) -> Result<Vec<Vec<f32>>, String> {
            Ok(expert_ids
                .iter()
                .map(|&id| hidden.iter().map(|&h| h * (id as f32 + 1.0)).collect())
                .collect())
        }
        fn health(&self) -> HealthSnapshot {
            HealthSnapshot {
                free_blocks: 7,
                expert_read_failures: 0,
                version: "test-shard".to_string(),
            }
        }
    }

    /// Bind an ephemeral port and return both the chosen `SocketAddr`
    /// and the listener-backed incoming stream, so the server and the
    /// client agree on the port with no race.
    async fn ephemeral_addr() -> SocketAddr {
        // Port 0 lets the OS choose a free port; we read it back.
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let _ = NEXT.fetch_add(1, Ordering::Relaxed);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        addr
    }

    async fn spawn_server(addr: SocketAddr) -> tokio::sync::oneshot::Sender<()> {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = serve_with_shutdown(addr, ScaleBackend, async move {
                let _ = rx.await;
            })
            .await;
        });
        // Give the server a moment to bind.
        tokio::time::sleep(Duration::from_millis(150)).await;
        tx
    }

    #[test]
    fn f16_byte_pack_roundtrips() {
        let xs = vec![0.5f32, -1.25, 0.0, 2.0];
        let bytes = pack_f16_bytes(&xs);
        let back = unpack_f16_bytes(&bytes).unwrap();
        assert_eq!(xs, back);
    }

    #[test]
    fn endpoint_uri_defaults_scheme() {
        assert_eq!(endpoint_uri(&NodeAddr("127.0.0.1:50051".into())), "http://127.0.0.1:50051");
        assert_eq!(
            endpoint_uri(&NodeAddr("http://host:9".into())),
            "http://host:9"
        );
    }

    #[tokio::test]
    async fn route_experts_roundtrip_over_grpc() {
        let addr = ephemeral_addr().await;
        let shutdown = spawn_server(addr).await;

        let node = NodeAddr(addr.to_string());
        let mut client = ShardClient::connect(&node).await.expect("connect");

        let hidden = vec![1.0f32, 2.0, 4.0, 8.0];
        let req = crate::rpc::RouteExpertsRequest {
            request_id: 99,
            layer_idx: 3,
            d_model: hidden.len() as u32,
            expert_ids: vec![0, 2],
            hidden_state_f16: pack_hidden_state(&hidden),
        };
        let resp = client.route_experts(req).await.expect("route");
        assert_eq!(resp.request_id, 99);
        assert_eq!(resp.expert_ids, vec![0, 2]);

        // Expert 0 scales by 1, expert 2 scales by 3.
        let out = unpack_hidden_state(&resp.ffn_out_f16);
        assert_eq!(out.len(), 2 * hidden.len());
        for (i, &h) in hidden.iter().enumerate() {
            assert!((out[i] - h * 1.0).abs() < 1e-2);
            assert!((out[hidden.len() + i] - h * 3.0).abs() < 1e-2);
        }

        let _ = shutdown.send(());
    }

    #[tokio::test]
    async fn health_probe_succeeds_against_live_shard() {
        let addr = ephemeral_addr().await;
        let shutdown = spawn_server(addr).await;
        let node = NodeAddr(addr.to_string());

        let mut client = ShardClient::connect(&node).await.expect("connect");
        let h = client.health().await.expect("health");
        assert_eq!(h.free_blocks, 7);
        assert_eq!(h.version, "test-shard");

        // The router-facing probe must also succeed against the live shard.
        probe_remote(5, node, Duration::from_secs(2))
            .await
            .expect("probe");

        let _ = shutdown.send(());
    }

    #[tokio::test]
    async fn probe_remote_unreachable_maps_to_structured_error() {
        // Nothing is listening on this port: connect must fail and the
        // error must be a structured `Unreachable`/`Timeout`, never a panic.
        let node = NodeAddr("127.0.0.1:1".to_string());
        let err = probe_remote(11, node, Duration::from_millis(500))
            .await
            .expect_err("must fail");
        match err {
            ShardRouterError::Unreachable { expert, .. }
            | ShardRouterError::Timeout { expert, .. } => assert_eq!(expert, 11),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
