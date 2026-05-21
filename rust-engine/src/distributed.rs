//! Distributed Expert Sharding (gist Task 3).
//!
//! This module is the **scaling-infrastructure seam** between the
//! local SSD-resident expert cache and a future multi-node MoE
//! deployment. The premise is the same one the rest of the engine
//! already commits to: experts are commodity weight blobs that can
//! live anywhere — on an NVMe drive, in a peer's DRAM, or across an
//! RDMA fabric — provided the router can produce a *structured
//! instruction* describing where to find them.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────┐    expert id    ┌──────────────┐
//!   │  Router  ├───────────────▶│  ShardRouter │
//!   └──────────┘                 └──────┬───────┘
//!                                        │ ShardInstruction
//!              ┌─────────────────────────┴──────────────────────┐
//!              │                                                │
//!         Local{id}                                  Remote{id, node, …}
//!              │                                                │
//!     ┌────────▼─────────┐                       ┌──────────────▼─────────────┐
//!     │  ExpertCache /   │                       │  remote fetch (gRPC/RDMA), │
//!     │   NvmeStorage    │                       │  AlignedBuffer pointer      │
//!     │   (zero-copy)    │                       │  swap on arrival            │
//!     └──────────────────┘                       └─────────────────────────────┘
//! ```
//!
//! Planned wiring (follow-up PR): the `BatchScheduler` will consult
//! [`ShardRouter`] before issuing `Engine::warm_with`, so ids tagged
//! `Local` stay on the existing NVMe path while `Remote` ids can
//! surface structured remote-fetch failures instead of panicking.
//!
//! ## Zero-copy invariant
//!
//! Per the gist's CRITICAL constraint, the sharding layer **must not
//! copy weight blocks**. The `ShardInstruction::Remote` variant
//! carries only the metadata required to *orchestrate* a transfer —
//! it does not own bytes. When the transport (gRPC, RDMA, NVMe-oF,
//! …) eventually delivers the bytes, the canonical landing zone is
//! still the engine's `AlignedBuffer` slab pool, and ownership is
//! transferred by moving the `AlignedBuffer` (a pointer + length +
//! capacity), never by a `memcpy`.
//!
//! ## Why this lives outside `router.rs`
//!
//! The legacy `router::PredictiveLoader` decides *which* experts a
//! token is likely to need; that's a routing-policy question. The
//! `ShardRouter` decides *where* a selected expert lives; that's a
//! placement question. Keeping them in separate modules lets the
//! placement layer evolve independently — a future cluster could
//! pick `RoundRobinShardRouter`, `ConsistentHashShardRouter`, or a
//! Kubernetes-aware variant — without touching the predictor's
//! Markov / locality / speculator arithmetic.

// Scaffolding for future multi-node support. Most items are exposed
// only via the public trait surface today; wiring this into the
// scheduler hot path lands in a follow-up PR.
#![allow(dead_code)]

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// Object-safe future alias used by [`ShardRouter::fetch_remote`].
///
/// We do **not** use `async fn` in the trait directly: it would
/// require nightly `dyn` support (or [`async_trait`]). A boxed
/// future keeps the trait `Send + Sync` and dyn-compatible so the
/// engine can hold an `Arc<dyn ShardRouter>` without per-method
/// indirection contortions on the hot path.
pub type ShardFetchFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(), ShardRouterError>> + Send + 'a>>;

/// Logical address of a node hosting one or more experts.
///
/// The transport is intentionally *not* baked into this enum: a
/// future gRPC ShardRouter will read `host:port`, an RDMA one will
/// read a queue-pair id, an NVMe-oF one will read an NQN. The
/// scheduler only cares that the address round-trips through
/// `Debug` / `Display` for structured logging on failure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeAddr(pub String);

impl std::fmt::Display for NodeAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What the [`ShardRouter`] tells the batch scheduler to do for a
/// given expert id.
///
/// Construction is lightweight and owns metadata only (expert id,
/// node address, timeout) — never expert weight buffers. `Remote`
/// stores a `NodeAddr(String)` and error types may carry textual
/// reasons, but no variant carries weight bytes. This preserves the
/// **zero-copy invariant** the engine commits to.
#[derive(Debug, Clone)]
pub enum ShardInstruction {
    /// Expert is locally resident or fetchable from local NVMe.
    /// The scheduler proceeds through the existing
    /// `Engine::warm_with` / `ExpertCache` path.
    Local { expert: u32 },
    /// Expert lives on a remote node. The scheduler must initiate a
    /// remote fetch through whatever transport [`ShardRouter::fetch_remote`]
    /// implements; on failure the scheduler surfaces an
    /// [`InferenceError::RemoteShardFetchFailed`] rather than
    /// panicking.
    Remote {
        expert: u32,
        node: NodeAddr,
        /// Per-call deadline for the remote fetch. The transport
        /// must abort and surface [`ShardRouterError::Timeout`]
        /// when this elapses so the scheduler can fall back / fail
        /// the request promptly.
        timeout: Duration,
    },
}

impl ShardInstruction {
    /// Expert id this instruction is about. Convenience accessor for
    /// the scheduler's expert-id de-duplication pre-pass — both
    /// variants ultimately reference a single id.
    pub fn expert(&self) -> u32 {
        match self {
            ShardInstruction::Local { expert } => *expert,
            ShardInstruction::Remote { expert, .. } => *expert,
        }
    }

    /// `true` when the scheduler can satisfy this instruction without
    /// touching the network. The hot path uses this as a cheap
    /// branch predicate so the remote-fetch slow path is only
    /// entered when truly necessary.
    pub fn is_local(&self) -> bool {
        matches!(self, ShardInstruction::Local { .. })
    }
}

/// Errors a [`ShardRouter`] implementation can surface.
///
/// Modelled as a small enum (not `Box<dyn Error>`) so callers can
/// pattern-match and map to retry / fallback policy without a
/// downcast. Every variant carries enough context for structured
/// logging on the scheduler's "remote fetch failed" log line.
#[derive(Debug)]
pub enum ShardRouterError {
    /// The remote node did not respond within the per-call deadline
    /// that the [`ShardInstruction::Remote::timeout`] field carries.
    Timeout { expert: u32, node: NodeAddr },
    /// The transport could not establish a session with the named
    /// node at all (DNS failure, connection refused, partitioned
    /// network, …).
    Unreachable { expert: u32, node: NodeAddr, reason: String },
    /// The node responded but did not have the requested expert.
    /// Usually indicates a stale placement map; the scheduler may
    /// re-query [`ShardRouter::route_expert`] after a refresh.
    NotFound { expert: u32, node: NodeAddr },
    /// Any other transport-level failure not captured by the
    /// variants above. Used by future RDMA / gRPC implementations
    /// for protocol-specific errors.
    Transport { expert: u32, node: NodeAddr, reason: String },
}

impl std::fmt::Display for ShardRouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardRouterError::Timeout { expert, node } => write!(
                f,
                "shard fetch timed out: expert {expert} on node {node}"
            ),
            ShardRouterError::Unreachable { expert, node, reason } => write!(
                f,
                "shard node unreachable: expert {expert} on node {node}: {reason}"
            ),
            ShardRouterError::NotFound { expert, node } => write!(
                f,
                "expert {expert} not present on node {node} (stale placement map?)"
            ),
            ShardRouterError::Transport { expert, node, reason } => write!(
                f,
                "shard transport error: expert {expert} on node {node}: {reason}"
            ),
        }
    }
}

impl std::error::Error for ShardRouterError {}

/// Top-level inference error surfaced to the batch scheduler when a
/// remote shard fetch fails. The scheduler converts this into a
/// `BatchError` so HTTP callers see a structured failure instead of
/// a panic at the I/O / network boundary.
///
/// Kept distinct from [`ShardRouterError`] so callers can choose
/// whether to retry the *same* shard or re-route to a different
/// replica.
#[derive(Debug)]
pub enum InferenceError {
    /// A remote expert could not be fetched. Carries the underlying
    /// [`ShardRouterError`] verbatim so logs preserve the full
    /// transport context.
    RemoteShardFetchFailed(ShardRouterError),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::RemoteShardFetchFailed(e) => {
                write!(f, "remote shard fetch failed: {e}")
            }
        }
    }
}

impl std::error::Error for InferenceError {}

impl From<ShardRouterError> for InferenceError {
    fn from(e: ShardRouterError) -> Self {
        InferenceError::RemoteShardFetchFailed(e)
    }
}

/// Pluggable expert-placement layer.
///
/// Implementations decide whether a given expert id is resident on
/// this node ([`ShardInstruction::Local`]) or on a peer
/// ([`ShardInstruction::Remote`]), and provide a `fetch_remote`
/// transport hook the scheduler calls when it actually needs the
/// bytes.
///
/// The trait is intentionally object-safe (`Send + Sync + 'static`,
/// no generic methods) so the engine can hold an
/// `Arc<dyn ShardRouter>` and operators can swap in a real cluster
/// implementation without recompiling the rest of the crate.
///
/// ### Safety contract
/// - `route_expert` is on the hot path and **must not block**.
///   Cluster topology lookups should use atomic / lock-free state
///   internally (e.g. `arc_swap::ArcSwap` over a placement map).
/// - `fetch_remote` is allowed to suspend (it's `async`) but **must
///   honour its own timeout**. The default
///   [`LocalShardRouter::fetch_remote`] never enters this path
///   because it always emits `Local`.

pub trait ShardRouter: Send + Sync + 'static {
    /// Human-readable identifier (e.g. `"local"`, `"grpc-mesh"`,
    /// `"rdma"`). Logged once at startup.
    fn name(&self) -> &'static str;

    /// Decide where `expert` lives. **Non-blocking, hot-path.**
    fn route_expert(&self, expert: u32) -> ShardInstruction;

    /// Initiate a remote fetch. The transport must honour the
    /// deadline carried in `instruction` (if `Remote`) and surface a
    /// structured [`ShardRouterError`] on failure — *never* panic.
    ///
    /// Default impl returns `NotFound` for any `Remote` variant — it
    /// exists so single-node deployments can implement `ShardRouter`
    /// without writing a network path.
    ///
    /// Returns a [`ShardFetchFuture`] (a boxed `Send` future) so the
    /// trait stays object-safe; the engine holds shard routers
    /// behind an `Arc<dyn ShardRouter>`.
    fn fetch_remote<'a>(
        &'a self,
        instruction: &'a ShardInstruction,
    ) -> ShardFetchFuture<'a> {
        Box::pin(async move {
            match instruction {
                ShardInstruction::Local { .. } => Ok(()),
                ShardInstruction::Remote { expert, node, .. } => {
                    Err(ShardRouterError::NotFound {
                        expert: *expert,
                        node: node.clone(),
                    })
                }
            }
        })
    }
}

/// Default single-node [`ShardRouter`]: every expert is local.
///
/// This is the implementation the engine ships with today: the
/// `BatchScheduler` consults it for symmetry with the multi-node
/// future, but every instruction is `Local` so the existing
/// NVMe-streamed path runs verbatim.
pub struct LocalShardRouter;

impl ShardRouter for LocalShardRouter {
    fn name(&self) -> &'static str {
        "local"
    }

    fn route_expert(&self, expert: u32) -> ShardInstruction {
        ShardInstruction::Local { expert }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_router_emits_local_instructions() {
        let r = LocalShardRouter;
        for id in [0u32, 1, 7, 12345] {
            let inst = r.route_expert(id);
            assert!(inst.is_local());
            assert_eq!(inst.expert(), id);
        }
    }

    #[tokio::test]
    async fn local_router_remote_fetch_is_noop_for_local_instructions() {
        let r = LocalShardRouter;
        let inst = r.route_expert(3);
        r.fetch_remote(&inst).await.expect("local fetch");
    }

    #[tokio::test]
    async fn default_remote_fetch_returns_structured_not_found() {
        // A degenerate `ShardRouter` whose `route_expert` claims an
        // expert is remote but inherits the trait's default
        // `fetch_remote`. The scheduler must see a structured
        // `NotFound`, never a panic.
        struct RemoteOnly;
        impl ShardRouter for RemoteOnly {
            fn name(&self) -> &'static str { "remote-only" }
            fn route_expert(&self, expert: u32) -> ShardInstruction {
                ShardInstruction::Remote {
                    expert,
                    node: NodeAddr("peer-1".to_string()),
                    timeout: Duration::from_millis(10),
                }
            }
        }
        let r = RemoteOnly;
        let inst = r.route_expert(42);
        let err = r.fetch_remote(&inst).await.expect_err("must fail");
        match err {
            ShardRouterError::NotFound { expert, node } => {
                assert_eq!(expert, 42);
                assert_eq!(node.0, "peer-1");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn inference_error_displays_underlying_shard_error() {
        let se = ShardRouterError::Timeout {
            expert: 7,
            node: NodeAddr("peer-2".into()),
        };
        let ie: InferenceError = se.into();
        let s = format!("{ie}");
        assert!(s.contains("remote shard fetch failed"), "got: {s}");
        assert!(s.contains("expert 7"), "got: {s}");
        assert!(s.contains("peer-2"), "got: {s}");
    }
}
