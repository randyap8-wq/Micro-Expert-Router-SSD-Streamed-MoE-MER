//! Production-readiness HTTP middleware: API-key gate, in-process
//! token-bucket rate limit, admission control, and per-request UUID
//! tracing spans.
//!
//! Wired by [`crate::server::build_router`] via `axum::middleware::from_fn_with_state`.
//! Each piece is independently configurable from the TOML config —
//! defaults are fully permissive so the legacy benchmark / development
//! flow is preserved bit-for-bit.

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Per-request identifier propagated through the tracing span so every
/// log line emitted on behalf of a request can be correlated. The
/// value is a hex-rendered random `u128` (de-facto v4 UUID layout
/// without the dashes — fine for log correlation, not intended as a
/// stable database key). We hand-roll this rather than pulling in the
/// `uuid` crate to keep the dependency footprint small.
pub fn new_request_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    // Set version (v4) and variant bits per RFC 4122 §4.4 so the
    // string is structurally identical to a real UUID v4 even though
    // we render it without dashes.
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    let mut out = String::with_capacity(32);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

// ---------------------------- API-key gate ----------------------------

/// API-key validator. `None` (empty key set) disables the gate.
#[derive(Clone, Debug, Default)]
pub struct ApiKeyGate {
    keys: Arc<HashSet<String>>,
}

impl ApiKeyGate {
    pub fn new(keys: &[String]) -> Self {
        Self {
            keys: Arc::new(keys.iter().cloned().collect()),
        }
    }

    pub fn enabled(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Returns the matching key if the request carries a valid token
    /// in either the `Authorization: Bearer <key>` header or the
    /// `X-API-Key: <key>` header.
    pub fn match_token<'a>(&'a self, req: &Request<Body>) -> Option<&'a String> {
        let headers = req.headers();
        let candidate = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")))
            .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));
        let candidate = candidate?;
        self.keys.get(candidate)
    }
}

// ------------------------ Token-bucket rate limit ----------------------

#[derive(Debug)]
struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

/// Per-key in-process token bucket. When `rps == 0` the limiter is
/// disabled.
#[derive(Clone, Debug)]
pub struct RateLimiter {
    inner: Arc<RateLimiterInner>,
}

#[derive(Debug)]
struct RateLimiterInner {
    rps: u32,
    burst: u32,
    buckets: DashMap<String, Mutex<Bucket>>,
}

impl RateLimiter {
    pub fn new(rps: u32, burst: u32) -> Self {
        let burst = if burst == 0 { rps } else { burst };
        Self {
            inner: Arc::new(RateLimiterInner {
                rps,
                burst,
                buckets: DashMap::new(),
            }),
        }
    }

    pub fn enabled(&self) -> bool {
        self.inner.rps > 0
    }

    /// Attempt to charge one token for `key`. Returns `true` on
    /// success, `false` when the bucket is empty (caller should
    /// emit 429).
    pub fn allow(&self, key: &str) -> bool {
        if !self.enabled() {
            return true;
        }
        let now = Instant::now();
        let entry = self
            .inner
            .buckets
            .entry(key.to_string())
            .or_insert_with(|| {
                Mutex::new(Bucket {
                    tokens: self.inner.burst as f64,
                    last_refill: now,
                })
            });
        let mut b = entry.value().lock();
        let elapsed = now.duration_since(b.last_refill).as_secs_f64();
        let refill = elapsed * self.inner.rps as f64;
        b.tokens = (b.tokens + refill).min(self.inner.burst as f64);
        b.last_refill = now;
        if b.tokens >= 1.0 {
            b.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

// --------------------------- Admission control ------------------------

/// Concurrency + paged-pool admission controller. Cheap to clone.
#[derive(Clone)]
pub struct Admission {
    inner: Arc<AdmissionInner>,
}

impl std::fmt::Debug for Admission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Admission")
            .field("max_concurrent", &self.inner.max_concurrent)
            .field("in_flight", &self.in_flight())
            .field("min_free_blocks", &self.inner.min_free_blocks)
            .field("free_blocks_probe", &self.inner.free_blocks.is_some())
            .finish()
    }
}

struct AdmissionInner {
    max_concurrent: usize,
    in_flight: Arc<AtomicUsize>,
    min_free_blocks: usize,
    /// Snapshot fn for the configured block pool — `None` when no
    /// `[real_transformer.block_pool_*]` is configured. Returns the
    /// number of free blocks in the primary slab.
    free_blocks: Option<Arc<dyn Fn() -> usize + Send + Sync>>,
}

impl Admission {
    pub fn new(
        max_concurrent: usize,
        min_free_blocks: usize,
        free_blocks: Option<Arc<dyn Fn() -> usize + Send + Sync>>,
    ) -> Self {
        Self {
            inner: Arc::new(AdmissionInner {
                max_concurrent,
                in_flight: Arc::new(AtomicUsize::new(0)),
                min_free_blocks,
                free_blocks,
            }),
        }
    }

    pub fn in_flight(&self) -> usize {
        self.inner.in_flight.load(Ordering::Relaxed)
    }

    /// Try to acquire one slot. Returns `None` when admission is
    /// denied — the caller surfaces this to the client as `503
    /// Service Unavailable`. Otherwise returns a guard that releases
    /// the slot on `Drop`.
    pub fn try_admit(&self) -> Option<AdmissionGuard> {
        if self.inner.max_concurrent > 0 {
            // Reserve a slot via CAS. The previous `fetch_add` +
            // post-hoc `fetch_sub` was correct in terms of *which*
            // callers were admitted (every caller gets a unique
            // `prev`, so the limit was never actually exceeded), but
            // it caused a *transient over-increment* of `in_flight`
            // between the `fetch_add` and the compensating `fetch_sub`.
            // Observers reading the counter during that window
            // (`Debug` impl, the `mer_in_flight_requests` Prometheus
            // gauge, `Self::in_flight()`) could see a value greater
            // than `max_concurrent`, which is surprising for an
            // admission controller that advertises a hard cap.
            // The CAS loop avoids the spurious bump entirely.
            let mut cur = self.inner.in_flight.load(Ordering::Acquire);
            loop {
                if cur >= self.inner.max_concurrent {
                    return None;
                }
                match self.inner.in_flight.compare_exchange_weak(
                    cur,
                    cur + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(observed) => cur = observed,
                }
            }
        }
        if self.inner.min_free_blocks > 0 {
            if let Some(probe) = &self.inner.free_blocks {
                let free = probe();
                if free < self.inner.min_free_blocks {
                    if self.inner.max_concurrent > 0 {
                        self.inner.in_flight.fetch_sub(1, Ordering::AcqRel);
                    }
                    return None;
                }
            }
        }
        Some(AdmissionGuard {
            counter: if self.inner.max_concurrent > 0 {
                Some(self.inner.in_flight.clone())
            } else {
                None
            },
        })
    }
}

/// Active admission slot. Releases the concurrency token on `Drop`.
pub struct AdmissionGuard {
    counter: Option<Arc<AtomicUsize>>,
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        if let Some(c) = &self.counter {
            c.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

// Public re-export shim: expose the rate-limit / admission state as a
// single object stored in AppState's middleware bundle.
#[derive(Clone, Debug)]
pub struct MiddlewareState {
    pub api_keys: ApiKeyGate,
    pub rate_limit: RateLimiter,
    pub admission: Admission,
}

// ----------------------- axum middleware adapters ---------------------

/// Attach a fresh request id to the tracing span, and emit a single
/// `tracing::info_span` at the request boundary. The id is also set
/// as an `X-Request-Id` response header so clients can correlate
/// logs and metrics.
pub async fn request_id_layer(
    mut req: Request<Body>,
    next: Next,
) -> Response {
    use tracing::Instrument;
    let id = new_request_id();
    let span = tracing::info_span!("http_request", request_id = %id);
    req.extensions_mut().insert(RequestId(id.clone()));
    // `Instrument` (rather than a `span.enter()` guard) so the span is
    // correctly entered/exited around every poll — holding an `enter()`
    // guard across `.await` corrupts span nesting on a multi-threaded
    // runtime.
    let mut resp = next.run(req).instrument(span).await;
    if let Ok(hv) = axum::http::HeaderValue::from_str(&id) {
        resp.headers_mut().insert("x-request-id", hv);
    }
    resp
}

/// Newtype for the per-request UUID so handlers can extract it via
/// `Extension<RequestId>` if needed.
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

/// API-key check middleware. Skips entirely when the gate is
/// disabled (no keys configured), so the default config keeps the
/// legacy behaviour.
pub async fn api_key_layer(
    State(state): State<MiddlewareState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if !state.api_keys.enabled() {
        return next.run(req).await;
    }
    if state.api_keys.match_token(&req).is_some() {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [("content-type", "application/json")],
            r#"{"error":"missing or invalid API key"}"#,
        )
            .into_response()
    }
}

/// Token-bucket rate limit middleware, keyed by API key (or
/// "anonymous" when API-key gating is disabled). Skipped when
/// `rate_limit_rps == 0`.
pub async fn rate_limit_layer(
    State(state): State<MiddlewareState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if !state.rate_limit.enabled() {
        return next.run(req).await;
    }
    let key = state
        .api_keys
        .match_token(&req)
        .cloned()
        .unwrap_or_else(|| "anonymous".to_string());
    if state.rate_limit.allow(&key) {
        next.run(req).await
    } else {
        (
            StatusCode::TOO_MANY_REQUESTS,
            [("content-type", "application/json")],
            r#"{"error":"rate limit exceeded"}"#,
        )
            .into_response()
    }
}

/// Admission control: rejects requests with `503 Service Unavailable`
/// when either the global concurrency cap or the paged-pool free-block
/// watermark is exceeded.
pub async fn admission_layer(
    State(state): State<MiddlewareState>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    let guard = match state.admission.try_admit() {
        Some(g) => g,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                [
                    ("content-type", "application/json"),
                    ("retry-after", "1"),
                ],
                r#"{"error":"server at capacity; retry shortly"}"#,
            )
                .into_response();
        }
    };
    // Store the guard so it lives for the whole request lifetime —
    // dropped after `next` returns the response.
    req.extensions_mut().insert(AdmissionTicket(Arc::new(guard)));
    next.run(req).await
}

/// Arc-wrapped admission guard so it can sit inside the request
/// extensions (which require `Send + Sync + Clone`).
#[derive(Clone)]
pub struct AdmissionTicket(#[allow(dead_code)] pub Arc<AdmissionGuard>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_id_is_32_lowercase_hex() {
        let id = new_request_id();
        assert_eq!(id.len(), 32);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn rate_limiter_respects_burst() {
        let l = RateLimiter::new(1, 3);
        assert!(l.allow("k1"));
        assert!(l.allow("k1"));
        assert!(l.allow("k1"));
        assert!(!l.allow("k1"));
    }

    #[test]
    fn rate_limiter_disabled_when_rps_zero() {
        let l = RateLimiter::new(0, 0);
        for _ in 0..1_000 {
            assert!(l.allow("k1"));
        }
    }

    #[test]
    fn api_key_gate_recognises_bearer_and_x_api_key() {
        let gate = ApiKeyGate::new(&["sekret".into()]);
        assert!(gate.enabled());
        let mut req = Request::builder().uri("/").body(Body::empty()).unwrap();
        req.headers_mut()
            .insert("authorization", "Bearer sekret".parse().unwrap());
        assert!(gate.match_token(&req).is_some());
        let mut req = Request::builder().uri("/").body(Body::empty()).unwrap();
        req.headers_mut()
            .insert("x-api-key", "sekret".parse().unwrap());
        assert!(gate.match_token(&req).is_some());
        let req = Request::builder().uri("/").body(Body::empty()).unwrap();
        assert!(gate.match_token(&req).is_none());
    }

    #[test]
    fn admission_max_concurrent_is_enforced() {
        let a = Admission::new(2, 0, None);
        let g1 = a.try_admit().expect("first admitted");
        let g2 = a.try_admit().expect("second admitted");
        assert!(a.try_admit().is_none(), "third must be rejected");
        drop(g1);
        let _g3 = a.try_admit().expect("admitted after release");
        drop(g2);
    }

    #[test]
    fn admission_respects_min_free_blocks() {
        let free = Arc::new(AtomicUsize::new(3));
        let f2 = free.clone();
        let probe = Arc::new(move || f2.load(Ordering::Relaxed));
        let a = Admission::new(0, 5, Some(probe));
        assert!(a.try_admit().is_none(), "below watermark must reject");
        free.store(10, Ordering::Relaxed);
        assert!(a.try_admit().is_some(), "above watermark must admit");
    }
}
