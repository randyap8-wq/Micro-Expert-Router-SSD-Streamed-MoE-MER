//! OpenAI-compatible HTTP server (gist Phase 6).
//!
//! Endpoints:
//! - `POST /v1/completions`      — text completion
//! - `POST /v1/chat/completions` — chat completion (single-turn, no streaming yet)
//! - `GET  /health`              — liveness
//! - `GET  /metrics`             — Prometheus metrics
//!
//! The server is stateless per request at the HTTP layer, but the
//! real-transformer path routes per-token decoder steps through a
//! shared [`crate::batch_scheduler::BatchScheduler`]: an `mpsc`-fed
//! background task fuses concurrent requests' steps into a single
//! batch (up to `max_batch_size` or `batch_timeout_ms`) and runs them
//! concurrently against the shared engine, so multiple HTTP clients
//! amortise each round of SSD-streamed expert FFN compute. Per-request
//! KV caches are moved into the scheduler and back, so attention state
//! stays strictly per-request.
//!
//! Generation strategy: the synthesised hidden state from
//! [`crate::inference::synth_hidden_state`] drives one
//! `Engine::generate` call per token. The "decoded" response text is
//! the round-tripped token ids through the configured tokenizer (real
//! HF tokenizer if `tokenizer.json` is present, byte tokenizer otherwise).
//! Once a real transformer decoder is wired (the [`crate::transformer`]
//! pieces), this loop swaps to producing real next-token logits — the
//! HTTP shape doesn't change.

use crate::batch_scheduler::BatchScheduler;
use crate::config::LiveConfig;
use crate::engine::Engine;
use crate::metrics::Metrics;
use crate::middleware::{
    admission_layer, api_key_layer, rate_limit_layer, request_id_layer, MiddlewareState,
};
use crate::model::RealModel;
use crate::sampling::SamplingParams;
use crate::session::{SessionCheckoutToken, SessionState, SessionStore};
use crate::tokenizer::Tokenizer;
use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;

/// Shared handler state. Cheap to clone — everything is `Arc`.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub tokenizer: Arc<Tokenizer>,
    pub metrics: Metrics,
    /// Optional real transformer. When set, [`generate`] runs the
    /// embedding → stacked layers → LM head pipeline (with experts
    /// streamed from SSD by the engine on every layer's MoE step) and
    /// samples next-token ids from real logits. When `None`, the
    /// legacy deterministic generator is used (the engine still drives
    /// SSD-streamed expert FFN compute either way, so cache / I/O
    /// metrics are populated identically).
    pub real_model: Option<Arc<RealModel>>,
    /// Optional continuous-batching scheduler. Always set together
    /// with `real_model` (see `cmd_serve` in `main.rs`); when `Some`,
    /// the real-transformer generation paths submit each per-token
    /// step to the scheduler instead of calling `RealModel::step`
    /// directly. The scheduler fuses concurrent requests' steps into a
    /// single batch so multiple HTTP clients share each round of
    /// SSD-streamed expert FFN compute.
    pub batch_scheduler: Option<Arc<BatchScheduler>>,
    /// Live, atomically-swappable runtime configuration. Reads on the
    /// hot path (sampling defaults, `max_tokens` cap) go through a
    /// single relaxed atomic load — no mutex, no contention across
    /// Tokio worker threads. SIGHUP refreshes this in place via
    /// [`LiveConfig::try_reload`]; in-flight requests keep observing
    /// their snapshot until they drop it. See
    /// [`crate::config::RuntimeConfig`].
    pub runtime: LiveConfig,
    /// Optional session store for KV-cache persistence between
    /// requests. `None` when `server.session_ttl_secs == 0`.
    pub sessions: Option<SessionStore>,
    /// Production-readiness middleware bundle: API-key gate, in-process
    /// rate limit, and admission controller. All three are no-ops
    /// when not configured, so this is safe to construct
    /// unconditionally.
    pub middleware: MiddlewareState,
}

/// Build the axum [`Router`] for the API.
///
/// Routes are split into two sub-routers:
///
/// * **Public**: `/health`, `/metrics`. These are required by
///   Kubernetes liveness/readiness probes and Prometheus scrapers,
///   which generally cannot present API keys and must not be subject
///   to per-tenant rate-limiting or admission shedding. They only
///   receive the request-id layer so logs remain correlated.
/// * **Protected**: every other route (`/v1/...`). These pass
///   through the full middleware stack: API key, in-process rate
///   limit, and admission control.
pub fn build_router(state: AppState) -> Router {
    let mw_state = state.middleware.clone();

    // Operational endpoints — always reachable, no auth/rate-limit/admission.
    let public = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics));

    // Tenant-facing endpoints — protected by the full middleware stack.
    let protected = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/sessions/:id", delete(delete_session))
        .route("/v1/admin/health/experts", get(admin_health_experts))
        .route("/v1/admin/evict", post(admin_evict_overflow))
        .layer(axum::middleware::from_fn_with_state(
            mw_state.clone(),
            admission_layer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            mw_state.clone(),
            rate_limit_layer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            mw_state,
            api_key_layer,
        ));

    public
        .merge(protected)
        .layer(axum::middleware::from_fn(request_id_layer))
        .with_state(state)
}

// ----------------------------- /health ------------------------------

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

// ----------------------------- /metrics ------------------------------

async fn metrics(State(state): State<AppState>) -> Response {
    match state.metrics.render() {
        Ok(body) => (
            StatusCode::OK,
            [("content-type", "text/plain; version=0.0.4")],
            body,
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("metrics render error: {e}"),
        )
            .into_response(),
    }
}

// ------------------------ /v1/completions ----------------------------

#[derive(Deserialize, Debug)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Optional session id. When present, the per-layer KV caches for
    /// this session are restored before the prompt is ingested and
    /// stored back when the request completes — see `crate::session`
    /// for the persistence model. When absent, the request is
    /// stateless (matches the legacy behaviour bit-for-bit).
    #[serde(default)]
    pub session_id: Option<String>,
    /// When `true`, the response is streamed as Server-Sent Events
    /// (`text/event-stream`) — one OpenAI-style `text_completion` chunk
    /// per generated token, terminated by `data: [DONE]`. Defaults to
    /// `false` (single-shot JSON response).
    #[serde(default)]
    pub stream: Option<bool>,
}

fn default_max_tokens() -> usize { 64 }
fn default_model() -> String { "micro-expert-router".to_string() }

#[derive(Serialize, Debug)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: &'static str,
}

#[derive(Serialize, Debug)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize, Debug)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageStats,
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let started = Instant::now();
    let params = resolve_params(&state, req.temperature, req.top_p, req.top_k, req.seed);
    let session_id = req.session_id.clone();
    if req.stream.unwrap_or(false) {
        // Server-Sent Events streaming path: emit one OpenAI-style chunk
        // per generated token, terminated with `data: [DONE]`. The whole
        // SSD-streaming substrate is exercised inside the generator
        // exactly as in the non-streaming path.
        match build_completion_stream(
            &state,
            &req.prompt,
            req.max_tokens,
            req.model.clone(),
            params,
            session_id,
        )
        .await
        {
            Ok(s) => {
                state
                    .metrics
                    .record_request("/v1/completions", started.elapsed().as_secs_f64());
                return Sse::new(s)
                    .keep_alive(KeepAlive::default())
                    .into_response();
            }
            Err(e) => {
                state
                    .metrics
                    .record_request("/v1/completions", started.elapsed().as_secs_f64());
                return error_response(e);
            }
        }
    }
    match generate(&state, &req.prompt, req.max_tokens, &req.model, params, session_id).await {
        Ok(resp) => {
            state.metrics.record_request("/v1/completions", started.elapsed().as_secs_f64());
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            state.metrics.record_request("/v1/completions", started.elapsed().as_secs_f64());
            error_response(e)
        }
    }
}

// ---------------------- /v1/chat/completions -------------------------

#[derive(Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Serialize, Debug)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub finish_reason: &'static str,
}

#[derive(Serialize, Debug)]
pub struct ChatResponseMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageStats,
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let started = Instant::now();
    let prompt = flatten_messages(&req.messages);
    let params = resolve_params(&state, req.temperature, req.top_p, req.top_k, req.seed);
    let session_id = req.session_id.clone();
    if req.stream.unwrap_or(false) {
        match build_chat_stream(
            &state,
            &prompt,
            req.max_tokens,
            req.model.clone(),
            params,
            session_id,
        )
        .await
        {
            Ok(s) => {
                state
                    .metrics
                    .record_request("/v1/chat/completions", started.elapsed().as_secs_f64());
                return Sse::new(s)
                    .keep_alive(KeepAlive::default())
                    .into_response();
            }
            Err(e) => {
                state
                    .metrics
                    .record_request("/v1/chat/completions", started.elapsed().as_secs_f64());
                return error_response(e);
            }
        }
    }
    // Flatten messages into a single prompt — exactly the same shape
    // simple OpenAI-compatible servers (vLLM, llama.cpp's HTTP) do when
    // no chat template is configured.
    match generate(&state, &prompt, req.max_tokens, &req.model, params, session_id).await {
        Ok(comp) => {
            let resp = ChatCompletionResponse {
                id: comp.id,
                object: "chat.completion",
                model: comp.model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant",
                        content: comp.choices.into_iter().next().map(|c| c.text).unwrap_or_default(),
                    },
                    finish_reason: "length",
                }],
                usage: comp.usage,
            };
            state.metrics.record_request("/v1/chat/completions", started.elapsed().as_secs_f64());
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            state.metrics.record_request("/v1/chat/completions", started.elapsed().as_secs_f64());
            error_response(e)
        }
    }
}

fn flatten_messages(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&m.role);
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    out
}

// ------------------------ generation core ----------------------------

#[derive(Debug)]
pub enum GenerateError {
    Tokenizer(String),
    InvalidRequest(String),
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerateError::Tokenizer(m) => write!(f, "tokenizer error: {m}"),
            GenerateError::InvalidRequest(m) => write!(f, "invalid request: {m}"),
        }
    }
}

/// Resolve per-request sampling parameters from the request fields,
/// falling back to the server-wide defaults pulled from the
/// atomically-swappable [`crate::config::RuntimeConfig`]. OpenAI's API
/// treats `temperature: 0` as "deterministic" — we mirror that by
/// passing it through verbatim (the sampler in [`crate::sampling`]
/// degrades to greedy `argmax`).
fn resolve_params(
    state: &AppState,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
) -> SamplingParams {
    // One relaxed atomic load — no lock, no contention with concurrent
    // SIGHUP reloads. The guard drops at the end of this function and
    // never crosses an `await` point.
    let mut p = state.runtime.snapshot().sampling;
    if let Some(t) = temperature {
        p.temperature = t;
    }
    if let Some(t) = top_p {
        p.top_p = t;
    }
    if let Some(t) = top_k {
        p.top_k = t;
    }
    if let Some(t) = seed {
        p.seed = t;
    }
    p
}

async fn generate(
    state: &AppState,
    prompt: &str,
    requested_max: usize,
    model_name: &str,
    params: SamplingParams,
    session_id: Option<String>,
) -> Result<CompletionResponse, GenerateError> {
    if prompt.is_empty() {
        return Err(GenerateError::InvalidRequest("prompt must be non-empty".into()));
    }
    let max_tokens = requested_max
        .min(state.runtime.snapshot().max_tokens_cap)
        .max(1);

    // 1) Tokenize the prompt. The token ids drive the engine's deterministic
    //    routing seed so completions are reproducible for a given prompt.
    let prompt_ids = state
        .tokenizer
        .encode(prompt)
        .map_err(|e| GenerateError::Tokenizer(e.to_string()))?;
    let prompt_tokens = prompt_ids.len();

    // 2) Drive next-token generation. Two paths:
    //
    //   * **Real-transformer path** (`real_model` is `Some`): for each
    //     output token, run the full decoder forward — embedding →
    //     stacked layers (each calling the engine's `moe_step` to load
    //     and run experts from SSD) → final RMSNorm → LM head — and
    //     sample the next token id from real logits via `params`.
    //     This is the gist's "actual generated text" path.
    //
    //   * **Legacy benchmark path** (`real_model` is `None`): drive
    //     `Engine::generate` for `max_tokens` cycles and synthesise a
    //     deterministic id stream. Same SSD-streaming behaviour, no
    //     real logits.
    //
    //   Both paths populate the same hits / misses / I/O counters, so
    //   the run summary stats look identical.
    let mut hits_total = 0u64;
    let mut misses_total = 0u64;
    let mut completion_ids: Vec<u32> = Vec::with_capacity(max_tokens);

    if let Some(model) = state.real_model.as_ref() {
        // Resolve session: take any existing KV state, then put it
        // back at the end. When no session is configured the request
        // is fully stateless (legacy behaviour).
        let (mut kv, mut start_pos, checkout) = load_session_kv(state, model, session_id.as_deref());
        let pre_hits = state.engine.report().hits;
        let pre_misses = state.engine.report().misses;

        // Prime the KV cache with the prompt tokens so the first
        // generated token actually attends over the prompt. We discard
        // the sampled ids during the prompt sweep — those would be the
        // model's predictions of the *next* prompt token, not part of
        // the completion. When a session is active and `start_pos > 0`
        // the prompt continues from where the previous request left
        // off (multi-turn chat).
        for &tid in prompt_ids.iter() {
            let _ = step_through_scheduler(state, model, tid, start_pos, &mut kv, &params).await;
            start_pos += 1;
        }

        // Now generate `max_tokens` real tokens.
        let mut last = *prompt_ids.last().unwrap_or(&0u32);
        for _ in 0..max_tokens {
            let next = step_through_scheduler(state, model, last, start_pos, &mut kv, &params).await;
            completion_ids.push(next);
            last = next;
            start_pos += 1;
        }
        save_session_kv(state, session_id.as_deref(), kv, start_pos, checkout);
        let post = state.engine.report();
        hits_total = post.hits.saturating_sub(pre_hits);
        misses_total = post.misses.saturating_sub(pre_misses);
    } else {
        // Legacy benchmark path — sampling params are unused (no real
        // logits) but we still respect `seed` for the deterministic
        // synthetic token stream so requests are reproducible.
        let base = prompt_ids
            .last()
            .copied()
            .unwrap_or(0)
            .wrapping_add(params.seed as u32) as u64;
        for i in 0..max_tokens {
            let stats = state.engine.generate(base.wrapping_add(i as u64)).await;
            hits_total += stats.hits;
            misses_total += stats.misses;
            // Map engine cycle stats to a deterministic next-token id.
            let vocab = state.tokenizer.vocab_size().max(1) as u64;
            let next = ((base.wrapping_add(i as u64).wrapping_mul(0x9E3779B97F4A7C15)) % vocab) as u32;
            completion_ids.push(next);
        }
    }
    state.metrics.record_tokens(max_tokens as u64);
    state.metrics.record_cache(hits_total, misses_total);

    // 3) Decode and respond.
    let text = state
        .tokenizer
        .decode(&completion_ids)
        .map_err(|e| GenerateError::Tokenizer(e.to_string()))?;
    info!(
        prompt_tokens,
        completion_tokens = max_tokens,
        cache_hits = hits_total,
        cache_misses = misses_total,
        "completed request"
    );
    Ok(CompletionResponse {
        id: format!("cmpl-{:x}", rand_request_id()),
        object: "text_completion",
        model: model_name.to_string(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: "length",
        }],
        usage: UsageStats {
            prompt_tokens,
            completion_tokens: max_tokens,
            total_tokens: prompt_tokens + max_tokens,
        },
    })
}

// ------------------------ scheduler helper --------------------------

/// Drive one decoder step for a single request, routing through the
/// [`BatchScheduler`] when one is configured and falling back to a
/// direct `RealModel::step` otherwise. The KV cache is moved into the
/// scheduler and back so attention state stays strictly per-request
/// even though many requests share the underlying scheduler task.
async fn step_through_scheduler(
    state: &AppState,
    model: &Arc<RealModel>,
    token_id: u32,
    pos: usize,
    kv: &mut Vec<crate::transformer::KvCache>,
    params: &crate::sampling::SamplingParams,
) -> u32 {
    if let Some(sched) = state.batch_scheduler.as_ref() {
        // Hand the KV caches to the scheduler's registry for the
        // duration of one step; the scheduler returns the next
        // token (the cache stays owned by the registry across the
        // mpsc channel, so no `Vec<KvCache>` clone happens on the
        // hot path). When the scheduler has shut down (server
        // teardown) we fall back to a direct call against a freshly
        // allocated KV cache so the request can finish — losing
        // history is the price of a clean shutdown path here.
        let owned = std::mem::take(kv);
        let id = sched.register(owned);
        let result = sched.step_registered(id, token_id, pos, *params).await;
        // Always reclaim the cache, even on error, so the registry
        // doesn't leak entries.
        match sched.release(id) {
            Some(returned) => *kv = returned,
            None => *kv = model.fresh_kv_caches(),
        }
        match result {
            Ok(next_token) => next_token,
            Err(_) => model.step(&state.engine, token_id, pos, kv, params).await,
        }
    } else {
        model.step(&state.engine, token_id, pos, kv, params).await
    }
}

/// Fetch the persisted KV state for a session, or build a fresh one.
/// Returns `(kv_caches, next_position)`. The next position is `0` for
/// fresh sessions and the saved cursor otherwise — multi-turn chats
/// then continue token absolute positions across requests, which is
/// what RoPE expects.
fn load_session_kv(
    state: &AppState,
    model: &Arc<RealModel>,
    session_id: Option<&str>,
) -> (
    Vec<crate::transformer::KvCache>,
    usize,
    Option<SessionCheckoutToken>,
) {
    if let (Some(id), Some(store)) = (session_id, state.sessions.as_ref()) {
        if let Some((prev, checkout)) = store.take(id) {
            // Validate shape: layer count must match the live model.
            // Mismatches happen if the server is restarted with a
            // different config; treat as a fresh session.
            if prev.kv.len() == model.config.num_layers {
                return (prev.kv, prev.position, Some(checkout));
            }
        }
    }
    (model.fresh_kv_caches(), 0, None)
}

/// Persist KV state back to the store at request completion. No-op
/// when no session is configured.
fn save_session_kv(
    state: &AppState,
    session_id: Option<&str>,
    kv: Vec<crate::transformer::KvCache>,
    position: usize,
    checkout: Option<SessionCheckoutToken>,
) {
    if let (Some(id), Some(store)) = (session_id, state.sessions.as_ref()) {
        store.put(
            id.to_string(),
            SessionState {
                kv,
                position,
                last_used: Instant::now(),
            },
            checkout,
        );
    }
}

#[derive(Serialize, Debug)]
struct SessionDeleteResponse {
    id: String,
    deleted: bool,
}

async fn delete_session(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    match state.sessions.as_ref() {
        Some(store) => {
            let deleted = store.delete(&id);
            (
                StatusCode::OK,
                Json(SessionDeleteResponse { id, deleted }),
            )
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "session store disabled (server.session_ttl_secs == 0)"
            })),
        )
            .into_response(),
    }
}

// ---------------------------- /v1/admin/* ----------------------------

#[derive(Serialize, Debug)]
struct ExpertHealth {
    /// `"ok"` when no expert reads have failed since startup,
    /// `"degraded"` when one or more failed (i.e. the engine has
    /// served at least one request with a missing top-K expert).
    status: &'static str,
    expert_read_failures: u64,
    cache_hits: u64,
    cache_misses: u64,
    cache_pinned: usize,
    cache_capacity: usize,
    in_flight_requests: usize,
    block_pool_free: Option<usize>,
    block_pool_capacity: Option<usize>,
    block_pool_overflow_in_use: Option<usize>,
    tokens_generated: u64,
    /// Phase 1 / 3-tier hierarchy: whether the VRAM (GPU) expert
    /// cache is configured. When `false` the remaining `vram_*` and
    /// `gpu_*` fields are still present (0) for stable schema.
    gpu_cache_enabled: bool,
    gpu_cache_hits: u64,
    gpu_cache_misses: u64,
    gpu_promotions_total: u64,
    vram_used_bytes: u64,
    vram_capacity_bytes: u64,
    gpu_anchor_count: usize,
    gpu_lru_count: usize,
}

/// Lightweight production health probe that surfaces the engine's
/// per-process counters in a format alerting systems can consume
/// without scraping Prometheus.
async fn admin_health_experts(State(state): State<AppState>) -> Response {
    let report = state.engine.report();
    let status = if report.expert_read_failures > 0 {
        "degraded"
    } else {
        "ok"
    };
    let (free, cap, overflow) = match state.batch_scheduler.as_ref().and_then(|s| s.block_pool()) {
        Some(p) => (
            Some(p.free_blocks()),
            Some(p.capacity()),
            Some(p.overflow_in_use()),
        ),
        None => (None, None, None),
    };
    let body = ExpertHealth {
        status,
        expert_read_failures: report.expert_read_failures,
        cache_hits: report.hits,
        cache_misses: report.misses,
        cache_pinned: report.pinned_count,
        cache_capacity: report.cache_capacity,
        in_flight_requests: state.middleware.admission.in_flight(),
        block_pool_free: free,
        block_pool_capacity: cap,
        block_pool_overflow_in_use: overflow,
        tokens_generated: report.tokens_processed,
        gpu_cache_enabled: report.gpu_cache_enabled,
        gpu_cache_hits: report.gpu_cache_hits,
        gpu_cache_misses: report.gpu_cache_misses,
        gpu_promotions_total: report.gpu_promotions,
        vram_used_bytes: report.vram_used_bytes,
        vram_capacity_bytes: report.vram_capacity_bytes,
        gpu_anchor_count: report.gpu_anchor_count,
        gpu_lru_count: report.gpu_lru_count,
    };
    // Surface degraded health via HTTP status so naive uptime probes
    // (HAProxy health checks, K8s liveness) light up without parsing
    // the body.
    let code = if status == "ok" {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, Json(body)).into_response()
}

#[derive(Serialize, Debug)]
struct EvictResponse {
    reclaimed_overflow_blocks: usize,
}

/// Admin endpoint: trigger a one-shot reclaim pass on the paged-KV
/// pool's overflow slab. Useful for operators after a transient
/// burst inflated the heap-backed fallback — calling this returns
/// the memory to the allocator immediately rather than waiting for
/// the periodic background sweep.
async fn admin_evict_overflow(State(state): State<AppState>) -> Response {
    let reclaimed = match state.batch_scheduler.as_ref().and_then(|s| s.block_pool()) {
        Some(p) => p.shrink_overflow_to_fit(),
        None => 0,
    };
    (
        StatusCode::OK,
        Json(EvictResponse {
            reclaimed_overflow_blocks: reclaimed,
        }),
    )
        .into_response()
}

// ----------------------- generation helpers --------------------------

/// One streamed completion chunk. Mirrors OpenAI's `text_completion` SSE
/// event shape so any OpenAI-compatible client can consume it.
#[derive(Serialize, Debug)]
struct CompletionChunk {
    id: String,
    object: &'static str,
    model: String,
    choices: Vec<CompletionChunkChoice>,
}
#[derive(Serialize, Debug)]
struct CompletionChunkChoice {
    text: String,
    index: u32,
    finish_reason: Option<&'static str>,
}

/// One streamed chat-completion chunk. Same shape OpenAI uses for
/// `chat.completion.chunk`: each event carries a *delta* with the
/// incremental content rather than the full message.
#[derive(Serialize, Debug)]
struct ChatChunk {
    id: String,
    object: &'static str,
    model: String,
    choices: Vec<ChatChunkChoice>,
}
#[derive(Serialize, Debug)]
struct ChatChunkChoice {
    index: u32,
    delta: ChatDelta,
    finish_reason: Option<&'static str>,
}
#[derive(Serialize, Debug, Default)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

/// Per-token output yielded by the streaming generators below: the new
/// piece of decoded text since the previous token, plus the cache
/// hits/misses recorded by the engine for that step.
struct StreamChunk {
    text: String,
    finished: bool,
    hits: u64,
    misses: u64,
}

/// Drive token-by-token generation, yielding incremental decoded text
/// after each step. Both the real-transformer and legacy paths are
/// supported (mirroring `generate`). Returns a boxed Stream so the
/// SSE handler can wrap it.
async fn stream_tokens(
    state: AppState,
    prompt: String,
    requested_max: usize,
    params: SamplingParams,
    session_id: Option<String>,
) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>, GenerateError> {
    if prompt.is_empty() {
        return Err(GenerateError::InvalidRequest("prompt must be non-empty".into()));
    }
    let max_tokens = requested_max
        .min(state.runtime.snapshot().max_tokens_cap)
        .max(1);
    let prompt_ids = state
        .tokenizer
        .encode(&prompt)
        .map_err(|e| GenerateError::Tokenizer(e.to_string()))?;

    // Generator state passed through `stream::unfold`. Holds everything a
    // single-stream generator needs across `await` points.
    enum GenMode {
        Real {
            kv: Vec<crate::transformer::KvCache>,
            last_token: u32,
            position: usize,
            checkout: Option<SessionCheckoutToken>,
        },
        Legacy {
            base: u64,
            i: u64,
        },
    }
    let mode = if let Some(model) = state.real_model.as_ref() {
        // Resume from session state if configured — otherwise start
        // fresh, matching the non-streaming `generate` path.
        let (mut kv, mut pos, checkout) = load_session_kv(&state, model, session_id.as_deref());
        for &tid in prompt_ids.iter() {
            let _ = step_through_scheduler(&state, model, tid, pos, &mut kv, &params).await;
            pos += 1;
        }
        GenMode::Real {
            kv,
            last_token: *prompt_ids.last().unwrap_or(&0u32),
            position: pos,
            checkout,
        }
    } else {
        GenMode::Legacy {
            base: prompt_ids
                .last()
                .copied()
                .unwrap_or(0)
                .wrapping_add(params.seed as u32) as u64,
            i: 0,
        }
    };

    // Carry the cumulative completion ids so we can decode each step's
    // *delta* (a new id may extend a multi-byte UTF-8 token from the
    // previous one; decoding the cumulative buffer and diffing is the
    // safe way to compute "what's new since last chunk").
    struct St {
        state: AppState,
        mode: GenMode,
        completion_ids: Vec<u32>,
        decoded_so_far: String,
        emitted: usize,
        max_tokens: usize,
        finished_emitted: bool,
        params: SamplingParams,
        session_id: Option<String>,
    }

    let st = St {
        state,
        mode,
        completion_ids: Vec::with_capacity(max_tokens),
        decoded_so_far: String::new(),
        emitted: 0,
        max_tokens,
        finished_emitted: false,
        params,
        session_id,
    };

    Ok(Box::pin(stream::unfold(st, move |mut st| async move {
        if st.finished_emitted {
            return None;
        }
        if st.emitted >= st.max_tokens {
            // Final chunk carries `finish_reason: length` and no new text.
            // Persist KV-cache state back to the session store so a
            // follow-up request can pick up where we stopped.
            if let GenMode::Real { kv, last_token: _, position, checkout } = &mut st.mode {
                let kv_take = std::mem::take(kv);
                save_session_kv(
                    &st.state,
                    st.session_id.as_deref(),
                    kv_take,
                    *position,
                    checkout.take(),
                );
            }
            st.finished_emitted = true;
            return Some((
                StreamChunk { text: String::new(), finished: true, hits: 0, misses: 0 },
                st,
            ));
        }
        let pre_hits = st.state.engine.report().hits;
        let pre_misses = st.state.engine.report().misses;
        let next: u32 = match &mut st.mode {
            GenMode::Real { kv, last_token, position, checkout: _ } => {
                let model = st
                    .state
                    .real_model
                    .as_ref()
                    .expect("Real mode requires real_model");
                let n = step_through_scheduler(&st.state, model, *last_token, *position, kv, &st.params).await;
                *last_token = n;
                *position += 1;
                n
            }
            GenMode::Legacy { base, i } => {
                let stats = st.state.engine.generate(base.wrapping_add(*i)).await;
                let _ = stats;
                let vocab = st.state.tokenizer.vocab_size().max(1) as u64;
                let id = ((base.wrapping_add(*i).wrapping_mul(0x9E3779B97F4A7C15)) % vocab) as u32;
                *i = i.wrapping_add(1);
                id
            }
        };
        let post = st.state.engine.report();
        let hits = post.hits.saturating_sub(pre_hits);
        let misses = post.misses.saturating_sub(pre_misses);

        st.completion_ids.push(next);
        st.emitted += 1;

        // Decode the cumulative ids and diff against what we've already
        // sent — robust to multi-byte tokens. Tokenizer errors fall back
        // to "no new text this step" rather than aborting the stream.
        let new_decoded = st
            .state
            .tokenizer
            .decode(&st.completion_ids)
            .unwrap_or_else(|_| st.decoded_so_far.clone());
        let delta = if new_decoded.starts_with(&st.decoded_so_far) {
            new_decoded[st.decoded_so_far.len()..].to_string()
        } else {
            // Re-tokenized text changed earlier characters — emit the
            // full new text and reset the cursor. Rare but possible
            // with BPE tokenizers.
            new_decoded.clone()
        };
        st.decoded_so_far = new_decoded;

        Some((
            StreamChunk { text: delta, finished: false, hits, misses },
            st,
        ))
    })))
}

async fn build_completion_stream(
    state: &AppState,
    prompt: &str,
    requested_max: usize,
    model_name: String,
    params: SamplingParams,
    session_id: Option<String>,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, GenerateError> {
    let id = format!("cmpl-{:x}", rand_request_id());
    let metrics = state.metrics.clone();
    let inner = stream_tokens(state.clone(), prompt.to_string(), requested_max, params, session_id).await?;
    let s = stream::unfold(
        (inner, id, model_name, metrics, 0u64, 0u64, 0u64, false),
        |(mut inner, id, model_name, metrics, mut hits, mut misses, mut tokens_done, terminated)| async move {
            if terminated {
                return None;
            }
            // Pull next token from the inner generator.
            use futures::stream::StreamExt;
            match inner.next().await {
                None => {
                    // Inner exhausted unexpectedly — emit DONE.
                    metrics.record_tokens(tokens_done);
                    metrics.record_cache(hits, misses);
                    let ev = Event::default().data("[DONE]");
                    Some((Ok(ev), (inner, id, model_name, metrics, hits, misses, tokens_done, true)))
                }
                Some(chunk) => {
                    hits += chunk.hits;
                    misses += chunk.misses;
                    if chunk.finished {
                        // End of stream: emit `[DONE]` and terminate.
                        // (We could optionally precede it with a chunk
                        // carrying `finish_reason: length` and empty
                        // text; OpenAI-compatible clients handle either
                        // shape, so we keep the wire output minimal.)
                        let done = Event::default().data("[DONE]");
                        metrics.record_tokens(tokens_done);
                        metrics.record_cache(hits, misses);
                        info!(
                            tokens = tokens_done,
                            cache_hits = hits,
                            cache_misses = misses,
                            "streamed completion finished"
                        );
                        Some((Ok(done), (inner, id, model_name, metrics, hits, misses, tokens_done, true)))
                    } else {
                        tokens_done += 1;
                        let payload = CompletionChunk {
                            id: id.clone(),
                            object: "text_completion",
                            model: model_name.clone(),
                            choices: vec![CompletionChunkChoice {
                                text: chunk.text,
                                index: 0,
                                finish_reason: None,
                            }],
                        };
                        let ev = Event::default()
                            .data(serde_json::to_string(&payload).unwrap_or_default());
                        Some((
                            Ok(ev),
                            (inner, id, model_name, metrics, hits, misses, tokens_done, false),
                        ))
                    }
                }
            }
        },
    );
    Ok(s)
}

async fn build_chat_stream(
    state: &AppState,
    prompt: &str,
    requested_max: usize,
    model_name: String,
    params: SamplingParams,
    session_id: Option<String>,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, GenerateError> {
    let id = format!("chatcmpl-{:x}", rand_request_id());
    let metrics = state.metrics.clone();
    let inner = stream_tokens(state.clone(), prompt.to_string(), requested_max, params, session_id).await?;

    // OpenAI emits a first "delta: { role: assistant }" event before any
    // content tokens. We do the same so streaming chat clients see the
    // role before the first content delta.
    let role_chunk = ChatChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        model: model_name.clone(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta { role: Some("assistant"), content: None },
            finish_reason: None,
        }],
    };
    let role_event =
        Event::default().data(serde_json::to_string(&role_chunk).unwrap_or_default());

    let s = stream::unfold(
        (Some(role_event), inner, id, model_name, metrics, 0u64, 0u64, 0u64, false),
        |(role_ev, mut inner, id, model_name, metrics, mut hits, mut misses, mut tokens_done, terminated)| async move {
            if let Some(ev) = role_ev {
                // Emit the role event first.
                return Some((
                    Ok(ev),
                    (None, inner, id, model_name, metrics, hits, misses, tokens_done, terminated),
                ));
            }
            if terminated {
                return None;
            }
            use futures::stream::StreamExt;
            match inner.next().await {
                None => {
                    metrics.record_tokens(tokens_done);
                    metrics.record_cache(hits, misses);
                    let ev = Event::default().data("[DONE]");
                    Some((
                        Ok(ev),
                        (None, inner, id, model_name, metrics, hits, misses, tokens_done, true),
                    ))
                }
                Some(chunk) => {
                    hits += chunk.hits;
                    misses += chunk.misses;
                    if chunk.finished {
                        // End of stream. We could optionally precede the
                        // terminator with a `ChatChunk { delta: {},
                        // finish_reason: "length" }` event; OpenAI-
                        // compatible clients accept either shape, so we
                        // emit only the `[DONE]` terminator to keep the
                        // wire output minimal.
                        let done = Event::default().data("[DONE]");
                        metrics.record_tokens(tokens_done);
                        metrics.record_cache(hits, misses);
                        info!(
                            tokens = tokens_done,
                            cache_hits = hits,
                            cache_misses = misses,
                            "streamed chat completion finished"
                        );
                        Some((
                            Ok(done),
                            (None, inner, id, model_name, metrics, hits, misses, tokens_done, true),
                        ))
                    } else {
                        tokens_done += 1;
                        let payload = ChatChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            model: model_name.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: ChatDelta {
                                    role: None,
                                    content: Some(chunk.text),
                                },
                                finish_reason: None,
                            }],
                        };
                        let ev = Event::default()
                            .data(serde_json::to_string(&payload).unwrap_or_default());
                        Some((
                            Ok(ev),
                            (None, inner, id, model_name, metrics, hits, misses, tokens_done, false),
                        ))
                    }
                }
            }
        },
    );
    Ok(s)
}

/// 64-bit pseudo-random id derived from the wall clock and a per-call
/// counter. Good enough for a request id; not used for security.
fn rand_request_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    // 2^64 / phi — the standard "golden ratio" odd multiplier used by
    // Knuth's multiplicative hash (also used in Java's HashMap and
    // Linux's hash_64). Mixes the per-call counter into the high bits
    // before XORing with the wall clock so two requests issued in the
    // same nanosecond still get distinct ids.
    const GOLDEN_RATIO_U64: u64 = 0x9E3779B97F4A7C15;
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ n.wrapping_mul(GOLDEN_RATIO_U64)
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorBodyInner,
}
#[derive(Serialize)]
struct ErrorBodyInner {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

fn error_response(e: GenerateError) -> Response {
    let (status, kind) = match &e {
        GenerateError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        GenerateError::Tokenizer(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
    };
    (
        status,
        Json(ErrorBody {
            error: ErrorBodyInner { message: e.to_string(), kind },
        }),
    )
        .into_response()
}

// ------------------------------ run ---------------------------------

/// **Cold-start mitigation orchestrator (gist Task 1).**
///
/// Runs a synthetic pass through the full token pipeline so the
/// very first user request does not pay the one-time costs of:
///
/// * faulting the `AlignedBuffer` slab pool into the resident set
///   (a `BufferPool` slot is `posix_memalign`'d up front but the
///   pages are only zero-filled on first touch),
/// * registering `io_uring` fixed buffers / fixed files with the
///   kernel (`IORING_REGISTER_BUFFERS` is amortised across every
///   subsequent read, but the first read pays the syscall),
/// * JIT-warming the math backend kernels (AVX-512 dispatch
///   tables, candle Tensor allocators, future GPU command queues).
///
/// Implementation strategy:
///
/// 1. If a real-transformer pipeline + batch scheduler is wired in,
///    invoke [`BatchScheduler::submit_internal_warmup`] which drives
///    one synthetic decoder step through the existing scheduler
///    plumbing — bypassing every network / auth / rate-limit /
///    admission layer that gates real HTTP requests.
/// 2. Otherwise, fall back to a direct `engine.warm_with(0..8)` +
///    a tiny synthetic backend `swiglu_into` call. This still
///    primes the slab pool, the io_uring fixed-buffer table, and
///    the math kernel dispatch — just without the
///    `mpsc → scheduler_loop` exercise.
///
/// **Safety contract.** If any warm-up step fails the server must
/// still bind. We log the failure via [`tracing::warn!`] and
/// continue startup — a warm-up failure is *not* a fatal startup
/// error (gist Task 1, "Constraint: The server must still bind if
/// warm-up fails").
///
/// **Performance contract.** This routine runs **before** the
/// listener binds in [`serve`], so the listening socket only
/// accepts connections once warm-up has completed. The first
/// real user therefore sees a 0-ms-start system.
pub async fn run_engine_warmup(state: &AppState) {
    const SCHEDULER_WARMUP_TIMEOUT: Duration = Duration::from_secs(5);
    let start = Instant::now();
    tracing::info!("engine warm-up starting (cold-start mitigation)");

    let mut result: Result<&'static str, String> = Ok("noop");

    if let (Some(model), Some(scheduler)) = (state.real_model.as_ref(), state.batch_scheduler.as_ref()) {
        // Full path: drive a synthetic decoder step through the
        // batch scheduler. Bypasses HTTP/auth/admission by design.
        match tokio::time::timeout(
            SCHEDULER_WARMUP_TIMEOUT,
            scheduler.submit_internal_warmup(model, &state.engine),
        )
        .await
        {
            Ok(Ok(())) => result = Ok("scheduler-synthetic-step"),
            Ok(Err(e)) => result = Err(format!("scheduler warm-up failed: {e}")),
            Err(_) => {
                result = Err(format!(
                    "scheduler warm-up timed out after {}s",
                    SCHEDULER_WARMUP_TIMEOUT.as_secs()
                ))
            }
        }
    } else {
        // Fallback path (no real-transformer / no scheduler wired in
        // — e.g. legacy benchmark generator). Still primes the slab
        // pool + io_uring + math backend so even the legacy path
        // sees a 0-ms start.
        let num_experts = state.engine.num_experts();
        if num_experts > 0 {
            let cap = num_experts.min(8);
            let ids: Vec<u32> = (0..cap).collect();
            if let Err(e) = state.engine.warm_with(&ids).await {
                result = Err(format!("engine.warm_with failed: {e}"));
            } else {
                result = Ok("engine-warm_with");
            }
        }
        // JIT-warm the math backend kernels regardless.
        let backend = crate::backend::current();
        let rows = 4usize;
        let cols = 8usize;
        let gate: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01).collect();
        let up: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.02).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.03).collect();
        let mut y = vec![0.0f32; rows];
        backend.swiglu_into(&gate, &up, &x, rows, cols, &mut y);
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(path) => tracing::info!(
            elapsed_ms = format!("{elapsed_ms:.2}"),
            path,
            "engine warm-up complete — first request sees a primed slab pool, io_uring \
             registrations, and JIT-warmed math kernels"
        ),
        Err(reason) => tracing::warn!(
            elapsed_ms = format!("{elapsed_ms:.2}"),
            reason,
            "engine warm-up failed; continuing to bind listener (cold-start cost will be \
             paid on the first user request)"
        ),
    }
}

/// Bind the server, listen, and run until the runtime is shut down.
pub async fn serve(state: AppState, bind: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Speculative engine warm-up — runs *before* the listener
    // binds so the first user request sees a 0-ms start. Failures
    // here are logged via `tracing::warn!` and the bind still
    // proceeds (gist Task 1 safety constraint).
    run_engine_warmup(&state).await;

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    info!(bind, "HTTP server listening");
    // Graceful shutdown on SIGTERM / Ctrl-C.
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        info!("shutdown signal received; draining...");
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::engine::{Engine, EngineOptions, ModelShape};
    use crate::expert_cache::ExpertCache;
    use crate::multi_layer_cache::MultiLayerExpertCache;
    use crate::io_provider::{generate_synthetic_experts, NvmeStorage, StorageConfig};
    use crate::router::{PredictiveLoader, TopKRouter};
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use std::sync::Arc;
    use tempdir::TempDir;
    use tower::ServiceExt;

    /// Minimal valid [`crate::config::Config`] for the server tests. The
    /// per-test `make_state` overrides `server.max_tokens` to vary the
    /// runtime cap exercised by [`generate`].
    fn test_minimal_cfg() -> crate::config::Config {
        use std::path::PathBuf;
        crate::config::Config {
            server: crate::config::ServerConfig {
                bind: "127.0.0.1:0".into(),
                max_tokens: 32,
                session_ttl_secs: 0,
                max_concurrent_requests: 0,
                admission_min_free_blocks: 0,
            },
            model: crate::config::ModelConfig {
                data_dir: PathBuf::from("./data"),
                num_experts: 8,
                top_k: 2,
                d_model: 8,
                d_ff: 16,
                expert_size: 4096,
                num_layers: 1,
                dtype: crate::inference::WeightDtype::F32,
            },
            storage: crate::config::StorageConfigToml {
                cache_slots: 4,
                block_align: 4096,
                no_direct: true,
                predict_fanout: 2,
                predict_min_prob: 0.05,
                partial_load_fraction: 1.0,
                pin_after_observations: 0,
            },
            tokenizer: crate::config::TokenizerConfig::default(),
            real_transformer: crate::config::RealTransformerConfig::default(),
            sampling: crate::config::SamplingConfig::default(),
            predictive: crate::config::PredictiveConfig::default(),
            security: crate::config::SecurityConfig::default(),
            gpu_cache: crate::config::GpuCacheConfig::default(),
        }
    }

    // We need a tempdir helper but don't want to add `tempfile` for one
    // test; reuse `std::env::temp_dir()` with a unique subpath.
    mod tempdir {
        use std::path::PathBuf;
        pub struct TempDir { path: PathBuf }
        impl TempDir {
            pub fn new(tag: &str) -> std::io::Result<Self> {
                let mut path = std::env::temp_dir();
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                path.push(format!("mer-server-test-{tag}-{}-{nanos}", std::process::id()));
                std::fs::create_dir_all(&path)?;
                Ok(Self { path })
            }
            pub fn path(&self) -> &std::path::Path { &self.path }
        }
        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.path);
            }
        }
    }

    async fn make_state() -> (AppState, TempDir) {
        let dir = TempDir::new("server").unwrap();
        // Tiny shape so the test stays cheap.
        let num_experts = 4u32;
        let d_model = 8;
        let d_ff = 16;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block = 4096usize;
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        generate_synthetic_experts(dir.path(), num_experts, expert_size, d_model, d_ff).unwrap();
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path().to_path_buf(),
                expert_size,
                block_align: block,
                use_direct_io: false, // tmpfs
                num_experts_per_layer: None,
            })
            .unwrap(),
        );
        let cache = Arc::new(MultiLayerExpertCache::single_layer(2));
        let pool = BufferPool::new(3, expert_size, block);
        let router = crate::gating::Router::Markov(Arc::new(TopKRouter::clustered(num_experts, 2, 2, 0.9, 1)));
        let predictor = Arc::new(PredictiveLoader::new(num_experts, 1, 0.05, 1));
        let engine = Arc::new(Engine::with_options(
            cache, pool, storage, router, predictor,
            ModelShape { d_model, d_ff, hidden_seed: 1 },
            EngineOptions::default(),
        ));
        let state = AppState {
            engine,
            tokenizer: Arc::new(Tokenizer::bytes()),
            metrics: Metrics::new(),
            real_model: None,
            batch_scheduler: None,
            runtime: crate::config::LiveConfig::from_config(&{
                let mut c = test_minimal_cfg();
                c.server.max_tokens = 32;
                c
            }),
            sessions: None,
            middleware: MiddlewareState {
                api_keys: crate::middleware::ApiKeyGate::default(),
                rate_limit: crate::middleware::RateLimiter::new(0, 0),
                admission: crate::middleware::Admission::new(0, 0, None),
            },
        };
        (state, dir)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn health_endpoint_returns_ok() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let resp = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        let s = String::from_utf8(body.to_vec()).unwrap();
        assert!(s.contains("\"status\":\"ok\""));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn completions_endpoint_generates_response() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let body = serde_json::json!({
            "prompt": "Once upon a time",
            "max_tokens": 4
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["index"], 0);
        assert_eq!(v["usage"]["completion_tokens"], 4);
        assert!(v["choices"][0]["text"].is_string());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metrics_endpoint_exposes_prometheus_format() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state.clone());
        // Generate one request first so the counters are non-zero.
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("{\"prompt\":\"hi\",\"max_tokens\":2}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let resp = app
            .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 16 * 1024).await.unwrap();
        let s = String::from_utf8(body.to_vec()).unwrap();
        assert!(s.contains("mer_requests_total"));
        assert!(s.contains("mer_tokens_generated_total"));
    }

    /// When `MiddlewareState.api_keys` is configured (non-empty), the
    /// router must continue to expose `/health` and `/metrics` without
    /// any authentication so Kubernetes liveness/readiness probes and
    /// Prometheus scrapers keep working, while every `/v1/...` route
    /// is gated behind the API-key check and returns
    /// `401 Unauthorized` when no valid key is presented.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn public_endpoints_bypass_api_key_gate_while_protected_routes_require_it() {
        let (mut state, _tmp) = make_state().await;
        // Enable the API-key gate with a single key the test will
        // (deliberately) not present on the protected request.
        state.middleware.api_keys =
            crate::middleware::ApiKeyGate::new(&["sekret".to_string()]);
        let app = build_router(state);

        // /health: no auth header, must succeed.
        let resp = app
            .clone()
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/health must be reachable without an API key"
        );

        // /metrics: no auth header, must succeed.
        let resp = app
            .clone()
            .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/metrics must be reachable without an API key"
        );

        // Protected /v1/... route without a key → 401.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("{\"prompt\":\"hi\",\"max_tokens\":1}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "/v1/completions must reject requests without a valid API key"
        );

        // And with the correct key the protected route is reachable
        // again (sanity-check that the gate isn't simply broken).
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer sekret")
                    .body(Body::from("{\"prompt\":\"hi\",\"max_tokens\":1}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/completions must succeed when a valid API key is presented"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn empty_prompt_returns_400() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("{\"prompt\":\"\",\"max_tokens\":2}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chat_completions_round_trips() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 3
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn completions_with_stream_returns_sse_events() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let body = serde_json::json!({
            "prompt": "Once upon",
            "max_tokens": 3,
            "stream": true,
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .map(|v| v.to_str().unwrap_or("").to_string())
            .unwrap_or_default();
        assert!(
            ct.starts_with("text/event-stream"),
            "expected SSE content-type, got {ct:?}"
        );
        // Read enough body to capture all events for max_tokens=3.
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = String::from_utf8(body.to_vec()).unwrap();
        // Should contain at least one data: line with text_completion shape and a [DONE] terminator.
        assert!(s.contains("data: "), "expected SSE data: lines, got {s}");
        assert!(s.contains("text_completion"), "expected event payload to be a text_completion chunk; got {s}");
        assert!(s.contains("[DONE]"), "expected [DONE] terminator; got {s}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chat_completions_with_stream_returns_sse_events() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "stream": true,
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = String::from_utf8(body.to_vec()).unwrap();
        // Role event should appear before any content delta.
        assert!(
            s.contains("\"role\":\"assistant\""),
            "expected leading role event; got {s}"
        );
        assert!(s.contains("chat.completion.chunk"), "expected chunk objects; got {s}");
        assert!(s.contains("[DONE]"));
    }

    /// Build an `AppState` whose engine + storage are sized for a real
    /// transformer with the given config. Returns the state plus the
    /// temp dir that holds the synthesised expert weight files.
    async fn make_state_with_real_model(
        cfg: crate::model::RealModelConfig,
    ) -> (AppState, TempDir) {
        let dir = TempDir::new("server-real").unwrap();
        let total = cfg.num_layers as u32 * cfg.num_experts as u32;
        let weight_bytes = crate::inference::expert_weight_bytes(cfg.d_model, cfg.d_ff);
        let block = 4096usize;
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        crate::io_provider::generate_synthetic_experts(
            dir.path(),
            total,
            expert_size,
            cfg.d_model,
            cfg.d_ff,
        )
        .unwrap();
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path().to_path_buf(),
                expert_size,
                block_align: block,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .unwrap(),
        );
        let cache = Arc::new(MultiLayerExpertCache::single_layer((total as usize).max(2)));
        let pool = BufferPool::new(total as usize + 2, expert_size, block);
        let router = crate::gating::Router::Markov(Arc::new(TopKRouter::new(total, cfg.top_k, 1)));
        let predictor = Arc::new(PredictiveLoader::new(total, 0, 0.05, 1));
        let engine = Arc::new(Engine::with_options(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape { d_model: cfg.d_model, d_ff: cfg.d_ff, hidden_seed: 1 },
            EngineOptions::default(),
        ));
        let model = Arc::new(crate::model::RealModel::new_seeded(cfg.clone(), 0xBEEF));
        let scheduler = Arc::new(crate::batch_scheduler::BatchScheduler::spawn(
            model.clone(),
            engine.clone(),
            crate::batch_scheduler::BatchConfig {
                max_batch_size: 4,
                batch_timeout: std::time::Duration::from_millis(2),
                ..Default::default()
            },
        ));
        let state = AppState {
            engine,
            tokenizer: Arc::new(Tokenizer::bytes()),
            metrics: Metrics::new(),
            real_model: Some(model),
            batch_scheduler: Some(scheduler),
            runtime: crate::config::LiveConfig::from_config(&{
                let mut c = test_minimal_cfg();
                c.server.max_tokens = 16;
                c
            }),
            sessions: Some(crate::session::SessionStore::new(std::time::Duration::from_secs(60))),
            middleware: MiddlewareState {
                api_keys: crate::middleware::ApiKeyGate::default(),
                rate_limit: crate::middleware::RateLimiter::new(0, 0),
                admission: crate::middleware::Admission::new(0, 0, None),
            },
        };
        (state, dir)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn completions_with_real_model_returns_logit_sampled_tokens() {
        let cfg = crate::model::RealModelConfig {
            // vocab=256 matches the byte tokenizer.
            vocab_size: 256,
            d_model: 16,
            d_ff: 32,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 4,
            num_layers: 2,
            num_experts: 4,
            top_k: 2,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
        };
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        let app = build_router(state.clone());
        let body = serde_json::json!({
            "prompt": "Hi",
            "max_tokens": 3
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["usage"]["completion_tokens"], 3);
        assert!(v["choices"][0]["text"].is_string());
        // The engine's hits/misses counters were populated by the
        // real-transformer path's `moe_step` calls.
        let r = state.engine.report();
        assert!(
            r.misses + r.hits > 0,
            "engine cache should be touched by real transformer path"
        );
        assert!(r.bytes_read > 0, "engine should have read expert bytes from SSD");
    }
}
