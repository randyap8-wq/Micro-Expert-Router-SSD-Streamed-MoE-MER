//! OpenAI-compatible HTTP server (gist Phase 6).
//!
//! Endpoints:
//! - `POST /v1/completions`      — text completion
//! - `POST /v1/chat/completions` — chat completion (single-turn, no streaming yet)
//! - `GET  /health`              — liveness
//! - `GET  /metrics`             — Prometheus metrics
//!
//! The server is intentionally **stateless per request**: each request
//! drives the existing [`crate::engine::Engine`] for `max_tokens` token
//! cycles. The engine in turn uses the SSD-streaming expert cache as
//! before — that's the whole point of this codebase. Continuous batching
//! is *not* implemented in this PR (gist Phase 7); the simple per-request
//! generator is the foundation a batched scheduler can be added on top of.
//!
//! Generation strategy: the synthesised hidden state from
//! [`crate::inference::synth_hidden_state`] drives one
//! `Engine::generate` call per token. The "decoded" response text is
//! the round-tripped token ids through the configured tokenizer (real
//! HF tokenizer if `tokenizer.json` is present, byte tokenizer otherwise).
//! Once a real transformer decoder is wired (the [`crate::transformer`]
//! pieces), this loop swaps to producing real next-token logits — the
//! HTTP shape doesn't change.

use crate::engine::Engine;
use crate::metrics::Metrics;
use crate::tokenizer::Tokenizer;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

/// Shared handler state. Cheap to clone — everything is `Arc`.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub tokenizer: Arc<Tokenizer>,
    pub metrics: Metrics,
    pub max_tokens_cap: usize,
}

/// Build the axum [`Router`] for the API.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
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
    /// We accept `stream` for OpenAI compatibility but currently ignore
    /// it; streaming is on the roadmap (Phase 7).
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
    if req.stream.unwrap_or(false) {
        warn!("client requested streaming, but streaming is not yet implemented; returning non-streaming response");
    }
    match generate(&state, &req.prompt, req.max_tokens, &req.model).await {
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
    if req.stream.unwrap_or(false) {
        warn!("client requested streaming, but streaming is not yet implemented; returning non-streaming response");
    }
    // Flatten messages into a single prompt — exactly the same shape
    // simple OpenAI-compatible servers (vLLM, llama.cpp's HTTP) do when
    // no chat template is configured.
    let prompt = flatten_messages(&req.messages);
    match generate(&state, &prompt, req.max_tokens, &req.model).await {
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

async fn generate(
    state: &AppState,
    prompt: &str,
    requested_max: usize,
    model_name: &str,
) -> Result<CompletionResponse, GenerateError> {
    if prompt.is_empty() {
        return Err(GenerateError::InvalidRequest("prompt must be non-empty".into()));
    }
    let max_tokens = requested_max.min(state.max_tokens_cap).max(1);

    // 1) Tokenize the prompt. The token ids drive the engine's deterministic
    //    routing seed so completions are reproducible for a given prompt.
    let prompt_ids = state
        .tokenizer
        .encode(prompt)
        .map_err(|e| GenerateError::Tokenizer(e.to_string()))?;
    let prompt_tokens = prompt_ids.len();

    // 2) Drive `Engine::generate` for `max_tokens` cycles. Each cycle
    //    streams its routed experts from SSD (cache-miss) or from the
    //    LRU (hit), runs the SwiGLU FFN over real f32 weights, and
    //    updates the predictor / metrics. We use the prompt's last
    //    token id (or 0) as the base for our token-index stream so
    //    different prompts get different routing trajectories.
    //
    //    NOTE: this is the "single-stream, scalar inference" path. A
    //    future PR adds the real transformer decoder; the HTTP shape
    //    above is unchanged when that happens.
    let base = prompt_ids.last().copied().unwrap_or(0) as u64;
    let mut hits_total = 0u64;
    let mut misses_total = 0u64;
    let mut completion_ids: Vec<u32> = Vec::with_capacity(max_tokens);
    for i in 0..max_tokens {
        let stats = state.engine.generate(base.wrapping_add(i as u64)).await;
        hits_total += stats.hits;
        misses_total += stats.misses;
        // Map engine cycle stats to a deterministic next-token id. The
        // synthesis is deliberately simple (vocab modulo); when the
        // real LM head is added, this is where its argmax lands.
        let vocab = state.tokenizer.vocab_size().max(1) as u64;
        let next = ((base.wrapping_add(i as u64).wrapping_mul(0x9E3779B97F4A7C15)) % vocab) as u32;
        completion_ids.push(next);
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

/// 64-bit pseudo-random id derived from the wall clock and a per-call
/// counter. Good enough for a request id; not used for security.
fn rand_request_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ n.wrapping_mul(0x9E3779B97F4A7C15)
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

/// Bind the server, listen, and run until the runtime is shut down.
pub async fn serve(state: AppState, bind: &str) -> Result<(), Box<dyn std::error::Error>> {
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
    use crate::io_provider::{generate_synthetic_experts, NvmeStorage, StorageConfig};
    use crate::router::{PredictiveLoader, TopKRouter};
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use std::sync::Arc;
    use tempdir::TempDir;
    use tower::ServiceExt;

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
            })
            .unwrap(),
        );
        let cache = Arc::new(ExpertCache::new(2));
        let pool = BufferPool::new(3, expert_size, block);
        let router = Arc::new(TopKRouter::clustered(num_experts, 2, 2, 0.9, 1));
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
            max_tokens_cap: 32,
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
}
