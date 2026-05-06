use std::cell::RefCell;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::engine::AegisEngine;
use aegisllm_base::error::{AegisError, Result};
use crate::executor::ExecutorReadiness;
use aegisllm_base::generation::{ChatMessage, GenerateRequest, SamplingConfig};
use aegisllm_base::text::TextProcessor;

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

#[derive(Debug, Default)]
struct ServerMetrics {
    requests_total: u64,
    generation_requests_total: u64,
    generation_errors_total: u64,
    generation_rejected_total: u64,
    prompt_tokens_total: u64,
    completion_tokens_total: u64,
    generation_latency_ms_total: f64,
    last_generation_latency_ms: Option<f64>,
}

/// RAII guard that increments the concurrent-generation counter on creation
/// and decrements it on drop — gives accurate kv_pages_used metrics.
struct GenerationGuard {
    counter: Arc<AtomicI64>,
}
impl GenerationGuard {
    fn new(counter: Arc<AtomicI64>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self { counter }
    }
}
impl Drop for GenerationGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Debug)]
struct ServerState {
    metrics: RefCell<ServerMetrics>,
    /// Counts how many generation requests are currently in flight.
    /// Exposed as `kv_pages_used` (one active generation ↔ KV cache in use).
    active_generations: Arc<AtomicI64>,
}

impl ServerState {
    fn new() -> Self {
        Self {
            metrics: RefCell::new(ServerMetrics::default()),
            active_generations: Arc::new(AtomicI64::new(0)),
        }
    }

    fn record_request(&self) {
        self.metrics.borrow_mut().requests_total += 1;
    }

    fn record_generation(&self, started: Instant, stats: Option<GenerateStats>) {
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut metrics = self.metrics.borrow_mut();
        metrics.generation_requests_total += 1;
        metrics.generation_latency_ms_total += latency_ms;
        metrics.last_generation_latency_ms = Some(latency_ms);
        match stats {
            Some(stats) => {
                metrics.prompt_tokens_total += stats.prompt_tokens as u64;
                metrics.completion_tokens_total += stats.completion_tokens as u64;
            }
            None => metrics.generation_errors_total += 1,
        }
    }

    fn record_generation_rejected(&self) {
        let mut metrics = self.metrics.borrow_mut();
        metrics.generation_requests_total += 1;
        metrics.generation_errors_total += 1;
        metrics.generation_rejected_total += 1;
    }
}

#[derive(Debug)]
struct GenerateStats {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerApiCompatibility {
    OpenAi,
    Anthropic,
    Google,
}

impl ServerApiCompatibility {
    fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
        }
    }
}

pub fn serve_http(
    host: String,
    port: u16,
    api: String,
    engine: AegisEngine,
    readiness: ExecutorReadiness,
    default_sampling: SamplingConfig,
) -> Result<()> {
    let api = normalize_api_compatibility(&api)?;
    let listener = TcpListener::bind(format!("{host}:{port}"))?;
    let state = ServerState::new();
    eprintln!(
        "serve: listening on http://{}:{} api={} runnable={} selected={}",
        host,
        port,
        api.name(),
        readiness.runnable,
        readiness.selected_backend
    );
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(error) = handle_http_connection(
                    &mut stream,
                    api,
                    &engine,
                    &readiness,
                    default_sampling,
                    &state,
                ) {
                    let _ = write_json_response(
                        &mut stream,
                        500,
                        serde_json::json!({
                            "error": {
                                "message": error.to_string(),
                                "type": "internal_error"
                            }
                        }),
                    );
                    eprintln!("serve: request failed: {error}");
                }
            }
            Err(error) => eprintln!("serve: accept failed: {error}"),
        }
    }
    Ok(())
}

fn handle_http_connection(
    stream: &mut TcpStream,
    api: ServerApiCompatibility,
    engine: &AegisEngine,
    readiness: &ExecutorReadiness,
    default_sampling: SamplingConfig,
    state: &ServerState,
) -> Result<()> {
    let request = read_http_request(stream)?;
    state.record_request();
    if request.method == "GET" && request.path == "/metrics" {
        let body = metrics_prometheus(state);
        return write_text_response(stream, 200, "text/plain; version=0.0.4; charset=utf-8", &body);
    }
    // Detect SSE streaming: stream:true in body, or Google streamGenerateContent path
    let is_streaming = request.method == "POST"
        && (request.path.ends_with(":streamGenerateContent")
            || serde_json::from_slice::<serde_json::Value>(&request.body)
                .ok()
                .and_then(|v| v.get("stream")?.as_bool())
                .unwrap_or(false));
    if is_streaming {
        return handle_streaming_request(stream, api, engine, readiness, &request, default_sampling, state);
    }
    let (status, payload) =
        route_http_request(api, engine, readiness, request, default_sampling, state);
    write_json_response(stream, status, payload)
}

fn route_http_request(
    api: ServerApiCompatibility,
    engine: &AegisEngine,
    readiness: &ExecutorReadiness,
    request: HttpRequest,
    default_sampling: SamplingConfig,
    state: &ServerState,
) -> (u16, serde_json::Value) {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") | ("GET", "/health") | ("GET", "/healthz") => (
            200,
            serde_json::json!({
                "status": if readiness.runnable { "ok" } else { "degraded" },
                "model": engine.placement.model,
                "api": {
                    "compatibility": api.name(),
                    "streaming": true,
                    "chat_completions": api == ServerApiCompatibility::OpenAi,
                    "completions": api == ServerApiCompatibility::OpenAi,
                    "messages": api == ServerApiCompatibility::Anthropic,
                    "generate_content": api == ServerApiCompatibility::Google
                },
                "executor": readiness_json(readiness),
                "metrics": metrics_json(state),
            }),
        ),
        ("GET", "/ready") | ("GET", "/readyz") => (
            if readiness.runnable { 200 } else { 503 },
            serde_json::json!({
                "ready": readiness.runnable,
                "executor": readiness_json(readiness),
            }),
        ),
        ("GET", "/metrics") => (
            200,
            serde_json::json!({
                "model": engine.placement.model,
                "api": api.name(),
                "executor": readiness_json(readiness),
                "metrics": metrics_json(state),
            }),
        ),
        ("GET", "/v1/models") => (
            200,
            serde_json::json!({
                "object": "list",
                "data": [{
                    "id": engine.placement.model,
                    "object": "model",
                    "owned_by": "aegisllm",
                    "metadata": {
                        "backend": readiness.selected_backend,
                        "runnable": readiness.runnable,
                        "api_compatibility": api.name()
                    }
                }]
            }),
        ),
        ("POST", "/v1/completions") if api == ServerApiCompatibility::OpenAi => {
            generate_http_response(
                engine,
                readiness,
                &request.body,
                false,
                default_sampling,
                state,
            )
        }
        ("POST", "/v1/chat/completions") if api == ServerApiCompatibility::OpenAi => {
            generate_http_response(
                engine,
                readiness,
                &request.body,
                true,
                default_sampling,
                state,
            )
        }
        ("POST", "/v1/messages") if api == ServerApiCompatibility::Anthropic => {
            generate_anthropic_response(engine, readiness, &request.body, default_sampling, state)
        }
        ("POST", path)
            if api == ServerApiCompatibility::Google
                && path.starts_with("/v1beta/models/")
                && path.ends_with(":generateContent") =>
        {
            generate_google_response(engine, readiness, &request.body, default_sampling, state)
        }
        ("OPTIONS", _) => (200, serde_json::json!({})),
        _ => (
            404,
            serde_json::json!({
                "error": {
                    "message": format!("unknown route `{}` {}", request.method, request.path),
                    "type": "not_found"
                }
            }),
        ),
    }
}

fn generate_http_response(
    engine: &AegisEngine,
    readiness: &ExecutorReadiness,
    body: &[u8],
    chat: bool,
    default_sampling: SamplingConfig,
    state: &ServerState,
) -> (u16, serde_json::Value) {
    if !readiness.runnable {
        state.record_generation_rejected();
        return executor_not_ready(readiness);
    }
    let parsed = match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(value) => value,
        Err(error) => {
            return (
                400,
                serde_json::json!({
                    "error": {
                        "message": format!("invalid json body: {error}"),
                        "type": "invalid_request_error"
                    }
                }),
            );
        }
    };
    if let Err(error) = validate_openai_request(engine, &parsed) {
        return json_error(400, error);
    }
    let prompt = if chat {
        match chat_prompt_from_json(engine, &parsed) {
            Ok(prompt) => prompt,
            Err(error) => return json_error(400, error),
        }
    } else {
        match completion_prompt_from_json(&parsed) {
            Ok(prompt) => prompt,
            Err(error) => return json_error(400, error),
        }
    };
    let request = GenerateRequest {
        prompt,
        max_tokens: json_usize_any(&parsed, &["max_tokens", "max_completion_tokens"], 32),
        sampling: SamplingConfig {
            temperature: json_f32(&parsed, "temperature", default_sampling.temperature),
            top_p: json_f32(&parsed, "top_p", default_sampling.top_p),
            top_k: json_usize(&parsed, "top_k", default_sampling.top_k),
            min_p: json_f32(&parsed, "min_p", default_sampling.min_p),
        },
    };
    let started = Instant::now();
    let _kv_guard = GenerationGuard::new(Arc::clone(&state.active_generations));
    match engine.generate(request) {
        Ok(output) if chat => {
            let stats = GenerateStats {
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
            };
            state.record_generation(started, Some(stats));
            (
                200,
                serde_json::json!({
                "id": completion_id("chatcmpl"),
                "object": "chat.completion",
                "created": unix_timestamp(),
                "model": engine.placement.model,
                "system_fingerprint": system_fingerprint(readiness),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output.text
                    },
                    "finish_reason": openai_finish_reason(&output.finish_reason)
                }],
                "usage": {
                    "prompt_tokens": output.prompt_tokens,
                    "completion_tokens": output.completion_tokens,
                    "total_tokens": output.prompt_tokens + output.completion_tokens
                }
                }),
            )
        }
        Ok(output) => {
            let stats = GenerateStats {
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
            };
            state.record_generation(started, Some(stats));
            (
                200,
                serde_json::json!({
                "id": completion_id("cmpl"),
                "object": "text_completion",
                "created": unix_timestamp(),
                "model": engine.placement.model,
                "system_fingerprint": system_fingerprint(readiness),
                "choices": [{
                    "index": 0,
                    "text": output.text,
                    "finish_reason": openai_finish_reason(&output.finish_reason)
                }],
                "usage": {
                    "prompt_tokens": output.prompt_tokens,
                    "completion_tokens": output.completion_tokens,
                    "total_tokens": output.prompt_tokens + output.completion_tokens
                }
                }),
            )
        }
        Err(error) => {
            state.record_generation(started, None);
            json_error(503, error.to_string())
        }
    }
}

fn generate_anthropic_response(
    engine: &AegisEngine,
    readiness: &ExecutorReadiness,
    body: &[u8],
    default_sampling: SamplingConfig,
    state: &ServerState,
) -> (u16, serde_json::Value) {
    if !readiness.runnable {
        state.record_generation_rejected();
        return executor_not_ready(readiness);
    }
    let parsed = match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(value) => value,
        Err(error) => return json_error(400, format!("invalid json body: {error}")),
    };
    let prompt = match chat_prompt_from_json(engine, &parsed) {
        Ok(prompt) => prompt,
        Err(error) => return json_error(400, error),
    };
    let request = GenerateRequest {
        prompt,
        max_tokens: json_usize_any(&parsed, &["max_tokens", "max_completion_tokens"], 32),
        sampling: SamplingConfig {
            temperature: json_f32(&parsed, "temperature", default_sampling.temperature),
            top_p: json_f32(&parsed, "top_p", default_sampling.top_p),
            top_k: json_usize(&parsed, "top_k", default_sampling.top_k),
            min_p: json_f32(&parsed, "min_p", default_sampling.min_p),
        },
    };
    let started = Instant::now();
    let _kv_guard = GenerationGuard::new(Arc::clone(&state.active_generations));
    match engine.generate(request) {
        Ok(output) => {
            let stats = GenerateStats {
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
            };
            state.record_generation(started, Some(stats));
            (
                200,
                serde_json::json!({
                "id": completion_id("msg"),
                "type": "message",
                "role": "assistant",
                "model": engine.placement.model,
                "content": [{
                    "type": "text",
                    "text": output.text
                }],
                "stop_reason": anthropic_stop_reason(&output.finish_reason),
                "usage": {
                    "input_tokens": output.prompt_tokens,
                    "output_tokens": output.completion_tokens
                }
                }),
            )
        }
        Err(error) => {
            state.record_generation(started, None);
            json_error(500, error.to_string())
        }
    }
}

fn generate_google_response(
    engine: &AegisEngine,
    readiness: &ExecutorReadiness,
    body: &[u8],
    default_sampling: SamplingConfig,
    state: &ServerState,
) -> (u16, serde_json::Value) {
    if !readiness.runnable {
        state.record_generation_rejected();
        return executor_not_ready(readiness);
    }
    let parsed = match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(value) => value,
        Err(error) => return json_error(400, format!("invalid json body: {error}")),
    };
    let prompt = match google_prompt_from_json(&parsed) {
        Ok(prompt) => prompt,
        Err(error) => return json_error(400, error),
    };
    let generation_config = parsed
        .get("generationConfig")
        .unwrap_or(&serde_json::Value::Null);
    let request = GenerateRequest {
        prompt,
        max_tokens: json_usize_any(
            generation_config,
            &["maxOutputTokens", "max_tokens", "max_completion_tokens"],
            32,
        ),
        sampling: SamplingConfig {
            temperature: json_f32(
                generation_config,
                "temperature",
                default_sampling.temperature,
            ),
            top_p: json_f32(generation_config, "topP", default_sampling.top_p),
            top_k: json_usize(generation_config, "topK", default_sampling.top_k),
            min_p: json_f32(generation_config, "minP", default_sampling.min_p),
        },
    };
    let started = Instant::now();
    let _kv_guard = GenerationGuard::new(Arc::clone(&state.active_generations));
    match engine.generate(request) {
        Ok(output) => {
            let stats = GenerateStats {
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
            };
            state.record_generation(started, Some(stats));
            (
                200,
                serde_json::json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{ "text": output.text }]
                    },
                    "finishReason": google_finish_reason(&output.finish_reason)
                }],
                "usageMetadata": {
                    "promptTokenCount": output.prompt_tokens,
                    "candidatesTokenCount": output.completion_tokens,
                    "totalTokenCount": output.prompt_tokens + output.completion_tokens
                },
                "modelVersion": engine.placement.model
                }),
            )
        }
        Err(error) => {
            state.record_generation(started, None);
            json_error(500, error.to_string())
        }
    }
}

// ── SSE streaming ────────────────────────────────────────────────────────────

fn handle_streaming_request(
    stream: &mut TcpStream,
    api: ServerApiCompatibility,
    engine: &AegisEngine,
    readiness: &ExecutorReadiness,
    request: &HttpRequest,
    default_sampling: SamplingConfig,
    state: &ServerState,
) -> Result<()> {
    if !readiness.runnable {
        state.record_generation_rejected();
        let (status, payload) = executor_not_ready(readiness);
        return write_json_response(stream, status, payload);
    }
    let parsed = match serde_json::from_slice::<serde_json::Value>(&request.body) {
        Ok(v) => v,
        Err(e) => {
            return write_json_response(
                stream,
                400,
                serde_json::json!({
                    "error": {"message": format!("invalid json body: {e}"), "type": "invalid_request_error"}
                }),
            );
        }
    };
    match api {
        ServerApiCompatibility::OpenAi if request.path == "/v1/chat/completions" => {
            sse_openai(stream, engine, readiness, &parsed, true, default_sampling, state)
        }
        ServerApiCompatibility::OpenAi if request.path == "/v1/completions" => {
            sse_openai(stream, engine, readiness, &parsed, false, default_sampling, state)
        }
        ServerApiCompatibility::Anthropic if request.path == "/v1/messages" => {
            sse_anthropic(stream, engine, readiness, &parsed, default_sampling, state)
        }
        ServerApiCompatibility::Google
            if request.path.starts_with("/v1beta/models/")
                && (request.path.ends_with(":streamGenerateContent")
                    || request.path.ends_with(":generateContent")) =>
        {
            sse_google(stream, engine, &parsed, default_sampling, state)
        }
        _ => write_json_response(
            stream,
            404,
            serde_json::json!({
                "error": {
                    "message": format!("streaming not supported for {} {}", request.method, request.path),
                    "type": "not_found"
                }
            }),
        ),
    }
}

fn sse_openai(
    stream: &mut TcpStream,
    engine: &AegisEngine,
    readiness: &ExecutorReadiness,
    parsed: &serde_json::Value,
    chat: bool,
    default_sampling: SamplingConfig,
    state: &ServerState,
) -> Result<()> {
    if let Err(error) = validate_openai_request(engine, parsed) {
        return write_json_response(
            stream,
            400,
            serde_json::json!({"error": {"message": error, "type": "invalid_request_error"}}),
        );
    }
    let prompt = if chat {
        match chat_prompt_from_json(engine, parsed) {
            Ok(p) => p,
            Err(e) => {
                return write_json_response(
                    stream,
                    400,
                    serde_json::json!({"error": {"message": e, "type": "invalid_request_error"}}),
                );
            }
        }
    } else {
        match completion_prompt_from_json(parsed) {
            Ok(p) => p,
            Err(e) => {
                return write_json_response(
                    stream,
                    400,
                    serde_json::json!({"error": {"message": e, "type": "invalid_request_error"}}),
                );
            }
        }
    };
    let gen_request = GenerateRequest {
        prompt,
        max_tokens: json_usize_any(parsed, &["max_tokens", "max_completion_tokens"], 32),
        sampling: SamplingConfig {
            temperature: json_f32(parsed, "temperature", default_sampling.temperature),
            top_p: json_f32(parsed, "top_p", default_sampling.top_p),
            top_k: json_usize(parsed, "top_k", default_sampling.top_k),
            min_p: json_f32(parsed, "min_p", default_sampling.min_p),
        },
    };

    write_sse_headers(stream)?;

    let _kv_guard = GenerationGuard::new(Arc::clone(&state.active_generations));
    let id = completion_id(if chat { "chatcmpl" } else { "cmpl" });
    let created = unix_timestamp();
    let model = engine.placement.model.clone();
    let fingerprint = system_fingerprint(readiness);
    let object = if chat { "chat.completion.chunk" } else { "text_completion.chunk" };

    // Initial role chunk (OpenAI chat convention)
    if chat {
        let first = serde_json::json!({
            "id": id, "object": object, "created": created, "model": model,
            "system_fingerprint": fingerprint,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        });
        write_sse_chunk(stream, &format!("data: {}\n\n", first)).ok();
    }

    let started = Instant::now();
    let mut finish_str = "stop".to_string();

    let result = engine.generate_streaming(&gen_request, &mut |_tok, tok_text| {
        let chunk = if chat {
            serde_json::json!({
                "id": id, "object": object, "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": tok_text}, "finish_reason": null}]
            })
        } else {
            serde_json::json!({
                "id": id, "object": object, "created": created, "model": model,
                "choices": [{"index": 0, "text": tok_text, "finish_reason": null}]
            })
        };
        if write_sse_chunk(stream, &format!("data: {}\n\n", chunk)).is_err() {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });

    match result {
        Ok(ref output) => {
            finish_str = openai_finish_reason(&output.finish_reason).to_string();
            state.record_generation(
                started,
                Some(GenerateStats {
                    prompt_tokens: output.prompt_tokens,
                    completion_tokens: output.completion_tokens,
                }),
            );
        }
        Err(_) => state.record_generation(started, None),
    }

    // Final stop chunk
    let final_chunk = if chat {
        serde_json::json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_str}]
        })
    } else {
        serde_json::json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": finish_str}]
        })
    };
    write_sse_chunk(stream, &format!("data: {}\n\n", final_chunk)).ok();
    write_sse_chunk(stream, "data: [DONE]\n\n").ok();
    write_sse_end(stream).ok();
    Ok(())
}

fn sse_anthropic(
    stream: &mut TcpStream,
    engine: &AegisEngine,
    readiness: &ExecutorReadiness,
    parsed: &serde_json::Value,
    default_sampling: SamplingConfig,
    state: &ServerState,
) -> Result<()> {
    let prompt = match chat_prompt_from_json(engine, parsed) {
        Ok(p) => p,
        Err(e) => {
            return write_json_response(
                stream,
                400,
                serde_json::json!({"error": {"message": e, "type": "invalid_request_error"}}),
            );
        }
    };
    let gen_request = GenerateRequest {
        prompt,
        max_tokens: json_usize_any(parsed, &["max_tokens", "max_completion_tokens"], 32),
        sampling: SamplingConfig {
            temperature: json_f32(parsed, "temperature", default_sampling.temperature),
            top_p: json_f32(parsed, "top_p", default_sampling.top_p),
            top_k: json_usize(parsed, "top_k", default_sampling.top_k),
            min_p: json_f32(parsed, "min_p", default_sampling.min_p),
        },
    };

    write_sse_headers(stream)?;

    let _kv_guard = GenerationGuard::new(Arc::clone(&state.active_generations));
    let msg_id = completion_id("msg");
    let model = engine.placement.model.clone();
    let fingerprint = system_fingerprint(readiness);

    // message_start
    let msg_start = serde_json::json!({
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "model": model, "system_fingerprint": fingerprint,
            "content": [], "stop_reason": null,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }
    });
    write_sse_chunk(stream, &format!("event: message_start\ndata: {}\n\n", msg_start)).ok();

    // content_block_start
    let block_start = serde_json::json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}});
    write_sse_chunk(stream, &format!("event: content_block_start\ndata: {}\n\n", block_start)).ok();
    write_sse_chunk(stream, "event: ping\ndata: {\"type\":\"ping\"}\n\n").ok();

    let started = Instant::now();
    let mut output_tokens = 0usize;
    let mut stop_reason = "end_turn".to_string();

    let result = engine.generate_streaming(&gen_request, &mut |_tok, tok_text| {
        let delta = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": tok_text}
        });
        if write_sse_chunk(stream, &format!("event: content_block_delta\ndata: {}\n\n", delta)).is_err() {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });

    match result {
        Ok(ref output) => {
            output_tokens = output.completion_tokens;
            stop_reason = anthropic_stop_reason(&output.finish_reason).to_string();
            state.record_generation(
                started,
                Some(GenerateStats {
                    prompt_tokens: output.prompt_tokens,
                    completion_tokens: output.completion_tokens,
                }),
            );
        }
        Err(_) => state.record_generation(started, None),
    }

    // content_block_stop
    write_sse_chunk(stream, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n").ok();

    // message_delta
    let msg_delta = serde_json::json!({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
        "usage": {"output_tokens": output_tokens}
    });
    write_sse_chunk(stream, &format!("event: message_delta\ndata: {}\n\n", msg_delta)).ok();

    // message_stop
    write_sse_chunk(stream, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n").ok();
    write_sse_end(stream).ok();
    Ok(())
}

fn sse_google(
    stream: &mut TcpStream,
    engine: &AegisEngine,
    parsed: &serde_json::Value,
    default_sampling: SamplingConfig,
    state: &ServerState,
) -> Result<()> {
    let prompt = match google_prompt_from_json(parsed) {
        Ok(p) => p,
        Err(e) => {
            return write_json_response(
                stream,
                400,
                serde_json::json!({"error": {"message": e, "type": "invalid_request_error"}}),
            );
        }
    };
    let generation_config = parsed
        .get("generationConfig")
        .unwrap_or(&serde_json::Value::Null);
    let gen_request = GenerateRequest {
        prompt,
        max_tokens: json_usize_any(
            generation_config,
            &["maxOutputTokens", "max_tokens", "max_completion_tokens"],
            32,
        ),
        sampling: SamplingConfig {
            temperature: json_f32(generation_config, "temperature", default_sampling.temperature),
            top_p: json_f32(generation_config, "topP", default_sampling.top_p),
            top_k: json_usize(generation_config, "topK", default_sampling.top_k),
            min_p: json_f32(generation_config, "minP", default_sampling.min_p),
        },
    };

    write_sse_headers(stream)?;

    let _kv_guard = GenerationGuard::new(Arc::clone(&state.active_generations));
    let model = engine.placement.model.clone();
    let started = Instant::now();
    let mut finish_reason_str = "STOP".to_string();

    let result = engine.generate_streaming(&gen_request, &mut |_tok, tok_text| {
        let chunk = serde_json::json!({
            "candidates": [{"content": {"role": "model", "parts": [{"text": tok_text}]}, "index": 0}],
            "modelVersion": model
        });
        if write_sse_chunk(stream, &format!("data: {}\n\n", chunk)).is_err() {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });

    match result {
        Ok(ref output) => {
            finish_reason_str = google_finish_reason(&output.finish_reason).to_string();
            state.record_generation(
                started,
                Some(GenerateStats {
                    prompt_tokens: output.prompt_tokens,
                    completion_tokens: output.completion_tokens,
                }),
            );
        }
        Err(_) => state.record_generation(started, None),
    }

    // Final usage chunk
    if let Ok(ref output) = result {
        let final_chunk = serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": ""}]},
                "finishReason": finish_reason_str, "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": output.prompt_tokens,
                "candidatesTokenCount": output.completion_tokens,
                "totalTokenCount": output.prompt_tokens + output.completion_tokens
            },
            "modelVersion": engine.placement.model
        });
        write_sse_chunk(stream, &format!("data: {}\n\n", final_chunk)).ok();
    }
    write_sse_end(stream).ok();
    Ok(())
}

fn write_sse_headers(stream: &mut TcpStream) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\naccess-control-allow-origin: *\r\naccess-control-allow-headers: content-type, authorization\r\naccess-control-allow-methods: GET, POST, OPTIONS\r\ntransfer-encoding: chunked\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n"
    )?;
    Ok(())
}

fn write_sse_chunk(stream: &mut TcpStream, frame: &str) -> std::io::Result<()> {
    write!(stream, "{:x}\r\n", frame.len())?;
    stream.write_all(frame.as_bytes())?;
    write!(stream, "\r\n")
}

fn write_sse_end(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"0\r\n\r\n")
}

// ─────────────────────────────────────────────────────────────────────────────

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 8192];
    let header_end = loop {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(AegisError::InvalidConfig("empty http request".into()));
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(index) = find_bytes(&buffer, b"\r\n\r\n") {
            break index;
        }
        if buffer.len() > 1024 * 1024 {
            return Err(AegisError::InvalidConfig(
                "http request headers exceed 1 MiB".into(),
            ));
        }
    };
    let header_bytes = &buffer[..header_end];
    let headers = String::from_utf8_lossy(header_bytes);
    let mut lines = headers.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| AegisError::InvalidConfig("missing http request line".into()))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| AegisError::InvalidConfig("missing http method".into()))?
        .to_string();
    let path = request_parts
        .next()
        .ok_or_else(|| AegisError::InvalidConfig("missing http path".into()))?
        .split('?')
        .next()
        .unwrap_or("/")
        .to_string();
    let content_length = headers
        .lines()
        .skip(1)
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    if content_length > 16 * 1024 * 1024 {
        return Err(AegisError::InvalidConfig(format!(
            "http body exceeds 16 MiB limit: {content_length} bytes"
        )));
    }
    let body_start = header_end + 4;
    let total_len = body_start + content_length;
    while buffer.len() < total_len {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    if buffer.len() < total_len {
        return Err(AegisError::InvalidConfig(format!(
            "http body truncated: expected {content_length} bytes"
        )));
    }
    Ok(HttpRequest {
        method,
        path,
        body: buffer[body_start..total_len].to_vec(),
    })
}

fn write_json_response(
    stream: &mut TcpStream,
    status: u16,
    payload: serde_json::Value,
) -> Result<()> {
    let body = serde_json::to_vec(&payload)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\naccess-control-allow-origin: *\r\naccess-control-allow-headers: content-type, authorization\r\naccess-control-allow-methods: GET, POST, OPTIONS\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    Ok(())
}

fn normalize_api_compatibility(api: &str) -> Result<ServerApiCompatibility> {
    match api.trim().to_ascii_lowercase().as_str() {
        "openai" | "openai-compatible" | "openai-chat" | "v1" => Ok(ServerApiCompatibility::OpenAi),
        "anthropic" | "claude" => Ok(ServerApiCompatibility::Anthropic),
        "google" | "gemini" => Ok(ServerApiCompatibility::Google),
        other => Err(AegisError::InvalidConfig(format!(
            "unsupported server-api `{other}`; expected openai|anthropic|google"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_compatible_aliases_normalize_to_openai() {
        assert_eq!(
            normalize_api_compatibility("openai-compatible").unwrap(),
            ServerApiCompatibility::OpenAi
        );
        assert_eq!(
            normalize_api_compatibility("openai").unwrap(),
            ServerApiCompatibility::OpenAi
        );
        assert_eq!(
            normalize_api_compatibility("v1").unwrap(),
            ServerApiCompatibility::OpenAi
        );
    }

    #[test]
    fn api_compatibility_modes_are_selectable() {
        assert_eq!(
            normalize_api_compatibility("anthropic").unwrap(),
            ServerApiCompatibility::Anthropic
        );
        assert_eq!(
            normalize_api_compatibility("google").unwrap(),
            ServerApiCompatibility::Google
        );
    }

    #[test]
    fn openai_request_validation_rejects_unsupported_fields() {
        let wrong_model = serde_json::json!({ "model": "other" });
        assert!(validate_openai_request_for_model("loaded", &wrong_model).is_err());

        let multi_choice = serde_json::json!({ "model": "loaded", "n": 2 });
        assert!(validate_openai_request_for_model("loaded", &multi_choice).is_err());

        // stream:true is now supported — must not be rejected
        let streaming = serde_json::json!({ "model": "loaded", "stream": true });
        assert!(validate_openai_request_for_model("loaded", &streaming).is_ok());
    }

    #[test]
    fn openai_finish_reason_maps_internal_eos_to_stop() {
        assert_eq!(openai_finish_reason("eos_token"), "stop");
        assert_eq!(openai_finish_reason("length"), "length");
    }

    #[test]
    fn server_metrics_record_success_and_error() {
        let state = ServerState::new();
        state.record_request();
        state.record_generation(
            Instant::now(),
            Some(GenerateStats {
                prompt_tokens: 7,
                completion_tokens: 3,
            }),
        );
        state.record_generation(Instant::now(), None);
        state.record_generation_rejected();

        let metrics = metrics_json(&state);
        assert_eq!(metrics["requests_total"], 1);
        assert_eq!(metrics["generation_requests_total"], 3);
        assert_eq!(metrics["generation_errors_total"], 2);
        assert_eq!(metrics["generation_rejected_total"], 1);
        assert_eq!(metrics["prompt_tokens_total"], 7);
        assert_eq!(metrics["completion_tokens_total"], 3);
        assert!(metrics["generation_latency_ms_avg"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn prometheus_metrics_emits_all_required_aegis_names() {
        let state = ServerState::new();
        state.record_request();
        state.record_generation(
            Instant::now(),
            Some(GenerateStats { prompt_tokens: 10, completion_tokens: 5 }),
        );
        let prom = metrics_prometheus(&state);
        let required = [
            "aegis_http_requests_total",
            "aegis_generation_requests_total",
            "aegis_generation_errors_total",
            "aegis_generation_rejected_total",
            "aegis_engine_prefill_tokens_total",
            "aegis_engine_decode_tokens_total",
            "aegis_engine_request_latency_seconds_total",
            "aegis_engine_request_latency_seconds_avg",
            "aegis_engine_request_latency_seconds_last",
            "aegis_engine_kv_pages_used",
            "aegis_engine_kv_pages_free",
        ];
        for name in &required {
            assert!(prom.contains(name), "missing metric: {name}");
        }
        // Prometheus format: each metric must have # HELP and # TYPE lines
        assert!(prom.contains("# HELP aegis_engine_prefill_tokens_total"));
        assert!(prom.contains("# TYPE aegis_engine_prefill_tokens_total counter"));
        // Value check: 10 prompt tokens recorded
        assert!(prom.contains("aegis_engine_prefill_tokens_total 10\n"));
    }
}

fn completion_prompt_from_json(value: &serde_json::Value) -> std::result::Result<String, String> {
    if let Some(prompt) = value.get("prompt").and_then(serde_json::Value::as_str) {
        return Ok(prompt.to_string());
    }
    if let Some(prompts) = value.get("prompt").and_then(serde_json::Value::as_array) {
        if prompts.len() != 1 {
            return Err("prompt arrays with more than one item are not supported yet".into());
        }
        if let Some(prompt) = prompts.first().and_then(serde_json::Value::as_str) {
            return Ok(prompt.to_string());
        }
    }
    Err("request requires string `prompt`".into())
}

fn validate_openai_request(
    engine: &AegisEngine,
    value: &serde_json::Value,
) -> std::result::Result<(), String> {
    validate_openai_request_for_model(&engine.placement.model, value)
}

fn validate_openai_request_for_model(
    loaded_model: &str,
    value: &serde_json::Value,
) -> std::result::Result<(), String> {
    if let Some(model) = value.get("model").and_then(serde_json::Value::as_str)
        && model != loaded_model
        && model != "aegisllm"
    {
        return Err(format!(
            "requested model `{model}` does not match loaded model `{loaded_model}`"
        ));
    }
    if let Some(n) = value.get("n").and_then(serde_json::Value::as_u64)
        && n != 1
    {
        return Err("only n=1 is supported".into());
    }
    for key in [
        "stop",
        "logprobs",
        "top_logprobs",
        "presence_penalty",
        "frequency_penalty",
        "tools",
        "tool_choice",
        "response_format",
    ] {
        if value.get(key).is_some() {
            return Err(format!("request field `{key}` is not supported yet"));
        }
    }
    Ok(())
}

fn chat_prompt_from_json(
    engine: &AegisEngine,
    value: &serde_json::Value,
) -> std::result::Result<String, String> {
    let messages = value
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "request requires array `messages`".to_string())?;
    let mut parsed = Vec::with_capacity(messages.len());
    for message in messages {
        let role = message
            .get("role")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("user");
        let content = message
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "each message requires string `content`".to_string())?;
        parsed.push(ChatMessage {
            role: role.into(),
            content: content.into(),
        });
    }
    TextProcessor::render_chat_messages_for_artifact(&engine.artifact, &parsed)
        .map_err(|error| error.to_string())
}

fn google_prompt_from_json(value: &serde_json::Value) -> std::result::Result<String, String> {
    let contents = value
        .get("contents")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "request requires array `contents`".to_string())?;
    let mut text = String::new();
    for content in contents {
        let role = content
            .get("role")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("user");
        let parts = content
            .get("parts")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "each content item requires array `parts`".to_string())?;
        for part in parts {
            let part_text = part
                .get("text")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "only text parts are supported".to_string())?;
            text.push_str(role);
            text.push_str(": ");
            text.push_str(part_text);
            text.push('\n');
        }
    }
    if text.is_empty() {
        Err("request contains no text parts".into())
    } else {
        Ok(text)
    }
}

fn readiness_json(readiness: &ExecutorReadiness) -> serde_json::Value {
    serde_json::json!({
        "selected": readiness.selected_backend,
        "runnable": readiness.runnable,
        "planned_cpu_regions": readiness.planned_cpu_regions,
        "planned_cuda_regions": readiness.planned_cuda_regions,
        "limitations": readiness.limitations,
    })
}

fn metrics_json(state: &ServerState) -> serde_json::Value {
    let metrics = state.metrics.borrow();
    let measured_generation_requests = metrics
        .generation_requests_total
        .saturating_sub(metrics.generation_rejected_total);
    let average_generation_latency_ms = if measured_generation_requests == 0 {
        None
    } else {
        Some(metrics.generation_latency_ms_total / measured_generation_requests as f64)
    };
    serde_json::json!({
        "requests_total": metrics.requests_total,
        "generation_requests_total": metrics.generation_requests_total,
        "generation_errors_total": metrics.generation_errors_total,
        "generation_rejected_total": metrics.generation_rejected_total,
        "generation_measured_requests_total": measured_generation_requests,
        "prompt_tokens_total": metrics.prompt_tokens_total,
        "completion_tokens_total": metrics.completion_tokens_total,
        "generation_latency_ms_total": metrics.generation_latency_ms_total,
        "generation_latency_ms_avg": average_generation_latency_ms,
        "generation_latency_ms_last": metrics.last_generation_latency_ms,
    })
}

fn metrics_prometheus(state: &ServerState) -> String {
    let metrics = state.metrics.borrow();
    let measured = metrics
        .generation_requests_total
        .saturating_sub(metrics.generation_rejected_total);
    let latency_seconds_total = metrics.generation_latency_ms_total / 1000.0;
    let latency_seconds_avg = if measured == 0 {
        0.0
    } else {
        latency_seconds_total / measured as f64
    };
    let latency_seconds_last = metrics.last_generation_latency_ms.unwrap_or(0.0) / 1000.0;
    let active = state.active_generations.load(Ordering::Relaxed).max(0) as u64;

    let mut out = String::with_capacity(1024);
    macro_rules! prom {
        ($name:literal, $help:literal, $kind:literal, $val:expr) => {
            out.push_str(concat!("# HELP ", $name, " ", $help, "\n"));
            out.push_str(concat!("# TYPE ", $name, " ", $kind, "\n"));
            out.push_str(&format!(concat!($name, " {}\n"), $val));
        };
    }
    prom!("aegis_http_requests_total", "Total HTTP requests received", "counter", metrics.requests_total);
    prom!("aegis_generation_requests_total", "Total generation requests", "counter", metrics.generation_requests_total);
    prom!("aegis_generation_errors_total", "Total generation errors", "counter", metrics.generation_errors_total);
    prom!("aegis_generation_rejected_total", "Total rejected generation requests", "counter", metrics.generation_rejected_total);
    prom!("aegis_engine_prefill_tokens_total", "Total prompt tokens processed", "counter", metrics.prompt_tokens_total);
    prom!("aegis_engine_decode_tokens_total", "Total completion tokens generated", "counter", metrics.completion_tokens_total);
    prom!("aegis_engine_request_latency_seconds_total", "Total generation latency in seconds", "counter", latency_seconds_total);
    prom!("aegis_engine_request_latency_seconds_avg", "Average generation latency in seconds", "gauge", latency_seconds_avg);
    prom!("aegis_engine_request_latency_seconds_last", "Last generation latency in seconds", "gauge", latency_seconds_last);
    prom!("aegis_engine_kv_pages_used", "KV cache pages currently in use", "gauge", active);
    prom!("aegis_engine_kv_pages_free", "KV cache pages currently free", "gauge", 0u64);
    out
}

fn write_text_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &str,
) -> Result<()> {
    let reason = match status {
        200 => "OK",
        _ => "OK",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\naccess-control-allow-origin: *\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body.as_bytes())?;
    Ok(())
}

fn json_error(status: u16, message: impl Into<String>) -> (u16, serde_json::Value) {
    (
        status,
        serde_json::json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error"
            }
        }),
    )
}

fn executor_not_ready(readiness: &ExecutorReadiness) -> (u16, serde_json::Value) {
    (
        503,
        serde_json::json!({
            "error": {
                "message": "executor plan is not runnable yet",
                "type": "executor_not_ready",
                "executor": readiness_json(readiness)
            }
        }),
    )
}

fn json_usize(value: &serde_json::Value, key: &str, default: usize) -> usize {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(default)
}

fn json_usize_any(value: &serde_json::Value, keys: &[&str], default: usize) -> usize {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(serde_json::Value::as_u64))
        .map(|value| value as usize)
        .unwrap_or(default)
}

fn json_f32(value: &serde_json::Value, key: &str, default: f32) -> f32 {
    value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or(default)
}

fn openai_finish_reason(reason: &str) -> &'static str {
    match reason {
        "eos_token" => "stop",
        "length" => "length",
        _ => "stop",
    }
}

fn anthropic_stop_reason(reason: &str) -> &'static str {
    match reason {
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

fn google_finish_reason(reason: &str) -> &'static str {
    match reason {
        "length" => "MAX_TOKENS",
        _ => "STOP",
    }
}

fn completion_id(prefix: &str) -> String {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("{prefix}-{millis}")
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn system_fingerprint(readiness: &ExecutorReadiness) -> String {
    format!("aegis-{}", readiness.selected_backend)
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
