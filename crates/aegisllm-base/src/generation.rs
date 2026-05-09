use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    /// Additional stop tokens beyond the model's intrinsic EOS — for tool
    /// calling (`<tool_call|>` etc.) and per-request "stop" sequences.
    /// Empty by default; populated by the chat server when tools are
    /// present so the model halts cleanly after a tool_call instead of
    /// hallucinating a fake tool response.
    pub stop_token_ids: Vec<usize>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 0,
            sampling: SamplingConfig::default(),
            stop_token_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Minimum-probability filter: keep candidates with `p >= min_p * p_max`,
    /// where `p_max` is the highest post-temperature probability. `0.0`
    /// disables the filter. Applied after `top_k` and before `top_p`.
    pub min_p: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimedGenerateOutput {
    pub output: GenerateOutput,
    pub tokenize_elapsed: Duration,
    pub prefill_elapsed: Duration,
    pub decode_elapsed: Duration,
    pub total_elapsed: Duration,
    pub prefill_stage_timings: Option<PrefillStageTimings>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefillStageTimings {
    pub chunks: usize,
    pub prepare_us: u128,
    pub embed_us: u128,
    pub qkv_us: u128,
    pub qkv_tflops: f64,
    pub rope_us: u128,
    pub kv_store_us: u128,
    pub attention_us: u128,
    pub o_proj_us: u128,
    pub mlp_us: u128,
    pub mlp_tflops: f64,
    pub layer_total_us: u128,
    pub sample_us: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChatMessage {
    pub role: String,
    /// Plain text content. May be empty when the assistant turn carries only
    /// `tool_calls`, or when a tool turn uses content-parts.
    pub content: String,
    /// Tool calls emitted by an assistant turn (OpenAI format). Empty for
    /// human/system/tool turns and pure-text assistant turns.
    pub tool_calls: Vec<ToolCall>,
    /// For role="tool": which assistant tool_call this is responding to.
    pub tool_call_id: Option<String>,
    /// For role="tool": the function name (some clients pass this in addition
    /// to tool_call_id; chat templates use it as a fallback).
    pub name: Option<String>,
    /// Reasoning / chain-of-thought content. When set, the chat template
    /// emits this in the appropriate thinking-channel for the model.
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    /// OpenAI tool type — currently always "function".
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallFunction {
    pub name: String,
    /// Per OpenAI spec, arguments is a JSON-encoded string (clients are
    /// responsible for parsing). Templates need it as either a string or a
    /// dict — we render it raw and let the template `is mapping` branch
    /// when callers pre-parse.
    pub arguments: String,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        }
    }
}
