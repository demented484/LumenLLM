use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}
