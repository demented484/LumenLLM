use std::ops::ControlFlow;
use std::time::Instant;

use crate::error::{AegisError, Result};
use crate::generation::{GenerateOutput, GenerateRequest, SamplingConfig, TimedGenerateOutput};

use super::traits::{GenerationBackendPrimitives, GenerationState};

pub fn generate_with_backend<B: GenerationBackendPrimitives + ?Sized>(
    backend: &B,
    request: &GenerateRequest,
) -> Result<GenerateOutput> {
    Ok(generate_with_backend_timed(backend, request)?.output)
}

pub fn generate_with_backend_timed<B: GenerationBackendPrimitives + ?Sized>(
    backend: &B,
    request: &GenerateRequest,
) -> Result<TimedGenerateOutput> {
    let total_start = Instant::now();
    let tokenize_start = Instant::now();
    let prompt_tokens = backend.encode_prompt(&request.prompt)?;
    let tokenize_elapsed = tokenize_start.elapsed();
    if prompt_tokens.is_empty() {
        return Err(AegisError::InvalidConfig(
            "prompt produced no tokens".into(),
        ));
    }

    let mut state = backend.new_sequence_state()?;
    // Stage I.2 multimodal: attach image embeddings to the state so the
    // prefill embed step can splice them at the image-token placeholder
    // positions. No-op for text-only backends / requests.
    if let Some(ref injection) = request.image_injection {
        backend.set_image_injection(state.as_mut(), injection)?;
    }
    let prefill_start = Instant::now();
    let mut next = backend.prefill_prompt(state.as_mut(), &prompt_tokens, &request.sampling)?;
    let prefill_elapsed = prefill_start.elapsed();
    let prefill_stage_timings = backend.prefill_stage_timings(state.as_mut());

    let decode_start = Instant::now();
    let mut generated = Vec::new();
    let mut finish_reason = "length".to_string();
    for _ in 0..request.max_tokens {
        if backend.is_eos(next) {
            finish_reason = "eos_token".into();
            break;
        }
        if request.stop_token_ids.contains(&next) {
            generated.push(next); // include the stop token so parsers see the closing marker
            finish_reason = "stop".into();
            break;
        }
        generated.push(next);
        if generated.len() < request.max_tokens {
            next = backend.forward_next_token(state.as_mut(), next, &request.sampling)?;
        }
    }
    let decode_elapsed = decode_start.elapsed();

    Ok(TimedGenerateOutput {
        output: GenerateOutput {
            text: backend.decode_tokens(&generated)?,
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: generated.len(),
            finish_reason,
        },
        tokenize_elapsed,
        prefill_elapsed,
        decode_elapsed,
        total_elapsed: total_start.elapsed(),
        prefill_stage_timings,
    })
}

pub fn generate_streaming_with_backend<B: GenerationBackendPrimitives + ?Sized>(
    backend: &B,
    request: &GenerateRequest,
    callback: &mut dyn FnMut(usize, &str) -> ControlFlow<()>,
) -> Result<GenerateOutput> {
    let prompt_tokens = backend.encode_prompt(&request.prompt)?;
    if prompt_tokens.is_empty() {
        return Err(AegisError::InvalidConfig(
            "prompt produced no tokens".into(),
        ));
    }
    let mut state = backend.new_sequence_state()?;
    if let Some(ref injection) = request.image_injection {
        backend.set_image_injection(state.as_mut(), injection)?;
    }
    let mut next = backend.prefill_prompt(state.as_mut(), &prompt_tokens, &request.sampling)?;

    let mut generated = Vec::new();
    let mut finish_reason = "length".to_string();
    for _ in 0..request.max_tokens {
        if backend.is_eos(next) {
            finish_reason = "eos_token".into();
            break;
        }
        let is_stop = request.stop_token_ids.contains(&next);
        generated.push(next);
        // Always call the callback for the token — even on a stop token —
        // so streaming clients (and any downstream parser) see the closing
        // marker. Without this, e.g. `<tool_call|>` would push the model
        // into the stop branch and the parser would never observe the
        // close, dropping the entire tool_call block.
        let token_text = backend.decode_tokens(&[next]).unwrap_or_default();
        if callback(next, &token_text).is_break() {
            break;
        }
        if is_stop {
            finish_reason = "stop".into();
            break;
        }
        if generated.len() < request.max_tokens {
            next = backend.forward_next_token(state.as_mut(), next, &request.sampling)?;
        }
    }
    Ok(GenerateOutput {
        text: backend.decode_tokens(&generated)?,
        prompt_tokens: prompt_tokens.len(),
        completion_tokens: generated.len(),
        finish_reason,
    })
}

#[allow(dead_code)]
fn prefill_prompt_logits<B: GenerationBackendPrimitives + ?Sized>(
    backend: &B,
    state: &mut dyn GenerationState,
    prompt_tokens: &[usize],
) -> Result<Vec<f32>> {
    let Some((&last, prefix)) = prompt_tokens.split_last() else {
        return Err(AegisError::InvalidConfig(
            "prompt produced no tokens".into(),
        ));
    };
    for &token in prefix {
        backend.forward_hidden(state, token)?;
    }
    backend.forward_logits(state, last)
}

pub fn prefill_prompt_token_by_token<B: GenerationBackendPrimitives + ?Sized>(
    backend: &B,
    state: &mut dyn GenerationState,
    prompt_tokens: &[usize],
    sampling: &SamplingConfig,
) -> Result<usize> {
    let Some((&last, prefix)) = prompt_tokens.split_last() else {
        return Err(AegisError::InvalidConfig(
            "prompt produced no tokens".into(),
        ));
    };
    for &token in prefix {
        backend.forward_hidden(state, token)?;
    }
    backend.forward_next_token(state, last, sampling)
}

/// In-place logit soft-cap: `logits[i] = cap * tanh(logits[i] / cap)`.
/// Used by Gemma 4 (lm_head output) and inside attention (attn_logit_softcap).
pub fn apply_logit_softcap(logits: &mut [f32], cap: f32) {
    let inv_cap = 1.0 / cap;
    for x in logits.iter_mut() {
        *x = cap * (*x * inv_cap).tanh();
    }
}

pub fn sample_next_token(logits: &[f32], sampling: &SamplingConfig) -> Result<usize> {
    if logits.is_empty() {
        return Err(AegisError::InvalidPlan(
            "cannot sample from empty logits".into(),
        ));
    }
    if sampling.temperature <= 0.0 || sampling.top_k == 1 {
        return Ok(argmax(logits));
    }

    let temperature = sampling.temperature.max(1e-6);
    let mut candidates = logits
        .iter()
        .enumerate()
        .filter_map(|(idx, &logit)| logit.is_finite().then_some((idx, logit)))
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Ok(argmax(logits));
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    if sampling.top_k > 0 && sampling.top_k < candidates.len() {
        candidates.truncate(sampling.top_k);
    }

    let max_logit = candidates
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weighted = candidates
        .into_iter()
        .map(|(idx, logit)| (idx, ((logit - max_logit) / temperature).exp()))
        .filter(|(_, weight)| weight.is_finite() && *weight > 0.0)
        .collect::<Vec<_>>();
    if weighted.is_empty() {
        return Ok(argmax(logits));
    }

    // Filter order matches HF transformers: temperature → top_k → top_p →
    // min_p (min_p LAST). top_k was applied above (pre-exp); top_p then min_p
    // here. `weighted` is sorted desc by logit, so weighted[0] stays the max
    // after top_p's prefix truncation, keeping min_p's `p/p_max` ratio correct.
    let total: f32 = weighted.iter().map(|(_, weight)| *weight).sum();
    if sampling.top_p > 0.0 && sampling.top_p < 1.0 && total > 0.0 {
        let mut cumulative = 0.0_f32;
        let cutoff = total * sampling.top_p;
        let mut keep = 0usize;
        for (_, weight) in &weighted {
            cumulative += *weight;
            keep += 1;
            if cumulative >= cutoff {
                break;
            }
        }
        weighted.truncate(keep.max(1));
    }

    // min_p: keep tokens with `weight >= min_p * weight_max`. Since weights are
    // post-temperature and proportional to probabilities, the ratio equals
    // `p / p_max`. Applied AFTER top_p (HF order).
    if sampling.min_p > 0.0 {
        let max_weight = weighted[0].1;
        let cutoff = max_weight * sampling.min_p;
        weighted.retain(|(_, weight)| *weight >= cutoff);
        if weighted.is_empty() {
            return Ok(argmax(logits));
        }
    }

    let total: f32 = weighted.iter().map(|(_, weight)| *weight).sum();
    if total <= 0.0 {
        return Ok(argmax(logits));
    }
    let mut draw = rand::random::<f32>() * total;
    for (idx, weight) in weighted {
        if draw <= weight {
            return Ok(idx);
        }
        draw -= weight;
    }
    Ok(argmax(logits))
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_zero_sampling_is_argmax() {
        let sampling = SamplingConfig {
            temperature: 0.0,
            top_k: 64,
            top_p: 0.95,
            min_p: 0.0,
        };
        assert_eq!(sample_next_token(&[0.0, 3.0, 1.0], &sampling).unwrap(), 1);
    }
}
