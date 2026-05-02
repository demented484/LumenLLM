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
        };
        assert_eq!(sample_next_token(&[0.0, 3.0, 1.0], &sampling).unwrap(), 1);
    }
}
