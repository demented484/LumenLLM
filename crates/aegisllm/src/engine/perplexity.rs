use super::{AegisEngine, EngineConfig};
use aegisllm_base::error::{AegisError, Result};

/// Built-in calibration text. Public domain (Pride and Prejudice, opening
/// paragraph). Deliberately short — long enough for a meaningful PPL number
/// but tokenizes to ~80 tokens so a `forward_logits` loop completes in a
/// few seconds even at 30 tok/s decode.
pub const DEFAULT_PPL_TEXT: &str = "It is a truth universally acknowledged, \
    that a single man in possession of a good fortune, must be in want of a \
    wife. However little known the feelings or views of such a man may be \
    on his first entering a neighbourhood, this truth is so well fixed in \
    the minds of the surrounding families, that he is considered the \
    rightful property of some one or other of their daughters.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerplexityRequest {
    /// If `None`, use [`DEFAULT_PPL_TEXT`].
    pub text: Option<String>,
    /// If `Some(n)`, truncate the scored text to `n` tokens. Useful for
    /// fast iteration during quant tuning.
    pub max_tokens: Option<usize>,
    /// Ignored. `compute_perplexity` always frames the text as an
    /// assistant turn and uses the whole chat frame as the warm-up window.
    /// Retained so the CLI surface does not change.
    pub context_tokens: Option<usize>,
    /// Ignored. The text is always scored as assistant-turn content (the
    /// in-distribution task for instruction-tuned models); raw-text scoring
    /// produced meaningless PPL on `-it` models. Retained for CLI stability.
    pub raw_text: bool,
}

#[derive(Debug, Clone)]
pub struct PerplexityResult {
    pub num_tokens_scored: usize,
    pub mean_neg_log_prob: f64,
    pub perplexity: f64,
}

/// Compute perplexity via teacher forcing: for each position `i` from `0`
/// to `len-2`, feed `tokens[i]` through the model and score the actual
/// `tokens[i+1]` against the resulting next-position logits.
///
/// PPL = exp(mean(-log P(t_{i+1} | t_0..t_i))).
pub fn compute_perplexity(
    config: EngineConfig,
    request: PerplexityRequest,
) -> Result<PerplexityResult> {
    let engine = AegisEngine::build(config)?;
    let executor = engine.executor().ok_or_else(|| {
        AegisError::Unsupported("engine was built without executor".into())
    })?;
    let backend = executor.as_primitives();

    let owned;
    let text: &str = match request.text.as_deref() {
        Some(s) => s,
        None => {
            owned = DEFAULT_PPL_TEXT.to_string();
            owned.as_str()
        }
    };
    // Score the text as ASSISTANT-TURN content:
    //   <bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{TEXT}
    //
    // aegisllm's target models are all heavily instruction-tuned. Such models
    // are trained exclusively on the chat format; scoring them on RAW text
    // (no chat frame) measures an out-of-distribution task and produces
    // meaningless perplexity — verified on Gemma-4-26B: raw-text PPL ~5000+
    // vs ~1.1 with this assistant-turn frame, on text the model reproduces
    // verbatim. The frame tokens establish the chat context (they are
    // prefilled) but never count toward PPL; only the {TEXT} tokens are
    // scored.
    let frame = backend.encode_prompt("Write out a passage of classic English prose.")?;
    let frame_len = frame.len();
    let mut text_tokens = backend.encode_text_raw(text)?;
    // `encode_text_raw` prepends its own BOS; the frame already opens with one.
    if text_tokens.first().copied() == frame.first().copied() {
        text_tokens.remove(0);
    }
    if let Some(limit) = request.max_tokens {
        text_tokens.truncate(limit);
    }
    if text_tokens.len() < 2 || frame_len < 2 {
        return Err(AegisError::InvalidPlan(format!(
            "perplexity needs ≥2 text tokens and a non-trivial frame; got \
             text={} frame={}",
            text_tokens.len(),
            frame_len,
        )));
    }
    let mut tokens = frame;
    tokens.extend_from_slice(&text_tokens);

    let ppl_debug = std::env::var("AEGIS_PPL_DEBUG").is_ok();
    if ppl_debug {
        eprintln!(
            "[PPL] frame_len={frame_len} text_tokens={} total={}",
            text_tokens.len(),
            tokens.len(),
        );
    }

    let greedy = aegisllm_base::generation::SamplingConfig {
        temperature: 0.0, top_k: 1, top_p: 1.0, min_p: 0.0,
    };
    let mut state = backend.new_sequence_state()?;
    // Prefill positions `0..frame_len-1` (all frame tokens but the last) via
    // the PREFILL path, then teacher-force the rest: feed `tokens[i]`, score
    // `tokens[i+1]`. The first decode step feeds the last frame token, so the
    // first scored target is `tokens[frame_len]` — the first text token —
    // and every text token contributes to PPL.
    let prefill_len = frame_len - 1;
    backend.prefill_prompt(state.as_mut(), &tokens[..prefill_len], &greedy)?;

    let mut total_logp: f64 = 0.0;
    let mut count: usize = 0;
    for i in prefill_len..tokens.len() - 1 {
        let logits = backend.forward_logits(state.as_mut(), tokens[i])?;
        let logp = log_softmax_at(&logits, tokens[i + 1])?;
        if ppl_debug {
            let (argmax, _) = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            let dec = |t: usize| backend.decode_tokens(&[t]).unwrap_or_default();
            eprintln!(
                "[PPL] i={i:>3} target={:>6}({:?}) argmax={:>6}({:?}) logp={:.3}",
                tokens[i + 1], dec(tokens[i + 1]),
                argmax, dec(argmax),
                logp,
            );
        }
        total_logp += logp as f64;
        count += 1;
    }
    if count == 0 {
        return Err(AegisError::InvalidPlan(
            "perplexity: nothing scored — text too short".into(),
        ));
    }

    let mean_neg = -total_logp / count as f64;
    Ok(PerplexityResult {
        num_tokens_scored: count,
        mean_neg_log_prob: mean_neg,
        perplexity: mean_neg.exp(),
    })
}

fn log_softmax_at(logits: &[f32], idx: usize) -> Result<f32> {
    if idx >= logits.len() {
        return Err(AegisError::InvalidPlan(format!(
            "PPL target token id {idx} >= vocab size {}",
            logits.len()
        )));
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits
        .iter()
        .copied()
        .map(|x| (x - max_logit).exp())
        .sum();
    let log_sum_exp = max_logit + sum_exp.ln();
    Ok(logits[idx] - log_sum_exp)
}
