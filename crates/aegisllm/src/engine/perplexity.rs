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
    /// If `Some(n)`, truncate the tokenized sequence to `n` tokens before
    /// scoring. Useful for fast iteration during quant tuning.
    pub max_tokens: Option<usize>,
    /// Number of leading tokens to feed through the model as context
    /// without scoring. The first ~16 tokens after BOS have very low
    /// predictability simply because the model has no prior context, and
    /// scoring them dominates the average. Standard llama.cpp practice.
    /// Defaults to 16 if `None`.
    pub context_tokens: Option<usize>,
    /// When `true`, bypass the model's chat template and score raw-text
    /// language modeling (pretrain ability). Defaults to `true` because
    /// PPL with the chat wrap on a chat-tuned model measures "predict
    /// continuation of user-role text", which is artificially hard.
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
    let mut tokens = if request.raw_text {
        backend.encode_text_raw(text)?
    } else {
        backend.encode_prompt(text)?
    };
    if let Some(limit) = request.max_tokens {
        tokens.truncate(limit);
    }
    if tokens.len() < 2 {
        return Err(AegisError::InvalidPlan(format!(
            "perplexity needs at least 2 tokens; got {} after truncation",
            tokens.len(),
        )));
    }

    let context_tokens = request.context_tokens.unwrap_or(16).min(tokens.len() - 1);
    let mut state = backend.new_sequence_state()?;
    let mut total_logp: f64 = 0.0;
    let mut count: usize = 0;
    for i in 0..tokens.len() - 1 {
        let logits = backend.forward_logits(state.as_mut(), tokens[i])?;
        if i + 1 < context_tokens {
            // Warm-up window: the first ~N positions have artificially low
            // log-prob because the model has no preceding context. Feed
            // through to advance state, but don't score.
            continue;
        }
        let logp = log_softmax_at(&logits, tokens[i + 1])?;
        total_logp += logp as f64;
        count += 1;
    }
    if count == 0 {
        return Err(AegisError::InvalidPlan(
            "perplexity: nothing scored — try smaller --context-tokens or longer text".into(),
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
