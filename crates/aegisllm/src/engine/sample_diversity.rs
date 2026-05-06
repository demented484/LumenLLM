use std::collections::BTreeMap;

use super::{AegisEngine, EngineConfig};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::{GenerateRequest, SamplingConfig};

/// Diagnostic for token-probability sanity: run the same prompt N times
/// with the same sampling config (non-greedy expected) and report the
/// distribution of generated continuations.
///
/// Healthy model under temp=1.0 + top-k=50 + min-p=0.05: a factual prompt
/// like "What is the capital of France?" should hit "Paris" >90% of the
/// time but still show variation in *how* it answers (continuation
/// phrasing). Pathological cases:
/// - Same exact output every run → probabilities collapsed (one token
///   dominating); could indicate sampler bug or extreme logit divergence.
/// - First-token chaos (no clear majority) → the model's logit
///   distribution is too flat → numerical degradation in forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleDiversityRequest {
    pub prompt: String,
    pub runs: usize,
    pub max_tokens: usize,
    /// If `Some`, override the default sampling config (from parameters).
    pub sampling: Option<SamplingConfig>,
}

#[derive(Debug, Clone)]
pub struct SampleDiversityResult {
    pub runs: usize,
    pub sampling: SamplingConfig,
    pub completions: Vec<String>,
    /// `(unique_completion, count)` sorted by count desc.
    pub distribution: Vec<(String, usize)>,
    /// `(first_token_text, count)` sorted by count desc — useful when full
    /// completions are too noisy to compare directly.
    pub first_token_distribution: Vec<(String, usize)>,
}

pub fn run_sample_diversity(
    config: EngineConfig,
    request: SampleDiversityRequest,
) -> Result<SampleDiversityResult> {
    if request.runs == 0 {
        return Err(AegisError::InvalidConfig(
            "sample-diversity: --runs must be >= 1".into(),
        ));
    }
    let engine = AegisEngine::build(config)?;
    let mut completions = Vec::with_capacity(request.runs);
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut first_token_counts: BTreeMap<String, usize> = BTreeMap::new();

    let sampling = request
        .sampling
        .unwrap_or_else(|| engine_default_sampling(&engine));

    for _ in 0..request.runs {
        let output = engine.generate(GenerateRequest {
            prompt: request.prompt.clone(),
            max_tokens: request.max_tokens,
            sampling,
        })?;
        // First "word" of the output is usually a stable proxy for the
        // model's first sampled token (greedy-tokenizers concatenate
        // word-piece sub-tokens, so we trim and take the first whitespace
        // chunk).
        let first_token: String = output
            .text
            .trim_start()
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
        *first_token_counts.entry(first_token).or_insert(0) += 1;
        *counts.entry(output.text.clone()).or_insert(0) += 1;
        completions.push(output.text);
    }

    let mut distribution: Vec<(String, usize)> = counts.into_iter().collect();
    distribution.sort_by(|a, b| b.1.cmp(&a.1));
    let mut first_token_distribution: Vec<(String, usize)> =
        first_token_counts.into_iter().collect();
    first_token_distribution.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(SampleDiversityResult {
        runs: request.runs,
        sampling,
        completions,
        distribution,
        first_token_distribution,
    })
}

fn engine_default_sampling(_engine: &AegisEngine) -> SamplingConfig {
    // Engine itself doesn't store the parsed sampling config (it's
    // request-scoped). We fall back to library defaults; callers should
    // override via the CLI flags when using non-greedy sampling.
    SamplingConfig::default()
}
