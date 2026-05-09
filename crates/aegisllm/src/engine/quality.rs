use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::{GenerateOutput, GenerateRequest, SamplingConfig};

use super::{AegisEngine, EngineConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualityCase {
    pub name: &'static str,
    pub prompt: &'static str,
    pub max_tokens: usize,
    pub expected_any: &'static [&'static str],
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualitySmokeResult {
    pub case: QualityCase,
    pub output: GenerateOutput,
}

pub fn run_quality_smoke(config: EngineConfig) -> Result<Vec<QualitySmokeResult>> {
    let engine = AegisEngine::build(config)?;
    quality_cases()
        .into_iter()
        .map(|case| {
            let output = engine.generate(GenerateRequest {
                prompt: case.prompt.into(),
                max_tokens: case.max_tokens,
                sampling: SamplingConfig {
                    temperature: 0.0,
                    top_k: 1,
                    top_p: 1.0,
                    min_p: 0.0,
                },
                stop_token_ids: Vec::new(),
            })?;
            validate_quality_output(&case, &output.text)?;
            Ok(QualitySmokeResult { case, output })
        })
        .collect()
}

pub fn quality_cases() -> Vec<QualityCase> {
    vec![
        QualityCase {
            name: "english_hello",
            prompt: "Say hello in one short English sentence.",
            max_tokens: 16,
            expected_any: &["hello", "hi"],
        },
        QualityCase {
            name: "russian_greeting",
            prompt: "Напиши короткое приветствие на русском языке.",
            max_tokens: 24,
            expected_any: &["привет", "здрав", "добро", "добрый"],
        },
    ]
}

pub fn validate_quality_output(case: &QualityCase, text: &str) -> Result<()> {
    let normalized = text.trim().to_lowercase();
    if normalized.is_empty() {
        return Err(AegisError::InvalidPlan(format!(
            "quality-smoke case `{}` produced empty text",
            case.name
        )));
    }
    if has_repeated_piece(&normalized) {
        return Err(AegisError::InvalidPlan(format!(
            "quality-smoke case `{}` produced suspicious repetition: {:?}",
            case.name, text
        )));
    }
    if !case
        .expected_any
        .iter()
        .any(|needle| normalized.contains(needle))
    {
        return Err(AegisError::InvalidPlan(format!(
            "quality-smoke case `{}` expected one of {:?}, got {:?}",
            case.name, case.expected_any, text
        )));
    }
    Ok(())
}

fn has_repeated_piece(text: &str) -> bool {
    let compact = text
        .chars()
        .filter(|ch| !ch.is_whitespace() && !ch.is_ascii_punctuation())
        .collect::<String>();
    if compact.len() < 12 {
        return false;
    }
    for width in 3..=8 {
        let Some(piece) = compact.get(0..width) else {
            continue;
        };
        if compact.starts_with(&piece.repeat(3)) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_gate_rejects_repeated_garbage() {
        let case = QualityCase {
            name: "x",
            prompt: "x",
            max_tokens: 8,
            expected_any: &["hello"],
        };
        let err = validate_quality_output(&case, "aguaaguaaguaagua").unwrap_err();
        assert!(err.to_string().contains("suspicious repetition"));
    }

    #[test]
    fn russian_quality_gate_accepts_common_greetings() {
        let case = QualityCase {
            name: "russian",
            prompt: "x",
            max_tokens: 8,
            expected_any: &["привет", "здрав", "добро", "добрый"],
        };
        validate_quality_output(&case, "Добрый день!").unwrap();
    }
}
