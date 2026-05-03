use std::fs;

use tokenizers::Tokenizer;

use crate::artifact::ModelArtifact;
use crate::error::{AegisError, Result};
use crate::generation::ChatMessage;

#[derive(Debug)]
pub struct TextProcessor {
    tokenizer: Tokenizer,
    bos_token_id: Option<usize>,
    eos_token_ids: Vec<usize>,
    chat_template: ChatTemplate,
}

impl TextProcessor {
    pub fn from_artifact(artifact: &ModelArtifact) -> Result<Self> {
        let tokenizer_path = artifact.root.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            AegisError::Unsupported(format!("failed to load tokenizer: {error}"))
        })?;
        Ok(Self {
            tokenizer,
            bos_token_id: artifact.config.bos_token_id.map(|id| id as usize),
            eos_token_ids: extract_eos_token_ids(artifact),
            chat_template: ChatTemplate::from_artifact(artifact),
        })
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<usize>> {
        let prompt = self.chat_template.apply(prompt);
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|error| {
                AegisError::Unsupported(format!("tokenizer encode failed: {error}"))
            })?;
        let mut ids = encoding
            .get_ids()
            .iter()
            .map(|id| *id as usize)
            .collect::<Vec<_>>();
        if ids.is_empty()
            && let Some(bos) = self.bos_token_id
        {
            ids.push(bos);
        }
        Ok(ids)
    }

    pub fn decode_tokens(&self, token_ids: &[usize]) -> Result<String> {
        let ids = token_ids
            .iter()
            .map(|id| {
                u32::try_from(*id).map_err(|_| AegisError::Unsupported("token id overflow".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        self.tokenizer
            .decode(&ids, true)
            .map_err(|error| AegisError::Unsupported(format!("tokenizer decode failed: {error}")))
    }

    pub fn is_eos(&self, token_id: usize) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    pub fn render_chat_messages_for_artifact(
        artifact: &ModelArtifact,
        messages: &[ChatMessage],
    ) -> Result<String> {
        ChatTemplate::from_artifact(artifact).render_messages(messages)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatTemplate {
    None,
    Llama3Instruct,
}

impl ChatTemplate {
    fn from_artifact(artifact: &ModelArtifact) -> Self {
        let chat_template_path = artifact.root.join("chat_template.jinja");
        let template = fs::read_to_string(&chat_template_path).unwrap_or_else(|_| {
            artifact
                .tokenizer_config
                .as_ref()
                .and_then(|config| config.chat_template.clone())
                .unwrap_or_default()
        });
        if template.contains("<|start_header_id|>")
            && template.contains("<|eot_id|>")
            && artifact.config.model_type == "llama"
        {
            Self::Llama3Instruct
        } else {
            Self::None
        }
    }

    fn apply(self, prompt: &str) -> String {
        match self {
            Self::None => prompt.to_string(),
            Self::Llama3Instruct if looks_preformatted_chat(prompt) => prompt.to_string(),
            Self::Llama3Instruct => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                prompt.trim()
            ),
        }
    }

    fn render_messages(self, messages: &[ChatMessage]) -> Result<String> {
        match self {
            Self::None => Err(AegisError::Unsupported(
                "chat completions require a supported model chat template".into(),
            )),
            Self::Llama3Instruct => render_llama3_messages(messages),
        }
    }
}

fn looks_preformatted_chat(prompt: &str) -> bool {
    prompt.contains("<|start_header_id|>")
        || prompt.contains("<|begin_of_text|>")
        || prompt.contains("<|eot_id|>")
}

fn render_llama3_messages(messages: &[ChatMessage]) -> Result<String> {
    let mut prompt = String::from("<|begin_of_text|>");
    for message in messages {
        if !matches!(
            message.role.as_str(),
            "system" | "user" | "assistant" | "tool"
        ) {
            return Err(AegisError::InvalidConfig(format!(
                "unsupported chat message role `{}`",
                message.role
            )));
        }
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(&message.role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(message.content.trim());
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    Ok(prompt)
}

fn extract_eos_token_ids(artifact: &ModelArtifact) -> Vec<usize> {
    let mut ids = Vec::new();
    match artifact.config.eos_token_id.as_ref() {
        Some(serde_json::Value::Number(value)) => {
            if let Some(id) = value.as_u64() {
                ids.push(id as usize);
            }
        }
        Some(serde_json::Value::Array(values)) => {
            ids.extend(
                values
                    .iter()
                    .filter_map(|value| value.as_u64().map(|id| id as usize)),
            );
        }
        _ => {}
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_template_wraps_plain_user_prompt() {
        let rendered = ChatTemplate::Llama3Instruct.apply("Hello");
        assert!(rendered.starts_with("<|begin_of_text|><|start_header_id|>user"));
        assert!(rendered.contains("\n\nHello<|eot_id|>"));
        assert!(rendered.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn llama3_template_preserves_preformatted_prompt() {
        let prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello";
        assert_eq!(ChatTemplate::Llama3Instruct.apply(prompt), prompt);
    }

    #[test]
    fn llama3_template_renders_structured_messages() {
        let rendered = ChatTemplate::Llama3Instruct
            .render_messages(&[
                ChatMessage {
                    role: "system".into(),
                    content: "Be brief.".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Hello".into(),
                },
            ])
            .unwrap();
        assert!(rendered.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(rendered.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(rendered.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }
}
