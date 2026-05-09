use std::fs;
use std::sync::Arc;

use minijinja::{Environment, Value, context};
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
        let bos_token_id = artifact.config.bos_token_id.map(|id| id as usize);
        let bos_token_str = bos_token_id
            .and_then(|id| u32::try_from(id).ok())
            .and_then(|id| tokenizer.id_to_token(id))
            .unwrap_or_default();
        Ok(Self {
            tokenizer,
            bos_token_id,
            eos_token_ids: extract_eos_token_ids(artifact),
            chat_template: ChatTemplate::from_artifact(artifact, bos_token_str)?,
        })
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<usize>> {
        self.encode_with_options(prompt, true)
    }

    /// Encode the input as raw text — bypasses the chat template even for
    /// chat-tuned models. BOS is still prepended. Used by tooling that
    /// needs to score the model's pretrain language-modeling ability
    /// (e.g. perplexity), where the chat wrap would introduce role tokens
    /// that pollute the measurement.
    pub fn encode_text_raw(&self, text: &str) -> Result<Vec<usize>> {
        self.encode_with_options(text, false)
    }

    fn encode_with_options(&self, prompt: &str, apply_chat_template: bool) -> Result<Vec<usize>> {
        let prompt = if apply_chat_template {
            self.chat_template.apply_user_prompt(prompt)?
        } else {
            prompt.to_string()
        };
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
        // Prepend BOS if the tokenizer didn't add one and the model has a BOS token defined.
        // Models like Gemma 4 absolutely require BOS at sequence start; without it the
        // first token's attention context is wrong and decoding collapses to gibberish.
        if let Some(bos) = self.bos_token_id
            && ids.first().copied() != Some(bos)
        {
            ids.insert(0, bos);
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

    /// Render a sequence of chat messages (with optional tool definitions)
    /// using the model's actual chat_template.jinja. Used by the HTTP server
    /// to honor OpenAI/Anthropic chat-completions semantics.
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
        enable_thinking: bool,
    ) -> Result<String> {
        self.chat_template
            .render(messages, tools, enable_thinking, true)
    }

    pub fn render_chat_messages_for_artifact(
        artifact: &ModelArtifact,
        messages: &[ChatMessage],
    ) -> Result<String> {
        // Same fallback path the constructor would take so callers without a
        // built TextProcessor get consistent rendering.
        let bos_token_str = artifact
            .config
            .bos_token_id
            .map(|id| id as usize)
            .and_then(|id| {
                u32::try_from(id).ok().and_then(|id| {
                    let path = artifact.root.join("tokenizer.json");
                    Tokenizer::from_file(&path)
                        .ok()
                        .and_then(|t| t.id_to_token(id))
                })
            })
            .unwrap_or_default();
        ChatTemplate::from_artifact(artifact, bos_token_str)?.render(messages, None, false, true)
    }
}

/// Real Jinja2 chat template renderer. Loads the model's
/// `chat_template.jinja` (or `tokenizer_config.json#chat_template`) and uses
/// `minijinja` to render with messages, tools, and BOS/generation flags as
/// HuggingFace transformers does.
#[derive(Debug, Clone)]
struct ChatTemplate {
    inner: Arc<ChatTemplateInner>,
}

#[derive(Debug)]
struct ChatTemplateInner {
    template: String,
    bos_token: String,
    /// `true` when no model-supplied template was found. We then synthesize a
    /// minimal user-only prompt so callers that only care about pretraining-
    /// style continuation still work.
    is_fallback: bool,
}

impl ChatTemplate {
    fn from_artifact(artifact: &ModelArtifact, bos_token: String) -> Result<Self> {
        let template_path = artifact.root.join("chat_template.jinja");
        let template = fs::read_to_string(&template_path).ok().or_else(|| {
            artifact
                .tokenizer_config
                .as_ref()
                .and_then(|cfg| cfg.chat_template.clone())
        });
        let (template, is_fallback) = match template {
            Some(t) if !t.is_empty() => (t, false),
            _ => (String::new(), true),
        };
        Ok(Self {
            inner: Arc::new(ChatTemplateInner {
                template,
                bos_token,
                is_fallback,
            }),
        })
    }

    fn apply_user_prompt(&self, prompt: &str) -> Result<String> {
        if self.inner.is_fallback {
            return Ok(prompt.to_string());
        }
        // If the prompt is already wrapped in chat-template markup (caller
        // pre-rendered), pass it through as-is. We detect this by looking
        // for any of the well-known turn/header tokens.
        if looks_preformatted(prompt) {
            return Ok(prompt.to_string());
        }
        self.render(
            &[ChatMessage {
                role: "user".into(),
                content: prompt.trim().to_string(),
            }],
            None,
            false,
            true,
        )
    }

    fn render(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
        enable_thinking: bool,
        add_generation_prompt: bool,
    ) -> Result<String> {
        if self.inner.is_fallback {
            return self.fallback_render(messages, add_generation_prompt);
        }
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        // HuggingFace chat templates routinely call Python-style methods on
        // dicts and strings (.get(), .split(), .strip(), .upper()) that
        // minijinja doesn't ship by default.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        env.add_template("chat", &self.inner.template).map_err(|e| {
            AegisError::Unsupported(format!("chat template parse failed: {e}"))
        })?;
        let tmpl = env.get_template("chat").map_err(|e| {
            AegisError::Unsupported(format!("chat template lookup failed: {e}"))
        })?;
        let messages_value = messages_to_jinja(messages)?;
        let tools_value = tools
            .map(Value::from_serialize)
            .unwrap_or_else(|| Value::from(()));
        let ctx = context! {
            messages => messages_value,
            tools => tools_value,
            bos_token => &self.inner.bos_token,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
        };
        tmpl.render(ctx).map_err(|e| {
            AegisError::Unsupported(format!("chat template render failed: {e}"))
        })
    }

    fn fallback_render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        // No chat template: stitch role-prefixed lines so debugging output
        // remains readable. Real models always have a template; this is just
        // for raw / pretrained checkpoints used in tests.
        let mut out = String::new();
        for m in messages {
            out.push_str(&m.role);
            out.push_str(": ");
            out.push_str(m.content.trim());
            out.push('\n');
        }
        if add_generation_prompt {
            out.push_str("assistant: ");
        }
        Ok(out)
    }
}

/// Convert ChatMessage values to a minijinja-friendly representation. Today's
/// ChatMessage is `{role, content}` only; richer fields (tool_calls,
/// reasoning, content-parts) get layered on top of this conversion as the
/// struct grows.
fn messages_to_jinja(messages: &[ChatMessage]) -> Result<Value> {
    let mut items: Vec<Value> = Vec::with_capacity(messages.len());
    for m in messages {
        items.push(context! {
            role => &m.role,
            content => &m.content,
        });
    }
    Ok(Value::from(items))
}

fn looks_preformatted(prompt: &str) -> bool {
    [
        "<|start_header_id|>",
        "<|begin_of_text|>",
        "<|eot_id|>",
        "<|turn>",
        "<turn|>",
        "<|im_start|>",
        "<|im_end|>",
    ]
    .iter()
    .any(|tok| prompt.contains(tok))
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

    fn build_template(template: &str) -> ChatTemplate {
        ChatTemplate {
            inner: Arc::new(ChatTemplateInner {
                template: template.to_string(),
                bos_token: "<bos>".into(),
                is_fallback: false,
            }),
        }
    }

    #[test]
    fn jinja_renders_simple_user_prompt() {
        let tmpl = build_template(
            "{{ bos_token }}{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}\
             {% if add_generation_prompt %}<|assistant|>{% endif %}",
        );
        let out = tmpl.apply_user_prompt("Hello").unwrap();
        assert_eq!(out, "<bos><|user|>Hello<|assistant|>");
    }

    #[test]
    fn jinja_passes_through_preformatted_prompt() {
        let tmpl = build_template("{{ bos_token }}{{ messages[0].content }}");
        let pre = "<|turn>user\nalready wrapped<turn|>";
        assert_eq!(tmpl.apply_user_prompt(pre).unwrap(), pre);
    }

    #[test]
    fn jinja_renders_message_list() {
        let tmpl = build_template(
            "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}\
             {% if add_generation_prompt %}go{% endif %}",
        );
        let out = tmpl
            .render(
                &[
                    ChatMessage {
                        role: "system".into(),
                        content: "be brief".into(),
                    },
                    ChatMessage {
                        role: "user".into(),
                        content: "Hi".into(),
                    },
                ],
                None,
                false,
                true,
            )
            .unwrap();
        assert_eq!(out, "system:be brief;user:Hi;go");
    }

    #[test]
    fn fallback_renders_when_no_template() {
        let tmpl = ChatTemplate {
            inner: Arc::new(ChatTemplateInner {
                template: String::new(),
                bos_token: String::new(),
                is_fallback: true,
            }),
        };
        let out = tmpl
            .render(
                &[ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                }],
                None,
                false,
                true,
            )
            .unwrap();
        assert_eq!(out, "user: Hi\nassistant: ");
    }
}
