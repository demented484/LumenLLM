//! Output parser for chat-tuned model assistant turns.
//!
//! Different model families emit reasoning and tool calls with different
//! special-token wrappers. This module recognises the formats used by the
//! target architectures (Gemma 4 today; Qwen 3.5/3.6 and Nemotron Nano can
//! be added by extending [`ParserKind::detect`]).
//!
//! The parser is conservative: it only consumes tokens it recognises, and
//! everything else falls through to `text`. So callers can run it
//! unconditionally and get sensible output even on unknown formats.

use crate::generation::{ToolCall, ToolCallFunction};

/// Result of splitting an assistant turn's raw decoded text into parts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedAssistant {
    /// User-visible text after stripping reasoning channels and tool-call
    /// blocks. Whitespace-trimmed.
    pub content: String,
    /// Concatenated content of any thinking/reasoning channels. None if the
    /// model didn't emit any.
    pub reasoning: Option<String>,
    /// Tool calls extracted from the output, in emission order.
    pub tool_calls: Vec<ToolCall>,
}

/// Which model-family parser to apply. Detect from the rendered
/// chat_template text or the artifact's model_type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserKind {
    /// `<|channel>thought\n…<channel|>` for reasoning,
    /// `<|tool_call>call:name{key:val,…}<tool_call|>` for tool calls.
    Gemma,
    /// Pass-through: the whole output is `content`. Used when the model
    /// either has no special markers or its chat template doesn't expose
    /// any. Tool calls will not be detected.
    None,
}

impl ParserKind {
    /// Detect the parser kind by scanning the model's chat_template source
    /// for tokens it emits in the assistant turn. Falls back to `None`.
    pub fn detect(chat_template: &str) -> Self {
        if chat_template.contains("<|tool_call>") || chat_template.contains("<|channel>") {
            Self::Gemma
        } else {
            Self::None
        }
    }

    pub fn parse_assistant(self, raw: &str) -> ParsedAssistant {
        match self {
            Self::Gemma => parse_gemma(raw),
            Self::None => ParsedAssistant {
                content: raw.trim().to_string(),
                ..Default::default()
            },
        }
    }
}

fn parse_gemma(raw: &str) -> ParsedAssistant {
    // 1. Extract and strip thinking channels: `<|channel>thought\n…\n<channel|>`.
    //    Concatenate all thought-channel bodies into a single reasoning string.
    let mut content_buf = String::with_capacity(raw.len());
    let mut reasoning_buf = String::new();
    let mut cursor = raw;
    while let Some(open_idx) = cursor.find("<|channel>") {
        content_buf.push_str(&cursor[..open_idx]);
        let after_open = &cursor[open_idx + "<|channel>".len()..];
        // The channel body starts after the channel name + newline. The
        // typical pattern is `<|channel>thought\n…\n<channel|>` but to be
        // safe we tolerate any name and any leading whitespace before the
        // body.
        let body_start = after_open
            .find('\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        let body_window = &after_open[body_start..];
        if let Some(close_rel) = body_window.find("<channel|>") {
            let body = body_window[..close_rel].trim();
            if !body.is_empty() {
                if !reasoning_buf.is_empty() {
                    reasoning_buf.push('\n');
                }
                reasoning_buf.push_str(body);
            }
            cursor = &body_window[close_rel + "<channel|>".len()..];
        } else {
            // Unterminated channel — model was cut off mid-thought. Treat
            // the rest as reasoning.
            let body = body_window.trim();
            if !body.is_empty() {
                if !reasoning_buf.is_empty() {
                    reasoning_buf.push('\n');
                }
                reasoning_buf.push_str(body);
            }
            cursor = "";
            break;
        }
    }
    content_buf.push_str(cursor);

    // 2. Extract tool calls from whatever remains: `<|tool_call>call:NAME{…}<tool_call|>`.
    let mut tool_calls = Vec::new();
    let mut clean_content = String::with_capacity(content_buf.len());
    let mut cursor: &str = &content_buf;
    let mut tc_idx = 0usize;
    while let Some(open_idx) = cursor.find("<|tool_call>") {
        clean_content.push_str(&cursor[..open_idx]);
        let after_open = &cursor[open_idx + "<|tool_call>".len()..];
        if let Some(close_rel) = after_open.find("<tool_call|>") {
            let body = &after_open[..close_rel];
            if let Some(tc) = parse_gemma_tool_call(body, tc_idx) {
                tool_calls.push(tc);
                tc_idx += 1;
            }
            cursor = &after_open[close_rel + "<tool_call|>".len()..];
        } else {
            // Unterminated tool call — drop it; caller will see the raw
            // tail in content if they want to debug.
            cursor = after_open;
            break;
        }
    }
    clean_content.push_str(cursor);

    // 3. Final cleanup. Models will sometimes overshoot a tool_call by
    //    role-playing the tool response (`<|tool_response>...<tool_response|>`)
    //    or starting a new turn (`<|turn>...`). Those are protocol bugs from
    //    the model's perspective: real tool execution happens server-side
    //    after this generation completes. Drop everything from the first
    //    such structural marker onward when we already extracted a tool_call,
    //    so the user-visible `content` doesn't carry fake tool output.
    let mut final_content = clean_content;
    if !tool_calls.is_empty() {
        for marker in ["<|tool_response>", "<|turn>", "<|tool_call>"] {
            if let Some(idx) = final_content.find(marker) {
                final_content.truncate(idx);
            }
        }
    }
    // Even when no tool_call was emitted, strip stray structural markers
    // that the tokenizer didn't already (BOS/EOS handled at decode time).
    for marker in [
        "<|tool_response>",
        "<tool_response|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "<|tool>",
        "<tool|>",
    ] {
        final_content = final_content.replace(marker, "");
    }

    ParsedAssistant {
        content: final_content.trim().to_string(),
        reasoning: if reasoning_buf.is_empty() {
            None
        } else {
            Some(reasoning_buf)
        },
        tool_calls,
    }
}

/// Parse the body of a single `<|tool_call>...<tool_call|>` block.
/// Body looks like: `call:func_name{key:value,key2:value2}` (or with
/// `<|"|>...<|"|>` quoted strings inside values).
fn parse_gemma_tool_call(body: &str, index: usize) -> Option<ToolCall> {
    let body = body.trim();
    let after_call = body.strip_prefix("call:")?;
    // Function name runs until the first `{`.
    let brace = after_call.find('{')?;
    let name = after_call[..brace].trim().to_string();
    let args_body = &after_call[brace + 1..];
    // Strip the matching trailing `}`. Tolerate whitespace.
    let args_body = args_body.trim_end();
    let args_body = args_body.strip_suffix('}').unwrap_or(args_body);
    let arguments_json = gemma_args_to_json(args_body);
    Some(ToolCall {
        id: format!("call_{index}"),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name,
            arguments: arguments_json,
        },
    })
}

/// Convert Gemma's argument DSL to a JSON-encoded string.
/// Input: `key1:<|"|>val<|"|>,key2:42`
/// Output: `{"key1":"val","key2":42}`
fn gemma_args_to_json(input: &str) -> String {
    // The DSL is close enough to JSON that a small character-level rewriter
    // works for the common cases (string values quoted via `<|"|>`,
    // numeric/bool values raw, top-level keys bare). Nested objects/arrays
    // also follow the same rules — the rewrite is uniform.
    let mut out = String::with_capacity(input.len() + 2);
    out.push('{');
    let mut chars = input.chars().peekable();
    let mut state = ArgState::Key;
    let mut depth: i32 = 0;
    while let Some(c) = chars.next() {
        match (state, c) {
            (ArgState::Key, ':') => {
                out.push(':');
                state = ArgState::Value;
            }
            (ArgState::Key, ',') if depth == 0 => {
                out.push(',');
                state = ArgState::Key;
            }
            (ArgState::Key, _) => {
                if needs_quote_key(out.chars().last()) {
                    out.push('"');
                    out.push(c);
                    // Read the rest of the key.
                    while let Some(&nc) = chars.peek() {
                        if nc == ':' || nc == ',' || nc == '}' || nc == ']' {
                            break;
                        }
                        out.push(nc);
                        chars.next();
                    }
                    out.push('"');
                } else {
                    out.push(c);
                }
            }
            (ArgState::Value, ',') if depth == 0 => {
                out.push(',');
                state = ArgState::Key;
            }
            (ArgState::Value, '{') | (ArgState::Value, '[') => {
                depth += 1;
                out.push(c);
            }
            (ArgState::Value, '}') | (ArgState::Value, ']') => {
                depth = depth.saturating_sub(1);
                out.push(c);
            }
            (ArgState::Value, '<') => {
                // Detect `<|"|>` opening quote.
                if try_consume(&mut chars, "|\"|>") {
                    out.push('"');
                    // Read until matching `<|"|>` close.
                    let mut s = String::new();
                    let mut closed = false;
                    while let Some(c2) = chars.next() {
                        if c2 == '<' && peek_match(&mut chars, "|\"|>") {
                            for _ in 0.."|\"|>".len() {
                                chars.next();
                            }
                            closed = true;
                            break;
                        }
                        s.push(c2);
                    }
                    out.push_str(&json_escape(&s));
                    out.push('"');
                    if !closed {
                        // unterminated — bail
                        break;
                    }
                } else {
                    out.push(c);
                }
            }
            (ArgState::Value, _) => {
                out.push(c);
            }
        }
    }
    out.push('}');
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArgState {
    Key,
    Value,
}

fn needs_quote_key(prev: Option<char>) -> bool {
    matches!(prev, Some('{') | Some(',') | None)
}

fn try_consume(it: &mut std::iter::Peekable<std::str::Chars<'_>>, pat: &str) -> bool {
    let snapshot: String = it.clone().take(pat.len()).collect();
    if snapshot == pat {
        for _ in 0..pat.len() {
            it.next();
        }
        true
    } else {
        false
    }
}

fn peek_match(it: &mut std::iter::Peekable<std::str::Chars<'_>>, pat: &str) -> bool {
    let snapshot: String = it.clone().take(pat.len()).collect();
    snapshot == pat
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_no_markers() {
        let p = ParserKind::Gemma.parse_assistant("hello world");
        assert_eq!(p.content, "hello world");
        assert!(p.reasoning.is_none());
        assert!(p.tool_calls.is_empty());
    }

    #[test]
    fn extracts_thought_channel() {
        let raw = "<|channel>thought\nlet me think hard\n<channel|>final answer";
        let p = ParserKind::Gemma.parse_assistant(raw);
        assert_eq!(p.content, "final answer");
        assert_eq!(p.reasoning.as_deref(), Some("let me think hard"));
        assert!(p.tool_calls.is_empty());
    }

    #[test]
    fn extracts_tool_call_with_string_args() {
        let raw =
            "I'll check the weather. <|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>,units:<|\"|>celsius<|\"|>}<tool_call|>";
        let p = ParserKind::Gemma.parse_assistant(raw);
        assert_eq!(p.content, "I'll check the weather.");
        assert_eq!(p.tool_calls.len(), 1);
        let tc = &p.tool_calls[0];
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(
            tc.function.arguments,
            r#"{"city":"Tokyo","units":"celsius"}"#
        );
    }

    #[test]
    fn extracts_tool_call_with_numeric_args() {
        let raw = "<|tool_call>call:add{a:1,b:2}<tool_call|>";
        let p = ParserKind::Gemma.parse_assistant(raw);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].function.arguments, r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn handles_multiple_tool_calls() {
        let raw =
            "<|tool_call>call:f{a:1}<tool_call|><|tool_call>call:g{b:<|\"|>x<|\"|>}<tool_call|>";
        let p = ParserKind::Gemma.parse_assistant(raw);
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(p.tool_calls[0].id, "call_0");
        assert_eq!(p.tool_calls[1].id, "call_1");
        assert_eq!(p.tool_calls[1].function.arguments, r#"{"b":"x"}"#);
    }

    #[test]
    fn detect_chooses_gemma_when_markers_present() {
        let template = "...{{- '<|tool_call>call:' + name + '{' -}}...";
        assert_eq!(ParserKind::detect(template), ParserKind::Gemma);
        assert_eq!(ParserKind::detect("{{ messages }}"), ParserKind::None);
    }
}
