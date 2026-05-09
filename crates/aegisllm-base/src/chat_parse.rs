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

/// Incremental events produced by [`StreamingParser`] as raw decoded
/// chunks arrive from the model. Mapped to API-specific deltas by the
/// HTTP server (OpenAI `delta.content` / `delta.tool_calls`, Anthropic
/// content_block_start/_delta/_stop, …).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// Plain assistant text — append to the user-visible content stream.
    Text(String),
    /// Chain-of-thought / thinking content. Sent between
    /// `<|channel>thought\n` and `<channel|>`. Append to
    /// `reasoning_content` (OpenAI) or a `thinking` block (Anthropic).
    Reasoning(String),
    /// First event for a tool call — gives the function name. `index` is
    /// the position in the response's `tool_calls` array.
    ToolCallBegin { index: usize, id: String, name: String },
    /// JSON-encoded arguments for the tool call. Currently emitted as a
    /// single full delta when the tool_call body finishes — true
    /// per-character streaming of the DSL→JSON rewrite is non-trivial.
    ToolCallArgsDelta { index: usize, partial: String },
    /// Tool call body fully read.
    ToolCallEnd { index: usize },
}

/// Streaming parser modelled after llama.cpp's
/// `common_chat_parse(input, is_partial=true)` + `compute_diffs`. On
/// each delta we re-parse the full accumulated text (tolerating
/// unterminated markers), then diff against the previous parse to
/// produce events. This keeps a single parser code path between
/// non-streaming and streaming, and naturally handles markers split
/// across token boundaries — the marker arrives whole in some later
/// snapshot of `accumulated`.
#[derive(Debug)]
pub struct StreamingParser {
    kind: ParserKind,
    accumulated: String,
    prev: ParsedAssistant,
    /// Counts the prev tool_calls that have been "closed" (i.e. their
    /// `<tool_call|>` close marker has been seen). Used so we can emit
    /// `ToolCallEnd` exactly once per call.
    prev_completed_tool_calls: usize,
}

impl StreamingParser {
    pub fn new(kind: ParserKind) -> Self {
        Self {
            kind,
            accumulated: String::new(),
            prev: ParsedAssistant::default(),
            prev_completed_tool_calls: 0,
        }
    }

    pub fn push(&mut self, delta_text: &str) -> Vec<StreamEvent> {
        if matches!(self.kind, ParserKind::None) {
            return if delta_text.is_empty() {
                Vec::new()
            } else {
                vec![StreamEvent::Text(delta_text.to_string())]
            };
        }
        if delta_text.is_empty() {
            return Vec::new();
        }
        self.accumulated.push_str(delta_text);
        let cur = parse_gemma_streaming(&self.accumulated);
        let cur_completed = count_occurrences(&self.accumulated, "<tool_call|>");
        let mut events = Vec::new();

        // Content delta: cur.content extends prev.content.
        if cur.content.starts_with(&self.prev.content) {
            let new_content = &cur.content[self.prev.content.len()..];
            if !new_content.is_empty() {
                events.push(StreamEvent::Text(new_content.to_string()));
            }
        } else if cur.content != self.prev.content {
            // Content diverged (rare: e.g. trailing space trimmed). Send
            // a corrective overwrite. Most clients tolerate this.
            events.push(StreamEvent::Text(cur.content.clone()));
        }

        // Reasoning delta. Both Some, cur extends prev.
        match (&self.prev.reasoning, &cur.reasoning) {
            (None, Some(r)) if !r.is_empty() => {
                events.push(StreamEvent::Reasoning(r.clone()));
            }
            (Some(p), Some(c)) if c.starts_with(p) && c.len() > p.len() => {
                events.push(StreamEvent::Reasoning(c[p.len()..].to_string()));
            }
            _ => {}
        }

        // Tool calls: emit Begin for any new ones, ArgsDelta when args
        // grow, End when the cur block transitions from "open" to
        // "closed" (close-marker count grew).
        for (i, tc) in cur.tool_calls.iter().enumerate() {
            match self.prev.tool_calls.get(i) {
                None => {
                    events.push(StreamEvent::ToolCallBegin {
                        index: i,
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                    });
                    if !tc.function.arguments.is_empty() {
                        events.push(StreamEvent::ToolCallArgsDelta {
                            index: i,
                            partial: tc.function.arguments.clone(),
                        });
                    }
                }
                Some(prev_tc) => {
                    let prev_args = &prev_tc.function.arguments;
                    let cur_args = &tc.function.arguments;
                    if cur_args.starts_with(prev_args) && cur_args.len() > prev_args.len() {
                        events.push(StreamEvent::ToolCallArgsDelta {
                            index: i,
                            partial: cur_args[prev_args.len()..].to_string(),
                        });
                    } else if cur_args != prev_args {
                        // Args reshaped (DSL→JSON conversion produced a
                        // different prefix). Send the full new value.
                        events.push(StreamEvent::ToolCallArgsDelta {
                            index: i,
                            partial: cur_args.clone(),
                        });
                    }
                }
            }
        }
        // Emit ToolCallEnd for every newly-closed tool call.
        for i in self.prev_completed_tool_calls..cur_completed {
            events.push(StreamEvent::ToolCallEnd { index: i });
        }

        self.prev = cur;
        self.prev_completed_tool_calls = cur_completed;
        events
    }

    /// Drain final state. With diff-on-parse there's nothing buffered to
    /// flush (the last `push` already emitted everything reachable).
    pub fn flush(&mut self) -> Vec<StreamEvent> {
        Vec::new()
    }
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

/// Streaming-tolerant variant of [`parse_gemma`] that emits partial
/// state for unterminated markers (channel still open, tool_call still
/// being typed). Used by [`StreamingParser`].
fn parse_gemma_streaming(raw: &str) -> ParsedAssistant {
    // 1. Split out reasoning channels — same as parse_gemma but tolerant
    //    of unterminated channels.
    let mut content_buf = String::with_capacity(raw.len());
    let mut reasoning_buf = String::new();
    let mut cursor = raw;
    while let Some(open_idx) = cursor.find("<|channel>") {
        content_buf.push_str(&cursor[..open_idx]);
        let after_open = &cursor[open_idx + "<|channel>".len()..];
        let body_start = after_open.find('\n').map(|i| i + 1).unwrap_or(0);
        // If `\n` not yet arrived after `<|channel>`, treat the entire
        // remainder as a partial channel name; flush nothing and stop.
        if after_open.find('\n').is_none() {
            cursor = "";
            break;
        }
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
            // Open channel — emit body so far as in-progress reasoning.
            let body = body_window.trim_end();
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

    // 2. Tool calls — extract completed AND in-progress.
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
            // Partial tool call. Streaming partial DSL→JSON args produces
            // non-prefix-monotonic strings (e.g. `{"city"}` then
            // `{"city":}`), which forces full overwrites every delta and
            // breaks OpenAI clients that concat `arguments` deltas. Skip
            // the partial: emit the tool_call atomically when its
            // `<tool_call|>` close arrives.
            cursor = "";
            break;
        }
    }
    clean_content.push_str(cursor);

    // 3. Strip stray structural markers from content (model overshooting).
    let mut final_content = clean_content;
    if !tool_calls.is_empty() {
        for marker in ["<|tool_response>", "<|turn>", "<|tool_call>"] {
            if let Some(idx) = final_content.find(marker) {
                final_content.truncate(idx);
            }
        }
    }
    final_content = strip_residual_markers(&final_content);
    // Trim trailing partial-marker prefix so streaming clients don't see
    // bytes that could later become part of a structural marker.
    final_content = trim_partial_marker_suffix(&final_content).to_string();

    ParsedAssistant {
        content: final_content.trim_end().to_string(),
        reasoning: if reasoning_buf.is_empty() {
            None
        } else {
            Some(reasoning_buf)
        },
        tool_calls,
    }
}

/// Like [`parse_gemma_tool_call`] but accepts an unterminated body.
/// Returns None until at least `call:NAME{` has arrived.
fn parse_gemma_tool_call_partial(body: &str, index: usize) -> Option<ToolCall> {
    let rest = body.trim_start().strip_prefix("call:")?;
    let brace = rest.find('{')?;
    let name = rest[..brace].trim();
    if name.is_empty() {
        return None;
    }
    let args_body = &rest[brace + 1..];
    let args_body = args_body.trim_end_matches('}');
    // Empty partial args → emit an empty arguments string. Clients see
    // the function name first and progressively get args via subsequent
    // `ToolCallArgsDelta` events; an initial empty `{}` would force
    // them to clear-and-replace, which is wasteful.
    let args_json = if args_body.trim().is_empty() {
        String::new()
    } else {
        gemma_args_to_json(args_body)
    };
    Some(ToolCall {
        id: format!("call_{index}"),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: name.to_string(),
            arguments: args_json,
        },
    })
}

/// Trim from the end any prefix of a known structural marker. Used so
/// streamed content doesn't contain bytes that may later turn out to be
/// the start of `<|tool_call>` / `<|channel>` / etc.
fn trim_partial_marker_suffix(s: &str) -> &str {
    const MARKERS: &[&str] = &[
        "<|tool_call>",
        "<tool_call|>",
        "<|channel>",
        "<channel|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|turn>",
        "<turn|>",
        "<|tool>",
        "<tool|>",
    ];
    let mut max_trim = 0;
    for m in MARKERS {
        // Look at every prefix of m (excluding the full marker — that
        // case is already handled by strip_residual_markers).
        let mlen = m.len();
        for plen in 1..mlen {
            let prefix = &m[..plen];
            if s.ends_with(prefix) && plen > max_trim {
                max_trim = plen;
            }
        }
    }
    &s[..s.len() - max_trim]
}

/// Strip residual structural markers from a tail of text that arrived
/// after generation finished. Mirrors the cleanup in [`parse_gemma`].
fn strip_residual_markers(buf: &str) -> String {
    let mut s = buf.to_string();
    for m in [
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
        s = s.replace(m, "");
    }
    s
}

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

    fn drive(events: &mut Vec<StreamEvent>, parser: &mut StreamingParser, chunks: &[&str]) {
        for c in chunks {
            events.extend(parser.push(c));
        }
        events.extend(parser.flush());
    }

    #[test]
    fn streaming_passes_plain_text_through() {
        let mut p = StreamingParser::new(ParserKind::Gemma);
        let mut ev = Vec::new();
        drive(&mut ev, &mut p, &["Hello", " world", "!"]);
        let texts: String = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, "Hello world!");
    }

    #[test]
    fn streaming_extracts_thought_channel_split_across_chunks() {
        let mut p = StreamingParser::new(ParserKind::Gemma);
        let mut ev = Vec::new();
        // Marker straddling chunk boundaries
        drive(
            &mut ev,
            &mut p,
            &["pre<|cha", "nnel>thought\nlet me ", "think\n<channel|>", "answer"],
        );
        let texts: String = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let reasoning: String = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Reasoning(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts.trim(), "preanswer");
        assert!(reasoning.contains("let me"));
    }

    #[test]
    fn streaming_emits_tool_call_atomic() {
        let mut p = StreamingParser::new(ParserKind::Gemma);
        let mut ev = Vec::new();
        drive(
            &mut ev,
            &mut p,
            &[
                "<|tool_call>call:get_weather{",
                "city:<|\"|>Tokyo<|\"|>}<tool_call|>",
            ],
        );
        let begins: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallBegin { name, id, .. } => Some((name.clone(), id.clone())),
                _ => None,
            })
            .collect();
        let args: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallArgsDelta { partial, .. } => Some(partial.clone()),
                _ => None,
            })
            .collect();
        let ends: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallEnd { index } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(begins, vec![("get_weather".to_string(), "call_0".to_string())]);
        assert_eq!(args, vec![r#"{"city":"Tokyo"}"#.to_string()]);
        assert_eq!(ends, vec![0usize]);
    }

    #[test]
    fn streaming_skips_when_kind_none() {
        let mut p = StreamingParser::new(ParserKind::None);
        let mut ev = Vec::new();
        drive(&mut ev, &mut p, &["raw <|channel>", "passes through"]);
        let texts: String = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, "raw <|channel>passes through");
    }

    #[test]
    fn detect_chooses_gemma_when_markers_present() {
        let template = "...{{- '<|tool_call>call:' + name + '{' -}}...";
        assert_eq!(ParserKind::detect(template), ParserKind::Gemma);
        assert_eq!(ParserKind::detect("{{ messages }}"), ParserKind::None);
    }
}
