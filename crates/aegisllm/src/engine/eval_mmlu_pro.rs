//! MMLU-Pro evaluation harness.
//!
//! Runs the [TIGER-Lab MMLU-Pro] benchmark against the loaded engine and
//! reports accuracy (overall + per-subject). Purpose: validate the engine
//! end-to-end against NVIDIA's published number for Gemma-4-26B-A4B-NVFP4
//! (MMLU-Pro = 84.8 %) and later measure the accuracy cost of FP8
//! attention / KV quantization.
//!
//! [TIGER-Lab MMLU-Pro]: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
//!
//! # Protocol implemented (matches `TIGER-AI-Lab/MMLU-Pro/evaluate_from_*.py`)
//!
//! * **5-shot CoT** by default. The prompt is a SINGLE user-role message
//!   containing:
//!   1. The instruction line — `"The following are multiple choice questions
//!      (with answers) about {subject}. Think step by step and then finish
//!      your answer with \"the answer is (X)\" where X is the correct letter
//!      choice.\n\n"` — with `{subject}` set to the test question's category.
//!   2. `--shots` worked examples drawn from the *validation* split of the
//!      SAME category (the validation split holds exactly 5 examples per
//!      category — the canonical few-shot pool). Each example is rendered by
//!      [`format_example`] with its `cot_content` reasoning.
//!   3. The test question rendered by [`format_example`] with an empty
//!      `cot_content`, i.e. ending in `"Answer: Let's think step by step."`.
//! * The full string is then wrapped by the model's own chat template
//!   (Gemma-4 chat format) as one user turn — matching how a chat-tuned
//!   model is actually deployed.
//! * **Greedy decode** (temperature 0) — deterministic since commit 5a5b106.
//! * **Answer extraction**: three fallback regexes, identical order to the
//!   upstream script — see [`extract_answer`]. A question whose generation
//!   yields no parseable letter is scored WRONG (and logged), never randomly
//!   assigned (upstream randomizes; we count it wrong so a harness/engine
//!   failure can never be masked by luck).
//!
//! # Deviations from upstream (documented so a score gap can be diagnosed)
//!
//! * Upstream sends the prompt as a raw `user` message to a hosted API; we
//!   additionally apply the model's chat template. For a chat-tuned model
//!   this is the correct (and NVIDIA-comparable) setup, but it is a
//!   deviation from the bare `evaluate_from_api.py` path.
//! * `--cot false` switches to direct-answer: the instruction asks for just
//!   the letter and few-shot answers are reduced to `"Answer: (X)"`. This is
//!   NOT the standard protocol — it is a fast-iteration mode and will score
//!   lower than 5-shot CoT.
//! * Unparseable generations are scored wrong, not randomized.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::{ChatMessage, SamplingConfig};
use aegisllm_base::text::TextProcessor;

use super::{AegisEngine, EngineConfig};

/// The 14 MMLU-Pro categories (lowercase, as they appear in the dataset).
pub const CATEGORIES: [&str; 14] = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "philosophy",
    "physics",
    "psychology",
    "other",
];

const CHOICE_LETTERS: &[u8] = b"ABCDEFGHIJ";

/// HuggingFace datasets-server JSON rows endpoint. Returns rows as JSON
/// (100/page) so we never need a parquet reader. `num_rows_total` in the
/// first page tells us how many pages to fetch.
const HF_ROWS_API: &str = "https://datasets-server.huggingface.co/rows";
const HF_DATASET: &str = "TIGER-Lab/MMLU-Pro";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalMmluProRequest {
    /// Pre-downloaded dataset directory. If `None`, fetch+cache from HF.
    pub dataset_path: Option<PathBuf>,
    /// Evaluate only the first N test questions (after subject filtering).
    /// `None` = the full ~12k test split.
    pub subset: Option<usize>,
    /// Restrict to these categories (lowercased). Empty = all 14.
    pub subjects: Vec<String>,
    /// Few-shot example count. Default 5 (the standard protocol).
    pub shots: usize,
    /// Chain-of-thought on/off. Default on.
    pub cot: bool,
    /// Native model thinking-channel on/off. Default on — reasoning models
    /// (Gemma 4) publish benchmark numbers with thinking enabled; disabling it
    /// pushes the model off-distribution into prompt-steered pseudo-CoT.
    pub thinking: bool,
    /// Per-question generation cap. Default 4000 (CoT) — see [`default_max_tokens`].
    pub max_tokens: usize,
    /// Per-question results sink (JSON). `None` = no per-question file.
    pub output: Option<PathBuf>,
    /// Print a running-accuracy line every N questions.
    pub progress_every: usize,
}

/// Default per-question token cap. CoT generations run long (the upstream
/// script uses 4000); direct-answer needs only a handful.
pub fn default_max_tokens(cot: bool) -> usize {
    if cot {
        4000
    } else {
        16
    }
}

#[derive(Debug, Clone)]
struct Question {
    question_id: i64,
    category: String,
    question: String,
    options: Vec<String>,
    answer: String,
    cot_content: String,
}

#[derive(Debug, Clone)]
pub struct PerQuestionResult {
    pub question_id: i64,
    pub category: String,
    pub gold: String,
    pub predicted: Option<String>,
    pub correct: bool,
    pub completion_tokens: usize,
    /// Truncated generation tail (for offline inspection of failures).
    pub generation_preview: String,
}

#[derive(Debug, Clone)]
pub struct EvalMmluProResult {
    pub total: usize,
    pub correct: usize,
    pub unparseable: usize,
    /// (category, correct, total) sorted by category.
    pub per_subject: Vec<(String, usize, usize)>,
    pub elapsed_secs: f64,
}

impl EvalMmluProResult {
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Prompt construction — mirrors TIGER-AI-Lab/MMLU-Pro format_example().
// ---------------------------------------------------------------------------

/// Render one question block. `cot_content` empty => the trailing
/// "Answer: Let's think step by step." stub (the test question). When
/// non-empty it is the worked example's reasoning.
///
/// Upstream `format_example`:
/// ```text
/// Question: {q}
/// Options: A. ...
/// B. ...
/// ...
/// Answer: {cot}\n\n            (few-shot)
/// Answer: Let's think step by step.   (test question, cot=on)
/// ```
fn format_example(q: &Question, cot_content: &str, cot: bool) -> String {
    let mut s = format!("Question: {}\nOptions: ", q.question.trim());
    for (i, opt) in q.options.iter().enumerate() {
        let letter = CHOICE_LETTERS.get(i).copied().unwrap_or(b'?') as char;
        s.push_str(&format!("{letter}. {opt}\n"));
    }
    let trimmed = cot_content.trim();
    if trimmed.is_empty() {
        // Test question stub.
        if cot {
            s.push_str("Answer: Let's think step by step.");
        } else {
            s.push_str("Answer:");
        }
    } else {
        // Few-shot example. Upstream strips a leading "A: " from cot_content.
        let body = trimmed.strip_prefix("A: ").unwrap_or(trimmed);
        if cot {
            s.push_str(&format!("Answer: {body}\n\n"));
        } else {
            // Direct-answer fast mode: collapse the worked reasoning to just
            // the final letter. NOT the standard protocol.
            s.push_str(&format!("Answer: ({})\n\n", q.answer));
        }
    }
    s
}

/// The instruction line. `{$}` in the upstream `initial_prompt.txt` is
/// replaced by the subject.
fn instruction_line(subject: &str, cot: bool) -> String {
    if cot {
        format!(
            "The following are multiple choice questions (with answers) about \
             {subject}. Think step by step and then finish your answer with \
             \"the answer is (X)\" where X is the correct letter choice.\n\n"
        )
    } else {
        format!(
            "The following are multiple choice questions (with answers) about \
             {subject}. Answer with \"the answer is (X)\" where X is the correct \
             letter choice.\n\n"
        )
    }
}

/// Build the full single-user-message prompt body for a test question.
fn build_prompt_body(test: &Question, shots: &[Question], cot: bool) -> String {
    let mut body = instruction_line(&test.category, cot);
    for ex in shots {
        body.push_str(&format_example(ex, &ex.cot_content, cot));
    }
    body.push_str(&format_example(test, "", cot));
    body
}

// ---------------------------------------------------------------------------
// Answer extraction — three fallback regexes, upstream order.
// ---------------------------------------------------------------------------

/// Extract the predicted letter from a generation. Returns `None` if no
/// pattern matches (scored wrong).
///
/// Mirrors `extract_answer` / `extract_again` / `extract_final` from
/// `evaluate_from_local.py`:
/// 1. `answer is \(?([A-J])\)?`
/// 2. `[aA]nswer:\s*([A-J])`
/// 3. last isolated `\b[A-J]\b`
///
/// Implemented without the `regex` crate (not a dependency) — hand-rolled
/// scanners that match the same languages.
pub fn extract_answer(text: &str) -> Option<char> {
    if let Some(c) = scan_answer_is(text) {
        return Some(c);
    }
    if let Some(c) = scan_answer_colon(text) {
        return Some(c);
    }
    scan_last_isolated_letter(text)
}

fn is_choice_letter(c: char) -> bool {
    ('A'..='J').contains(&c)
}

/// Pattern 1: `answer is \(?([A-J])\)?` (case-insensitive on "answer is").
/// Returns the LAST match — the model's final answer line — so a CoT that
/// says "the answer is (B)" mid-reasoning then concludes "(D)" scores D.
fn scan_answer_is(text: &str) -> Option<char> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let needle = b"answer is";
    let mut found = None;
    let mut i = 0;
    while i + needle.len() <= bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            // Skip spaces, then an optional '('.
            let mut j = i + needle.len();
            while j < bytes.len() && bytes[j] == b' ' {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'(' {
                j += 1;
            }
            if j < bytes.len() {
                // Read from the ORIGINAL (cased) text for the letter.
                let c = text.as_bytes()[j] as char;
                if is_choice_letter(c.to_ascii_uppercase()) {
                    found = Some(c.to_ascii_uppercase());
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    found
}

/// Pattern 2: `[aA]nswer:\s*([A-J])` — last match.
fn scan_answer_colon(text: &str) -> Option<char> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let needle = b"answer:";
    let mut found = None;
    let mut i = 0;
    while i + needle.len() <= bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            let mut j = i + needle.len();
            while j < bytes.len() && (bytes[j] as char).is_ascii_whitespace() {
                j += 1;
            }
            // Allow an optional '(' before the letter.
            if j < bytes.len() && bytes[j] == b'(' {
                j += 1;
            }
            if j < bytes.len() {
                let c = (text.as_bytes()[j] as char).to_ascii_uppercase();
                if is_choice_letter(c) {
                    found = Some(c);
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    found
}

/// Pattern 3: the last standalone A-J (word-boundary on both sides).
fn scan_last_isolated_letter(text: &str) -> Option<char> {
    let chars: Vec<char> = text.chars().collect();
    let mut found = None;
    for i in 0..chars.len() {
        let c = chars[i].to_ascii_uppercase();
        if !is_choice_letter(c) {
            continue;
        }
        let prev_ok = i == 0 || !chars[i - 1].is_alphanumeric();
        let next_ok = i + 1 >= chars.len() || !chars[i + 1].is_alphanumeric();
        if prev_ok && next_ok {
            found = Some(c);
        }
    }
    found
}

// ---------------------------------------------------------------------------
// Dataset fetch / cache.
// ---------------------------------------------------------------------------

fn cache_dir() -> PathBuf {
    PathBuf::from(".aegis-cache").join("mmlu-pro")
}

/// Load the test + validation splits. If `dataset_path` is given, read JSON
/// from there; otherwise fetch from HF (via `curl`) and cache under
/// `.aegis-cache/mmlu-pro/`.
fn load_splits(request: &EvalMmluProRequest) -> Result<(Vec<Question>, Vec<Question>)> {
    let base = match &request.dataset_path {
        Some(p) => p.clone(),
        None => {
            let dir = cache_dir();
            ensure_cached(&dir)?;
            dir
        }
    };
    let test = read_split_file(&base, "test")?;
    let validation = read_split_file(&base, "validation")?;
    Ok((test, validation))
}

/// Read a split from `<dir>/<split>.json`. Accepts either the HF rows-API
/// shape (`{"rows":[{"row":{...}}]}`) or a plain array of row objects.
fn read_split_file(dir: &Path, split: &str) -> Result<Vec<Question>> {
    let path = dir.join(format!("{split}.json"));
    let raw = std::fs::read_to_string(&path).map_err(|e| {
        AegisError::InvalidConfig(format!(
            "eval-mmlu-pro: cannot read dataset split {}: {e}\n\
             Provide --dataset-path to a directory containing test.json and \
             validation.json, or omit it to fetch from HuggingFace.",
            path.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        AegisError::InvalidConfig(format!("eval-mmlu-pro: bad JSON in {}: {e}", path.display()))
    })?;
    parse_rows(&value)
}

/// Parse questions out of either `{"rows":[{"row":{...}}]}` (HF rows API) or
/// a bare `[{...}]` array, or `{"data":[...]}`.
fn parse_rows(value: &serde_json::Value) -> Result<Vec<Question>> {
    let rows: Vec<&serde_json::Value> = if let Some(arr) = value.get("rows").and_then(|v| v.as_array()) {
        arr.iter()
            .map(|entry| entry.get("row").unwrap_or(entry))
            .collect()
    } else if let Some(arr) = value.get("data").and_then(|v| v.as_array()) {
        arr.iter().collect()
    } else if let Some(arr) = value.as_array() {
        arr.iter().collect()
    } else {
        return Err(AegisError::InvalidConfig(
            "eval-mmlu-pro: dataset JSON has neither `rows`, `data`, nor a top-level array".into(),
        ));
    };
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        out.push(parse_question(row)?);
    }
    Ok(out)
}

fn parse_question(row: &serde_json::Value) -> Result<Question> {
    let question = row
        .get("question")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AegisError::InvalidConfig("eval-mmlu-pro: row missing `question`".into()))?
        .to_string();
    let options = row
        .get("options")
        .and_then(|v| v.as_array())
        .ok_or_else(|| AegisError::InvalidConfig("eval-mmlu-pro: row missing `options`".into()))?
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect::<Vec<_>>();
    let answer = row
        .get("answer")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AegisError::InvalidConfig("eval-mmlu-pro: row missing `answer`".into()))?
        .trim()
        .to_string();
    let category = row
        .get("category")
        .and_then(|v| v.as_str())
        .unwrap_or("other")
        .to_lowercase();
    let cot_content = row
        .get("cot_content")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let question_id = row.get("question_id").and_then(|v| v.as_i64()).unwrap_or(-1);
    Ok(Question {
        question_id,
        category,
        question,
        options,
        answer,
        cot_content,
    })
}

/// Ensure `dir` has `test.json` + `validation.json`, fetching from HF if not.
fn ensure_cached(dir: &Path) -> Result<()> {
    let test = dir.join("test.json");
    let validation = dir.join("validation.json");
    if test.exists() && validation.exists() {
        eprintln!("[eval-mmlu-pro] using cached dataset at {}", dir.display());
        return Ok(());
    }
    std::fs::create_dir_all(dir).map_err(|e| {
        AegisError::InvalidConfig(format!(
            "eval-mmlu-pro: cannot create cache dir {}: {e}",
            dir.display()
        ))
    })?;
    eprintln!(
        "[eval-mmlu-pro] dataset not cached — fetching {HF_DATASET} from HuggingFace \
         (this needs network; pass --dataset-path to skip)"
    );
    for split in ["validation", "test"] {
        let target = dir.join(format!("{split}.json"));
        if target.exists() {
            continue;
        }
        let rows = fetch_split_from_hf(split)?;
        let json = serde_json::json!({ "rows": rows });
        std::fs::write(&target, serde_json::to_vec_pretty(&json).unwrap()).map_err(|e| {
            AegisError::InvalidConfig(format!(
                "eval-mmlu-pro: cannot write {}: {e}",
                target.display()
            ))
        })?;
        eprintln!(
            "[eval-mmlu-pro] cached {} ({} rows) -> {}",
            split,
            rows.len(),
            target.display()
        );
    }
    Ok(())
}

/// Fetch one split via the HF datasets-server rows API, paging 100 at a time.
/// Returns the raw `{"row":{...}}` entries so the cache file mirrors the API.
fn fetch_split_from_hf(split: &str) -> Result<Vec<serde_json::Value>> {
    let mut all: Vec<serde_json::Value> = Vec::new();
    let mut offset = 0usize;
    let page = 100usize;
    let mut total: Option<usize> = None;
    loop {
        let url = format!(
            "{HF_ROWS_API}?dataset={}&config=default&split={split}&offset={offset}&length={page}",
            urlencode(HF_DATASET)
        );
        let body = curl_get(&url)?;
        let value: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            AegisError::InvalidConfig(format!(
                "eval-mmlu-pro: HF rows API returned non-JSON for {split} @ offset {offset}: {e}"
            ))
        })?;
        if total.is_none() {
            total = value
                .get("num_rows_total")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
        }
        let rows = value.get("rows").and_then(|v| v.as_array()).ok_or_else(|| {
            AegisError::InvalidConfig(format!(
                "eval-mmlu-pro: HF rows API response for {split} has no `rows` array \
                 (offset {offset}); body starts: {}",
                body.chars().take(200).collect::<String>()
            ))
        })?;
        if rows.is_empty() {
            break;
        }
        all.extend(rows.iter().cloned());
        offset += rows.len();
        match total {
            Some(t) if offset >= t => break,
            _ if rows.len() < page => break,
            _ => {}
        }
    }
    if all.is_empty() {
        return Err(AegisError::InvalidConfig(format!(
            "eval-mmlu-pro: fetched 0 rows for split {split} — check network or pass --dataset-path"
        )));
    }
    Ok(all)
}

/// Minimal percent-encoding for the dataset path segment (`/` -> `%2F`).
fn urlencode(s: &str) -> String {
    s.replace('/', "%2F")
}

/// HTTP GET via the system `curl`. We shell out rather than add an HTTP
/// crate so the build stays dependency-stable (purely-additive constraint).
fn curl_get(url: &str) -> Result<String> {
    let output = std::process::Command::new("curl")
        .args(["-sSL", "--fail", "--max-time", "120", url])
        .output()
        .map_err(|e| {
            AegisError::InvalidConfig(format!(
                "eval-mmlu-pro: failed to invoke `curl` ({e}). Install curl or pass \
                 --dataset-path to a pre-downloaded dataset directory."
            ))
        })?;
    if !output.status.success() {
        return Err(AegisError::InvalidConfig(format!(
            "eval-mmlu-pro: curl failed for {url}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

// ---------------------------------------------------------------------------
// Eval loop.
// ---------------------------------------------------------------------------

/// Run the MMLU-Pro benchmark. Loads the engine (the USER runs this — it is a
/// model load) and drives one greedy generation per question.
pub fn run_eval_mmlu_pro(
    config: EngineConfig,
    request: EvalMmluProRequest,
) -> Result<EvalMmluProResult> {
    if request.shots == 0 && request.cot {
        eprintln!(
            "[eval-mmlu-pro] WARNING: --shots 0 with CoT is zero-shot — \
             the standard MMLU-Pro protocol is 5-shot"
        );
    }
    let (test, validation) = load_splits(&request)?;

    // Few-shot pool: validation split grouped by category.
    let mut shots_by_category: BTreeMap<String, Vec<Question>> = BTreeMap::new();
    for q in &validation {
        shots_by_category
            .entry(q.category.clone())
            .or_default()
            .push(q.clone());
    }

    // Subject filter.
    let subjects: Vec<String> = request.subjects.iter().map(|s| s.to_lowercase()).collect();
    let mut selected: Vec<&Question> = test
        .iter()
        .filter(|q| subjects.is_empty() || subjects.contains(&q.category))
        .collect();
    if subjects.is_empty() {
        // Keep dataset order for reproducibility; the dataset is already
        // interleaved by category.
    }
    if let Some(n) = request.subset {
        selected.truncate(n);
    }
    if selected.is_empty() {
        return Err(AegisError::InvalidConfig(
            "eval-mmlu-pro: no questions selected — check --subjects / --subset".into(),
        ));
    }

    eprintln!(
        "[eval-mmlu-pro] {} questions, shots={}, cot={}, max_tokens={}",
        selected.len(),
        request.shots,
        request.cot,
        request.max_tokens
    );

    let engine = AegisEngine::build(config)?;
    let executor = engine.executor().ok_or_else(|| {
        AegisError::Unsupported("eval-mmlu-pro: engine was built without executor".into())
    })?;
    let backend = executor.as_primitives();
    // Gemma-4's recommended sampling. Reasoning models MUST be sampled, not
    // run greedy: greedy (temp=0) degenerates into repetition loops on long
    // reasoning traces — verified against the official Gemma-4 API, which at
    // temp=0 also looped to MAX_TOKENS (32k thinking tokens, no answer) and at
    // temp=1.0 concluded cleanly in ~5k. Benchmark numbers (NVIDIA's 84.8%
    // MMLU-Pro) are sampled, not greedy. The run is therefore stochastic.
    let sampling = SamplingConfig {
        temperature: 1.0,
        top_k: 64,
        top_p: 0.95,
        min_p: 0.0,
    };

    // Per-question output sink — written incrementally as JSON Lines so a
    // long run that is interrupted still has partial results on disk.
    let mut output_writer = match &request.output {
        Some(path) => {
            let file = std::fs::File::create(path).map_err(|e| {
                AegisError::InvalidConfig(format!(
                    "eval-mmlu-pro: cannot create --output {}: {e}",
                    path.display()
                ))
            })?;
            Some(std::io::BufWriter::new(file))
        }
        None => None,
    };

    let start = Instant::now();
    let mut correct = 0usize;
    let mut unparseable = 0usize;
    let mut per_subject: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let total = selected.len();

    for (idx, test_q) in selected.iter().enumerate() {
        let shots: Vec<Question> = shots_by_category
            .get(&test_q.category)
            .map(|pool| pool.iter().take(request.shots).cloned().collect())
            .unwrap_or_default();
        if shots.len() < request.shots {
            eprintln!(
                "[eval-mmlu-pro] WARNING: category `{}` has only {} few-shot examples \
                 (requested {})",
                test_q.category,
                shots.len(),
                request.shots
            );
        }
        let body = build_prompt_body(test_q, &shots, request.cot);

        // Wrap the single-user-message body in the model's chat template, then
        // tokenize as raw text (the string is already chat-formatted — using
        // encode_text_raw avoids a second chat-template wrap).
        let rendered = TextProcessor::render_chat_for_artifact_with_tools(
            &engine.artifact,
            &[ChatMessage {
                role: "user".into(),
                content: body,
                ..Default::default()
            }],
            None,
            // Native thinking channel — reasoning models are benchmarked with
            // it ON; OFF forces prompt-steered pseudo-CoT (off-distribution).
            request.thinking,
        )?;
        let prompt_tokens = backend.encode_text_raw(&rendered)?;
        if prompt_tokens.is_empty() {
            return Err(AegisError::InvalidConfig(format!(
                "eval-mmlu-pro: question {} produced no tokens",
                test_q.question_id
            )));
        }

        // Greedy decode loop. Mirrors generate_with_backend but lets us stop
        // early once the model has emitted a clean "the answer is (X)".
        let mut state = backend.new_sequence_state()?;
        let mut next = backend.prefill_prompt(state.as_mut(), &prompt_tokens, &sampling)?;
        let mut generated: Vec<usize> = Vec::new();
        for _ in 0..request.max_tokens {
            if backend.is_eos(next) {
                break;
            }
            generated.push(next);
            if generated.len() >= request.max_tokens {
                break;
            }
            // Early stop: in thinking mode the model concludes ("Final Answer:
            // X") then loops that line to the token cap without ever emitting
            // EOS. Once a final-answer marker AND a parseable letter are both
            // present in the recent tail, the answer is locked — stop. Checked
            // every 32 tokens (decode of a ~220-token tail is cheap).
            if generated.len() % 32 == 0 {
                let tail_start = generated.len().saturating_sub(220);
                let tail = backend
                    .decode_tokens(&generated[tail_start..])
                    .unwrap_or_default();
                if tail.to_lowercase().contains("final answer")
                    && extract_answer(&tail).is_some()
                {
                    break;
                }
            }
            next = backend.forward_next_token(state.as_mut(), next, &sampling)?;
        }
        let generation = backend.decode_tokens(&generated).unwrap_or_default();

        let predicted = extract_answer(&generation);
        let gold = test_q.answer.chars().next().unwrap_or('?');
        let is_correct = predicted == Some(gold);
        if predicted.is_none() {
            unparseable += 1;
            eprintln!(
                "[eval-mmlu-pro] q{} ({}) — no parseable answer; tail: {:?}",
                test_q.question_id,
                test_q.category,
                generation.chars().rev().take(120).collect::<String>()
                    .chars().rev().collect::<String>()
            );
        }
        if is_correct {
            correct += 1;
        }
        let entry = per_subject
            .entry(test_q.category.clone())
            .or_insert((0, 0));
        entry.1 += 1;
        if is_correct {
            entry.0 += 1;
        }

        let preview: String = {
            let tail: String = generation.chars().rev().take(200).collect();
            tail.chars().rev().collect()
        };
        let result = PerQuestionResult {
            question_id: test_q.question_id,
            category: test_q.category.clone(),
            gold: gold.to_string(),
            predicted: predicted.map(|c| c.to_string()),
            correct: is_correct,
            completion_tokens: generated.len(),
            generation_preview: preview,
        };
        if let Some(writer) = output_writer.as_mut() {
            let json = serde_json::json!({
                "question_id": result.question_id,
                "category": result.category,
                "gold": result.gold,
                "predicted": result.predicted,
                "correct": result.correct,
                "completion_tokens": result.completion_tokens,
                "generation_preview": result.generation_preview,
            });
            writeln!(writer, "{json}").map_err(|e| {
                AegisError::InvalidConfig(format!("eval-mmlu-pro: failed to write --output: {e}"))
            })?;
            writer.flush().ok();
        }

        let done = idx + 1;
        if request.progress_every > 0
            && (done % request.progress_every == 0 || done == total)
        {
            let acc = correct as f64 / done as f64 * 100.0;
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed.max(1e-9);
            let eta = (total - done) as f64 / rate.max(1e-9);
            eprintln!(
                "[eval-mmlu-pro] {done}/{total} | acc={acc:.2}% ({correct}/{done}) | \
                 unparseable={unparseable} | {rate:.2} q/s | ETA {:.0}m",
                eta / 60.0
            );
        }
    }

    if let Some(writer) = output_writer.as_mut() {
        writer.flush().ok();
    }

    let per_subject: Vec<(String, usize, usize)> = per_subject
        .into_iter()
        .map(|(k, (c, t))| (k, c, t))
        .collect();

    Ok(EvalMmluProResult {
        total,
        correct,
        unparseable,
        per_subject,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

/// Pretty-print the summary table.
pub fn print_eval_summary(result: &EvalMmluProResult) {
    println!("\n=== MMLU-Pro results ===");
    println!("{:<22} {:>8} {:>8} {:>9}", "subject", "correct", "total", "accuracy");
    println!("{}", "-".repeat(49));
    for (subject, correct, total) in &result.per_subject {
        let acc = if *total == 0 {
            0.0
        } else {
            *correct as f64 / *total as f64 * 100.0
        };
        println!("{subject:<22} {correct:>8} {total:>8} {acc:>8.2}%");
    }
    println!("{}", "-".repeat(49));
    println!(
        "{:<22} {:>8} {:>8} {:>8.2}%",
        "OVERALL",
        result.correct,
        result.total,
        result.accuracy() * 100.0
    );
    println!(
        "unparseable={} | elapsed={:.1}s ({:.1}m)",
        result.unparseable,
        result.elapsed_secs,
        result.elapsed_secs / 60.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(category: &str, answer: &str) -> Question {
        Question {
            question_id: 1,
            category: category.into(),
            question: "What is 2+2?".into(),
            options: vec!["3".into(), "4".into(), "5".into()],
            answer: answer.into(),
            cot_content: String::new(),
        }
    }

    #[test]
    fn extract_answer_is_paren() {
        assert_eq!(extract_answer("So the answer is (C)."), Some('C'));
    }

    #[test]
    fn extract_answer_is_no_paren() {
        assert_eq!(extract_answer("Therefore the answer is D"), Some('D'));
    }

    #[test]
    fn extract_answer_is_case_insensitive() {
        assert_eq!(extract_answer("The Answer Is (B)"), Some('B'));
    }

    #[test]
    fn extract_answer_picks_last_match() {
        // CoT changes its mind: an intermediate "(A)" then a final "(F)".
        assert_eq!(
            extract_answer("first the answer is (A) but actually the answer is (F)"),
            Some('F')
        );
    }

    #[test]
    fn extract_answer_colon_fallback() {
        assert_eq!(extract_answer("Reasoning...\nAnswer: G"), Some('G'));
    }

    #[test]
    fn extract_answer_last_isolated_letter() {
        assert_eq!(extract_answer("I think it is option H here"), Some('H'));
    }

    #[test]
    fn extract_answer_ignores_letters_in_words() {
        // No "answer is" / "answer:" — falls to isolated-letter scan. "Cat"
        // and "Dog" contain C/D but not as isolated tokens.
        assert_eq!(extract_answer("Cat Dog elephant"), None);
    }

    #[test]
    fn extract_answer_none_when_absent() {
        assert_eq!(extract_answer("no choice here at all"), None);
    }

    #[test]
    fn extract_answer_out_of_range_letter() {
        // K is past J — not a valid choice.
        assert_eq!(extract_answer("the answer is (K)"), None);
    }

    #[test]
    fn format_example_test_stub_cot() {
        let s = format_example(&q("math", "B"), "", true);
        assert!(s.starts_with("Question: What is 2+2?\nOptions: A. 3\nB. 4\nC. 5\n"));
        assert!(s.ends_with("Answer: Let's think step by step."));
    }

    #[test]
    fn format_example_test_stub_direct() {
        let s = format_example(&q("math", "B"), "", false);
        assert!(s.ends_with("Answer:"));
    }

    #[test]
    fn format_example_fewshot_strips_a_prefix() {
        let s = format_example(&q("math", "B"), "A: because four is correct.", true);
        assert!(s.contains("Answer: because four is correct.\n\n"));
        assert!(!s.contains("A: because"));
    }

    #[test]
    fn format_example_direct_collapses_cot_to_letter() {
        let s = format_example(&q("math", "B"), "long reasoning here", false);
        assert!(s.contains("Answer: (B)\n\n"));
    }

    #[test]
    fn instruction_line_has_subject_and_format_request() {
        let line = instruction_line("physics", true);
        assert!(line.contains("about physics."));
        assert!(line.contains("the answer is (X)"));
        assert!(line.contains("step by step"));
    }

    #[test]
    fn build_prompt_body_order() {
        let test = q("math", "B");
        let shot = {
            let mut s = q("math", "A");
            s.cot_content = "A: trivial.".into();
            s
        };
        let body = build_prompt_body(&test, std::slice::from_ref(&shot), true);
        let instr_pos = body.find("multiple choice").unwrap();
        let shot_pos = body.find("Answer: trivial.").unwrap();
        let test_pos = body.rfind("Answer: Let's think step by step.").unwrap();
        assert!(instr_pos < shot_pos);
        assert!(shot_pos < test_pos);
    }

    #[test]
    fn parse_rows_hf_shape() {
        let json = serde_json::json!({
            "rows": [
                {"row": {
                    "question_id": 7,
                    "question": "Q?",
                    "options": ["a", "b"],
                    "answer": "B",
                    "answer_index": 1,
                    "cot_content": "A: think.",
                    "category": "Math"
                }}
            ]
        });
        let parsed = parse_rows(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].question_id, 7);
        assert_eq!(parsed[0].answer, "B");
        assert_eq!(parsed[0].category, "math");
        assert_eq!(parsed[0].options, vec!["a", "b"]);
    }

    #[test]
    fn parse_rows_bare_array() {
        let json = serde_json::json!([
            {"question": "Q?", "options": ["a"], "answer": "A", "category": "law"}
        ]);
        let parsed = parse_rows(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].category, "law");
    }

    #[test]
    fn urlencode_slash() {
        assert_eq!(urlencode("TIGER-Lab/MMLU-Pro"), "TIGER-Lab%2FMMLU-Pro");
    }

    #[test]
    fn default_max_tokens_cot_vs_direct() {
        assert_eq!(default_max_tokens(true), 4000);
        assert_eq!(default_max_tokens(false), 16);
    }
}
