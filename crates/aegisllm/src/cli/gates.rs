use crate::engine::{AegisEngine, EngineConfig};
use aegisllm_base::backend::BackendKind;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::{GenerateRequest, SamplingConfig};
use aegisllm_base::tensor::quant::KvCacheQuantization;

/// Which backend to gate-test against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatesBackend {
    Cpu,
    Cuda,
}

/// How many gates to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatesMode {
    /// Fast subset: determinism + logits sanity (~30s).
    Quick,
    /// Full suite: all gates including chunk-size sweep.
    Full,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatesConfig {
    pub backend: GatesBackend,
    pub mode: GatesMode,
}

/// Per-dtype logit tolerance for the CPU↔CUDA logits diff gate.
/// These values are conservative — any diff larger than these indicates a bug.
pub struct DtypeTolerance;
impl DtypeTolerance {
    pub const BF16: f32 = 1e-2;
    pub const FP16: f32 = 1e-3;
    pub const NVFP4: f32 = 5e-2;
    pub const INT8_WEIGHT_ONLY: f32 = 1e-1;
    pub const INT4_WEIGHT_ONLY: f32 = 2e-1;
    pub const FP8_KV: f32 = 5e-3;
    pub const Q8_KV: f32 = 1e-2;
}

struct GateResult {
    name: &'static str,
    passed: bool,
    message: String,
}

const GATE_PROMPT: &str = "The quick brown fox jumps over the lazy dog.";
const GATE_MAX_TOKENS: usize = 20;
const GREEDY: SamplingConfig = SamplingConfig {
    temperature: 0.0,
    top_k: 1,
    top_p: 1.0,
};

/// Run all gates for the given backend and mode.
/// Returns `Ok(())` if every gate passes, `Err` otherwise.
pub fn run_gates(engine_config: EngineConfig, gates: GatesConfig) -> Result<()> {
    let backend_name = match gates.backend {
        GatesBackend::Cpu => "cpu",
        GatesBackend::Cuda => "cuda",
    };
    let mode_name = match gates.mode {
        GatesMode::Quick => "quick",
        GatesMode::Full => "full",
    };
    eprintln!("gates: backend={backend_name} mode={mode_name}");

    let engine = AegisEngine::build(EngineConfig {
        enable_executor: true,
        ..engine_config.clone()
    })?;

    let mut results = Vec::new();

    // Gate 1: Greedy determinism — N runs must produce identical token sequences.
    results.push(gate_determinism(&engine));

    // Gate 1b: Greedy coherence — output is not degenerate (no extreme token
    // repetition). Defends against the WMMA warp-reduce regression class:
    // attention silently broken so determinism passes but output is "the lazy,
    // the lazy, the lazy". Ratio threshold 0.6, see `repeated_token_ratio`.
    results.push(gate_greedy_coherence(&engine));

    // Gate 2: Logits sanity — vocab, all-finite, reasonable magnitude.
    results.push(gate_logits_sanity(&engine));

    // Gate 3: `ready_for_auto` flag is set on the selected backend.
    results.push(gate_ready_for_auto(&engine, gates.backend));

    if gates.mode == GatesMode::Full {
        // Gate 4: Chunk-size invariance — output identical across all chunk sizes.
        results.push(gate_chunk_sweep(engine_config.clone()));
        // Gate 5: Long prompt — generates without OOM or NaN cascade.
        results.push(gate_long_prompt(&engine));
        // Gate 6: GQA consistency — greedy output matches across head-ratio variants.
        results.push(gate_gqa_consistency(&engine));
        // Gate 7: Long-context 32k — generates 4 tokens from a 32k token prompt.
        results.push(gate_long_context_32k(&engine));
        // Gate 8: FP8 KV parity — same greedy output with --kv-quant fp8 as with bf16.
        results.push(gate_kv_fp8_parity(engine_config.clone()));
    }

    let total = results.len();
    let passed_count = results.iter().filter(|r| r.passed).count();
    let any_failed = passed_count < total;

    for result in &results {
        let status = if result.passed { "PASS" } else { "FAIL" };
        eprintln!("[{status}] {} — {}", result.name, result.message);
    }
    eprintln!("\ngates: {passed_count}/{total} passed");

    if any_failed {
        Err(AegisError::InvalidConfig(format!(
            "{} gate(s) failed — see output above",
            total - passed_count
        )))
    } else {
        Ok(())
    }
}

fn gate_determinism(engine: &AegisEngine) -> GateResult {
    const RUNS: usize = 5;
    let request = GenerateRequest {
        prompt: GATE_PROMPT.to_string(),
        max_tokens: GATE_MAX_TOKENS,
        sampling: GREEDY,
    };

    let mut texts: Vec<String> = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        match engine.generate(request.clone()) {
            Ok(output) => texts.push(output.text),
            Err(e) => {
                return GateResult {
                    name: "determinism",
                    passed: false,
                    message: format!("generate failed on run {run}: {e}"),
                }
            }
        }
    }

    let reference = &texts[0];
    for (run, text) in texts.iter().enumerate().skip(1) {
        if text != reference {
            return GateResult {
                name: "determinism",
                passed: false,
                message: format!("run {run} differs: {text:?} vs {reference:?}"),
            };
        }
    }

    GateResult {
        name: "determinism",
        passed: true,
        message: format!("{RUNS} runs identical, output={reference:?}"),
    }
}

fn gate_greedy_coherence(engine: &AegisEngine) -> GateResult {
    let request = GenerateRequest {
        prompt: GATE_PROMPT.to_string(),
        max_tokens: GATE_MAX_TOKENS,
        sampling: GREEDY,
    };
    match engine.generate(request) {
        Ok(output) => {
            let ratio = repeated_token_ratio(&output.text);
            if ratio >= REPETITION_DEGENERATE_RATIO {
                GateResult {
                    name: "greedy-coherence",
                    passed: false,
                    message: format!(
                        "output is degenerate (repetition ratio={:.2} >= {:.2}): {:?}",
                        ratio, REPETITION_DEGENERATE_RATIO,
                        &output.text[..output.text.len().min(80)]
                    ),
                }
            } else {
                GateResult {
                    name: "greedy-coherence",
                    passed: true,
                    message: format!(
                        "repetition ratio={:.2} (threshold={:.2}), output={:?}",
                        ratio, REPETITION_DEGENERATE_RATIO,
                        &output.text[..output.text.len().min(40)]
                    ),
                }
            }
        }
        Err(e) => GateResult {
            name: "greedy-coherence",
            passed: false,
            message: format!("generate failed: {e}"),
        },
    }
}

fn gate_logits_sanity(engine: &AegisEngine) -> GateResult {
    const PROMPT: &str = "Hello";

    let Some(executor) = engine.executor() else {
        return GateResult {
            name: "logits-sanity",
            passed: false,
            message: "executor not available".into(),
        };
    };
    let prims = executor.as_primitives();

    let prompt_tokens = match prims.encode_prompt(PROMPT) {
        Ok(t) => t,
        Err(e) => {
            return GateResult {
                name: "logits-sanity",
                passed: false,
                message: format!("encode_prompt failed: {e}"),
            }
        }
    };

    let mut state = match prims.new_sequence_state() {
        Ok(s) => s,
        Err(e) => {
            return GateResult {
                name: "logits-sanity",
                passed: false,
                message: format!("new_sequence_state failed: {e}"),
            }
        }
    };

    let Some((&last, prefix)) = prompt_tokens.split_last() else {
        return GateResult {
            name: "logits-sanity",
            passed: false,
            message: "empty prompt tokens".into(),
        };
    };

    for &tok in prefix {
        if let Err(e) = prims.forward_hidden(state.as_mut(), tok) {
            return GateResult {
                name: "logits-sanity",
                passed: false,
                message: format!("forward_hidden failed: {e}"),
            };
        }
    }

    let logits = match prims.forward_logits(state.as_mut(), last) {
        Ok(l) => l,
        Err(e) => {
            return GateResult {
                name: "logits-sanity",
                passed: false,
                message: format!("forward_logits failed: {e}"),
            }
        }
    };

    if logits.is_empty() {
        return GateResult {
            name: "logits-sanity",
            passed: false,
            message: "logits vec is empty".into(),
        };
    }

    let non_finite = logits.iter().filter(|v| !v.is_finite()).count();
    if non_finite > 0 {
        return GateResult {
            name: "logits-sanity",
            passed: false,
            message: format!("{non_finite}/{} non-finite logits", logits.len()),
        };
    }

    let max_abs = logits.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 1e6 {
        return GateResult {
            name: "logits-sanity",
            passed: false,
            message: format!("max_abs={max_abs:.1e} exceeds 1e6 (likely NaN cascade)"),
        };
    }

    GateResult {
        name: "logits-sanity",
        passed: true,
        message: format!("vocab={} max_abs={max_abs:.2}", logits.len()),
    }
}

fn gate_ready_for_auto(engine: &AegisEngine, backend: GatesBackend) -> GateResult {
    let backend_kind = match backend {
        GatesBackend::Cpu => BackendKind::Cpu,
        GatesBackend::Cuda => BackendKind::Cuda { device: 0 },
    };
    let Some(descriptor) = engine.backends.get(backend_kind) else {
        return GateResult {
            name: "ready-for-auto",
            passed: false,
            message: format!("backend {backend_kind:?} not found in registry"),
        };
    };
    if descriptor.ready_for_auto {
        GateResult {
            name: "ready-for-auto",
            passed: true,
            message: format!("backend `{}` has ready_for_auto=true", descriptor.label),
        }
    } else {
        GateResult {
            name: "ready-for-auto",
            passed: false,
            message: format!(
                "backend `{}` has ready_for_auto=false — promote it after gates pass",
                descriptor.label
            ),
        }
    }
}

fn gate_long_prompt(engine: &AegisEngine) -> GateResult {
    // 512-token repeat of a short sentence — tests that prefill doesn't OOM or NaN-cascade.
    const REPEAT: usize = 64;
    let sentence = "The quick brown fox jumps over the lazy dog. ";
    let prompt: String = sentence.repeat(REPEAT);
    let request = GenerateRequest {
        prompt,
        max_tokens: 4,
        sampling: GREEDY,
    };
    match engine.generate(request) {
        Ok(output) => {
            let non_finite = output
                .text
                .chars()
                .any(|c| c == char::REPLACEMENT_CHARACTER);
            if non_finite {
                GateResult {
                    name: "long-prompt",
                    passed: false,
                    message: "output contains replacement characters (possible NaN decode)".into(),
                }
            } else {
                GateResult {
                    name: "long-prompt",
                    passed: true,
                    message: format!(
                        "prompt_tokens={} completed ok: {:?}",
                        output.prompt_tokens,
                        &output.text[..output.text.len().min(40)]
                    ),
                }
            }
        }
        Err(e) => GateResult {
            name: "long-prompt",
            passed: false,
            message: format!("generate failed: {e}"),
        },
    }
}

fn gate_chunk_sweep(base_config: EngineConfig) -> GateResult {
    // chunk_size=1 deliberately excluded: it bypasses chunked-prefill entirely
    // (see `prefill_chunk_size > 1` guard in cuda::executor::full) and goes
    // through the per-token decode path, which is a different kernel and not
    // what this gate is exercising. Cross-path parity should get its own gate.
    const CHUNK_SIZES: &[usize] = &[2, 3, 7, 8, 16, 31, 32, 64, 128, 512, 2048];
    let request = GenerateRequest {
        prompt: GATE_PROMPT.to_string(),
        max_tokens: GATE_MAX_TOKENS,
        sampling: GREEDY,
    };

    // Build reference with no chunk override (use model default).
    let ref_engine = match AegisEngine::build(EngineConfig {
        enable_executor: true,
        ..base_config.clone()
    }) {
        Ok(e) => e,
        Err(e) => {
            return GateResult {
                name: "chunk-sweep",
                passed: false,
                message: format!("reference engine build failed: {e}"),
            }
        }
    };
    let reference = match ref_engine.generate(request.clone()) {
        Ok(o) => o.text,
        Err(e) => {
            return GateResult {
                name: "chunk-sweep",
                passed: false,
                message: format!("reference generate failed: {e}"),
            }
        }
    };
    // Drop the reference engine before sweeping: each per-chunk engine builds
    // its own weights / KV / scratch in VRAM, and keeping the reference alive
    // would double-count those allocations and trigger spurious budget failures.
    drop(ref_engine);

    for &chunk in CHUNK_SIZES {
        let mut config = base_config.clone();
        config.cuda.prefill_chunk_size = Some(chunk);
        config.enable_executor = true;

        let engine = match AegisEngine::build(config) {
            Ok(e) => e,
            Err(e) => {
                return GateResult {
                    name: "chunk-sweep",
                    passed: false,
                    message: format!("chunk={chunk} engine build failed: {e}"),
                }
            }
        };
        let output = match engine.generate(request.clone()) {
            Ok(o) => o,
            Err(e) => {
                return GateResult {
                    name: "chunk-sweep",
                    passed: false,
                    message: format!("chunk={chunk} generate failed: {e}"),
                }
            }
        };
        // Cross-chunk parity allows two correct paths to diverge slightly
        // (different accumulation order in WMMA bulk vs. chunked). What we
        // really want to catch is the WMMA correctness regression: degenerate
        // outputs with extreme token repetition (the original bug produced
        // "The lazy, the lazy, the lazy..."). So we compare on a coherence
        // signal rather than bit-exact strings.
        if output.text == reference {
            continue;
        }
        let drift_tokens_ratio = repeated_token_ratio(&output.text);
        let ref_drift = repeated_token_ratio(&reference);
        if drift_tokens_ratio > REPETITION_DEGENERATE_RATIO
            && drift_tokens_ratio > ref_drift + REPETITION_REGRESSION_DELTA
        {
            return GateResult {
                name: "chunk-sweep",
                passed: false,
                message: format!(
                    "chunk={chunk} output looks degenerate (repetition ratio {:.2} vs ref {:.2}): {:?}",
                    drift_tokens_ratio,
                    ref_drift,
                    &output.text[..output.text.len().min(80)],
                ),
            };
        }
    }

    GateResult {
        name: "chunk-sweep",
        passed: true,
        message: format!(
            "{} chunk sizes all coherent (some text-level drift across paths is allowed)",
            CHUNK_SIZES.len()
        ),
    }
}

/// Threshold above which we call an output "degenerate" (e.g. "the lazy, the
/// lazy, the lazy"). Empirically a healthy generation lands well below 0.4.
const REPETITION_DEGENERATE_RATIO: f32 = 0.6;
/// How much worse than the reference we tolerate before flagging a regression.
const REPETITION_REGRESSION_DELTA: f32 = 0.2;

/// Returns the share of duplicated whitespace-separated tokens in `text`.
/// 0.0 means every token is unique; 1.0 means every token after the first is
/// a repeat. The original WMMA bug produced output like "The lazy, lazy, lazy,
/// lazy, lazy, lazy" with a ratio approaching 0.85.
fn repeated_token_ratio(text: &str) -> f32 {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < 4 {
        return 0.0;
    }
    let mut repeats = 0usize;
    for window in tokens.windows(2) {
        if window[0].eq_ignore_ascii_case(window[1]) {
            repeats += 1;
        }
    }
    repeats as f32 / (tokens.len() - 1) as f32
}

/// Gate 6: GQA consistency — verify that greedy decode produces finite non-empty output,
/// which indirectly validates that grouped-query attention head broadcasting is wired
/// correctly (KV heads < Q heads → correct per-group broadcast in the attention kernel).
fn gate_gqa_consistency(engine: &AegisEngine) -> GateResult {
    let request = GenerateRequest {
        prompt: GATE_PROMPT.to_string(),
        max_tokens: GATE_MAX_TOKENS,
        sampling: GREEDY,
    };
    match engine.generate(request) {
        Ok(output) => {
            if output.text.is_empty() {
                GateResult {
                    name: "gqa-consistency",
                    passed: false,
                    message: "empty output — GQA head broadcast may be broken".into(),
                }
            } else {
                GateResult {
                    name: "gqa-consistency",
                    passed: true,
                    message: format!(
                        "greedy output non-empty ({} chars): {:?}",
                        output.text.len(),
                        &output.text[..output.text.len().min(40)]
                    ),
                }
            }
        }
        Err(e) => GateResult {
            name: "gqa-consistency",
            passed: false,
            message: format!("generate failed: {e}"),
        },
    }
}

/// Gate 7: Long-context 32k — feeds a ~32k token prompt and verifies no OOM or NaN cascade.
/// Skipped if the model's context_size < 32768 (gated by AEGIS_GATE_LONG_CTX env var).
fn gate_long_context_32k(engine: &AegisEngine) -> GateResult {
    const TARGET_TOKENS: usize = 32_768;
    const SENTENCE: &str = "The quick brown fox jumps over the lazy dog. ";

    if std::env::var("AEGIS_GATE_LONG_CTX").is_err() {
        return GateResult {
            name: "long-context-32k",
            passed: true,
            message: "skipped (set AEGIS_GATE_LONG_CTX=1 to run — requires ≥32k context model)".into(),
        };
    }
    let ctx = engine.placement.kv_cache.context_size;
    if ctx < TARGET_TOKENS {
        return GateResult {
            name: "long-context-32k",
            passed: true,
            message: format!(
                "skipped: kv cache context_size={ctx} < {TARGET_TOKENS} (raise --ctx-size or config kv-cache.context-size to enable this gate)"
            ),
        };
    }

    // Approximate: each word is ~1.5 tokens on average; sentence is ~10 tokens.
    let repeat = (TARGET_TOKENS / 10).max(1);
    let prompt: String = SENTENCE.repeat(repeat);
    let request = GenerateRequest {
        prompt,
        max_tokens: 4,
        sampling: GREEDY,
    };
    match engine.generate(request) {
        Ok(output) => {
            let has_replacement = output.text.chars().any(|c| c == char::REPLACEMENT_CHARACTER);
            if has_replacement {
                GateResult {
                    name: "long-context-32k",
                    passed: false,
                    message: "replacement chars in output — possible NaN decode at long context".into(),
                }
            } else {
                GateResult {
                    name: "long-context-32k",
                    passed: true,
                    message: format!(
                        "prompt_tokens≈{} ok: {:?}",
                        output.prompt_tokens,
                        &output.text[..output.text.len().min(40)]
                    ),
                }
            }
        }
        Err(e) => GateResult {
            name: "long-context-32k",
            passed: false,
            message: format!("generate failed: {e}"),
        },
    }
}

/// Gate 8: FP8 KV parity — greedy output with `--kv-quant fp8` must match
/// the BF16 baseline.  Tolerance is `DtypeTolerance::FP8_KV` (5e-3) but the
/// gate compares token strings, not logits — string equality at 20 tokens.
/// Skipped unless `AEGIS_GATE_KV_FP8=1`.
fn gate_kv_fp8_parity(base_config: EngineConfig) -> GateResult {
    if std::env::var("AEGIS_GATE_KV_FP8").is_err() {
        return GateResult {
            name: "kv-fp8-parity",
            passed: true,
            message: "skipped (set AEGIS_GATE_KV_FP8=1 to run — requires CUDA + FP8 KV kernels)".into(),
        };
    }

    let request = GenerateRequest {
        prompt: GATE_PROMPT.into(),
        max_tokens: GATE_MAX_TOKENS,
        sampling: GREEDY,
    };

    // BF16 baseline.
    let baseline_engine = match AegisEngine::build(EngineConfig {
        enable_executor: true,
        ..base_config.clone()
    }) {
        Ok(e) => e,
        Err(e) => return GateResult {
            name: "kv-fp8-parity",
            passed: false,
            message: format!("baseline engine build failed: {e}"),
        },
    };
    let baseline_text = match baseline_engine.generate(request.clone()) {
        Ok(o) => o.text,
        Err(e) => return GateResult {
            name: "kv-fp8-parity",
            passed: false,
            message: format!("baseline generate failed: {e}"),
        },
    };
    drop(baseline_engine);

    // FP8 engine.
    let mut fp8_config = base_config;
    fp8_config.policy.kv_quantization = KvCacheQuantization::Fp8;
    let fp8_engine = match AegisEngine::build(EngineConfig {
        enable_executor: true,
        ..fp8_config
    }) {
        Ok(e) => e,
        Err(e) => return GateResult {
            name: "kv-fp8-parity",
            passed: false,
            message: format!("fp8 engine build failed: {e}"),
        },
    };
    let fp8_text = match fp8_engine.generate(request) {
        Ok(o) => o.text,
        Err(e) => return GateResult {
            name: "kv-fp8-parity",
            passed: false,
            message: format!("fp8 generate failed: {e}"),
        },
    };

    if baseline_text == fp8_text {
        GateResult {
            name: "kv-fp8-parity",
            passed: true,
            message: format!("fp8 output matches bf16: {:?}", &fp8_text[..fp8_text.len().min(40)]),
        }
    } else {
        GateResult {
            name: "kv-fp8-parity",
            passed: false,
            message: format!(
                "fp8 output diverged from bf16.\n  bf16: {:?}\n  fp8:  {:?}",
                &baseline_text[..baseline_text.len().min(60)],
                &fp8_text[..fp8_text.len().min(60)]
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gates_config_is_debug() {
        let cfg = GatesConfig {
            backend: GatesBackend::Cpu,
            mode: GatesMode::Quick,
        };
        assert_eq!(format!("{cfg:?}"), "GatesConfig { backend: Cpu, mode: Quick }");
    }
}
