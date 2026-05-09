use std::path::PathBuf;

use aegisllm_cuda::cuda::CudaRuntimeConfig;
use crate::engine::EngineConfig;
use crate::engine::bench::BenchGenerateRequest;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::{GenerateRequest, SamplingConfig};
use aegisllm_base::hardware::HardwareInventory;
use crate::params::{ParametersFile, ServeConfig};
use aegisllm_base::planning::placement::PlacementPolicy;

use super::Command;
use super::command::BenchOutputFormat;
use super::flags::{flag_takes_value, is_engine_flag, parse_engine_flags, parse_value, take_value};

pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command> {
    let args = args.into_iter().collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        Some("inspect-hardware") => Ok(Command::InspectHardware),
        Some("serve") => parse_serve(&args[1..]),
        Some("show-plan") => parse_show_plan(&args[1..]),
        Some("mvp-check") => parse_mvp_check(&args[1..]),
        Some("quality-smoke") => parse_quality_smoke(&args[1..]),
        Some("storage-smoke") => parse_storage_smoke(&args[1..]),
        Some("cpu-smoke") => parse_cpu_smoke(&args[1..]),
        Some("cpu-materialize-smoke") => parse_cpu_materialize_smoke(&args[1..]),
        Some("cuda-smoke") => parse_cuda_smoke(&args[1..]),
        Some("cuda-dense-smoke") => parse_cuda_dense_smoke(&args[1..]),
        Some("cuda-chain-smoke") => parse_cuda_chain_smoke(&args[1..]),
        Some("cuda-compare") => parse_cuda_compare(&args[1..]),
        Some("cuda-prefill-compare") => parse_cuda_prefill_compare(&args[1..]),
        Some("cuda-prefill-sweep") => parse_cuda_prefill_sweep(&args[1..]),
        Some("generate") => parse_generate(&args[1..]),
        Some("quality-diff") => parse_quality_diff(&args[1..]),
        Some("bench-generate") => parse_bench_generate(&args[1..]),
        Some("perplexity") => parse_perplexity(&args[1..]),
        Some("sample-diversity") => parse_sample_diversity(&args[1..]),
        Some("gates") => parse_gates(&args[1..]),
        Some("--help") | Some("-h") | Some("help") | None => {
            Err(AegisError::InvalidConfig(usage()))
        }
        Some(other) => Err(AegisError::InvalidConfig(format!(
            "unknown subcommand `{other}`\n\n{}",
            usage()
        ))),
    }
}

fn parse_storage_smoke(args: &[String]) -> Result<Command> {
    Ok(Command::StorageSmoke(
        parse_engine_flags(args)?.engine_config(false),
    ))
}

fn parse_mvp_check(args: &[String]) -> Result<Command> {
    Ok(Command::MvpCheck(
        parse_engine_flags(args)?.engine_config(false),
    ))
}

fn parse_quality_smoke(args: &[String]) -> Result<Command> {
    Ok(Command::QualitySmoke(
        parse_engine_flags(args)?.engine_config(true),
    ))
}

fn parse_cpu_smoke(args: &[String]) -> Result<Command> {
    Ok(Command::CpuSmoke(
        parse_engine_flags(args)?.engine_config(false),
    ))
}

fn parse_cpu_materialize_smoke(args: &[String]) -> Result<Command> {
    Ok(Command::CpuMaterializeSmoke(
        parse_engine_flags(args)?.engine_config(false),
    ))
}

fn parse_cuda_smoke(args: &[String]) -> Result<Command> {
    Ok(Command::CudaSmoke(
        parse_engine_flags(args)?.engine_config(false),
    ))
}

fn parse_cuda_dense_smoke(args: &[String]) -> Result<Command> {
    Ok(Command::CudaDenseSmoke(
        parse_engine_flags(args)?.engine_config(false),
    ))
}

fn parse_cuda_chain_smoke(args: &[String]) -> Result<Command> {
    Ok(Command::CudaChainSmoke(
        parse_engine_flags(args)?.engine_config(false),
    ))
}

fn parse_cuda_compare(args: &[String]) -> Result<Command> {
    Ok(Command::CudaCompare(
        parse_engine_flags(args)?.engine_config(false),
    ))
}

fn parse_cuda_prefill_compare(args: &[String]) -> Result<Command> {
    Ok(Command::CudaPrefillCompare(
        parse_engine_flags(args)?.engine_config(true),
    ))
}

fn parse_cuda_prefill_sweep(args: &[String]) -> Result<Command> {
    Ok(Command::CudaPrefillSweep(
        parse_engine_flags(args)?.engine_config(true),
    ))
}

fn parse_show_plan(args: &[String]) -> Result<Command> {
    Ok(Command::ShowPlan(
        parse_engine_flags(args)?.engine_config(false),
    ))
}

fn parse_serve(args: &[String]) -> Result<Command> {
    let inventory = HardwareInventory::detect();
    let mut serve: Option<ServeConfig> = None;
    let mut host = None;
    let mut port = None;
    let mut model = None;
    let mut ctx_size = None;
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        match flag.as_str() {
            "--config" | "--parameters" => {
                let path = take_value(args, &mut i, flag)?;
                serve = Some(
                    ParametersFile::from_path(&path)?
                        .into_serve_config(PlacementPolicy::auto_for(&inventory))?,
                );
            }
            "--host" => host = Some(take_value(args, &mut i, flag)?),
            "--port" => port = Some(parse_value(args, &mut i, flag)?),
            "--model" => model = Some(PathBuf::from(take_value(args, &mut i, flag)?)),
            "--ctx-size" => ctx_size = Some(parse_value(args, &mut i, flag)?),
            "--threads" => {
                let _threads: usize = parse_value(args, &mut i, flag)?;
            }
            other => {
                return Err(AegisError::InvalidConfig(format!(
                    "unknown serve flag `{other}`"
                )));
            }
        }
        i += 1;
    }
    let mut serve = serve.unwrap_or_else(|| ServeConfig {
        host: "127.0.0.1".into(),
        port: 1337,
        api: "openai".into(),
        engine: crate::params::EngineConfigFragment {
            model_path: PathBuf::new(),
            policy: PlacementPolicy::auto_for(&inventory),
            cuda: CudaRuntimeConfig::from_env(),
            generation: SamplingConfig::default(),
        },
    });
    if let Some(host) = host {
        serve.host = host;
    }
    if let Some(port) = port {
        serve.port = port;
    }
    if let Some(model) = model {
        serve.engine.model_path = model;
    }
    if let Some(ctx_size) = ctx_size {
        serve.engine.policy.context_size = ctx_size;
    }
    if serve.engine.model_path.as_os_str().is_empty() {
        return Err(AegisError::InvalidConfig(
            "serve requires model.path in --config or --model".into(),
        ));
    }
    Ok(Command::Serve(serve))
}

fn parse_generate(args: &[String]) -> Result<Command> {
    let (config, request) = parse_generate_request(args, "generate")?;
    Ok(Command::Generate(config, request))
}

fn parse_quality_diff(args: &[String]) -> Result<Command> {
    // Strip out --reference PATH, leave the rest for parse_generate_request.
    let mut filtered = Vec::with_capacity(args.len());
    let mut reference_path: Option<std::path::PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--reference" {
            i += 1;
            if i >= args.len() {
                return Err(aegisllm_base::error::AegisError::InvalidConfig(
                    "quality-diff: --reference requires a path".into(),
                ));
            }
            reference_path = Some(std::path::PathBuf::from(&args[i]));
            i += 1;
        } else {
            filtered.push(args[i].clone());
            i += 1;
        }
    }
    let reference_path = reference_path.ok_or_else(|| {
        aegisllm_base::error::AegisError::InvalidConfig(
            "quality-diff requires --reference PATH (snapshot file written on first run, \
             diffed against on subsequent runs)"
                .into(),
        )
    })?;
    let (config, request) = parse_generate_request(&filtered, "quality-diff")?;
    Ok(Command::QualityDiff(config, request, reference_path))
}

fn parse_bench_generate(args: &[String]) -> Result<Command> {
    let mut generate_args = Vec::new();
    let mut prompt_repeat = 1;
    let mut prompt_repeats = Vec::new();
    let mut chunk_sizes = Vec::new();
    let mut warmup_runs = 0;
    let mut measured_runs = 1;
    let mut format = BenchOutputFormat::Text;

    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        match flag.as_str() {
            "--prompt-repeat" => prompt_repeat = parse_value(args, &mut i, flag)?,
            "--prompt-repeats" => {
                prompt_repeats = parse_usize_list(&take_value(args, &mut i, flag)?, flag)?
            }
            "--chunk-sizes" => {
                chunk_sizes = parse_usize_list(&take_value(args, &mut i, flag)?, flag)?
            }
            "--warmup" | "--warmup-runs" => warmup_runs = parse_value(args, &mut i, flag)?,
            "--runs" | "--measured-runs" => measured_runs = parse_value(args, &mut i, flag)?,
            "--format" => format = parse_bench_format(&take_value(args, &mut i, flag)?)?,
            other => {
                generate_args.push(flag.clone());
                if other == "--prompt"
                    || other == "--max-tokens"
                    || other == "--temp"
                    || other == "--temperature"
                    || other == "--top-k"
                    || other == "--top-p"
                    || other == "--warmup"
                    || other == "--warmup-runs"
                    || other == "--runs"
                    || other == "--measured-runs"
                    || (is_engine_flag(other) && flag_takes_value(other))
                {
                    generate_args.push(take_value(args, &mut i, flag)?);
                } else if !is_engine_flag(other) {
                    return Err(AegisError::InvalidConfig(format!(
                        "unknown bench-generate flag `{other}`"
                    )));
                }
            }
        }
        i += 1;
    }
    if prompt_repeat == 0 || prompt_repeats.contains(&0) {
        return Err(AegisError::InvalidConfig(
            "bench-generate requires prompt repeat values greater than 0".into(),
        ));
    }
    if measured_runs == 0 {
        return Err(AegisError::InvalidConfig(
            "bench-generate requires --runs greater than 0".into(),
        ));
    }

    let (config, mut request) = parse_generate_request(&generate_args, "bench-generate")?;
    if !prompt_repeats.is_empty() || !chunk_sizes.is_empty() {
        if prompt_repeats.is_empty() {
            prompt_repeats.push(prompt_repeat);
        }
        if chunk_sizes.is_empty() {
            chunk_sizes.push(config.cuda.prefill_chunk_size.unwrap_or(1));
        }
        return Ok(Command::BenchGenerateSweep(
            config,
            request,
            prompt_repeats,
            chunk_sizes,
            warmup_runs,
            measured_runs,
            format,
        ));
    }
    request.prompt = repeat_prompt(&request.prompt, prompt_repeat);
    Ok(Command::BenchGenerate(
        config,
        BenchGenerateRequest {
            generate: request,
            warmup_runs,
            measured_runs,
        },
        prompt_repeat,
        format,
    ))
}

fn parse_usize_list(value: &str, flag: &str) -> Result<Vec<usize>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>().map_err(|error| {
                AegisError::InvalidConfig(format!("bad {flag} item `{part}`: {error}"))
            })
        })
        .collect()
}

fn parse_perplexity(args: &[String]) -> Result<Command> {
    use crate::engine::perplexity::PerplexityRequest;
    let mut text: Option<String> = None;
    let mut text_file: Option<PathBuf> = None;
    let mut max_tokens: Option<usize> = None;
    let mut context_tokens: Option<usize> = None;
    let mut apply_chat_template = false;
    let mut filtered: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        match flag.as_str() {
            "--text" => {
                text = Some(take_value(args, &mut i, flag)?);
            }
            "--text-file" => {
                text_file = Some(PathBuf::from(take_value(args, &mut i, flag)?));
            }
            "--max-tokens" => {
                max_tokens = Some(parse_value(args, &mut i, flag)?);
            }
            "--context-tokens" => {
                context_tokens = Some(parse_value(args, &mut i, flag)?);
            }
            "--apply-chat-template" => {
                apply_chat_template = true;
            }
            other if is_engine_flag(other) => {
                filtered.push(args[i].clone());
                if flag_takes_value(other) {
                    i += 1;
                    if i >= args.len() {
                        return Err(AegisError::InvalidConfig(format!(
                            "perplexity flag `{other}` requires a value"
                        )));
                    }
                    filtered.push(args[i].clone());
                }
            }
            other => {
                return Err(AegisError::InvalidConfig(format!(
                    "unknown perplexity flag `{other}`"
                )));
            }
        }
        i += 1;
    }
    if text.is_some() && text_file.is_some() {
        return Err(AegisError::InvalidConfig(
            "perplexity: pass either --text or --text-file, not both".into(),
        ));
    }
    let resolved_text = match (text, text_file) {
        (Some(s), _) => Some(s),
        (None, Some(path)) => Some(std::fs::read_to_string(&path).map_err(|e| {
            AegisError::InvalidConfig(format!(
                "perplexity: cannot read --text-file {}: {e}",
                path.display()
            ))
        })?),
        _ => None,
    };
    let config = parse_engine_flags(&filtered)?.engine_config(true);
    Ok(Command::Perplexity(
        config,
        PerplexityRequest {
            text: resolved_text,
            max_tokens,
            context_tokens,
            raw_text: !apply_chat_template,
        },
    ))
}

fn parse_sample_diversity(args: &[String]) -> Result<Command> {
    use crate::engine::sample_diversity::SampleDiversityRequest;
    let mut prompt: Option<String> = None;
    let mut runs: usize = 10;
    let mut max_tokens: usize = 12;
    // Per-CLI overrides for sampling. If unset, fall through to the values
    // baked into the parsed config (parameters.json `other-parameters`).
    let mut temperature: Option<f32> = None;
    let mut top_k: Option<usize> = None;
    let mut top_p: Option<f32> = None;
    let mut min_p: Option<f32> = None;
    let mut filtered: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        match flag.as_str() {
            "--prompt" => prompt = Some(take_value(args, &mut i, flag)?),
            "--runs" => runs = parse_value(args, &mut i, flag)?,
            "--max-tokens" => max_tokens = parse_value(args, &mut i, flag)?,
            "--temperature" => temperature = Some(parse_value(args, &mut i, flag)?),
            "--top-k" => top_k = Some(parse_value(args, &mut i, flag)?),
            "--top-p" => top_p = Some(parse_value(args, &mut i, flag)?),
            "--min-p" => min_p = Some(parse_value(args, &mut i, flag)?),
            other if is_engine_flag(other) => {
                filtered.push(args[i].clone());
                if flag_takes_value(other) {
                    i += 1;
                    if i >= args.len() {
                        return Err(AegisError::InvalidConfig(format!(
                            "sample-diversity flag `{other}` requires a value"
                        )));
                    }
                    filtered.push(args[i].clone());
                }
            }
            other => {
                return Err(AegisError::InvalidConfig(format!(
                    "unknown sample-diversity flag `{other}`"
                )));
            }
        }
        i += 1;
    }
    let prompt = prompt.ok_or_else(|| {
        AegisError::InvalidConfig("sample-diversity requires --prompt <text>".into())
    })?;
    let parsed = parse_engine_flags(&filtered)?;
    let mut sampling = parsed.generation;
    if let Some(v) = temperature {
        sampling.temperature = v;
    }
    if let Some(v) = top_k {
        sampling.top_k = v;
    }
    if let Some(v) = top_p {
        sampling.top_p = v;
    }
    if let Some(v) = min_p {
        sampling.min_p = v;
    }
    let config = parsed.engine_config(true);
    Ok(Command::SampleDiversity(
        config,
        SampleDiversityRequest {
            prompt,
            runs,
            max_tokens,
            sampling: Some(sampling),
        },
    ))
}

fn parse_bench_format(value: &str) -> Result<BenchOutputFormat> {
    match value {
        "text" => Ok(BenchOutputFormat::Text),
        "json" => Ok(BenchOutputFormat::Json),
        "csv" => Ok(BenchOutputFormat::Csv),
        other => Err(AegisError::InvalidConfig(format!(
            "unsupported bench output format `{other}`; expected text|json|csv"
        ))),
    }
}

fn parse_generate_request(
    args: &[String],
    command: &str,
) -> Result<(EngineConfig, GenerateRequest)> {
    let mut engine_args = Vec::new();
    let mut prompt = None;
    let mut max_tokens = 32;
    let mut temperature = None;
    let mut top_k = None;
    let mut top_p = None;

    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        match flag.as_str() {
            "--prompt" => prompt = Some(take_value(args, &mut i, flag)?),
            "--max-tokens" => max_tokens = parse_value(args, &mut i, flag)?,
            "--temp" | "--temperature" => temperature = Some(parse_value(args, &mut i, flag)?),
            "--top-k" => top_k = Some(parse_value(args, &mut i, flag)?),
            "--top-p" => top_p = Some(parse_value(args, &mut i, flag)?),
            other if is_engine_flag(other) => {
                engine_args.push(flag.clone());
                if flag_takes_value(other) {
                    engine_args.push(take_value(args, &mut i, flag)?);
                }
            }
            other => {
                return Err(AegisError::InvalidConfig(format!(
                    "unknown generate flag `{other}`"
                )));
            }
        }
        i += 1;
    }

    let flags = parse_engine_flags(&engine_args)?;
    let mut sampling = flags.generation;
    if let Some(value) = temperature {
        sampling.temperature = value;
    }
    if let Some(value) = top_k {
        sampling.top_k = value;
    }
    if let Some(value) = top_p {
        sampling.top_p = value;
    }
    let prompt =
        prompt.ok_or_else(|| AegisError::InvalidConfig(format!("{command} requires --prompt")))?;
    Ok((
        flags.engine_config(true),
        GenerateRequest {
            prompt,
            max_tokens,
            sampling,
            stop_token_ids: Vec::new(),
        },
    ))
}

fn repeat_prompt(prompt: &str, repeat: usize) -> String {
    std::iter::repeat_n(prompt, repeat)
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_gates(args: &[String]) -> Result<Command> {
    use crate::cli::gates::{GatesBackend, GatesConfig, GatesMode};

    let mut backend = GatesBackend::Cuda;
    let mut mode = GatesMode::Quick;

    // First pass: extract gates-specific flags.
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| AegisError::InvalidConfig("missing value for --backend".into()))?;
                backend = match val.as_str() {
                    "cpu" => GatesBackend::Cpu,
                    "cuda" => GatesBackend::Cuda,
                    other => {
                        return Err(AegisError::InvalidConfig(format!(
                            "--backend must be cpu|cuda, got `{other}`"
                        )))
                    }
                };
            }
            "--quick" => mode = GatesMode::Quick,
            "--full" => mode = GatesMode::Full,
            _ => {}
        }
        i += 1;
    }

    // Second pass: strip gates-specific flags for engine-flag parser.
    let mut engine_args = args.to_vec();
    engine_args.retain(|a| a != "--quick" && a != "--full");
    if let Some(pos) = engine_args.iter().position(|a| a == "--backend") {
        if pos + 1 < engine_args.len() {
            engine_args.remove(pos + 1);
        }
        engine_args.remove(pos);
    }

    let engine_config = parse_engine_flags(&engine_args)?.engine_config(true);
    Ok(Command::Gates(engine_config, GatesConfig { backend, mode }))
}

fn usage() -> String {
    "usage:\n  aegisllm inspect-hardware\n  aegisllm serve --config <parameters.json>\n  aegisllm show-plan --config <parameters.json>\n  aegisllm mvp-check --config <parameters.json>\n  aegisllm quality-smoke --config <parameters.json>\n  aegisllm storage-smoke --config <parameters.json>\n  aegisllm cpu-smoke --config <parameters.json>\n  aegisllm cpu-materialize-smoke --config <parameters.json>\n  aegisllm cuda-smoke --config <parameters.json>\n  aegisllm cuda-dense-smoke --config <parameters.json>\n  aegisllm cuda-chain-smoke --config <parameters.json>\n  aegisllm cuda-compare --config <parameters.json>\n  aegisllm cuda-prefill-compare --config <parameters.json>\n  aegisllm cuda-prefill-sweep --config <parameters.json>\n  aegisllm show-plan --model <path> [placement flags] [--native-mxfp4-repack] [--native-mxfp4-inference] [--cuda-prefill-attention auto|off|fa2|fa3|fa4|aegis-varlen] [--cuda-prefill-chunk-size N]\n  aegisllm generate --model <path> --prompt <text> [--max-tokens N] [placement flags]\n  aegisllm bench-generate --config <parameters.json> --prompt <text> [--prompt-repeat N] [--max-tokens N] [--temperature T] [--format text|json|csv]\n  aegisllm gates --model <path> [--backend cpu|cuda] [--quick|--full]".into()
}
