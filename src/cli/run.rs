use std::env;

use super::generate::print_generate_bench;
use super::smoke::{
    cpu_materialize_smoke, cpu_smoke, cuda_chain_smoke, cuda_compare, cuda_dense_smoke,
    cuda_prefill_compare, cuda_prefill_sweep, cuda_smoke, inspect_hardware, mvp_check,
    quality_smoke, storage_smoke,
};
use super::{Command, parse_args};
use crate::engine::bench::run_generation_bench;
use crate::engine::{AegisEngine, EngineConfig};
use crate::error::Result;
use crate::executor::readiness_for_plan;

pub fn run_env() -> Result<()> {
    match parse_args(env::args().skip(1))? {
        Command::InspectHardware => inspect_hardware(),
        Command::ShowPlan(config) => {
            let engine = AegisEngine::build(config)?;
            print!("{}", engine.report());
        }
        Command::MvpCheck(config) => mvp_check(config)?,
        Command::QualitySmoke(config) => quality_smoke(config)?,
        Command::StorageSmoke(config) => storage_smoke(config)?,
        Command::CpuSmoke(config) => cpu_smoke(config)?,
        Command::CpuMaterializeSmoke(config) => cpu_materialize_smoke(config)?,
        Command::CudaSmoke(config) => cuda_smoke(config)?,
        Command::CudaDenseSmoke(config) => cuda_dense_smoke(config)?,
        Command::CudaChainSmoke(config) => cuda_chain_smoke(config)?,
        Command::CudaCompare(config) => cuda_compare(config)?,
        Command::CudaPrefillCompare(config) => cuda_prefill_compare(config)?,
        Command::CudaPrefillSweep(config) => cuda_prefill_sweep(config)?,
        Command::Generate(config, request) => {
            let engine = AegisEngine::build(config)?;
            let output = engine.generate(request)?;
            println!("{}", output.text);
            eprintln!(
                "finish={} prompt_tokens={} completion_tokens={}",
                output.finish_reason, output.prompt_tokens, output.completion_tokens
            );
        }
        Command::BenchGenerate(config, request, prompt_repeat, format) => {
            let metrics = run_generation_bench(config, request)?;
            print_generate_bench(&metrics, prompt_repeat, format);
        }
        Command::Serve(config) => {
            let default_sampling = config.engine.generation;
            let engine_config = EngineConfig {
                model_path: config.engine.model_path,
                policy: config.engine.policy,
                enable_executor: false,
                cuda: config.engine.cuda,
            };
            let preview = AegisEngine::build(engine_config.clone())?;
            let readiness = readiness_for_plan(&preview.placement, &preview.runtime);
            let engine = if readiness.runnable {
                AegisEngine::build(EngineConfig {
                    enable_executor: true,
                    ..engine_config
                })?
            } else {
                preview
            };
            eprintln!("{}", engine.report());
            crate::server::serve_http(
                config.host,
                config.port,
                config.api,
                engine,
                readiness,
                default_sampling,
            )?;
        }
    }
    Ok(())
}
