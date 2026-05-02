use std::time::Instant;

use crate::cuda::CudaRuntime;
use aegisllm_base::error::Result;
use crate::executor::state::CudaPrefillStageTimings;

pub(super) fn record_prefill_stage(
    runtime: &CudaRuntime,
    timings: &mut CudaPrefillStageTimings,
    start: Instant,
    apply: impl FnOnce(&mut CudaPrefillStageTimings, u128),
) -> Result<()> {
    if timings.enabled {
        runtime.synchronize()?;
        apply(timings, start.elapsed().as_micros());
    }
    Ok(())
}

pub(super) fn print_prefill_stage_timings(timings: CudaPrefillStageTimings) {
    if timings.enabled {
        eprintln!(
            "cuda-prefill-stages: chunks={} prepare_us={} embed_us={} qkv_us={} qkv_tflops={:.3} rope_us={} kv_store_us={} attention_us={} o_proj_us={} mlp_us={} mlp_tflops={:.3} layer_total_us={} sample_us={}",
            timings.chunks,
            timings.prepare_us,
            timings.embed_us,
            timings.qkv_us,
            timings.qkv_tflops,
            timings.rope_us,
            timings.kv_store_us,
            timings.attention_us,
            timings.o_proj_us,
            timings.mlp_us,
            timings.mlp_tflops,
            timings.layer_total_us,
            timings.sample_us
        );
    }
}
