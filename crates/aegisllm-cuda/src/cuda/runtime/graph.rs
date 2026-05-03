use cudarc::driver::{CudaGraph, sys};

use super::{CudaRuntime, map_cuda_err};
use aegisllm_base::error::Result;

impl CudaRuntime {
    /// Start capturing all subsequent kernel launches on this stream into a CUDA Graph.
    pub fn begin_decode_graph_capture(&self) -> Result<()> {
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(map_cuda_err("begin graph capture"))
    }

    /// Stop capturing and return the compiled CUDA Graph ready for replay.
    pub fn end_decode_graph_capture(&self) -> Result<Option<CudaGraph>> {
        // 0 = no special instantiation flags
        let flags = unsafe { std::mem::transmute::<u32, sys::CUgraphInstantiate_flags>(0u32) };
        self.stream
            .end_capture(flags)
            .map_err(map_cuda_err("end graph capture"))
    }

    /// Replay a previously captured decode graph. All device buffers must still be live
    /// and at the same addresses as when the graph was captured.
    pub fn replay_decode_graph(&self, graph: &CudaGraph) -> Result<()> {
        graph.launch().map_err(map_cuda_err("replay decode graph"))
    }
}
