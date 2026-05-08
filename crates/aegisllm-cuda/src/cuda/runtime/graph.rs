use cudarc::driver::{CudaGraph, sys};

use super::{CudaRuntime, map_cuda_err};
use aegisllm_base::error::{AegisError, Result};

impl CudaRuntime {
    /// Start capturing all subsequent kernel launches on this stream into a CUDA Graph.
    pub fn begin_decode_graph_capture(&self) -> Result<()> {
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(map_cuda_err("begin graph capture"))
    }

    /// Stop capturing and return the compiled CUDA Graph ready for replay.
    pub fn end_decode_graph_capture(&self) -> Result<Option<CudaGraph>> {
        // `AUTO_FREE_ON_LAUNCH` only affects graphs containing memory-
        // allocation nodes (`cuGraphAddMemAllocNode`). Our captures
        // don't allocate, so this is effectively a no-op flag — but it
        // gives us a real `CUgraphInstantiate_flags` variant to pass,
        // avoiding the soundness hole of transmuting a literal 0u32 into
        // a Rust `#[repr(u32)] enum` whose first variant is `1`.
        let flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
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

/// One node from a captured graph, with its handle and kind.
///
/// Node handles refer into the underlying `CUgraph` returned by capture; they
/// remain valid for the lifetime of that graph and of any `CUgraphExec`
/// instantiated from it.
///
/// API is wired but not yet consumed by the executor — the next step
/// (parameterised decode replay) is what will use it. Suppress `dead_code`
/// in the meantime.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct GraphNode {
    pub(crate) handle: sys::CUgraphNode,
    pub(crate) kind: GraphNodeKind,
}

/// The subset of `CUgraphNodeType` we care about for parameterised replay.
/// Other node types are mapped to `Other` and ignored by the param-update API.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphNodeKind {
    Kernel,
    Memcpy,
    Memset,
    Other,
}

/// Walk a captured graph and return every node with its classification.
///
/// Used by the parameterised-replay path: after capture we record the handle
/// of each kernel/memcpy node we plan to mutate per-token (e.g. expert
/// weight pointers, staging-slot memcpy sources), and on each replay we
/// patch those nodes via [`set_kernel_node_params_in_exec`] /
/// [`set_memcpy_node_params_in_exec`] before launching.
///
/// Order is implementation-defined by the driver; callers that need to
/// correlate nodes with their high-level meaning must do so by capturing
/// node handles inline as kernels are launched (see CUDA Graphs docs §3.2.6.6).
#[allow(dead_code)]
pub(crate) fn enumerate_graph_nodes(graph: &CudaGraph) -> Result<Vec<GraphNode>> {
    let cu_graph = graph.cu_graph();
    // First call with NULL nodes to query the count, per CUDA driver API.
    let mut count: usize = 0;
    let rc = unsafe { sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut count) };
    cu_check("cuGraphGetNodes (count)", rc)?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut handles: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); count];
    let rc = unsafe { sys::cuGraphGetNodes(cu_graph, handles.as_mut_ptr(), &mut count) };
    cu_check("cuGraphGetNodes (fetch)", rc)?;
    handles.truncate(count);
    let mut out = Vec::with_capacity(count);
    for handle in handles {
        let mut ty: sys::CUgraphNodeType =
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
        let rc = unsafe { sys::cuGraphNodeGetType(handle, &mut ty) };
        cu_check("cuGraphNodeGetType", rc)?;
        let kind = match ty {
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => GraphNodeKind::Kernel,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => GraphNodeKind::Memcpy,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => GraphNodeKind::Memset,
            _ => GraphNodeKind::Other,
        };
        out.push(GraphNode { handle, kind });
    }
    Ok(out)
}

/// Update a kernel node's parameters in the `CUgraphExec` instantiated
/// from this `CudaGraph`. Mutating the source graph itself is not the
/// CUDA Graphs flow — instead the `*Exec*SetParams` family lets you
/// swap kernel/memcpy arguments on the *executable* between replays.
///
/// Use to swap the device-pointer kernel arguments of a captured matvec/etc.
/// kernel between replays — e.g. point a captured expert GEMM at a different
/// expert's weights in VRAM staging without re-capturing the whole graph.
///
/// Safety:
///  * `params.kernelParams` is a `void**` pointing at an array of pointers,
///    each of which points at one kernel argument. The driver reads through
///    this array and the pointed-to argument slots once, during this call
///    — neither the array nor the slot pointees need to outlive the call.
///    They DO need to be valid for the duration of the call.
///  * `params.func` must be the same `CUfunction` (or one that's binary-
///    compatible: same signature, register usage, shared-memory budget) as
///    the kernel originally captured into `node`. Mismatches return
///    `CUDA_ERROR_INVALID_VALUE` at instantiation, but some driver
///    versions accept the mismatch and corrupt at replay time.
///  * `params.gridDim` / `blockDim` may change between replays, but
///    `sharedMemBytes` must not exceed what the original capture
///    requested via `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`.
///  * `node` must have been returned by [`enumerate_graph_nodes`] for the
///    graph that was instantiated into the exec held by `graph`. Passing
///    a node from a different graph corrupts state.
#[allow(dead_code)]
pub(crate) unsafe fn set_kernel_node_params_in_exec(
    graph: &CudaGraph,
    node: sys::CUgraphNode,
    params: &sys::CUDA_KERNEL_NODE_PARAMS,
) -> Result<()> {
    let cu_exec = graph.cu_graph_exec();
    // CUDA 12+ uses _v2; the build-system version selects the right
    // `CUDA_KERNEL_NODE_PARAMS` typedef so the v2 entry matches.
    let rc = unsafe { sys::cuGraphExecKernelNodeSetParams_v2(cu_exec, node, params) };
    cu_check("cuGraphExecKernelNodeSetParams_v2", rc)
}

/// Update a memcpy node's parameters in an instantiated graph.
///
/// Use to swap the source/destination of a captured H2D copy between
/// replays — e.g. point a captured expert weight upload at a different
/// expert's row in pinned host memory without re-capturing.
///
/// Safety: see [`set_kernel_node_params_in_exec`]. The memory pointed to by
/// the `srcDevice`/`dstDevice` (or `srcHost`/`dstHost`) fields of `params`
/// must outlive every replay that uses this node.
#[allow(dead_code)]
pub(crate) unsafe fn set_memcpy_node_params_in_exec(
    graph: &CudaGraph,
    ctx: &cudarc::driver::CudaContext,
    node: sys::CUgraphNode,
    params: &sys::CUDA_MEMCPY3D,
) -> Result<()> {
    let cu_exec = graph.cu_graph_exec();
    let cu_ctx = ctx.cu_ctx();
    let rc = unsafe { sys::cuGraphExecMemcpyNodeSetParams(cu_exec, node, params, cu_ctx) };
    cu_check("cuGraphExecMemcpyNodeSetParams", rc)
}

fn cu_check(label: &'static str, rc: sys::CUresult) -> Result<()> {
    if rc == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(AegisError::Unsupported(format!("{label}: CUresult={rc:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    #[test]
    fn enumerate_returns_empty_for_empty_capture() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("skip: no CUDA device");
            return;
        };
        // Stream capture is not allowed on the legacy/default stream — use a
        // dedicated user stream.
        let stream = ctx.new_stream().expect("new stream");
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .expect("begin capture");
        let flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph = stream
            .end_capture(flags)
            .expect("end capture")
            .expect("graph from empty capture");
        let nodes = enumerate_graph_nodes(&graph).expect("enumerate empty");
        assert!(nodes.is_empty(), "expected zero nodes, got {}", nodes.len());
    }
}
