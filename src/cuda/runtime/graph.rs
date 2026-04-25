#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub struct CudaGraphShapeKey {
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub sequences: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CudaGraphReplayMode {
    Disabled,
    Capture,
    Replay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct CudaGraphPolicy {
    pub mode: CudaGraphReplayMode,
    pub min_replay_tokens: usize,
}

impl Default for CudaGraphPolicy {
    fn default() -> Self {
        Self {
            mode: CudaGraphReplayMode::Disabled,
            min_replay_tokens: 1,
        }
    }
}
