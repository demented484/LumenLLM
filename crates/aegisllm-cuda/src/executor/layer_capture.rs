//! Per-layer post-attention hidden-state capture for the `cuda-attn-compare`
//! correctness gate (Stage A.3 of the attention-backend rewrite).
//!
//! The prefill layer code (`prefill/layer.rs`) calls [`capture_post_attn`]
//! after each layer's post-attention residual add. When capture is armed
//! (via [`arm`]), it downloads the row-0 hidden slice for every layer into a
//! thread-local buffer. The `cuda-attn-compare` CLI subcommand arms capture,
//! runs a prefill, then drains the buffer with [`take`] to diff the reference
//! attention backend against a fast backend layer by layer.
//!
//! When capture is not armed (every normal run — perplexity, serve, generate)
//! the hook is a single relaxed atomic load and a cheap early return, so it is
//! perf-neutral for production paths.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global arm flag. Checked first in [`capture_post_attn`] so the common
/// (disarmed) case costs one relaxed atomic load and nothing else.
static ARMED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Per-layer captured hidden slices, in layer order. Each entry is the
    /// row-0 (first token) post-attention residual for one layer.
    static CAPTURED: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

/// Arm capture and clear any previously captured layers. Call before a
/// prefill whose per-layer hidden states should be recorded.
pub fn arm() {
    CAPTURED.with(|c| c.borrow_mut().clear());
    ARMED.store(true, Ordering::Relaxed);
}

/// Disarm capture. Subsequent [`capture_post_attn`] calls become no-ops.
pub fn disarm() {
    ARMED.store(false, Ordering::Relaxed);
}

/// Whether capture is currently armed.
pub fn is_armed() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// Drain and return the captured per-layer hidden slices, leaving the
/// thread-local buffer empty.
pub fn take() -> Vec<Vec<f32>> {
    CAPTURED.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// Record one layer's post-attention hidden slice. No-op unless capture is
/// armed. `row0` should be the first-token hidden vector (length = hidden
/// size). Called from the prefill layer hook.
pub fn capture_post_attn(row0: &[f32]) {
    if !ARMED.load(Ordering::Relaxed) {
        return;
    }
    CAPTURED.with(|c| c.borrow_mut().push(row0.to_vec()));
}
