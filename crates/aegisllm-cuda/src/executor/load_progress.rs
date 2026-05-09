//! Single-line in-place progress indicator for the CUDA model load.
//!
//! Prints `Loading model: 18/33 (54%) layer 15` and overwrites itself with
//! `\r` until the final step, when it emits a trailing newline. Goes to
//! stderr so it doesn't interleave with the planner's stdout `region: …`
//! lines, and stays silent when stderr is not a TTY (so `2>file` logs and
//! containers don't accumulate carriage-return lines).

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct LoadProgress {
    total: usize,
    current: AtomicUsize,
    is_tty: bool,
}

impl LoadProgress {
    pub(super) fn new(total: usize) -> Self {
        Self {
            total: total.max(1),
            current: AtomicUsize::new(0),
            is_tty: std::io::stderr().is_terminal(),
        }
    }

    /// Advance one step, render the line, and emit a trailing newline on the
    /// final step. `label` is the per-step descriptor (e.g. `"embed"`,
    /// `"layer 15"`).
    pub(super) fn step(&self, label: &str) {
        let n = self.current.fetch_add(1, Ordering::Relaxed) + 1;
        if !self.is_tty {
            return;
        }
        let pct = (n * 100) / self.total;
        let mut err = std::io::stderr().lock();
        // Pad to clear any leftover characters from a previous longer label.
        let _ = write!(err, "\rLoading model: {n}/{total} ({pct:>3}%) {label:<24}",
            total = self.total);
        if n >= self.total {
            let _ = writeln!(err);
        }
        let _ = err.flush();
    }
}
