//! Load progress indicator. On a TTY, overwrites itself in place with `\r`
//! and clears with a trailing newline on the final step. When stderr is
//! piped/redirected (containers, `2>file`, `nohup`, subprocess.run), prints
//! one line per step instead — keeps the same information visible in logs
//! without writing carriage returns that would render as garbage.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct LoadProgress {
    total: usize,
    current: AtomicUsize,
    is_tty: bool,
}

impl LoadProgress {
    pub(crate) fn new(total: usize) -> Self {
        Self {
            total: total.max(1),
            current: AtomicUsize::new(0),
            is_tty: std::io::stderr().is_terminal(),
        }
    }

    /// True when stderr is attached to a terminal — callers can use this to
    /// suppress redundant per-stage diagnostic prints that would push the
    /// in-place progress bar off-screen.
    pub(crate) fn is_tty(&self) -> bool {
        self.is_tty
    }

    /// Advance one step and emit a progress line. TTY: in-place via `\r`
    /// + trailing `\n` on the last step. Non-TTY: one line per step.
    pub(crate) fn step(&self, label: &str) {
        let n = self.current.fetch_add(1, Ordering::Relaxed) + 1;
        let pct = (n * 100) / self.total;
        let mut err = std::io::stderr().lock();
        if self.is_tty {
            let _ = write!(
                err,
                "\rLoading model: {n}/{total} ({pct:>3}%) {label:<40}",
                total = self.total
            );
            if n >= self.total {
                let _ = writeln!(err);
            }
        } else {
            let _ = writeln!(
                err,
                "Loading model: {n}/{total} ({pct:>3}%) {label}",
                total = self.total
            );
        }
        let _ = err.flush();
    }

    /// Update the displayed status text without advancing the step counter.
    /// Used by long-running sub-stages (e.g. MoE expert loops, big BF16
    /// uploads) so the user sees activity between `step` calls instead of
    /// the bar appearing frozen — particularly important for offloaded
    /// (host-resident) layers, where one "step" can take many seconds.
    ///
    /// TTY only: rewrites the current line in-place. On non-TTY (logs,
    /// pipes) this is a no-op so we don't flood the log with intermediate
    /// status updates — non-TTY users still see the per-step lines.
    pub(crate) fn tick(&self, label: &str) {
        if !self.is_tty {
            return;
        }
        let n = self.current.load(Ordering::Relaxed);
        let pct = (n * 100) / self.total;
        let mut err = std::io::stderr().lock();
        let _ = write!(
            err,
            "\rLoading model: {n}/{total} ({pct:>3}%) {label:<40}",
            total = self.total
        );
        let _ = err.flush();
    }
}
