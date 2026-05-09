//! Load progress indicator. On a TTY, overwrites itself in place with `\r`
//! and clears with a trailing newline on the final step. When stderr is
//! piped/redirected (containers, `2>file`, `nohup`, subprocess.run), prints
//! one line per step instead — keeps the same information visible in logs
//! without writing carriage returns that would render as garbage.

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

    /// Advance one step and emit a progress line. TTY: in-place via `\r`
    /// + trailing `\n` on the last step. Non-TTY: one line per step.
    pub(super) fn step(&self, label: &str) {
        let n = self.current.fetch_add(1, Ordering::Relaxed) + 1;
        let pct = (n * 100) / self.total;
        let mut err = std::io::stderr().lock();
        if self.is_tty {
            let _ = write!(
                err,
                "\rLoading model: {n}/{total} ({pct:>3}%) {label:<24}",
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
}
