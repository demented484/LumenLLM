use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrefillBudget {
    pub(super) max_prefill_tokens: usize,
    pub(super) max_decode_tokens: usize,
    pub(super) max_sequences: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefillRequestState {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PrefillRequest {
    pub(super) request_id: u64,
    pub(super) seq_id: u64,
    pub(super) prompt_tokens: usize,
    pub(super) decoded_tokens: usize,
    pub(super) state: PrefillRequestState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PrefillScheduleDecision {
    pub(super) prefill_request_ids: Vec<u64>,
    pub(super) decode_request_ids: Vec<u64>,
    pub(super) spans: Vec<ScheduledQuerySpan>,
    pub(super) prefill_tokens: usize,
    pub(super) decode_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScheduledQueryKind {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ScheduledQuerySpan {
    pub(super) request_id: u64,
    pub(super) seq_id: u64,
    pub(super) kind: ScheduledQueryKind,
    pub(super) tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PrefillScheduler {
    budget: PrefillBudget,
}

impl PrefillScheduler {
    pub(super) fn new(budget: PrefillBudget) -> Self {
        Self { budget }
    }

    pub(super) fn schedule(&self, requests: &[PrefillRequest]) -> PrefillScheduleDecision {
        let mut decision = PrefillScheduleDecision {
            prefill_request_ids: Vec::new(),
            decode_request_ids: Vec::new(),
            spans: Vec::new(),
            prefill_tokens: 0,
            decode_tokens: 0,
        };
        for request in requests {
            if decision.prefill_request_ids.len() + decision.decode_request_ids.len()
                >= self.budget.max_sequences
            {
                break;
            }
            match request.state {
                PrefillRequestState::Waiting | PrefillRequestState::Prefilling => {
                    let remaining = self.budget.max_prefill_tokens - decision.prefill_tokens;
                    if remaining == 0 {
                        continue;
                    }
                    let take = request.prompt_tokens.min(remaining);
                    decision.prefill_tokens += take;
                    decision.prefill_request_ids.push(request.request_id);
                    decision.spans.push(ScheduledQuerySpan {
                        request_id: request.request_id,
                        seq_id: request.seq_id,
                        kind: ScheduledQueryKind::Prefill,
                        tokens: take,
                    });
                }
                PrefillRequestState::Decoding => {
                    if decision.decode_tokens < self.budget.max_decode_tokens {
                        decision.decode_tokens += 1;
                        decision.decode_request_ids.push(request.request_id);
                        decision.spans.push(ScheduledQuerySpan {
                            request_id: request.request_id,
                            seq_id: request.seq_id,
                            kind: ScheduledQueryKind::Decode,
                            tokens: 1,
                        });
                    }
                }
                PrefillRequestState::Finished | PrefillRequestState::Cancelled => {}
            }
        }
        decision
    }
}

/// Per-request cancellation token.  Set to `true` to signal the decode loop
/// to stop after the current token.  Checked in `generate_streaming`.
#[derive(Debug, Clone)]
pub(super) struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub(super) fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub(super) fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub(super) fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub(super) fn clone_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0)
    }
}

/// Lifecycle wrapper for a single-request session passing through the scheduler.
/// Admitted via `admit`, advanced by `next_step`, completed by `finish`.
#[derive(Debug)]
pub(super) struct SingleRequestSession {
    pub(super) request: PrefillRequest,
    pub(super) cancel: CancelToken,
    scheduler: PrefillScheduler,
}

impl SingleRequestSession {
    /// Wrap a new request in a scheduler session.
    pub(super) fn admit(
        request_id: u64,
        seq_id: u64,
        prompt_tokens: usize,
        prefill_chunk_size: usize,
        max_decode_tokens: usize,
    ) -> Self {
        let scheduler = PrefillScheduler::new(PrefillBudget {
            max_prefill_tokens: prefill_chunk_size,
            max_decode_tokens,
            max_sequences: 1,
        });
        let request = PrefillRequest {
            request_id,
            seq_id,
            prompt_tokens,
            decoded_tokens: 0,
            state: PrefillRequestState::Waiting,
        };
        Self {
            request,
            cancel: CancelToken::new(),
            scheduler,
        }
    }

    /// Advance request state: Waiting → Prefilling → Decoding.
    /// Returns the scheduled decision for this step.
    pub(super) fn next_step(&mut self) -> PrefillScheduleDecision {
        match self.request.state {
            PrefillRequestState::Waiting => {
                self.request.state = PrefillRequestState::Prefilling;
            }
            PrefillRequestState::Prefilling => {
                self.request.state = PrefillRequestState::Decoding;
            }
            PrefillRequestState::Decoding => {
                self.request.decoded_tokens += 1;
            }
            _ => {}
        }
        self.scheduler.schedule(std::slice::from_ref(&self.request))
    }

    pub(super) fn finish(&mut self) {
        self.request.state = PrefillRequestState::Finished;
    }

    pub(super) fn cancel(&mut self) {
        self.cancel.cancel();
        self.request.state = PrefillRequestState::Cancelled;
    }

    pub(super) fn is_active(&self) -> bool {
        matches!(
            self.request.state,
            PrefillRequestState::Waiting
                | PrefillRequestState::Prefilling
                | PrefillRequestState::Decoding
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PrefillBudget, PrefillRequest, PrefillRequestState, PrefillScheduler,
        ScheduledQueryKind, SingleRequestSession,
    };

    #[test]
    fn scheduler_respects_token_and_sequence_budgets() {
        let scheduler = PrefillScheduler::new(PrefillBudget {
            max_prefill_tokens: 10,
            max_decode_tokens: 2,
            max_sequences: 3,
        });
        let decision = scheduler.schedule(&[
            PrefillRequest {
                request_id: 1,
                seq_id: 1,
                prompt_tokens: 8,
                decoded_tokens: 0,
                state: PrefillRequestState::Waiting,
            },
            PrefillRequest {
                request_id: 2,
                seq_id: 2,
                prompt_tokens: 8,
                decoded_tokens: 0,
                state: PrefillRequestState::Waiting,
            },
            PrefillRequest {
                request_id: 3,
                seq_id: 3,
                prompt_tokens: 0,
                decoded_tokens: 4,
                state: PrefillRequestState::Decoding,
            },
        ]);
        assert_eq!(decision.prefill_request_ids, [1, 2]);
        assert_eq!(decision.prefill_tokens, 10);
        assert_eq!(decision.decode_request_ids, [3]);
        assert_eq!(decision.decode_tokens, 1);
        assert_eq!(decision.spans.len(), 3);
        assert_eq!(decision.spans[0].kind, ScheduledQueryKind::Prefill);
        assert_eq!(decision.spans[2].kind, ScheduledQueryKind::Decode);
    }

    #[test]
    fn single_request_session_advances_state_correctly() {
        let mut session = SingleRequestSession::admit(1, 1, 64, 128, 1024);
        assert!(session.is_active());
        assert!(!session.cancel.is_cancelled());

        // First step: Waiting → Prefilling; schedule sees Prefilling → prefill slot
        let d = session.next_step();
        assert_eq!(d.prefill_request_ids, [1]);
        assert!(d.decode_request_ids.is_empty());

        // Second step: Prefilling → Decoding; schedule sees Decoding → decode slot
        let d = session.next_step();
        assert!(d.prefill_request_ids.is_empty());
        assert_eq!(d.decode_request_ids, [1]);

        // Decode step increments decoded_tokens
        let d = session.next_step();
        assert_eq!(d.decode_request_ids, [1]);
        assert_eq!(session.request.decoded_tokens, 1);

        // Cancellation marks session as inactive
        session.cancel();
        assert!(!session.is_active());
        assert!(session.cancel.is_cancelled());
    }
}
