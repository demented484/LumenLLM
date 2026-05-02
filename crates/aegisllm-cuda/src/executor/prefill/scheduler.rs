#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct PrefillBudget {
    pub(super) max_prefill_tokens: usize,
    pub(super) max_decode_tokens: usize,
    pub(super) max_sequences: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum PrefillRequestState {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct PrefillRequest {
    pub(super) request_id: u64,
    pub(super) seq_id: u64,
    pub(super) prompt_tokens: usize,
    pub(super) decoded_tokens: usize,
    pub(super) state: PrefillRequestState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct PrefillScheduleDecision {
    pub(super) prefill_request_ids: Vec<u64>,
    pub(super) decode_request_ids: Vec<u64>,
    pub(super) spans: Vec<ScheduledQuerySpan>,
    pub(super) prefill_tokens: usize,
    pub(super) decode_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum ScheduledQueryKind {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct ScheduledQuerySpan {
    pub(super) request_id: u64,
    pub(super) seq_id: u64,
    pub(super) kind: ScheduledQueryKind,
    pub(super) tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct PrefillScheduler {
    budget: PrefillBudget,
}

#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::{
        PrefillBudget, PrefillRequest, PrefillRequestState, PrefillScheduler, ScheduledQueryKind,
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
}
