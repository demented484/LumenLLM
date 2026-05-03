/// Modality fusion stub (Phase 8.5).
///
/// Planned implementation: replace `<image>`, `<audio>`, `<video>` placeholder
/// tokens in the text token sequence with the encoded token sequences from each
/// encoder. Position IDs are updated to account for inserted multimodal tokens.
///
/// **Phase 8 stub** — `fuse` returns `Unsupported`.
use crate::error::{AegisError, Result};
use super::EncodedTokens;

/// A text token sequence with optional placeholder spans for modality tokens.
#[derive(Debug, Clone)]
pub struct TextWithPlaceholders {
    /// The full token id sequence including placeholder tokens.
    pub token_ids: Vec<u32>,
    /// Spans within `token_ids` that should be replaced with modality tokens.
    pub spans: Vec<ModalitySpan>,
}

/// One placeholder span in the text token sequence.
#[derive(Debug, Clone)]
pub struct ModalitySpan {
    /// Inclusive start index in `TextWithPlaceholders::token_ids`.
    pub start: usize,
    /// Exclusive end index.
    pub end: usize,
    /// Which modality's encoded tokens should fill this span.
    pub kind: SpanKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    Vision,
    Audio,
    Video,
}

/// Result of fusion: final flat token ids and position ids.
#[derive(Debug, Clone)]
pub struct FusedSequence {
    /// Text token ids and modality token ids interleaved; modality tokens are
    /// represented as `u32::MAX` (a sentinel not in any real vocabulary).
    pub token_ids: Vec<u32>,
    /// Position ids (length = `token_ids.len()`).
    pub position_ids: Vec<u32>,
    /// The actual f32 embedding values for modality token slots.
    /// Indexed in slot order matching the `u32::MAX` sentinels.
    pub modality_embeddings: Vec<EncodedTokens>,
}

/// Sentinel token id used to mark slots in `FusedSequence::token_ids` that
/// hold modality embeddings rather than vocabulary token ids.
pub const MODALITY_SLOT_SENTINEL: u32 = u32::MAX;

/// Fuse text tokens with encoded modality tokens.
///
/// Walks `text.token_ids` and replaces each declared `ModalitySpan` with
/// `EncodedTokens::num_tokens` sentinel slots. The actual embeddings are
/// preserved in `modality_embeddings` in span order. Position ids are
/// monotonically increasing across the fused sequence.
///
/// `modality_tokens` must list one entry per span in `text.spans` order, and
/// the `SpanKind` must match.
pub fn fuse(
    text: &TextWithPlaceholders,
    modality_tokens: &[(SpanKind, EncodedTokens)],
) -> Result<FusedSequence> {
    if text.spans.len() != modality_tokens.len() {
        return Err(AegisError::InvalidPlan(format!(
            "fuse: span count mismatch — text has {} spans but {} modality tokens provided",
            text.spans.len(),
            modality_tokens.len()
        )));
    }
    let mut sorted_spans: Vec<(usize, &ModalitySpan, &(SpanKind, EncodedTokens))> = text
        .spans
        .iter()
        .zip(modality_tokens.iter())
        .enumerate()
        .map(|(idx, (span, mt))| (idx, span, mt))
        .collect();
    sorted_spans.sort_by_key(|(_, span, _)| span.start);
    for (_, span, (kind, _)) in &sorted_spans {
        if span.start > span.end || span.end > text.token_ids.len() {
            return Err(AegisError::InvalidPlan(format!(
                "fuse: span [{}, {}) out of bounds for {} text tokens",
                span.start, span.end, text.token_ids.len()
            )));
        }
        if &span.kind != kind {
            return Err(AegisError::InvalidPlan(format!(
                "fuse: span kind {:?} does not match provided modality kind {:?}",
                span.kind, kind
            )));
        }
    }
    // Verify spans don't overlap.
    for window in sorted_spans.windows(2) {
        if window[0].1.end > window[1].1.start {
            return Err(AegisError::InvalidPlan(format!(
                "fuse: overlapping spans [{}, {}) and [{}, {})",
                window[0].1.start, window[0].1.end, window[1].1.start, window[1].1.end
            )));
        }
    }

    // Stream through text, replacing each span's range with sentinel slots.
    let mut token_ids = Vec::with_capacity(text.token_ids.len());
    let mut position_ids = Vec::with_capacity(text.token_ids.len());
    let mut cursor = 0usize;
    let mut next_position: u32 = 0;
    let mut modality_embeddings = Vec::with_capacity(sorted_spans.len());
    let mut original_order: Vec<(usize, EncodedTokens)> = Vec::with_capacity(sorted_spans.len());
    for (orig_idx, span, (_, encoded)) in &sorted_spans {
        // Append text tokens before this span.
        for &tok in &text.token_ids[cursor..span.start] {
            token_ids.push(tok);
            position_ids.push(next_position);
            next_position += 1;
        }
        // Insert sentinel slots for the modality tokens.
        for _ in 0..encoded.num_tokens {
            token_ids.push(MODALITY_SLOT_SENTINEL);
            position_ids.push(next_position);
            next_position += 1;
        }
        original_order.push((*orig_idx, (*encoded).clone()));
        cursor = span.end;
    }
    // Tail text after the last span.
    for &tok in &text.token_ids[cursor..] {
        token_ids.push(tok);
        position_ids.push(next_position);
        next_position += 1;
    }

    // Restore original-input order for caller convenience.
    original_order.sort_by_key(|(idx, _)| *idx);
    modality_embeddings.extend(original_order.into_iter().map(|(_, et)| et));

    Ok(FusedSequence { token_ids, position_ids, modality_embeddings })
}

#[cfg(test)]
mod tests {
    use super::{
        EncodedTokens, MODALITY_SLOT_SENTINEL, ModalitySpan, SpanKind,
        TextWithPlaceholders, fuse,
    };

    fn et(num_tokens: usize, hidden: usize) -> EncodedTokens {
        EncodedTokens::new(vec![0.0; num_tokens * hidden], num_tokens, hidden).unwrap()
    }

    #[test]
    fn fuse_replaces_single_vision_span() {
        // text: [10, 11, <img>, 12]  span [2,3) → 4 image tokens
        let text = TextWithPlaceholders {
            token_ids: vec![10, 11, 99, 12],
            spans: vec![ModalitySpan { start: 2, end: 3, kind: SpanKind::Vision }],
        };
        let fused = fuse(&text, &[(SpanKind::Vision, et(4, 8))]).unwrap();
        assert_eq!(
            fused.token_ids,
            [10, 11, MODALITY_SLOT_SENTINEL, MODALITY_SLOT_SENTINEL,
                MODALITY_SLOT_SENTINEL, MODALITY_SLOT_SENTINEL, 12],
        );
        assert_eq!(fused.position_ids, [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(fused.modality_embeddings.len(), 1);
    }

    #[test]
    fn fuse_handles_no_spans() {
        let text = TextWithPlaceholders {
            token_ids: vec![1, 2, 3],
            spans: vec![],
        };
        let fused = fuse(&text, &[]).unwrap();
        assert_eq!(fused.token_ids, [1, 2, 3]);
        assert_eq!(fused.position_ids, [0, 1, 2]);
    }

    #[test]
    fn fuse_handles_two_spans_in_order() {
        let text = TextWithPlaceholders {
            token_ids: vec![5, 90, 6, 7, 91, 8],
            spans: vec![
                ModalitySpan { start: 1, end: 2, kind: SpanKind::Vision },
                ModalitySpan { start: 4, end: 5, kind: SpanKind::Audio },
            ],
        };
        let fused = fuse(
            &text,
            &[
                (SpanKind::Vision, et(2, 4)),
                (SpanKind::Audio, et(3, 4)),
            ],
        ).unwrap();
        // Expect: 5, V, V, 6, 7, A, A, A, 8
        assert_eq!(fused.token_ids.len(), 9);
        assert_eq!(fused.token_ids[0], 5);
        assert!(fused.token_ids[1..3].iter().all(|&t| t == MODALITY_SLOT_SENTINEL));
        assert_eq!(fused.token_ids[3], 6);
        assert_eq!(fused.token_ids[4], 7);
        assert!(fused.token_ids[5..8].iter().all(|&t| t == MODALITY_SLOT_SENTINEL));
        assert_eq!(fused.token_ids[8], 8);
    }

    #[test]
    fn fuse_rejects_kind_mismatch() {
        let text = TextWithPlaceholders {
            token_ids: vec![1, 99, 2],
            spans: vec![ModalitySpan { start: 1, end: 2, kind: SpanKind::Vision }],
        };
        let err = fuse(&text, &[(SpanKind::Audio, et(1, 4))]);
        assert!(err.is_err());
    }

    #[test]
    fn fuse_rejects_count_mismatch() {
        let text = TextWithPlaceholders {
            token_ids: vec![1, 99, 2],
            spans: vec![ModalitySpan { start: 1, end: 2, kind: SpanKind::Vision }],
        };
        assert!(fuse(&text, &[]).is_err());
    }

    #[test]
    fn fuse_rejects_overlapping_spans() {
        let text = TextWithPlaceholders {
            token_ids: vec![1, 2, 3, 4, 5],
            spans: vec![
                ModalitySpan { start: 1, end: 4, kind: SpanKind::Vision },
                ModalitySpan { start: 2, end: 5, kind: SpanKind::Audio },
            ],
        };
        assert!(fuse(&text, &[
            (SpanKind::Vision, et(1, 4)),
            (SpanKind::Audio, et(1, 4)),
        ]).is_err());
    }
}
