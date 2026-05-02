#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum PrefillAttentionPath {
    FirstPrefill,
    ContinuationPrefill,
    Decode,
    Mixed,
    Reference,
}
