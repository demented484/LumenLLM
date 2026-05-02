use super::state::CpuLayerState;
use aegisllm_base::error::Result;
use crate::attention::{
    ReferenceAttentionDecodeRequest, reference_attention_decode_f32_into,
};

pub(super) fn attention_into(
    state: &CpuLayerState,
    query: &[f32],
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<()> {
    reference_attention_decode_f32_into(
        ReferenceAttentionDecodeRequest {
            keys: &state.keys,
            values: &state.values,
            seq_len: state.seq_len,
            query,
            num_attention_heads,
            num_kv_heads,
            head_dim,
        },
        out,
    )
}
