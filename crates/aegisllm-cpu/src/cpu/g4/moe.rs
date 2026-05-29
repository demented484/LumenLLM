//! Gemma-4 MoE router (26B-A4B). Ports `softmax_top_k_normalized_into`
//! (`crates/aegisllm-cuda/src/executor/mlp.rs:620-643`):
//!
//!   probs   = softmax(logits)                 (over ALL experts)
//!   topk_w, topk_i = topk(probs, k)           (descending weight order)
//!   topk_w /= Σ topk_w                          (renormalize)
//!   topk_w *= per_expert_scale[topk_i]          (if provided)
//!
//! The expert FFN itself (gate/up → GeGLU-tanh → down, weighted-accumulate)
//! reuses the dense MLP primitives in the forward driver.

/// Softmax over all experts, take top-`k` by probability, renormalize the
/// selected weights to sum to 1, then apply optional per-expert scale.
/// Returns `(indices, weights)` in descending weight order, mirroring the
/// CUDA reference exactly.
pub(crate) fn router_softmax_topk_normalized(
    logits: &[f32],
    top_k: usize,
    per_expert_scale: Option<&[f32]>,
) -> (Vec<usize>, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    let k = top_k.min(probs.len());
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    if k > 0 {
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let top = &mut indexed[..k];
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut indices: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
    let mut weights: Vec<f32> = top.iter().map(|(_, w)| *w).collect();

    // renormalize so the top-k weights sum to 1.
    let wsum: f32 = weights.iter().sum();
    if wsum > 0.0 {
        for w in weights.iter_mut() {
            *w /= wsum;
        }
    }
    // per-expert calibration scale on the selected weights.
    if let Some(pes) = per_expert_scale {
        for (i, w) in indices.iter().zip(weights.iter_mut()) {
            if let Some(s) = pes.get(*i) {
                *w *= *s;
            }
        }
    }
    indices.shrink_to_fit();
    (indices, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_highest_prob_experts_descending() {
        let logits = [1.0f32, 0.0, 3.0, -1.0];
        let (idx, w) = router_softmax_topk_normalized(&logits, 2, None);
        assert_eq!(idx.len(), 2);
        // expert 2 highest, then expert 0.
        assert_eq!(idx[0], 2);
        assert_eq!(idx[1], 0);
        assert!(w[0] >= w[1]);
    }

    #[test]
    fn renormalized_weights_sum_to_one() {
        let logits = [1.0f32, 0.0, 3.0, -1.0];
        let (_idx, w) = router_softmax_topk_normalized(&logits, 2, None);
        let wsum: f32 = w.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-5, "renorm sum={wsum}");
    }

    #[test]
    fn per_expert_scale_applied_after_renorm() {
        let logits = [1.0f32, 0.0, 3.0, -1.0];
        // top-2 = experts 2,0. Scale doubles expert 2, halves expert 0.
        let mut scale = vec![1.0f32; 4];
        scale[2] = 2.0;
        scale[0] = 0.5;
        let (idx, w) = router_softmax_topk_normalized(&logits, 2, Some(&scale));
        assert_eq!(idx, vec![2, 0]);
        // Recompute expected: renormalized then scaled.
        let (_ridx, rw) = router_softmax_topk_normalized(&logits, 2, None);
        assert!((w[0] - rw[0] * 2.0).abs() < 1e-5);
        assert!((w[1] - rw[1] * 0.5).abs() < 1e-5);
    }

    #[test]
    fn hand_computed_top1_of_two() {
        // logits [0, ln(3)] → probs [1/4, 3/4]. top-1 = expert 1, renorm → 1.0.
        let logits = [0.0f32, 3.0f32.ln()];
        let (idx, w) = router_softmax_topk_normalized(&logits, 1, None);
        assert_eq!(idx, vec![1]);
        assert!((w[0] - 1.0).abs() < 1e-6);
    }
}
