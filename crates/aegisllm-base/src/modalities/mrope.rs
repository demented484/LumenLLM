//! Host-side M-RoPE (multimodal RoPE) position-id construction for Qwen3-VL
//! (Qwen3.5 / 3.6).
//!
//! Pure CPU port of HF `Qwen3_5Model.get_rope_index` /
//! `get_vision_position_ids` (`modeling_qwen3_5.py`). Given a token-id
//! sequence with image soft-tokens, this produces the 3 position-id
//! components `(T, H, W)` per token that the M-RoPE attention kernel consumes.
//!
//! Layout, per HF:
//!   * **Text** spans get `(p, p, p)` — a running `arange` starting at
//!     `current_pos`. Because all three components are equal, text-only input
//!     collapses to ordinary 1-D RoPE — this is what guarantees no regression
//!     on the existing text path.
//!   * Each **image**'s tokens get `(t, row, col)` from
//!     [`vision_position_ids`]: the post-merge grid `(gh/m, gw/m)` enumerated
//!     in the same 2×2-merge-block order the packer/merger use, offset by the
//!     running `current_pos`.
//!   * After an image, `current_pos += max(gh, gw) / spatial_merge_size`.
//!
//! This module only *constructs* the host position ids and is intentionally
//! NOT wired into the decode path — the GPU M-RoPE kernel + wiring is a later
//! phase. It exists so the GPU phase has a verified reference + the structure
//! is locked against HF.

use crate::error::{AegisError, Result};

/// The 3 position-id components for a sequence, row-major `[3][seq_len]`.
/// `comp[0]` = temporal (T), `comp[1]` = height (H), `comp[2]` = width (W).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MRopePositionIds {
    /// `[T_row, H_row, W_row]`, each of length `seq_len`.
    pub comp: [Vec<i64>; 3],
}

impl MRopePositionIds {
    pub fn seq_len(&self) -> usize {
        self.comp[0].len()
    }

    /// True iff all three components are identical at every position — i.e.
    /// the sequence has no multimodal tokens and M-RoPE degenerates to 1-D
    /// RoPE. The text-only path MUST satisfy this.
    pub fn is_collapsed(&self) -> bool {
        self.comp[0] == self.comp[1] && self.comp[1] == self.comp[2]
    }

    /// `mrope_position_delta = max(position) + 1 − seq_len`, matching HF's
    /// `mrope_position_deltas`. Used to continue positions during decode.
    pub fn position_delta(&self) -> i64 {
        let max = self
            .comp
            .iter()
            .flat_map(|c| c.iter().copied())
            .max()
            .unwrap_or(-1);
        max + 1 - self.seq_len() as i64
    }
}

/// Grid shape of one image after patch embedding: `(t, h, w)` in **pre-merge**
/// patches (`t = 1` for a still image; `h = H/patch`, `w = W/patch`).
/// This is exactly the `grid_thw` emitted by the Qwen2-VL packer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridThw {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl GridThw {
    pub fn new(t: usize, h: usize, w: usize) -> Self {
        Self { t, h, w }
    }
}

/// HF `get_vision_position_ids` for a single image/video grid.
///
/// Returns `[3][n]` where `n = (t/temp_merge)·(h/spatial_merge)·(w/spatial_merge)`
/// — the post-merge token count. Component order is `(temporal, height, width)`.
/// `start_position` is added to the spatial axes (and, after `time_interval`
/// scaling, to the temporal axis), matching HF exactly.
///
/// Iteration order (the "repeat patterns are important" note in HF): for each
/// temporal slot `t`, for each merged row `r`, for each merged col `c`, in that
/// nesting — i.e. width is the fastest-varying axis. This is the same
/// 2×2-merge-block grouped order the packer rows follow.
pub fn vision_position_ids(
    start_position: i64,
    grid: GridThw,
    temp_merge_size: usize,
    spatial_merge_size: usize,
    time_interval: i64,
) -> [Vec<i64>; 3] {
    let tm = temp_merge_size.max(1);
    let sm = spatial_merge_size.max(1);
    let llm_t = grid.t / tm;
    let llm_h = grid.h / sm;
    let llm_w = grid.w / sm;
    let n = llm_t * llm_h * llm_w;

    let mut temporal = Vec::with_capacity(n);
    let mut height = Vec::with_capacity(n);
    let mut width = Vec::with_capacity(n);

    // HF: position_width = arange(llm_w)+start, repeat(llm_h*llm_t)        (tile)
    //     position_height = (arange(llm_h)+start).repeat_interleave(llm_w).repeat(llm_t)
    //     position_temporal = arange(llm_t)*time_interval, repeat_interleave(llm_h*llm_w) + start
    // Equivalent triple loop t -> r -> c (c fastest):
    for ti in 0..llm_t {
        let t_val = ti as i64 * time_interval + start_position;
        for r in 0..llm_h {
            let h_val = r as i64 + start_position;
            for c in 0..llm_w {
                let w_val = c as i64 + start_position;
                temporal.push(t_val);
                height.push(h_val);
                width.push(w_val);
            }
        }
    }
    [temporal, height, width]
}

/// HF `get_rope_index` for a single (un-batched) sequence.
///
/// `token_ids` is the full prompt (text + image soft-tokens). Positions where
/// `token_ids[i] == image_token_id` are treated as image tokens (HF derives
/// this from `mm_token_type_ids`, which the processor sets to `1` exactly at
/// the expanded image-placeholder positions — identical to matching the
/// `image_token_id`). `image_grids` supplies one [`GridThw`] per image, in
/// order of appearance; its length must equal the number of contiguous image
/// runs in `token_ids`.
///
/// Returns the 3 position-id components `[3][seq_len]`.
pub fn get_rope_index(
    token_ids: &[u32],
    image_token_id: u32,
    image_grids: &[GridThw],
    spatial_merge_size: usize,
) -> Result<MRopePositionIds> {
    let sm = spatial_merge_size.max(1);
    let seq_len = token_ids.len();
    let mut t = vec![0i64; seq_len];
    let mut h = vec![0i64; seq_len];
    let mut w = vec![0i64; seq_len];

    let mut current_pos: i64 = 0;
    let mut img_idx = 0usize;
    let mut i = 0usize;
    while i < seq_len {
        let is_img = token_ids[i] == image_token_id;
        // Extend the current modality run.
        let start = i;
        while i < seq_len && (token_ids[i] == image_token_id) == is_img {
            i += 1;
        }
        let end = i;

        if !is_img {
            // Text run: (p, p, p) running arange.
            for (k, idx) in (start..end).enumerate() {
                let p = current_pos + k as i64;
                t[idx] = p;
                h[idx] = p;
                w[idx] = p;
            }
            current_pos += (end - start) as i64;
        } else {
            let grid = *image_grids.get(img_idx).ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "m-rope: image run #{img_idx} has no matching grid_thw \
                     (image_grids.len()={})",
                    image_grids.len()
                ))
            })?;
            img_idx += 1;
            let expected = grid.t * (grid.h / sm) * (grid.w / sm);
            if expected != end - start {
                return Err(AegisError::InvalidPlan(format!(
                    "m-rope: image run #{} has {} tokens but grid {:?} (merge {}) \
                     expects {}",
                    img_idx - 1,
                    end - start,
                    grid,
                    sm,
                    expected
                )));
            }
            let vp = vision_position_ids(current_pos, grid, 1, sm, 1);
            for (k, idx) in (start..end).enumerate() {
                t[idx] = vp[0][k];
                h[idx] = vp[1][k];
                w[idx] = vp[2][k];
            }
            // HF: current_pos += max(gh, gw) // spatial_merge_size
            current_pos += (grid.h.max(grid.w) / sm) as i64;
        }
    }

    if img_idx != image_grids.len() {
        return Err(AegisError::InvalidPlan(format!(
            "m-rope: {} image runs in token_ids but {} grids supplied",
            img_idx,
            image_grids.len()
        )));
    }

    Ok(MRopePositionIds { comp: [t, h, w] })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE_TOKEN: u32 = 248056;
    const MERGE: usize = 2;

    #[test]
    fn text_only_collapses_to_arange() {
        let ids = [10u32, 11, 12, 13, 14];
        let pos = get_rope_index(&ids, IMAGE_TOKEN, &[], MERGE).unwrap();
        // The defining property: all 3 components equal a plain arange.
        let want: Vec<i64> = (0..5).collect();
        assert_eq!(pos.comp[0], want);
        assert_eq!(pos.comp[1], want);
        assert_eq!(pos.comp[2], want);
        // ...and the collapse invariant the GPU path relies on.
        assert!(pos.is_collapsed(), "text-only must collapse to 1-D RoPE");
        assert_eq!(pos.position_delta(), 0);
    }

    #[test]
    fn vision_position_ids_match_hf_pattern() {
        // grid 1x4x4, merge 2 -> 2x2 post-merge, start_position 2.
        // HF reference: T=[2,2,2,2] H=[2,2,3,3] W=[2,3,2,3].
        let vp = vision_position_ids(2, GridThw::new(1, 4, 4), 1, MERGE, 1);
        assert_eq!(vp[0], vec![2, 2, 2, 2], "temporal");
        assert_eq!(vp[1], vec![2, 2, 3, 3], "height");
        assert_eq!(vp[2], vec![2, 3, 2, 3], "width");
    }

    #[test]
    fn text_image_text_matches_hf() {
        // seq = [100,101] + image(1x4x4 -> 4 tokens) + [200,201,202]
        // HF reference run (get_rope_index):
        //   T = [0,1, 2,2,2,2, 4,5,6]
        //   H = [0,1, 2,2,3,3, 4,5,6]
        //   W = [0,1, 2,3,2,3, 4,5,6]
        let mut ids = vec![100u32, 101];
        ids.extend([IMAGE_TOKEN; 4]); // (4/2)*(4/2) = 4 merged tokens
        ids.extend([200u32, 201, 202]);
        let grids = [GridThw::new(1, 4, 4)];
        let pos = get_rope_index(&ids, IMAGE_TOKEN, &grids, MERGE).unwrap();
        assert_eq!(pos.comp[0], vec![0, 1, 2, 2, 2, 2, 4, 5, 6], "T");
        assert_eq!(pos.comp[1], vec![0, 1, 2, 2, 3, 3, 4, 5, 6], "H");
        assert_eq!(pos.comp[2], vec![0, 1, 2, 3, 2, 3, 4, 5, 6], "W");
        // Multimodal sequence must NOT collapse.
        assert!(!pos.is_collapsed());
        // HF delta = max(6) + 1 - 9 = -2.
        assert_eq!(pos.position_delta(), -2);
    }

    #[test]
    fn non_square_image_matches_hf() {
        // grid 1x6x4 -> 3x2 post-merge (6 tokens), prefix [5], suffix [9,9].
        // HF reference run:
        //   T = [0, 1,1,1,1,1,1, 4,5]
        //   H = [0, 1,1,2,2,3,3, 4,5]
        //   W = [0, 1,2,1,2,1,2, 4,5]
        let mut ids = vec![5u32];
        ids.extend([IMAGE_TOKEN; 6]);
        ids.extend([9u32, 9]);
        let grids = [GridThw::new(1, 6, 4)];
        let pos = get_rope_index(&ids, IMAGE_TOKEN, &grids, MERGE).unwrap();
        assert_eq!(pos.comp[0], vec![0, 1, 1, 1, 1, 1, 1, 4, 5], "T");
        assert_eq!(pos.comp[1], vec![0, 1, 1, 2, 2, 3, 3, 4, 5], "H");
        assert_eq!(pos.comp[2], vec![0, 1, 2, 1, 2, 1, 2, 4, 5], "W");
        // current_pos after image advanced by max(6,4)//2 = 3 (from 1 -> 4).
        assert_eq!(pos.comp[0][7], 4);
    }

    #[test]
    fn two_images_advance_position() {
        // [a] img(1x4x4 -> 4) [b] img(1x4x4 -> 4) [c]
        // HF reference run:
        //   T = [0, 1,1,1,1, 3, 4,4,4,4, 6]
        //   H = [0, 1,1,2,2, 3, 4,4,5,5, 6]
        //   W = [0, 1,2,1,2, 3, 4,5,4,5, 6]
        let mut ids = vec![1u32];
        ids.extend([IMAGE_TOKEN; 4]);
        ids.push(2);
        ids.extend([IMAGE_TOKEN; 4]);
        ids.push(3);
        let grids = [GridThw::new(1, 4, 4), GridThw::new(1, 4, 4)];
        let pos = get_rope_index(&ids, IMAGE_TOKEN, &grids, MERGE).unwrap();
        assert_eq!(pos.comp[0], vec![0, 1, 1, 1, 1, 3, 4, 4, 4, 4, 6], "T");
        assert_eq!(pos.comp[1], vec![0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6], "H");
        assert_eq!(pos.comp[2], vec![0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6], "W");
        // text 'b' between the images takes the max-spatial offset (img1 ran
        // 1..3 spatially → 'b' lands at 3); text 'c' lands at 6.
        assert_eq!(pos.comp[0][5], 3, "text 'b' after image1");
        assert_eq!(*pos.comp[0].last().unwrap(), 6, "text 'c' after image2");
        assert!(!pos.is_collapsed());
    }

    #[test]
    fn image_starts_at_zero() {
        // Image with no text prefix: vision positions begin at current_pos=0.
        let ids = [IMAGE_TOKEN; 4];
        let grids = [GridThw::new(1, 4, 4)];
        let pos = get_rope_index(&ids, IMAGE_TOKEN, &grids, MERGE).unwrap();
        assert_eq!(pos.comp[0], vec![0, 0, 0, 0]);
        assert_eq!(pos.comp[1], vec![0, 0, 1, 1]);
        assert_eq!(pos.comp[2], vec![0, 1, 0, 1]);
    }

    #[test]
    fn grid_count_mismatch_errors() {
        let mut ids = vec![1u32];
        ids.extend([IMAGE_TOKEN; 4]);
        // No grid supplied for the image run.
        assert!(get_rope_index(&ids, IMAGE_TOKEN, &[], MERGE).is_err());
        // Too many grids.
        let grids = [GridThw::new(1, 4, 4), GridThw::new(1, 4, 4)];
        assert!(get_rope_index(&ids, IMAGE_TOKEN, &grids, MERGE).is_err());
    }

    #[test]
    fn token_count_grid_mismatch_errors() {
        // Image run has 4 tokens but grid 1x4x4 wants 4 — make it disagree.
        let mut ids = vec![1u32];
        ids.extend([IMAGE_TOKEN; 3]); // 3 != expected 4
        let grids = [GridThw::new(1, 4, 4)];
        assert!(get_rope_index(&ids, IMAGE_TOKEN, &grids, MERGE).is_err());
    }
}
