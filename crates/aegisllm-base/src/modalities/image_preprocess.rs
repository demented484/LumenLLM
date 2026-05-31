//! Generic image preprocessing for vision-encoder front-ends.
//!
//! Supports two resize strategies and per-channel normalization, covering the
//! preprocessors used by SigLIP / CLIP / InternViT / Gemma-* / Qwen-VL variants.
//! Resize strategies:
//!   * `Fixed { height, width }` — bilinear/bicubic resize to a fixed target
//!     (CLIP, classic SigLIP, fixed-resolution InternViT).
//!   * `AspectPreserving { max_patches, side_multiple }` — Gemma-4 / Qwen-2-VL
//!     style: largest dims that (a) keep aspect ratio, (b) are multiples of
//!     `side_multiple` (typically `pooling_kernel_size * patch_size`), and (c)
//!     produce ≤ `max_patches` patches with `patch_size`.
//!
//! Pipeline:
//!   1. Load RGB8 (PNG/JPEG via the `image` crate).
//!   2. Resize per strategy.
//!   3. Rescale u8 → f32 by `rescale_factor`.
//!   4. (Optional) normalize: `(x - mean) / std` per channel.
//!   5. HWC → CHW; (optional) patchify into `[n_patches, P*P*3]` row-major.
//!
//! Resampling: `BICUBIC` everywhere a HF processor uses `resample=3`. The
//! `image` crate's `CatmullRom` is the bicubic of that family.

use crate::error::{AegisError, Result};
use image::imageops::FilterType;
use std::path::Path;

/// Round `value` to the nearest multiple of `factor` (ties to even, matching
/// Python `round`). Used by `qwen2vl_smart_resize`.
fn round_by_factor(value: f64, factor: usize) -> usize {
    // Python `round` is banker's rounding; HF relies on `.5` cases being rare
    // (post-area-scaling). Replicate banker's rounding for exact parity.
    let q = value / factor as f64;
    let r = q.round();
    // f64::round is round-half-away-from-zero; convert to round-half-to-even
    // only on exact .5 to match Python.
    let r = if (q - q.floor() - 0.5).abs() < 1e-9 {
        let fl = q.floor();
        if (fl as i64) % 2 == 0 { fl } else { fl + 1.0 }
    } else {
        r
    };
    (r as usize) * factor
}

fn floor_by_factor(value: f64, factor: usize) -> usize {
    ((value / factor as f64).floor() as usize) * factor
}

fn ceil_by_factor(value: f64, factor: usize) -> usize {
    ((value / factor as f64).ceil() as usize) * factor
}

/// HF `Qwen2VLImageProcessor.smart_resize` (also used by Qwen3-VL). Returns
/// `(resized_height, resized_width)` such that:
///   1. both are multiples of `factor` (= patch_size · spatial_merge_size),
///   2. the total AREA `h·w` lies in `[min_pixels, max_pixels]`,
///   3. the aspect ratio is preserved as closely as the factor grid allows.
///
/// `min_pixels`/`max_pixels` are the `size.{shortest_edge,longest_edge}`
/// AREA bounds from `processor_config.json` (Qwen3-VL: 65536 / 16777216).
pub fn qwen2vl_smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let (hf, wf) = (height as f64, width as f64);
    // HF: initial `round(h/factor)*factor` with no floor clamp.
    let mut h_bar = round_by_factor(hf, factor);
    let mut w_bar = round_by_factor(wf, factor);
    if h_bar * w_bar > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        h_bar = floor_by_factor(hf / beta, factor).max(factor);
        w_bar = floor_by_factor(wf / beta, factor).max(factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = ceil_by_factor(hf * beta, factor);
        w_bar = ceil_by_factor(wf * beta, factor);
    }
    (h_bar, w_bar)
}

/// How to choose the resized (H, W).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeStrategy {
    /// Force a single fixed target. Patch grid = (H/P, W/P).
    Fixed {
        height: usize,
        width: usize,
    },
    /// Aspect-ratio-preserving with a patch budget.
    AspectPreserving {
        /// Maximum number of patches (pre-pool) allowed in the resized image.
        max_patches: usize,
        /// Resized H and W must each be a multiple of this. For Gemma-4 this
        /// is `pooling_kernel_size * patch_size = 3 * 16 = 48`. For models
        /// with no spatial pooling, set this to `patch_size`.
        side_multiple: usize,
    },
    /// Qwen2-VL / Qwen3-VL "smart resize": both sides snapped to multiples of
    /// `factor = patch_size * spatial_merge_size`, with the total pixel AREA
    /// clamped to `[min_pixels, max_pixels]`. Mirrors HF `smart_resize`.
    Qwen2VLSmartResize {
        /// `patch_size * spatial_merge_size` (Qwen3-VL: 16 * 2 = 32).
        factor: usize,
        /// AREA lower bound (`size.shortest_edge`; Qwen3-VL: 65536 px²).
        min_pixels: usize,
        /// AREA upper bound (`size.longest_edge`; Qwen3-VL: 16777216 px²).
        max_pixels: usize,
    },
}

/// Static image-preprocessing configuration. Same struct serves any
/// SigLIP/CLIP/Gemma/Qwen-VL family — just plug different numbers from the
/// model's `processor_config.json`.
#[derive(Debug, Clone, Copy)]
pub struct ImageProcessor {
    /// Side length (pixels) of one square patch.
    pub patch_size: usize,
    /// Spatial pool factor applied AFTER patchification (`1` for no pooling).
    pub pooling_kernel_size: usize,
    /// How to resize the input image.
    pub resize: ResizeStrategy,
    /// u8 → f32 scale (typically 1/255).
    pub rescale_factor: f32,
    /// Per-channel mean (RGB). Use `[0.0; 3]` to skip the subtract.
    pub mean: [f32; 3],
    /// Per-channel std (RGB). Use `[1.0; 3]` to skip the divide.
    pub std: [f32; 3],
    /// Qwen2/3-VL 2×2 patch-merge factor used by the packer (`spatial_merge_size`).
    /// Only consulted by the `preprocess_qwen2vl` path; `1` elsewhere.
    pub merge_size: usize,
    /// Qwen2/3-VL temporal patch depth (`temporal_patch_size`, 2). A still
    /// image is duplicated across these slots. `1` for non-Qwen processors.
    pub temporal_patch_size: usize,
}

impl ImageProcessor {
    /// Build the processor from the artifact's `vision_config`. All shape
    /// parameters (patch_size, pooling_kernel_size, max-soft-tokens budget)
    /// come from the model checkpoint — never hardcoded per-model here. The
    /// engine supports any Gemma-style aspect-preserving + patchify pipeline
    /// that the checkpoint advertises.
    ///
    /// Pre-pool patch budget = `vision_soft_tokens_per_image * pool²`
    /// (Gemma-4 native: 280 * 9 = 2520). `side_multiple = pool * patch_size`
    /// (Gemma-4: 3 * 16 = 48) ensures every resize produces a clean
    /// patch+pool tiling with no tail rows.
    ///
    /// `AEGIS_VISION_MAX_PATCHES` env var overrides the budget for OCR
    /// (matches llama.cpp's `--image-max-tokens` lever). The model's
    /// `position_embedding_size` (10240 for Gemma-4) is the hard upper bound.
    /// Construct directly from parameters — for tests / examples / engine
    /// callers that already have the values. Production callers should use
    /// `from_artifact_vision` to ensure the values match the loaded model.
    pub fn with_params(
        patch_size: usize,
        pooling_kernel_size: usize,
        max_soft_tokens: usize,
    ) -> Self {
        let computed_budget = max_soft_tokens
            .checked_mul(pooling_kernel_size * pooling_kernel_size)
            .expect("vision: max_soft_tokens × pool² overflow");
        let max_patches = std::env::var("AEGIS_VISION_MAX_PATCHES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(computed_budget);
        Self {
            patch_size,
            pooling_kernel_size,
            resize: ResizeStrategy::AspectPreserving {
                max_patches,
                side_multiple: pooling_kernel_size * patch_size,
            },
            rescale_factor: 1.0 / 255.0,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            merge_size: 1,
            temporal_patch_size: 1,
        }
    }

    pub fn from_artifact_vision(cfg: &crate::artifact::HfVisionConfig, max_soft_tokens: usize) -> Self {
        let pool = cfg.pooling_kernel_size;
        let patch = cfg.patch_size;
        let computed_budget = max_soft_tokens
            .checked_mul(pool * pool)
            .expect("vision: max_soft_tokens × pool² overflow");
        let max_patches = std::env::var("AEGIS_VISION_MAX_PATCHES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(computed_budget);
        Self {
            patch_size: patch,
            pooling_kernel_size: pool,
            resize: ResizeStrategy::AspectPreserving {
                max_patches,
                side_multiple: pool * patch,
            },
            // HF Gemma-4 image processor: rescale `1/255` to [0,1], then
            // `2*(x-0.5)` to [-1,1] inside the patch-embed kernel. We do
            // the same here; mean/std normalization is a no-op for Gemma-4.
            rescale_factor: 1.0 / 255.0,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            merge_size: 1,
            temporal_patch_size: 1,
        }
    }

    /// Build a Qwen2-VL / Qwen3-VL image processor from the artifact's
    /// `vision_config` + the processor AREA bounds.
    ///
    /// `min_pixels`/`max_pixels` are the `size.{shortest_edge,longest_edge}`
    /// AREA bounds from `processor_config.json` (Qwen3-VL: 65536 / 16777216).
    /// Normalization is `(x/255 - 0.5) / 0.5` → `[-1, 1]` (mean=std=0.5).
    pub fn qwen2vl_from_artifact_vision(
        cfg: &crate::artifact::HfVisionConfig,
        min_pixels: usize,
        max_pixels: usize,
    ) -> Self {
        Self::qwen2vl(
            cfg.patch_size,
            cfg.spatial_merge_size.max(1),
            cfg.temporal_patch_size.max(1),
            min_pixels,
            max_pixels,
        )
    }

    /// Build a Qwen2-VL / Qwen3-VL image processor from explicit parameters.
    ///
    /// `factor = patch_size · merge_size` is the smart-resize grid quantum.
    /// `temporal_patch_size` (2 on Qwen) is the depth a still image is
    /// duplicated to. Normalization is `(x/255 − 0.5)/0.5 → [-1, 1]`.
    pub fn qwen2vl(
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> Self {
        Self {
            patch_size,
            // Qwen has no SigLIP-style spatial pool; the 2×2 merge lives in
            // the projector. Keep `pooling_kernel_size = 1` here.
            pooling_kernel_size: 1,
            resize: ResizeStrategy::Qwen2VLSmartResize {
                factor: patch_size * merge_size,
                min_pixels,
                max_pixels,
            },
            rescale_factor: 1.0 / 255.0,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            merge_size,
            temporal_patch_size,
        }
    }

    /// Builder for SigLIP base (224×224, patch=14, no spatial pooling, ImageNet
    /// normalization). Many CLIP/SigLIP checkpoints use these.
    pub fn siglip_base_224() -> Self {
        Self {
            patch_size: 14,
            pooling_kernel_size: 1,
            resize: ResizeStrategy::Fixed { height: 224, width: 224 },
            rescale_factor: 1.0 / 255.0,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            merge_size: 1,
            temporal_patch_size: 1,
        }
    }
}

/// One preprocessed image, ready for the vision encoder.
///
/// `pixels` is CHW f32 (3 planes × H × W). `patches` is the same data
/// reshaped to `[n_patches, 3 * P * P]` row-major patch-by-patch.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub height: usize,
    pub width: usize,
    pub num_patches_h: usize,
    pub num_patches_w: usize,
    /// Post-pooling token grid (= patch grid when `pooling_kernel_size == 1`).
    pub num_tokens_h: usize,
    pub num_tokens_w: usize,
    /// CHW f32 pixels: `3 * height * width` elements.
    pub pixels: Vec<f32>,
    /// Patchified view: `num_patches() * 3 * patch_size²` elements.
    pub patches: Vec<f32>,
}

impl PreprocessedImage {
    pub fn num_tokens(&self) -> usize {
        self.num_tokens_h * self.num_tokens_w
    }
    pub fn num_patches(&self) -> usize {
        self.num_patches_h * self.num_patches_w
    }
}

impl ImageProcessor {
    /// Compute target (H, W) per the resize strategy.
    pub fn target_size(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        if height == 0 || width == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "image: zero dimension (got {height}x{width})"
            )));
        }
        match self.resize {
            ResizeStrategy::Fixed { height: h, width: w } => Ok((h, w)),
            ResizeStrategy::AspectPreserving { max_patches, side_multiple } => {
                let total_px = (height * width) as f64;
                let target_px = (max_patches * self.patch_size * self.patch_size) as f64;
                let factor = (target_px / total_px).sqrt();
                let ideal_h = factor * height as f64;
                let ideal_w = factor * width as f64;

                let mut target_h = ((ideal_h / side_multiple as f64).floor() as usize) * side_multiple;
                let mut target_w = ((ideal_w / side_multiple as f64).floor() as usize) * side_multiple;

                if target_h == 0 && target_w == 0 {
                    return Err(AegisError::InvalidPlan(format!(
                        "image {height}x{width}: resize rounds to 0 (side_mult={side_multiple})"
                    )));
                }
                let pool = self.pooling_kernel_size.max(1);
                let max_side_length = (max_patches / (pool * pool)) * side_multiple;
                if target_h == 0 {
                    target_h = side_multiple;
                    target_w = ((width as f64 / height as f64).floor() as usize * side_multiple)
                        .min(max_side_length);
                } else if target_w == 0 {
                    target_w = side_multiple;
                    target_h = ((height as f64 / width as f64).floor() as usize * side_multiple)
                        .min(max_side_length);
                }
                if target_h * target_w > max_patches * self.patch_size * self.patch_size {
                    return Err(AegisError::InvalidPlan(format!(
                        "resize [{height}x{width}] -> [{target_h}x{target_w}] exceeds {max_patches} patches"
                    )));
                }
                Ok((target_h, target_w))
            }
            ResizeStrategy::Qwen2VLSmartResize { factor, min_pixels, max_pixels } => {
                Ok(qwen2vl_smart_resize(height, width, factor, min_pixels, max_pixels))
            }
        }
    }

    /// Load image from disk, resize, rescale, normalize, patchify.
    pub fn load(&self, path: &Path) -> Result<PreprocessedImage> {
        let img = image::open(path)
            .map_err(|e| AegisError::InvalidPlan(format!("image open {path:?}: {e}")))?;
        let rgb = img.to_rgb8();
        let (src_w, src_h) = rgb.dimensions();
        let (tgt_h, tgt_w) = self.target_size(src_h as usize, src_w as usize)?;

        let resized = image::imageops::resize(
            &rgb,
            tgt_w as u32,
            tgt_h as u32,
            FilterType::CatmullRom,
        );

        let h = tgt_h;
        let w = tgt_w;
        let plane = h * w;
        let mut pixels = vec![0.0_f32; 3 * plane];
        for y in 0..h {
            for x in 0..w {
                let p = resized.get_pixel(x as u32, y as u32).0;
                for c in 0..3 {
                    let v = p[c] as f32 * self.rescale_factor;
                    let v = (v - self.mean[c]) / self.std[c];
                    pixels[c * plane + y * w + x] = v;
                }
            }
        }

        let pool = self.pooling_kernel_size.max(1);
        let num_patches_h = h / self.patch_size;
        let num_patches_w = w / self.patch_size;
        let num_tokens_h = num_patches_h / pool;
        let num_tokens_w = num_patches_w / pool;

        let p = self.patch_size;
        let n_patches = num_patches_h * num_patches_w;
        let patch_dim = 3 * p * p;
        let mut patches = vec![0.0_f32; n_patches * patch_dim];
        // Patch layout: HWC (height, width, channel) — innermost channel.
        // This matches HuggingFace `Gemma4ImageProcessor` and the trained
        // patch_embed weight `[hidden, P*P*3]` whose 768 input columns are
        // laid out as `(py, px, c)`. Earlier we used CHW here, which broke
        // semantic fidelity (model saw abstract patterns, not the scene).
        for ph in 0..num_patches_h {
            for pw in 0..num_patches_w {
                let patch_idx = ph * num_patches_w + pw;
                for py in 0..p {
                    for px in 0..p {
                        for c in 0..3 {
                            let src_off = c * plane + (ph * p + py) * w + (pw * p + px);
                            let dst_off = patch_idx * patch_dim + py * p * 3 + px * 3 + c;
                            patches[dst_off] = pixels[src_off];
                        }
                    }
                }
            }
        }

        Ok(PreprocessedImage {
            height: h,
            width: w,
            num_patches_h,
            num_patches_w,
            num_tokens_h,
            num_tokens_w,
            pixels,
            patches,
        })
    }
}

/// Qwen2-VL / Qwen3-VL preprocessed image: packed patch matrix + grid_thw.
///
/// `pixel_values` is `[grid_t · grid_h · grid_w, embed_dim]` row-major, where
/// `embed_dim = in_channels · temporal_patch_size · patch_size²`
/// (Qwen3-VL: 3·2·16·16 = 1536). Rows are ordered by 2×2 **merge block** so
/// the projector's adjacent-4 view (`reshape[-1, merge², dim]`) is contiguous.
/// `grid_thw = (grid_t, grid_h, grid_w)` is the pre-merge patch grid
/// (`grid_t = 1` for a still image; `grid_h = H/patch`, `grid_w = W/patch`).
#[derive(Debug, Clone)]
pub struct Qwen2VLPreprocessed {
    /// Resized pixel height (multiple of `patch_size · merge_size`).
    pub height: usize,
    /// Resized pixel width (multiple of `patch_size · merge_size`).
    pub width: usize,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    /// `in_channels · temporal_patch_size · patch_size²` (= 1536 for Qwen3-VL).
    pub embed_dim: usize,
    /// `[grid_t·grid_h·grid_w, embed_dim]` row-major, merge-block grouped.
    pub pixel_values: Vec<f32>,
}

impl Qwen2VLPreprocessed {
    /// Number of packed patch rows = `grid_t · grid_h · grid_w`.
    pub fn num_patches(&self) -> usize {
        self.grid_t * self.grid_h * self.grid_w
    }
    /// Number of LLM image tokens after the 2×2 projector merge.
    pub fn num_merged_tokens(&self, merge_size: usize) -> usize {
        let m = merge_size.max(1);
        self.grid_t * (self.grid_h / m) * (self.grid_w / m)
    }
    /// `[grid_t, grid_h, grid_w]` for `get_rope_index` / the merger.
    pub fn grid_thw(&self) -> (usize, usize, usize) {
        (self.grid_t, self.grid_h, self.grid_w)
    }
}

impl ImageProcessor {
    /// Pack a CHW f32 image (already resized + normalized) into the Qwen2-VL
    /// `[grid_t·grid_h·grid_w, embed_dim]` matrix, mirroring HF
    /// `Qwen2VLImageProcessor._preprocess` exactly:
    ///
    ///   reshape `[C, gh/m, m, P, gw/m, m, P]`
    ///   → permute to `[gh/m, gw/m, m, m, C, P, P]`
    ///   → insert temporal axis (duplicate the still image `tp` times) after C
    ///   → flatten to `[gh·gw, C·tp·P·P]`.
    ///
    /// The row order is therefore 2×2-merge-block grouped; the column order is
    /// `(C, tp, P_row, P_col)`. A still image has `grid_t = 1`.
    ///
    /// `chw` must be `3 · h · w` f32 in plane-major `[c][y][x]` order with the
    /// rescale + normalize already applied.
    pub fn pack_qwen2vl_chw(
        &self,
        chw: &[f32],
        h: usize,
        w: usize,
    ) -> Result<Qwen2VLPreprocessed> {
        let c = 3usize;
        let p = self.patch_size;
        let m = self.merge_size.max(1);
        let tp = self.temporal_patch_size.max(1);
        if p == 0 || h % p != 0 || w % p != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "qwen2vl pack: {h}x{w} not a multiple of patch {p}"
            )));
        }
        let gh = h / p;
        let gw = w / p;
        if m == 0 || gh % m != 0 || gw % m != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "qwen2vl pack: grid {gh}x{gw} not a multiple of merge {m}"
            )));
        }
        if chw.len() != c * h * w {
            return Err(AegisError::InvalidPlan(format!(
                "qwen2vl pack: chw len {} != {}",
                chw.len(),
                c * h * w
            )));
        }
        let plane = h * w;
        let embed_dim = c * tp * p * p;
        let n_rows = gh * gw; // grid_t == 1
        let mut out = vec![0.0_f32; n_rows * embed_dim];

        // Row index enumerates (block_gh, block_gw, m_row, m_col); column index
        // enumerates (C, tp, P_row, P_col). Temporal slots are identical copies
        // of the single still frame.
        let mut row = 0usize;
        for bgh in 0..(gh / m) {
            for bgw in 0..(gw / m) {
                for mr in 0..m {
                    for mc in 0..m {
                        // Original patch coordinates.
                        let ph = bgh * m + mr;
                        let pw = bgw * m + mc;
                        let base = row * embed_dim;
                        let mut col = 0usize;
                        for ci in 0..c {
                            for _t in 0..tp {
                                for py in 0..p {
                                    let y = ph * p + py;
                                    for px in 0..p {
                                        let x = pw * p + px;
                                        out[base + col] = chw[ci * plane + y * w + x];
                                        col += 1;
                                    }
                                }
                            }
                        }
                        debug_assert_eq!(col, embed_dim);
                        row += 1;
                    }
                }
            }
        }
        debug_assert_eq!(row, n_rows);

        Ok(Qwen2VLPreprocessed {
            height: h,
            width: w,
            grid_t: 1,
            grid_h: gh,
            grid_w: gw,
            embed_dim,
            pixel_values: out,
        })
    }

    /// Full Qwen2-VL / Qwen3-VL pipeline: load → convert RGB → smart-resize
    /// (bicubic / CatmullRom) → rescale → normalize → pack patches.
    ///
    /// Requires the processor to use the `Qwen2VLSmartResize` strategy
    /// (built via [`ImageProcessor::qwen2vl`] /
    /// [`ImageProcessor::qwen2vl_from_artifact_vision`]).
    pub fn preprocess_qwen2vl(&self, path: &Path) -> Result<Qwen2VLPreprocessed> {
        let img = image::open(path)
            .map_err(|e| AegisError::InvalidPlan(format!("image open {path:?}: {e}")))?;
        let rgb = img.to_rgb8();
        let (src_w, src_h) = rgb.dimensions();
        let (h, w) = self.target_size(src_h as usize, src_w as usize)?;
        let resized =
            image::imageops::resize(&rgb, w as u32, h as u32, FilterType::CatmullRom);

        let plane = h * w;
        let mut chw = vec![0.0_f32; 3 * plane];
        for y in 0..h {
            for x in 0..w {
                let px = resized.get_pixel(x as u32, y as u32).0;
                for c in 0..3 {
                    let v = px[c] as f32 * self.rescale_factor;
                    chw[c * plane + y * w + x] = (v - self.mean[c]) / self.std[c];
                }
            }
        }
        self.pack_qwen2vl_chw(&chw, h, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_landscape_resize() {
        // Mirror Gemma-4 vision-config: patch=16, pool=3, max 280 soft tokens.
        let p = ImageProcessor::with_params(16, 3, 280);
        let (h, w) = p.target_size(1080, 1920).unwrap();
        assert_eq!(h % 48, 0);
        assert_eq!(w % 48, 0);
        let patches = (h / 16) * (w / 16);
        assert!(patches <= 2520, "patches {patches} > max 2520");
    }

    #[test]
    fn fixed_strategy_returns_target() {
        let p = ImageProcessor::siglip_base_224();
        assert_eq!(p.target_size(123, 456).unwrap(), (224, 224));
    }

    #[test]
    fn rejects_zero_dim() {
        let p = ImageProcessor::with_params(16, 3, 280);
        assert!(p.target_size(0, 10).is_err());
    }

    // ───────────────────────── Qwen2-VL smart resize ────────────────────────
    // Ground truth generated with HF `Qwen2VLImageProcessor.smart_resize`
    // (factor=32, min=65536, max=16777216). See task notes.

    const QWEN_FACTOR: usize = 16 * 2; // patch · merge
    const QWEN_MIN: usize = 65536;
    const QWEN_MAX: usize = 16777216;

    #[test]
    fn qwen_smart_resize_matches_hf() {
        // (in_h, in_w) -> (out_h, out_w), straight from the HF reference run.
        let cases: &[((usize, usize), (usize, usize))] = &[
            ((100, 100), (256, 256)),   // upscaled to min area
            ((1080, 1920), (1088, 1920)),
            ((64, 64), (256, 256)),
            ((40, 30), (320, 224)),
            ((7, 5000), (32, 6848)),
            ((224, 224), (256, 256)),
            ((300, 400), (288, 384)),
            ((33, 77), (192, 416)),
        ];
        for &((ih, iw), (oh, ow)) in cases {
            let got = qwen2vl_smart_resize(ih, iw, QWEN_FACTOR, QWEN_MIN, QWEN_MAX);
            assert_eq!(got, (oh, ow), "smart_resize({ih}x{iw})");
            assert_eq!(got.0 % QWEN_FACTOR, 0, "h multiple of {QWEN_FACTOR}");
            assert_eq!(got.1 % QWEN_FACTOR, 0, "w multiple of {QWEN_FACTOR}");
            let area = got.0 * got.1;
            assert!(area >= QWEN_MIN, "area {area} below min");
            assert!(area <= QWEN_MAX, "area {area} above max");
        }
    }

    #[test]
    fn qwen_resize_dims_multiple_of_32() {
        let p = ImageProcessor::qwen2vl(16, 2, 2, QWEN_MIN, QWEN_MAX);
        for &(h, w) in &[(123usize, 456usize), (1000, 1), (2, 9000), (777, 333)] {
            let (th, tw) = p.target_size(h, w).unwrap();
            assert_eq!(th % 32, 0, "{h}x{w} -> h%32");
            assert_eq!(tw % 32, 0, "{h}x{w} -> w%32");
        }
    }

    /// Build a deterministic CHW buffer identical to the HF reference script:
    /// `value = ((c·H + y)·W + x) % 256`, then `(v/255 − 0.5)/0.5`.
    fn synthetic_normalized_chw(h: usize, w: usize) -> Vec<f32> {
        let mut chw = vec![0.0f32; 3 * h * w];
        let plane = h * w;
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let raw = (((c * h + y) * w + x) % 256) as f32;
                    chw[c * plane + y * w + x] = (raw / 255.0 - 0.5) / 0.5;
                }
            }
        }
        chw
    }

    #[test]
    fn qwen_pack_shapes_and_grid() {
        let p = ImageProcessor::qwen2vl(16, 2, 2, QWEN_MIN, QWEN_MAX);
        // 64x96 image: gh=4, gw=6, both multiples of merge=2.
        let (h, w) = (64usize, 96usize);
        let chw = synthetic_normalized_chw(h, w);
        let out = p.pack_qwen2vl_chw(&chw, h, w).unwrap();
        assert_eq!(out.grid_thw(), (1, 4, 6));
        assert_eq!(out.embed_dim, 3 * 2 * 16 * 16); // 1536
        assert_eq!(out.num_patches(), 4 * 6);
        assert_eq!(out.pixel_values.len(), out.num_patches() * out.embed_dim);
        assert_eq!(out.num_merged_tokens(2), 1 * 2 * 3); // (gh/2)·(gw/2)
        // N == gh·gw
        assert_eq!(out.num_patches(), out.grid_h * out.grid_w);
    }

    #[test]
    fn qwen_pack_normalize_range() {
        let p = ImageProcessor::qwen2vl(16, 2, 2, QWEN_MIN, QWEN_MAX);
        let (h, w) = (32usize, 32usize);
        let chw = synthetic_normalized_chw(h, w);
        let out = p.pack_qwen2vl_chw(&chw, h, w).unwrap();
        // mean=std=0.5 maps [0,255] -> [-1, 1].
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &v in &out.pixel_values {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(lo >= -1.0001 && hi <= 1.0001, "range [{lo},{hi}] outside [-1,1]");
        assert!((lo + 1.0).abs() < 1e-6, "min should hit -1, got {lo}");
    }

    #[test]
    fn qwen_pack_matches_hf_values() {
        // 32x32 synthetic image (gh=gw=2, one merge block) cross-checked
        // against HF `Qwen2VLImageProcessor` (/tmp/hf_pack.npy reference run).
        let p = ImageProcessor::qwen2vl(16, 2, 2, QWEN_MIN, QWEN_MAX);
        let (h, w) = (32usize, 32usize);
        let chw = synthetic_normalized_chw(h, w);
        let out = p.pack_qwen2vl_chw(&chw, h, w).unwrap();
        let v = &out.pixel_values;
        let dim = out.embed_dim;
        let at = |row: usize, col: usize| v[row * dim + col];
        // Exact values from the HF reference run.
        assert!((at(0, 0) - (-1.0)).abs() < 1e-6, "v[0][0]");
        assert!((at(0, 5) - (-0.96078431)).abs() < 1e-6, "v[0][5]");
        assert!((at(0, 1535) - 0.87450980).abs() < 1e-6, "v[0][1535]");
        assert!((at(1, 0) - (-0.87450980)).abs() < 1e-6, "v[1][0]");
        assert!((at(3, 768) - (-0.87450980)).abs() < 1e-6, "v[3][768]");
        // Whole-matrix checksum (HF sum == 0 for this symmetric synthetic).
        let sum: f64 = v.iter().map(|&x| x as f64).sum();
        assert!(sum.abs() < 1e-3, "checksum {sum} (HF: 0)");
    }

    #[test]
    fn qwen_pack_temporal_duplication() {
        // The two temporal slots must hold identical pixels for a still image.
        let p = ImageProcessor::qwen2vl(16, 2, 2, QWEN_MIN, QWEN_MAX);
        let (h, w) = (32usize, 32usize);
        let chw = synthetic_normalized_chw(h, w);
        let out = p.pack_qwen2vl_chw(&chw, h, w).unwrap();
        // Column layout per patch row: (C, tp, P, P). For C=0, tp slot 0
        // occupies cols [0, 256), tp slot 1 occupies [256, 512).
        let dim = out.embed_dim;
        for row in 0..out.num_patches() {
            for k in 0..(16 * 16) {
                let s0 = out.pixel_values[row * dim + k];
                let s1 = out.pixel_values[row * dim + 16 * 16 + k];
                assert_eq!(s0, s1, "temporal slots differ at row {row} k {k}");
            }
        }
    }

    #[test]
    fn qwen_pack_rejects_misaligned() {
        let p = ImageProcessor::qwen2vl(16, 2, 2, QWEN_MIN, QWEN_MAX);
        // 48x32: gh=3 not a multiple of merge=2.
        let chw = synthetic_normalized_chw(48, 32);
        assert!(p.pack_qwen2vl_chw(&chw, 48, 32).is_err());
    }
}
