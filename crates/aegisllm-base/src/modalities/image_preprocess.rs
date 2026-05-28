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
}

impl ImageProcessor {
    /// Builder for Gemma-4-it / Gemma-4-vision: aspect-preserving resize with
    /// max 280 post-pool tokens (= 2520 pre-pool patches @ pool=3), patch=16,
    /// rescale=1/255, no mean/std normalization.
    pub fn gemma4() -> Self {
        // Pre-pool patch budget = max_soft_tokens × pooling². Gemma-4 native
        // spec is max_soft_tokens=280, pooling=3 → 2520. For OCR-heavy
        // inputs, llama.cpp's `--image-max-tokens` lifts this cap by feeding
        // a higher-resolution resize (the trained projector tolerates more
        // tokens, the position table goes to 10240). We match that lever via
        // `AEGIS_VISION_MAX_PATCHES` (e.g. 9000 ≈ 1000 soft tokens).
        let max_patches = std::env::var("AEGIS_VISION_MAX_PATCHES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2520);
        Self {
            patch_size: 16,
            pooling_kernel_size: 3,
            resize: ResizeStrategy::AspectPreserving {
                max_patches,
                side_multiple: 48, // pooling * patch = 3 * 16
            },
            rescale_factor: 1.0 / 255.0,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_landscape_resize() {
        let p = ImageProcessor::gemma4();
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
        let p = ImageProcessor::gemma4();
        assert!(p.target_size(0, 10).is_err());
    }
}
