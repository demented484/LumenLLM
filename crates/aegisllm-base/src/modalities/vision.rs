/// Vision encoder stub (Phase 8.2).
///
/// Planned implementation: SigLIP / InternViT-style patch embedding
/// followed by a standard transformer encoder.
/// Output shape: `[num_patches, hidden_dim]`.
///
/// **Phase 8 stub** — `encode` returns `Unsupported` until the ViT
/// kernels and weight loader are implemented.
use crate::error::{AegisError, Result};
use super::EncodedTokens;

/// Configuration for the vision encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionEncoderConfig {
    /// Input image height in pixels.
    pub image_height: usize,
    /// Input image width in pixels.
    pub image_width: usize,
    /// Patch size (square); number of patches = (H/P) × (W/P).
    pub patch_size: usize,
    /// Hidden dimension of the vision transformer.
    pub hidden_dim: usize,
    /// Number of transformer layers in the encoder.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
}

impl VisionEncoderConfig {
    /// Number of spatial patches produced by this config.
    pub fn num_patches(&self) -> usize {
        (self.image_height / self.patch_size) * (self.image_width / self.patch_size)
    }
}

/// ViT-style vision encoder.
///
/// Phase 8 stub — holds config only; all operations return `Unsupported`.
#[derive(Debug, Clone)]
pub struct VisionEncoder {
    pub config: VisionEncoderConfig,
}

impl VisionEncoder {
    pub fn new(config: VisionEncoderConfig) -> Self {
        Self { config }
    }

    /// Extract raw patches from an `[H, W, 3]` row-major RGB image.
    ///
    /// Returns a flat `[num_patches, patch_size² × 3]` row-major array suitable
    /// for the input projection (`Conv2d` with kernel = stride = patch_size in
    /// PyTorch reference implementations).
    ///
    /// This is the deterministic, weight-free preprocessing step that runs
    /// before the patch-embedding linear projection.
    pub fn extract_patches(&self, image_hwc: &[f32]) -> Result<Vec<f32>> {
        let h = self.config.image_height;
        let w = self.config.image_width;
        let p = self.config.patch_size;
        if h % p != 0 || w % p != 0 || p == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "vision: image {}x{} not divisible by patch_size {}",
                h, w, p
            )));
        }
        if image_hwc.len() != h * w * 3 {
            return Err(AegisError::InvalidPlan(format!(
                "vision: image data has {} elements, expected {} = {}×{}×3",
                image_hwc.len(), h * w * 3, h, w
            )));
        }
        let patch_dim = p * p * 3;
        let num_patches = self.config.num_patches();
        let mut out = Vec::with_capacity(num_patches * patch_dim);
        let patches_per_row = w / p;
        for py in 0..h / p {
            for px in 0..patches_per_row {
                for dy in 0..p {
                    let row = py * p + dy;
                    let row_start = (row * w + px * p) * 3;
                    let len = p * 3;
                    out.extend_from_slice(&image_hwc[row_start..row_start + len]);
                }
            }
        }
        Ok(out)
    }

    /// Encode a raw image (RGB f32, shape `[H, W, 3]`) into patch tokens.
    ///
    /// **Phase 8 stub** — `extract_patches` (the deterministic preprocessing)
    /// is implemented; the patch-embedding projection + transformer encoder
    /// returns `Unsupported`.
    #[allow(unused_variables)]
    pub fn encode(&self, image_hwc: &[f32]) -> Result<EncodedTokens> {
        Err(AegisError::Unsupported(
            "vision encoder transformer not yet implemented (Phase 8.2); raw patch extraction is available via extract_patches".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{VisionEncoder, VisionEncoderConfig};

    #[test]
    fn vision_encoder_config_num_patches() {
        let cfg = VisionEncoderConfig {
            image_height: 224,
            image_width: 224,
            patch_size: 14,
            hidden_dim: 1024,
            num_layers: 24,
            num_heads: 16,
        };
        assert_eq!(cfg.num_patches(), 256);
    }

    #[test]
    fn vision_encoder_encode_returns_unsupported() {
        let enc = VisionEncoder::new(VisionEncoderConfig {
            image_height: 224,
            image_width: 224,
            patch_size: 14,
            hidden_dim: 1024,
            num_layers: 24,
            num_heads: 16,
        });
        let err = enc.encode(&[]).unwrap_err();
        assert!(matches!(err, crate::error::AegisError::Unsupported(_)));
    }

    #[test]
    fn extract_patches_produces_expected_shape() {
        let enc = VisionEncoder::new(VisionEncoderConfig {
            image_height: 4,
            image_width: 4,
            patch_size: 2,
            hidden_dim: 16,
            num_layers: 1,
            num_heads: 1,
        });
        let image: Vec<f32> = (0..(4 * 4 * 3)).map(|i| i as f32).collect();
        let patches = enc.extract_patches(&image).unwrap();
        // 4×4 image with patch=2 → 4 patches, each patch_dim = 2×2×3 = 12.
        assert_eq!(patches.len(), 4 * 12);
    }

    #[test]
    fn extract_patches_rejects_wrong_image_size() {
        let enc = VisionEncoder::new(VisionEncoderConfig {
            image_height: 4,
            image_width: 4,
            patch_size: 2,
            hidden_dim: 16,
            num_layers: 1,
            num_heads: 1,
        });
        assert!(enc.extract_patches(&[0.0; 5]).is_err());
    }

    #[test]
    fn extract_patches_first_patch_top_left_corner() {
        let enc = VisionEncoder::new(VisionEncoderConfig {
            image_height: 2,
            image_width: 2,
            patch_size: 2,
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 1,
        });
        // Single patch covering the whole image: 2*2*3 = 12 elements.
        let image: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let patches = enc.extract_patches(&image).unwrap();
        assert_eq!(patches, image);
    }
}
