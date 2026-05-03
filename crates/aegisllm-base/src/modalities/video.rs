/// Video encoder stub (Phase 8.4).
///
/// Planned implementation: uniform frame sampling + per-frame vision encoder
/// (shared weights with `vision.rs`) + temporal token aggregation.
/// Output shape: `[num_frames × patches_per_frame, hidden_dim]`.
///
/// **Phase 8 stub** — `encode` returns `Unsupported`.
use crate::error::{AegisError, Result};
use super::{EncodedTokens, vision::VisionEncoderConfig};

/// Video encoding configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoEncoderConfig {
    /// Per-frame vision encoder configuration.
    pub frame_encoder: VisionEncoderConfig,
    /// Maximum number of frames to sample from the input video.
    pub max_frames: usize,
    /// Frames per second to sample at.
    pub sample_fps: f32,
}

impl VideoEncoderConfig {
    /// Total patch tokens for `num_frames` sampled frames.
    pub fn tokens_for_frames(&self, num_frames: usize) -> usize {
        num_frames * self.frame_encoder.num_patches()
    }
}

/// Video encoder (frame sampling + temporal vision).
///
/// Phase 8 stub — holds config only; all operations return `Unsupported`.
#[derive(Debug, Clone)]
pub struct VideoEncoder {
    pub config: VideoEncoderConfig,
}

impl VideoEncoder {
    pub fn new(config: VideoEncoderConfig) -> Self {
        Self { config }
    }

    /// Encode a video given as a sequence of RGB frames (each `[H, W, 3]` f32)
    /// at the configured sample FPS.
    ///
    /// **Phase 8 stub** — returns `Unsupported`.
    #[allow(unused_variables)]
    pub fn encode(&self, frames_hwc: &[&[f32]]) -> Result<EncodedTokens> {
        Err(AegisError::Unsupported(
            "video encoder not yet implemented (Phase 8.4)".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{VideoEncoder, VideoEncoderConfig};
    use crate::modalities::vision::VisionEncoderConfig;

    fn make_config() -> VideoEncoderConfig {
        VideoEncoderConfig {
            frame_encoder: VisionEncoderConfig {
                image_height: 224,
                image_width: 224,
                patch_size: 14,
                hidden_dim: 1024,
                num_layers: 24,
                num_heads: 16,
            },
            max_frames: 8,
            sample_fps: 1.0,
        }
    }

    #[test]
    fn video_encoder_token_count() {
        let cfg = make_config();
        // 8 frames × 16×16 patches = 8 × 256 = 2048
        assert_eq!(cfg.tokens_for_frames(8), 2048);
    }

    #[test]
    fn video_encoder_encode_returns_unsupported() {
        let enc = VideoEncoder::new(make_config());
        let err = enc.encode(&[]).unwrap_err();
        assert!(matches!(err, crate::error::AegisError::Unsupported(_)));
    }
}
