/// Multimodal pipeline framework (Phase 8).
///
/// Each submodule owns one encoder type. The `Modality` enum selects which
/// encoders are active for a given inference session. All encoders produce a
/// `[num_tokens, hidden_dim]` token sequence that fuses into the LLM's input
/// embedding stream via `fusion.rs`.
///
/// Phase 8 status: all encoder implementations return `AegisError::Unsupported`.
/// The public types and trait surface are stable; kernels land in Phase 8.x.
pub mod audio;
pub mod fusion;
pub mod image_preprocess;
pub mod mrope;
pub mod video;
pub mod vision;

use crate::error::{AegisError, Result};

/// Which modality encoders are active for this session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Modality {
    /// Plain text input — no encoder needed.
    Text,
    /// Single image input (ViT-style patch embedding).
    Vision,
    /// Audio input (log-mel spectrogram + transformer encoder).
    Audio,
    /// Video input (sampled frames processed by vision encoder + temporal fusion).
    Video,
    /// Interleaved text + image.
    TextVision,
    /// Interleaved text + audio.
    TextAudio,
    /// Interleaved text + image + audio (Nemotron 3 Omni full path).
    TextVisionAudio,
}

impl Modality {
    pub fn has_vision(&self) -> bool {
        matches!(self, Self::Vision | Self::Video | Self::TextVision | Self::TextVisionAudio)
    }

    pub fn has_audio(&self) -> bool {
        matches!(self, Self::Audio | Self::TextAudio | Self::TextVisionAudio)
    }

    pub fn has_video(&self) -> bool {
        matches!(self, Self::Video)
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vision => "vision",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::TextVision => "text+vision",
            Self::TextAudio => "text+audio",
            Self::TextVisionAudio => "text+vision+audio",
        }
    }
}

/// Encoded token sequence produced by a modality encoder.
/// Shape: `[num_tokens, hidden_dim]` in row-major f32.
#[derive(Debug, Clone)]
pub struct EncodedTokens {
    pub data: Vec<f32>,
    pub num_tokens: usize,
    pub hidden_dim: usize,
}

impl EncodedTokens {
    pub fn new(data: Vec<f32>, num_tokens: usize, hidden_dim: usize) -> Result<Self> {
        if data.len() != num_tokens * hidden_dim {
            return Err(AegisError::InvalidPlan(format!(
                "EncodedTokens data length mismatch: data={} num_tokens={} hidden_dim={}",
                data.len(),
                num_tokens,
                hidden_dim
            )));
        }
        Ok(Self { data, num_tokens, hidden_dim })
    }
}

#[cfg(test)]
mod tests {
    use super::Modality;

    #[test]
    fn modality_flags_are_consistent() {
        assert!(Modality::TextVisionAudio.has_vision());
        assert!(Modality::TextVisionAudio.has_audio());
        assert!(!Modality::TextVisionAudio.has_video());
        assert!(!Modality::Text.has_vision());
        assert!(!Modality::Audio.has_vision());
        assert!(Modality::Video.has_vision());
        assert!(Modality::Video.has_video());
    }

    #[test]
    fn encoded_tokens_rejects_shape_mismatch() {
        let err = super::EncodedTokens::new(vec![0.0; 10], 3, 4);
        assert!(err.is_err());
    }

    #[test]
    fn encoded_tokens_accepts_valid_shape() {
        let t = super::EncodedTokens::new(vec![0.0; 12], 3, 4).unwrap();
        assert_eq!(t.num_tokens, 3);
        assert_eq!(t.hidden_dim, 4);
    }
}
