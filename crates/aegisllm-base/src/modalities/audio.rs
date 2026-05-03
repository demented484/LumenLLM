/// Audio encoder stub (Phase 8.3).
///
/// Planned implementation: log-mel spectrogram preprocessing (CPU SIMD via
/// pulp) + Whisper-style or Conformer transformer encoder.
/// Output shape: `[num_audio_tokens, hidden_dim]`.
///
/// **Phase 8 stub** — `encode` returns `Unsupported`.
use crate::error::{AegisError, Result};
use super::EncodedTokens;

/// Audio preprocessing + encoder configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioEncoderConfig {
    /// Sample rate in Hz (e.g. 16000 for Whisper).
    pub sample_rate: u32,
    /// Number of mel filter banks.
    pub n_mels: usize,
    /// FFT window size in samples.
    pub n_fft: usize,
    /// Hop size (stride) in samples.
    pub hop_length: usize,
    /// Hidden dimension of the audio transformer.
    pub hidden_dim: usize,
    /// Number of transformer layers in the encoder.
    pub num_layers: usize,
}

impl AudioEncoderConfig {
    /// Number of audio tokens produced for a given number of input samples.
    pub fn num_tokens(&self, num_samples: usize) -> usize {
        num_samples.div_ceil(self.hop_length)
    }
}

/// Whisper / Conformer-style audio encoder.
///
/// Preprocessing path (log-mel spectrogram) is fully implemented in Rust;
/// the transformer encoder forward pass returns `Unsupported` until the
/// kernels land.
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    pub config: AudioEncoderConfig,
}

impl AudioEncoder {
    pub fn new(config: AudioEncoderConfig) -> Self {
        Self { config }
    }

    /// Compute the log-mel spectrogram for `pcm_samples` (mono f32 at
    /// `config.sample_rate`). Returns a flat `[num_frames * n_mels]` row-major
    /// array with `num_frames = ceil(num_samples / hop_length)`.
    ///
    /// Spec matches Whisper's preprocessing convention:
    /// - frames are extracted with stride = `hop_length`,
    /// - per frame: Hann window of length `n_fft`, real-valued DFT,
    ///   magnitude squared, mel filterbank, `log10(max(mel, 1e-10))`.
    ///
    /// This implementation uses a naive O(n_fft²) DFT per frame — sufficient
    /// for short clips and for tests; production calls should swap in an FFT.
    pub fn log_mel_spectrogram(&self, pcm_samples: &[f32]) -> Vec<f32> {
        let num_frames = self.config.num_tokens(pcm_samples.len());
        let n_fft = self.config.n_fft;
        let n_mels = self.config.n_mels;
        let hop = self.config.hop_length;
        let sample_rate = self.config.sample_rate;

        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                let phase = (std::f32::consts::PI * 2.0 * i as f32) / (n_fft as f32 - 1.0);
                0.5 - 0.5 * phase.cos()
            })
            .collect();

        let mel_filters = mel_filterbank(n_mels, n_fft, sample_rate);
        let half = n_fft / 2 + 1;
        let mut out = Vec::with_capacity(num_frames * n_mels);
        let mut frame_buf = vec![0.0f32; n_fft];
        let mut power = vec![0.0f32; half];

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;
            for i in 0..n_fft {
                let pos = start + i;
                frame_buf[i] = if pos < pcm_samples.len() {
                    pcm_samples[pos] * window[i]
                } else {
                    0.0
                };
            }
            // Naive DFT: compute |X[k]|² for k in 0..half.
            for k in 0..half {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                let kf = k as f32;
                for (n, &x) in frame_buf.iter().enumerate() {
                    let theta = -2.0 * std::f32::consts::PI * kf * n as f32 / n_fft as f32;
                    re += x * theta.cos();
                    im += x * theta.sin();
                }
                power[k] = re * re + im * im;
            }
            // Apply mel filterbank.
            for m in 0..n_mels {
                let mut acc = 0.0f32;
                let filter = &mel_filters[m * half..(m + 1) * half];
                for k in 0..half {
                    acc += filter[k] * power[k];
                }
                out.push(acc.max(1e-10).log10());
            }
        }
        out
    }

    /// Encode raw PCM audio (mono f32 samples at `config.sample_rate`) into
    /// audio tokens.
    ///
    /// **Phase 8 stub** — preprocessing (`log_mel_spectrogram`) is implemented;
    /// the transformer encoder forward returns `Unsupported`.
    #[allow(unused_variables)]
    pub fn encode(&self, pcm_samples: &[f32]) -> Result<EncodedTokens> {
        Err(AegisError::Unsupported(
            "audio encoder transformer not yet implemented (Phase 8.3); log-mel preprocessing is available via log_mel_spectrogram".into(),
        ))
    }
}

/// Convert a frequency in Hz to mel scale (Slaney / Whisper convention).
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Inverse of `hz_to_mel`.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Build a triangular mel filterbank: `[n_mels * half_fft]` row-major.
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<f32> {
    let half = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(nyquist);

    // n_mels + 2 mel-spaced points → n_mels triangular filters.
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    // Map each hz to a fractional FFT bin.
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * n_fft as f32 / sample_rate as f32)
        .collect();

    let mut filters = vec![0.0f32; n_mels * half];
    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        for k in 0..half {
            let kf = k as f32;
            let value = if kf < left || kf > right {
                0.0
            } else if kf <= center {
                (kf - left) / (center - left).max(1e-10)
            } else {
                (right - kf) / (right - center).max(1e-10)
            };
            filters[m * half + k] = value;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::{AudioEncoder, AudioEncoderConfig};

    #[test]
    fn audio_encoder_num_tokens() {
        let cfg = AudioEncoderConfig {
            sample_rate: 16000,
            n_mels: 128,
            n_fft: 400,
            hop_length: 160,
            hidden_dim: 1280,
            num_layers: 32,
        };
        // 16000 samples (1 second) → ceil(16000/160) = 100 tokens
        assert_eq!(cfg.num_tokens(16000), 100);
    }

    #[test]
    fn audio_encoder_encode_returns_unsupported() {
        let enc = AudioEncoder::new(AudioEncoderConfig {
            sample_rate: 16000,
            n_mels: 128,
            n_fft: 400,
            hop_length: 160,
            hidden_dim: 1280,
            num_layers: 32,
        });
        let err = enc.encode(&[]).unwrap_err();
        assert!(matches!(err, crate::error::AegisError::Unsupported(_)));
    }

    #[test]
    fn log_mel_shape_matches_expected_frames() {
        // Use a small FFT to keep the naive DFT fast in tests.
        let enc = AudioEncoder::new(AudioEncoderConfig {
            sample_rate: 8000,
            n_mels: 16,
            n_fft: 64,
            hop_length: 32,
            hidden_dim: 64,
            num_layers: 1,
        });
        // 256 samples → ceil(256/32) = 8 frames
        let pcm = vec![0.5f32; 256];
        let mel = enc.log_mel_spectrogram(&pcm);
        assert_eq!(mel.len(), 8 * 16);
        // All values must be finite (silence epsilon prevents -inf).
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn log_mel_silence_produces_floor() {
        let enc = AudioEncoder::new(AudioEncoderConfig {
            sample_rate: 8000,
            n_mels: 8,
            n_fft: 64,
            hop_length: 32,
            hidden_dim: 64,
            num_layers: 1,
        });
        let pcm = vec![0.0f32; 128];
        let mel = enc.log_mel_spectrogram(&pcm);
        // log10(1e-10) = -10
        assert!(mel.iter().all(|&v| (v - (-10.0)).abs() < 1e-3));
    }

    #[test]
    fn log_mel_sinusoid_has_higher_energy_than_silence() {
        let enc = AudioEncoder::new(AudioEncoderConfig {
            sample_rate: 8000,
            n_mels: 8,
            n_fft: 64,
            hop_length: 32,
            hidden_dim: 64,
            num_layers: 1,
        });
        let pcm: Vec<f32> = (0..128)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 8000.0).sin())
            .collect();
        let mel = enc.log_mel_spectrogram(&pcm);
        let max_energy = mel.iter().cloned().fold(f32::MIN, f32::max);
        // 1 kHz sinusoid should produce a mel value well above silence floor.
        assert!(max_energy > -5.0, "max mel = {}", max_energy);
    }
}
