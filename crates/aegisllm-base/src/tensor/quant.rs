use std::fmt::{Display, Formatter};

use crate::error::{AegisError, Result};
use crate::tensor::{TensorDType, TensorInfo};

pub const QK_NVFP4: usize = 64;
pub const QK_NVFP4_SUB: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QuantFormat {
    DenseF32,
    F16,
    Bf16,
    Nvfp4,
    Fp8E4M3Block,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TensorCorePrecision {
    Tf32,
    F16,
    Bf16,
    Fp8,
    Fp4,
    Int8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleGranularity {
    None,
    Scalar,
    PerBlock { values: usize },
    TwoLevel { values: usize, super_values: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantFormatDescriptor {
    pub format: QuantFormat,
    pub label: &'static str,
    pub packed_weight_dtype: Option<TensorDType>,
    pub logical_bits_per_value: Option<u8>,
    pub values_per_packed_byte: Option<u8>,
    pub scale_dtype: Option<TensorDType>,
    pub scale_granularity: ScaleGranularity,
    pub input_granularity: ScaleGranularity,
    pub native_tensor_core_precision: Option<TensorCorePrecision>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4LinearSpec {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub packed_bytes: usize,
    pub scale_bytes: usize,
    pub input_scale: f32,
    pub output_scale: f32,
}

impl QuantFormat {
    pub fn descriptor(self) -> QuantFormatDescriptor {
        match self {
            Self::DenseF32 => QuantFormatDescriptor {
                format: self,
                label: "dense_f32",
                packed_weight_dtype: Some(TensorDType::F32),
                logical_bits_per_value: Some(32),
                values_per_packed_byte: None,
                scale_dtype: None,
                scale_granularity: ScaleGranularity::None,
                input_granularity: ScaleGranularity::None,
                native_tensor_core_precision: Some(TensorCorePrecision::Tf32),
            },
            Self::F16 => QuantFormatDescriptor {
                format: self,
                label: "f16",
                packed_weight_dtype: Some(TensorDType::F16),
                logical_bits_per_value: Some(16),
                values_per_packed_byte: None,
                scale_dtype: None,
                scale_granularity: ScaleGranularity::None,
                input_granularity: ScaleGranularity::None,
                native_tensor_core_precision: Some(TensorCorePrecision::F16),
            },
            Self::Bf16 => QuantFormatDescriptor {
                format: self,
                label: "bf16",
                packed_weight_dtype: Some(TensorDType::BF16),
                logical_bits_per_value: Some(16),
                values_per_packed_byte: None,
                scale_dtype: None,
                scale_granularity: ScaleGranularity::None,
                input_granularity: ScaleGranularity::None,
                native_tensor_core_precision: Some(TensorCorePrecision::Bf16),
            },
            Self::Nvfp4 => QuantFormatDescriptor {
                format: self,
                label: "nvfp4",
                packed_weight_dtype: Some(TensorDType::U8),
                logical_bits_per_value: Some(4),
                values_per_packed_byte: Some(2),
                scale_dtype: Some(TensorDType::F8E4M3),
                scale_granularity: ScaleGranularity::TwoLevel {
                    values: QK_NVFP4_SUB,
                    super_values: QK_NVFP4,
                },
                input_granularity: ScaleGranularity::PerBlock {
                    values: QK_NVFP4_SUB,
                },
                native_tensor_core_precision: Some(TensorCorePrecision::Fp4),
            },
            Self::Fp8E4M3Block => QuantFormatDescriptor {
                format: self,
                label: "fp8_e4m3_block",
                packed_weight_dtype: Some(TensorDType::F8E4M3),
                logical_bits_per_value: Some(8),
                values_per_packed_byte: Some(1),
                scale_dtype: Some(TensorDType::F8E4M3),
                scale_granularity: ScaleGranularity::PerBlock { values: 32 },
                input_granularity: ScaleGranularity::PerBlock { values: 32 },
                native_tensor_core_precision: Some(TensorCorePrecision::Fp8),
            },
            Self::Unknown => QuantFormatDescriptor {
                format: self,
                label: "unknown",
                packed_weight_dtype: None,
                logical_bits_per_value: None,
                values_per_packed_byte: None,
                scale_dtype: None,
                scale_granularity: ScaleGranularity::None,
                input_granularity: ScaleGranularity::None,
                native_tensor_core_precision: None,
            },
        }
    }

    pub fn label(self) -> &'static str {
        self.descriptor().label
    }

    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Nvfp4 | Self::Fp8E4M3Block)
    }

    pub fn is_dense(self) -> bool {
        matches!(self, Self::DenseF32 | Self::F16 | Self::Bf16)
    }
}

impl Display for QuantFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl TensorCorePrecision {
    pub fn label(self) -> &'static str {
        match self {
            Self::Tf32 => "tf32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Fp4 => "fp4",
            Self::Int8 => "int8",
        }
    }
}

impl Display for TensorCorePrecision {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightQuantization {
    None,
    Fp16,
    Bf16,
    Q8_0,
    /// GPTQ / AWQ INT8 weight-only with per-group scales (groupsize 128).
    Int8,
    /// GPTQ / AWQ INT4 weight-only with per-group scales (groupsize 128).
    Int4,
    Nvfp4,
    /// FP8 E4M3 block-quantized weights (DeepSeek-style 128x128 blocks).
    /// Used by Qwen3.5-9B-FP8 and similar checkpoints.
    Fp8E4M3Block,
    Unknown,
}

impl WeightQuantization {
    pub fn parse_guess(value: &str) -> Self {
        let lower = value.to_ascii_lowercase();
        if lower.contains("nvfp4") || lower.contains("fp4") {
            Self::Nvfp4
        } else if lower.contains("fp8") || lower.contains("float8") {
            // DeepSeek-style block FP8 quant_method ("fp8") or torch dtype "float8_e4m3fn".
            Self::Fp8E4M3Block
        } else if lower.contains("q8_0") || lower.contains("q8-0") {
            Self::Q8_0
        } else if lower.contains("int4") || lower.contains("gptq-4") || lower.contains("awq-4") {
            Self::Int4
        } else if lower.contains("int8") || lower.contains("gptq-8") || lower.contains("awq-8") {
            Self::Int8
        } else if lower.contains("bf16") || lower.contains("bfloat16") {
            Self::Bf16
        } else if lower.contains("fp16") || lower.contains("float16") {
            Self::Fp16
        } else if lower == "none" || lower == "f32" || lower == "float32" {
            Self::None
        } else {
            Self::Unknown
        }
    }

    pub fn bytes_per_weight_hint(self) -> Option<f32> {
        match self {
            Self::None => Some(4.0),
            Self::Fp16 | Self::Bf16 => Some(2.0),
            Self::Q8_0 | Self::Int8 => Some(1.125),
            // FP8: 1 byte per weight + per-block scales (~0.06 b extra for 128x128 blocks).
            Self::Fp8E4M3Block => Some(1.06),
            Self::Int4 => Some(0.5625),
            Self::Nvfp4 => Some(0.625),
            Self::Unknown => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Nvfp4 => "nvfp4",
            Self::Fp8E4M3Block => "fp8",
            Self::Unknown => "unknown",
        }
    }

    pub fn format_hint(self) -> QuantFormat {
        match self {
            Self::None => QuantFormat::DenseF32,
            Self::Fp16 => QuantFormat::F16,
            Self::Bf16 => QuantFormat::Bf16,
            Self::Nvfp4 => QuantFormat::Nvfp4,
            Self::Fp8E4M3Block => QuantFormat::Fp8E4M3Block,
            Self::Q8_0 | Self::Int8 | Self::Int4 | Self::Unknown => QuantFormat::Unknown,
        }
    }
}

impl Display for WeightQuantization {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvCacheQuantization {
    F16,
    Bf16,
    Q8_0,
    Fp8,
    /// Blackwell-native FP4 (NVFP4 / E2M1) — 4-bit per element with per-block scales.
    Nvfp4,
}

impl KvCacheQuantization {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "q8_0" | "q8-0" => Some(Self::Q8_0),
            "fp8" | "f8" => Some(Self::Fp8),
            "nvfp4" | "fp4" | "f4" => Some(Self::Nvfp4),
            _ => None,
        }
    }

    pub fn bytes_per_element(self) -> f32 {
        match self {
            Self::F16 | Self::Bf16 => 2.0,
            Self::Q8_0 | Self::Fp8 => 1.0,
            Self::Nvfp4 => 0.5,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

impl Display for KvCacheQuantization {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl Nvfp4LinearSpec {
    pub fn from_tensors(
        name: &str,
        weight: &TensorInfo,
        scales: &TensorInfo,
        input_scale: f32,
        output_scale: f32,
    ) -> Result<Self> {
        if weight.dtype != TensorDType::U8 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be U8 packed NVFP4, got {:?}",
                weight.name, weight.dtype
            )));
        }
        if scales.dtype != TensorDType::F8E4M3 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be F8_E4M3 NVFP4 scales, got {:?}",
                scales.name, scales.dtype
            )));
        }
        if weight.shape.len() != 2 || scales.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "NVFP4 linear `{name}` expects 2D weight and scale tensors"
            )));
        }
        let rows = weight.shape[0];
        let packed_cols = weight.shape[1];
        let cols = packed_cols.checked_mul(2).ok_or_else(|| {
            AegisError::InvalidPlan(format!("NVFP4 linear `{name}` column overflow"))
        })?;
        if cols % QK_NVFP4 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "NVFP4 linear `{name}` columns must be divisible by {QK_NVFP4}, got {cols}"
            )));
        }
        let expected_scale_cols = cols / QK_NVFP4 * (QK_NVFP4 / QK_NVFP4_SUB);
        if scales.shape != [rows, expected_scale_cols] {
            return Err(AegisError::InvalidPlan(format!(
                "NVFP4 scale shape mismatch for `{name}`: expected [{rows}, {expected_scale_cols}], got {:?}",
                scales.shape
            )));
        }
        Ok(Self {
            name: name.into(),
            rows,
            cols,
            packed_bytes: weight.data_len_bytes() as usize,
            scale_bytes: scales.data_len_bytes() as usize,
            input_scale,
            output_scale,
        })
    }

    pub fn packed_cols(&self) -> usize {
        self.cols / 2
    }

    pub fn scale_cols(&self) -> usize {
        self.cols / QK_NVFP4 * (QK_NVFP4 / QK_NVFP4_SUB)
    }
}

#[inline(always)]
pub fn decode_nvfp4_nibble_i8(nibble: u8) -> i8 {
    const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    KVALUES_MXFP4[nibble as usize]
}

#[inline(always)]
pub fn decode_ue4m3_with_half_lut(byte: u8) -> f32 {
    const LUT: [f32; 128] = build_ue4m3_with_half_lut();
    LUT[(byte & 0x7f) as usize]
}

const fn build_ue4m3_with_half_lut() -> [f32; 128] {
    let mut lut = [0.0_f32; 128];
    let mut byte = 0usize;
    while byte < 128 {
        let exponent = ((byte as u8 >> 3) & 0x0f) as i32;
        let mantissa = (byte as u8 & 0x07) as f32;
        let raw = if byte == 0 || byte == 0x7f {
            0.0
        } else if exponent == 0 {
            mantissa * 0.001953125
        } else {
            (1.0 + mantissa * 0.125) * pow2i_const(exponent - 7)
        };
        lut[byte] = raw * 0.5;
        byte += 1;
    }
    lut
}

const fn pow2i_const(exp: i32) -> f32 {
    let bits = ((exp + 127) as u32) << 23;
    f32::from_bits(bits)
}

pub fn quantize_input_nvfp4(input: &[f32], input_scale: f32) -> Option<Vec<f32>> {
    if input_scale <= 0.0 || input.is_empty() || !input.len().is_multiple_of(QK_NVFP4_SUB) {
        return None;
    }
    let mut out = vec![0.0_f32; input.len()];
    quantize_input_nvfp4_into(input, input_scale, &mut out);
    Some(out)
}

pub fn quantize_input_nvfp4_into(input: &[f32], input_scale: f32, out: &mut [f32]) {
    debug_assert_eq!(input.len(), out.len());
    let input_scale_inv = 1.0 / input_scale;
    for (src_block, dst_block) in input
        .chunks_exact(QK_NVFP4_SUB)
        .zip(out.chunks_exact_mut(QK_NVFP4_SUB))
    {
        let mut amax = 0.0_f32;
        let mut scaled = [0.0_f32; QK_NVFP4_SUB];
        for (slot, &value) in scaled.iter_mut().zip(src_block.iter()) {
            let scaled_value = value * input_scale_inv;
            *slot = scaled_value;
            amax = amax.max(scaled_value.abs());
        }
        if amax == 0.0 {
            dst_block.fill(0.0);
            continue;
        }
        let block_scale = decode_ue4m3_with_half_lut(fp32_to_ue4m3(amax / 6.0));
        for (dst, &value) in dst_block.iter_mut().zip(scaled.iter()) {
            let quant = decode_nvfp4_nibble_i8(best_index_mxfp4(value, block_scale)) as f32;
            *dst = quant * block_scale * input_scale;
        }
    }
}

fn best_index_mxfp4(x: f32, d: f32) -> u8 {
    if d == 0.0 {
        return 0;
    }
    let mut best = 0_u8;
    let mut best_err = f32::INFINITY;
    for idx in 0..16_u8 {
        let candidate = decode_nvfp4_nibble_i8(idx) as f32 * d;
        let err = (candidate - x).abs();
        if err < best_err {
            best = idx;
            best_err = err;
        }
    }
    best
}

fn fp32_to_ue4m3(mut x: f32) -> u8 {
    if x <= 0.0 {
        return 0;
    }
    if x > 448.0 {
        x = 448.0;
    }
    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xff) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x7) as i32;
    let mut ue4m3_exp = fp32_exp + 7;
    if ue4m3_exp <= 0 {
        let mut man = (x * 512.0 + 0.5) as i32;
        if man > 7 {
            man = 7;
        }
        if man < 1 {
            return 0;
        }
        return man as u8;
    }
    if ue4m3_exp >= 15 {
        return 0x7e;
    }
    let round_bit = ((bits >> 19) & 1) as i32;
    let mut ue4m3_man = fp32_man + round_bit;
    if ue4m3_man > 7 {
        ue4m3_man = 0;
        ue4m3_exp += 1;
        if ue4m3_exp >= 15 {
            return 0x7e;
        }
    }
    ((ue4m3_exp << 3) | ue4m3_man) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tensor(name: &str, dtype: TensorDType, shape: Vec<usize>, bytes: u64) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dtype,
            shape,
            num_elements: bytes as usize,
            data_offsets: (0, bytes),
            file_offsets: (0, bytes),
            shard_name: "s".into(),
            shard_path: PathBuf::from("s"),
        }
    }

    #[test]
    fn nvfp4_spec_validates_scale_shape() {
        let weight = tensor("x.weight", TensorDType::U8, vec![4096, 2048], 4096 * 2048);
        let scales = tensor(
            "x.weight_scale",
            TensorDType::F8E4M3,
            vec![4096, 256],
            4096 * 256,
        );
        let spec = Nvfp4LinearSpec::from_tensors("x", &weight, &scales, 1.0, 1.0).unwrap();
        assert_eq!(spec.rows, 4096);
        assert_eq!(spec.cols, 4096);
    }

    #[test]
    fn ue4m3_lut_applies_half_factor() {
        assert!((decode_ue4m3_with_half_lut(0x40) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nvfp4_nibble_matches_ggml_table() {
        assert_eq!(decode_nvfp4_nibble_i8(0x7), 12);
        assert_eq!(decode_nvfp4_nibble_i8(0xf), -12);
    }

    #[test]
    fn kv_cache_quant_nvfp4_parse_and_label() {
        for alias in ["nvfp4", "fp4", "f4"] {
            let q = KvCacheQuantization::parse(alias).unwrap();
            assert_eq!(q, KvCacheQuantization::Nvfp4);
            assert_eq!(q.label(), "nvfp4");
        }
        assert!((KvCacheQuantization::Nvfp4.bytes_per_element() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn kv_cache_quant_bytes_per_element() {
        assert!((KvCacheQuantization::F16.bytes_per_element() - 2.0).abs() < 1e-6);
        assert!((KvCacheQuantization::Bf16.bytes_per_element() - 2.0).abs() < 1e-6);
        assert!((KvCacheQuantization::Fp8.bytes_per_element() - 1.0).abs() < 1e-6);
        assert!((KvCacheQuantization::Q8_0.bytes_per_element() - 1.0).abs() < 1e-6);
        assert!((KvCacheQuantization::Nvfp4.bytes_per_element() - 0.5).abs() < 1e-6);
    }
}
