use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{AegisError, Result};
use crate::tensor::TensorInfo;
use crate::tensor::quant::Nvfp4LinearSpec;

const CACHE_VERSION: &str = "mxfp4-v1";
const CUTLASS_NVFP4_CACHE_VERSION: &str = "cutlass-nvfp4-e2m1-ue4m3-sm120-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CutlassNvfp4LinearLayout {
    pub logical_n: usize,
    pub logical_k: usize,
    pub packed_k_cols: usize,
    pub source_scale_cols: usize,
    pub scale_rows: usize,
    pub scale_cols: usize,
    pub scale_k_tiles: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CutlassNvfp4HostLinear {
    pub layout: CutlassNvfp4LinearLayout,
    pub payload_e2m1: Vec<u8>,
    pub scales_ue4m3: Vec<u8>,
}

impl CutlassNvfp4LinearLayout {
    pub const SCALE_VEC_SIZE: usize = 16;
    pub const SCALE_ROW_ALIGNMENT: usize = 128;
    pub const SCALE_COL_ALIGNMENT: usize = 4;
    pub const OUTPUT_CHANNEL_ALIGNMENT: usize = 32;

    pub fn for_weight(spec: &Nvfp4LinearSpec) -> Result<Self> {
        if spec.cols % 32 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` CUTLASS NVFP4 layout requires K divisible by 32, got {}",
                spec.name, spec.cols
            )));
        }
        if spec.rows % Self::OUTPUT_CHANNEL_ALIGNMENT != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` CUTLASS NVFP4 layout requires output channels divisible by {}, got {}",
                spec.name,
                Self::OUTPUT_CHANNEL_ALIGNMENT,
                spec.rows
            )));
        }
        let source_scale_cols = spec.cols / Self::SCALE_VEC_SIZE;
        let scale_rows = round_up(spec.rows, Self::SCALE_ROW_ALIGNMENT);
        let scale_cols = round_up(source_scale_cols, Self::SCALE_COL_ALIGNMENT);
        Ok(Self {
            logical_n: spec.rows,
            logical_k: spec.cols,
            packed_k_cols: spec.packed_cols(),
            source_scale_cols,
            scale_rows,
            scale_cols,
            scale_k_tiles: scale_cols / Self::SCALE_COL_ALIGNMENT,
        })
    }

    pub fn payload_bytes(self) -> usize {
        self.logical_n * self.packed_k_cols
    }

    pub fn scale_bytes(self) -> usize {
        self.scale_rows * self.scale_cols
    }

    pub fn swizzled_scale_offset(self, row: usize, k_scale_idx: usize) -> Result<usize> {
        if row >= self.scale_rows || k_scale_idx >= self.scale_cols {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS NVFP4 scale offset out of bounds: row={} k_scale_idx={} shape=[{}, {}]",
                row, k_scale_idx, self.scale_rows, self.scale_cols
            )));
        }
        let m_tile_idx = row >> 7;
        let outer_m_idx = row & 31;
        let inner_m_idx = (row >> 5) & 3;
        let k_tile_idx = k_scale_idx >> 2;
        let inner_k_idx = k_scale_idx & 3;
        Ok(((m_tile_idx * self.scale_k_tiles + k_tile_idx) << 9)
            | (outer_m_idx << 4)
            | (inner_m_idx << 2)
            | inner_k_idx)
    }
}

pub(crate) fn cached_repack_nvfp4_to_cutlass_e2m1_ue4m3_host(
    model_root: &Path,
    spec: &Nvfp4LinearSpec,
    weight: &TensorInfo,
    scales: &TensorInfo,
    packed: &[u8],
    scale_bytes: &[u8],
) -> Result<CutlassNvfp4HostLinear> {
    let layout = CutlassNvfp4LinearLayout::for_weight(spec)?;
    if native_mxfp4_cache_disabled() {
        return repack_nvfp4_to_cutlass_e2m1_ue4m3_host(spec, packed, scale_bytes);
    }
    let path = cutlass_nvfp4_cache_path(model_root, spec, weight, scales, layout);
    let expected_len = layout.payload_bytes() + layout.scale_bytes();
    if let Some(cached) = try_read_cache(&path, expected_len)? {
        let (payload_e2m1, scales_ue4m3) = cached.split_at(layout.payload_bytes());
        return Ok(CutlassNvfp4HostLinear {
            layout,
            payload_e2m1: payload_e2m1.to_vec(),
            scales_ue4m3: scales_ue4m3.to_vec(),
        });
    }
    let repacked = repack_nvfp4_to_cutlass_e2m1_ue4m3_host(spec, packed, scale_bytes)?;
    let mut cache_blob = Vec::with_capacity(expected_len);
    cache_blob.extend_from_slice(&repacked.payload_e2m1);
    cache_blob.extend_from_slice(&repacked.scales_ue4m3);
    write_cache(&path, &cache_blob)?;
    Ok(repacked)
}

pub(crate) fn repack_nvfp4_to_cutlass_e2m1_ue4m3_host(
    spec: &Nvfp4LinearSpec,
    packed: &[u8],
    scales: &[u8],
) -> Result<CutlassNvfp4HostLinear> {
    let layout = CutlassNvfp4LinearLayout::for_weight(spec)?;
    if packed.len() != layout.payload_bytes() || scales.len() != spec.rows * spec.scale_cols() {
        return Err(AegisError::InvalidPlan(format!(
            "`{}` CUTLASS NVFP4 shape mismatch: packed={} expected={} scales={} expected={}",
            spec.name,
            packed.len(),
            layout.payload_bytes(),
            scales.len(),
            spec.rows * spec.scale_cols()
        )));
    }

    let payload_e2m1 = packed.to_vec();
    let mut scales_ue4m3 = vec![0u8; layout.scale_bytes()];
    for row in 0..spec.rows {
        let src_base = row * layout.source_scale_cols;
        for k_scale_idx in 0..layout.source_scale_cols {
            let dst = layout.swizzled_scale_offset(row, k_scale_idx)?;
            scales_ue4m3[dst] = scales[src_base + k_scale_idx];
        }
    }

    Ok(CutlassNvfp4HostLinear {
        layout,
        payload_e2m1,
        scales_ue4m3,
    })
}

pub(crate) fn cached_repack_nvfp4_to_mxfp4_host(
    model_root: &Path,
    spec: &Nvfp4LinearSpec,
    weight: &TensorInfo,
    scales: &TensorInfo,
    packed: &[u8],
    scale_bytes: &[u8],
) -> Result<Vec<u8>> {
    if native_mxfp4_cache_disabled() {
        return repack_nvfp4_to_mxfp4_host(spec, packed, scale_bytes);
    }
    let expected_len = expected_mxfp4_bytes(spec)?;
    let path = native_mxfp4_cache_path(model_root, spec, weight, scales);
    if let Some(cached) = try_read_cache(&path, expected_len)? {
        return Ok(cached);
    }
    let repacked = repack_nvfp4_to_mxfp4_host(spec, packed, scale_bytes)?;
    write_cache(&path, &repacked)?;
    Ok(repacked)
}

pub(crate) fn repack_nvfp4_to_mxfp4_host(
    spec: &Nvfp4LinearSpec,
    packed: &[u8],
    scales: &[u8],
) -> Result<Vec<u8>> {
    if spec.cols % 32 != 0 {
        return Err(AegisError::InvalidPlan(format!(
            "`{}` native MXFP4 repack requires cols divisible by 32, got {}",
            spec.name, spec.cols
        )));
    }
    let packed_cols = spec.cols / 2;
    let nvfp4_scale_cols = spec.cols / 16;
    let mxfp4_blocks_per_row = spec.cols / 32;
    if packed.len() != spec.rows * packed_cols || scales.len() != spec.rows * nvfp4_scale_cols {
        return Err(AegisError::InvalidPlan(format!(
            "`{}` native MXFP4 repack shape mismatch: packed={} expected={} scales={} expected={}",
            spec.name,
            packed.len(),
            spec.rows * packed_cols,
            scales.len(),
            spec.rows * nvfp4_scale_cols
        )));
    }

    let mut repacked = vec![0u8; spec.rows * mxfp4_blocks_per_row * 17];
    let mut values = [0.0f32; 32];
    for row in 0..spec.rows {
        let packed_row = &packed[row * packed_cols..(row + 1) * packed_cols];
        let scale_row = &scales[row * nvfp4_scale_cols..(row + 1) * nvfp4_scale_cols];
        for block in 0..mxfp4_blocks_per_row {
            let col_base = block * 32;
            let mut amax = 0.0f32;
            for (lane, value) in values.iter_mut().enumerate() {
                let col = col_base + lane;
                let group = col / 16;
                let lane_in_group = col % 16;
                let byte = packed_row[group * 8 + lane_in_group / 2];
                let nibble = if lane_in_group % 2 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let scale = decode_ue4m3_half_host(scale_row[group]);
                *value = decode_nvfp4_nibble_host(nibble) * scale;
                amax = amax.max(value.abs());
            }

            let e = compute_e8m0_scale_host(amax);
            let d = e8m0_to_f32_half_host(e);
            let out_base = (row * mxfp4_blocks_per_row + block) * 17;
            repacked[out_base] = e;
            for lane in 0..16 {
                let lo = best_mxfp4_index(values[lane], d);
                let hi = best_mxfp4_index(values[lane + 16], d);
                repacked[out_base + 1 + lane] = lo | (hi << 4);
            }
        }
    }
    Ok(repacked)
}

fn expected_mxfp4_bytes(spec: &Nvfp4LinearSpec) -> Result<usize> {
    if spec.cols % 32 != 0 {
        return Err(AegisError::InvalidPlan(format!(
            "`{}` native MXFP4 repack requires cols divisible by 32, got {}",
            spec.name, spec.cols
        )));
    }
    Ok(spec.rows * (spec.cols / 32) * 17)
}

fn native_mxfp4_cache_disabled() -> bool {
    std::env::var("AEGISLLM_NATIVE_MXFP4_CACHE")
        .map(|value| matches!(value.as_str(), "0" | "false" | "off" | "no"))
        .unwrap_or(false)
}

fn native_mxfp4_cache_path(
    model_root: &Path,
    spec: &Nvfp4LinearSpec,
    weight: &TensorInfo,
    scales: &TensorInfo,
) -> PathBuf {
    let key = format!(
        "{}-r{}-c{}-pb{}-sb{}-is{:08x}-os{:08x}-w{}-{}-{}-s{}-{}-{}.bin",
        sanitize_cache_component(&spec.name),
        spec.rows,
        spec.cols,
        spec.packed_bytes,
        spec.scale_bytes,
        spec.input_scale.to_bits(),
        spec.output_scale.to_bits(),
        sanitize_cache_component(&weight.shard_name),
        weight.file_offsets.0,
        weight.file_offsets.1,
        sanitize_cache_component(&scales.shard_name),
        scales.file_offsets.0,
        scales.file_offsets.1,
    );
    model_root
        .join(".aegis-cache")
        .join(CACHE_VERSION)
        .join(key)
}

fn cutlass_nvfp4_cache_path(
    model_root: &Path,
    spec: &Nvfp4LinearSpec,
    weight: &TensorInfo,
    scales: &TensorInfo,
    layout: CutlassNvfp4LinearLayout,
) -> PathBuf {
    let key = format!(
        "{}-n{}-k{}-pk{}-sr{}-sc{}-pb{}-sb{}-is{:08x}-os{:08x}-w{}-{}-{}-s{}-{}-{}.bin",
        sanitize_cache_component(&spec.name),
        layout.logical_n,
        layout.logical_k,
        layout.packed_k_cols,
        layout.scale_rows,
        layout.scale_cols,
        spec.packed_bytes,
        spec.scale_bytes,
        spec.input_scale.to_bits(),
        spec.output_scale.to_bits(),
        sanitize_cache_component(&weight.shard_name),
        weight.file_offsets.0,
        weight.file_offsets.1,
        sanitize_cache_component(&scales.shard_name),
        scales.file_offsets.0,
        scales.file_offsets.1,
    );
    model_root
        .join(".aegis-cache")
        .join(CUTLASS_NVFP4_CACHE_VERSION)
        .join(key)
}

fn round_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn sanitize_cache_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn try_read_cache(path: &Path, expected_len: usize) -> Result<Option<Vec<u8>>> {
    let Ok(metadata) = fs::metadata(path) else {
        return Ok(None);
    };
    if metadata.len() != expected_len as u64 {
        return Ok(None);
    }
    let mut file = fs::File::open(path)?;
    let mut data = Vec::with_capacity(expected_len);
    file.read_to_end(&mut data)?;
    if data.len() == expected_len {
        Ok(Some(data))
    } else {
        Ok(None)
    }
}

fn write_cache(path: &Path, data: &[u8]) -> Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    fs::create_dir_all(parent)?;
    let tmp = path.with_extension("tmp");
    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(data)?;
        file.sync_data()?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

fn decode_nvfp4_nibble_host(nibble: u8) -> f32 {
    match nibble & 0x0f {
        0 => 0.0,
        1 => 1.0,
        2 => 2.0,
        3 => 3.0,
        4 => 4.0,
        5 => 6.0,
        6 => 8.0,
        7 => 12.0,
        8 => 0.0,
        9 => -1.0,
        10 => -2.0,
        11 => -3.0,
        12 => -4.0,
        13 => -6.0,
        14 => -8.0,
        _ => -12.0,
    }
}

fn decode_ue4m3_half_host(byte: u8) -> f32 {
    let byte = byte & 0x7f;
    if byte == 0 || byte == 0x7f {
        return 0.0;
    }
    let exponent = ((byte >> 3) & 0x0f) as i32;
    let mantissa = (byte & 0x07) as f32;
    let raw = if exponent == 0 {
        mantissa * 2.0f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
    };
    raw * 0.5
}

fn compute_e8m0_scale_host(amax: f32) -> u8 {
    if amax <= 0.0 {
        return 0;
    }
    let exponent = (amax / 6.0).log2().ceil() as i32 + 127;
    exponent.clamp(0, 254) as u8
}

fn e8m0_to_f32_half_host(e: u8) -> f32 {
    match e {
        0 => 2.0f32.powi(-128),
        1 => 2.0f32.powi(-127),
        value => 2.0f32.powi(value as i32 - 128),
    }
}

fn best_mxfp4_index(value: f32, scale: f32) -> u8 {
    if scale == 0.0 {
        return 0;
    }
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for idx in 0..16u8 {
        let candidate = decode_nvfp4_nibble_host(idx) * scale;
        let err = (candidate - value).abs();
        if err < best_err {
            best = idx;
            best_err = err;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repack_nvfp4_to_mxfp4_interleaves_32_value_blocks() {
        let spec = Nvfp4LinearSpec {
            name: "x".into(),
            rows: 1,
            cols: 64,
            packed_bytes: 32,
            scale_bytes: 4,
            input_scale: 1.0,
            output_scale: 1.0,
        };
        let mut packed = vec![0u8; spec.packed_bytes];
        for group in 0..4 {
            for lane in 0..8 {
                packed[group * 8 + lane] = (2 * lane) as u8 | ((2 * lane + 1) as u8) << 4;
            }
        }
        let scales = vec![0x40u8; spec.scale_bytes];

        let repacked = repack_nvfp4_to_mxfp4_host(&spec, &packed, &scales).unwrap();

        assert_eq!(repacked.len(), 34);
        for block in 0..2 {
            let base = block * 17;
            assert_eq!(repacked[base], 128);
            assert_eq!(
                &repacked[base + 1..base + 17],
                &[
                    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x00, 0x99, 0xaa, 0xbb, 0xcc,
                    0xdd, 0xee, 0xff,
                ]
            );
        }
    }

    #[test]
    fn cutlass_nvfp4_layout_matches_vllm_sm120_scale_shape() {
        let spec = Nvfp4LinearSpec {
            name: "proj".into(),
            rows: 96,
            cols: 64,
            packed_bytes: 96 * 32,
            scale_bytes: 96 * 4,
            input_scale: 1.0,
            output_scale: 1.0,
        };

        let layout = CutlassNvfp4LinearLayout::for_weight(&spec).unwrap();

        assert_eq!(layout.logical_n, 96);
        assert_eq!(layout.logical_k, 64);
        assert_eq!(layout.packed_k_cols, 32);
        assert_eq!(layout.source_scale_cols, 4);
        assert_eq!(layout.scale_rows, 128);
        assert_eq!(layout.scale_cols, 4);
        assert_eq!(layout.scale_bytes(), 512);
        assert_eq!(layout.swizzled_scale_offset(0, 0).unwrap(), 0);
        assert_eq!(layout.swizzled_scale_offset(0, 3).unwrap(), 3);
        assert_eq!(layout.swizzled_scale_offset(32, 0).unwrap(), 4);
        assert_eq!(layout.swizzled_scale_offset(1, 0).unwrap(), 16);
    }

    #[test]
    fn cutlass_nvfp4_scale_offsets_cross_row_and_k_tiles() {
        let spec = Nvfp4LinearSpec {
            name: "proj".into(),
            rows: 128,
            cols: 128,
            packed_bytes: 128 * 64,
            scale_bytes: 128 * 8,
            input_scale: 1.0,
            output_scale: 1.0,
        };

        let layout = CutlassNvfp4LinearLayout::for_weight(&spec).unwrap();

        assert_eq!(layout.scale_rows, 128);
        assert_eq!(layout.scale_cols, 8);
        assert_eq!(layout.scale_k_tiles, 2);
        assert_eq!(layout.swizzled_scale_offset(0, 4).unwrap(), 512);
        assert_eq!(layout.swizzled_scale_offset(31, 4).unwrap(), 1008);
        assert_eq!(layout.swizzled_scale_offset(32, 4).unwrap(), 516);
        assert_eq!(layout.swizzled_scale_offset(127, 7).unwrap(), 1023);
    }

    #[test]
    fn repack_nvfp4_to_cutlass_preserves_payload_and_swizzles_scales() {
        let spec = Nvfp4LinearSpec {
            name: "proj".into(),
            rows: 32,
            cols: 64,
            packed_bytes: 32 * 32,
            scale_bytes: 32 * 4,
            input_scale: 1.0,
            output_scale: 1.0,
        };
        let packed = (0..spec.packed_bytes)
            .map(|idx| (idx & 0xff) as u8)
            .collect::<Vec<_>>();
        let scales = (0..spec.scale_bytes)
            .map(|idx| (idx & 0x7f) as u8)
            .collect::<Vec<_>>();

        let repacked = repack_nvfp4_to_cutlass_e2m1_ue4m3_host(&spec, &packed, &scales).unwrap();

        assert_eq!(repacked.payload_e2m1, packed);
        assert_eq!(repacked.scales_ue4m3.len(), 512);
        for row in 0..spec.rows {
            for k_scale_idx in 0..spec.scale_cols() {
                let dst = repacked
                    .layout
                    .swizzled_scale_offset(row, k_scale_idx)
                    .unwrap();
                assert_eq!(
                    repacked.scales_ue4m3[dst],
                    scales[row * spec.scale_cols() + k_scale_idx]
                );
            }
        }
    }

    #[test]
    fn cutlass_nvfp4_rejects_unaligned_output_channels() {
        let spec = Nvfp4LinearSpec {
            name: "proj".into(),
            rows: 31,
            cols: 64,
            packed_bytes: 31 * 32,
            scale_bytes: 31 * 4,
            input_scale: 1.0,
            output_scale: 1.0,
        };

        assert!(CutlassNvfp4LinearLayout::for_weight(&spec).is_err());
    }

    #[test]
    fn cutlass_nvfp4_rejects_bad_k_alignment_and_lengths() {
        let bad_k = Nvfp4LinearSpec {
            name: "proj".into(),
            rows: 32,
            cols: 48,
            packed_bytes: 32 * 24,
            scale_bytes: 32 * 3,
            input_scale: 1.0,
            output_scale: 1.0,
        };
        assert!(CutlassNvfp4LinearLayout::for_weight(&bad_k).is_err());

        let spec = Nvfp4LinearSpec {
            name: "proj".into(),
            rows: 32,
            cols: 64,
            packed_bytes: 32 * 32,
            scale_bytes: 32 * 4,
            input_scale: 1.0,
            output_scale: 1.0,
        };
        let packed = vec![0u8; spec.packed_bytes - 1];
        let scales = vec![0u8; spec.scale_bytes];
        assert!(repack_nvfp4_to_cutlass_e2m1_ue4m3_host(&spec, &packed, &scales).is_err());
    }
}
