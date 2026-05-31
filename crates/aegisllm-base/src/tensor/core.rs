use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{AegisError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorDType {
    Bool,
    U8,
    I8,
    I16,
    U16,
    F16,
    BF16,
    F32,
    F64,
    I32,
    U32,
    I64,
    U64,
    F8E4M3,
    F8E5M2,
}

impl TensorDType {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "BOOL" => Ok(Self::Bool),
            "U8" => Ok(Self::U8),
            "I8" => Ok(Self::I8),
            "I16" => Ok(Self::I16),
            "U16" => Ok(Self::U16),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "F32" => Ok(Self::F32),
            "F64" => Ok(Self::F64),
            "I32" => Ok(Self::I32),
            "U32" => Ok(Self::U32),
            "I64" => Ok(Self::I64),
            "U64" => Ok(Self::U64),
            "F8_E4M3" => Ok(Self::F8E4M3),
            "F8_E5M2" => Ok(Self::F8E5M2),
            other => Err(AegisError::Unsupported(format!(
                "unsupported safetensors dtype `{other}`"
            ))),
        }
    }

    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: TensorDType,
    pub shape: Vec<usize>,
    pub num_elements: usize,
    pub data_offsets: (u64, u64),
    pub file_offsets: (u64, u64),
    pub shard_name: String,
    pub shard_path: PathBuf,
}

impl TensorInfo {
    pub fn data_len_bytes(&self) -> u64 {
        self.data_offsets.1.saturating_sub(self.data_offsets.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRegistry {
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl TensorRegistry {
    pub fn from_index_and_shards(
        root: &Path,
        weight_map: &BTreeMap<String, String>,
    ) -> Result<Self> {
        let mut by_shard = BTreeMap::<String, BTreeSet<String>>::new();
        for (tensor_name, shard_name) in weight_map {
            by_shard
                .entry(shard_name.clone())
                .or_default()
                .insert(tensor_name.clone());
        }

        let mut tensors = BTreeMap::new();
        for (shard_name, shard_tensors) in by_shard {
            let shard_path = root.join(&shard_name);
            let header = parse_safetensors_header(&shard_path)?;
            for tensor_name in shard_tensors {
                let raw = header.entries.get(&tensor_name).ok_or_else(|| {
                    AegisError::Unsupported(format!(
                        "tensor `{tensor_name}` is referenced in index but missing in `{shard_name}`"
                    ))
                })?;
                let dtype = TensorDType::parse(&raw.dtype)?;
                let num_elements = num_elements(&raw.shape)?;
                let expected_bytes = num_elements
                    .checked_mul(dtype.bytes_per_element())
                    .ok_or_else(|| AegisError::Unsupported("tensor byte size overflow".into()))?
                    as u64;
                let data_offsets = (raw.data_offsets[0], raw.data_offsets[1]);
                let actual_bytes = data_offsets.1.saturating_sub(data_offsets.0);
                if expected_bytes != actual_bytes {
                    return Err(AegisError::Unsupported(format!(
                        "tensor `{tensor_name}` size mismatch: expected {expected_bytes}, got {actual_bytes}"
                    )));
                }
                if data_offsets.1 > header.data_section_len {
                    return Err(AegisError::Unsupported(format!(
                        "tensor `{tensor_name}` exceeds shard payload"
                    )));
                }

                tensors.insert(
                    tensor_name.clone(),
                    TensorInfo {
                        name: tensor_name,
                        dtype,
                        shape: raw.shape.clone(),
                        num_elements,
                        data_offsets,
                        file_offsets: (
                            header.payload_base + data_offsets.0,
                            header.payload_base + data_offsets.1,
                        ),
                        shard_name: shard_name.clone(),
                        shard_path: shard_path.clone(),
                    },
                );
            }
        }

        // compressed-tensors (Qwen3-Next NVFP4) uses different tensor names than
        // the engine's Gemma-derived NVFP4 layout, but the byte layout is
        // identical (`nvfp4-pack-quantized`: 2 nibbles/byte, per-16 fp8-e4m3
        // group scales). Register name aliases that point at the SAME bytes so
        // the planner, graph builder, and CUDA loader (all keyed on Gemma's
        // names) work unchanged — no repack needed.
        //   weight_packed → weight   (packed 4-bit, u8 — identical bytes)
        // (`weight_scale`, the per-group fp8 scale, has the same name in both.)
        // The PER-TENSOR global scales (compressed-tensors `weight_global_scale`
        // / `input_global_scale`) are NOT byte-aliased: they use the RECIPROCAL
        // convention vs Gemma's `weight_scale_2` / `input_scale` (a DIVISOR, not
        // a multiplier), so the loader reads + inverts them by name instead.
        const ALIASES: [(&str, &str); 1] = [
            (".weight_packed", ".weight"),
        ];
        let mut to_add: Vec<(String, TensorInfo)> = Vec::new();
        for (name, info) in tensors.iter() {
            for (ct_suffix, gemma_suffix) in ALIASES {
                if let Some(stem) = name.strip_suffix(ct_suffix) {
                    let alias = format!("{stem}{gemma_suffix}");
                    // Don't clobber a real tensor that already carries this name.
                    if !tensors.contains_key(&alias) {
                        let mut cloned = info.clone();
                        cloned.name = alias.clone();
                        to_add.push((alias, cloned));
                    }
                }
            }
        }
        for (alias, info) in to_add {
            tensors.entry(alias).or_insert(info);
        }

        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn dtype_counts(&self) -> BTreeMap<TensorDType, usize> {
        let mut counts = BTreeMap::new();
        for tensor in self.tensors.values() {
            *counts.entry(tensor.dtype).or_insert(0) += 1;
        }
        counts
    }
}

#[derive(Debug, Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
struct SafetensorsHeader {
    entries: BTreeMap<String, RawTensorEntry>,
    data_section_len: u64,
    payload_base: u64,
}

fn parse_safetensors_header(path: &Path) -> Result<SafetensorsHeader> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut header_len_bytes = [0_u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    if 8 + header_len > file_len {
        return Err(AegisError::Unsupported(format!(
            "invalid safetensors header in {}",
            path.display()
        )));
    }
    let mut header_json = vec![0_u8; header_len as usize];
    file.read_exact(&mut header_json)?;
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header_json)?;
    let mut entries = BTreeMap::new();
    for (name, value) in raw {
        if name != "__metadata__" {
            entries.insert(name, serde_json::from_value(value)?);
        }
    }
    Ok(SafetensorsHeader {
        entries,
        data_section_len: file_len.saturating_sub(8 + header_len),
        payload_base: 8 + header_len,
    })
}

fn num_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or_else(|| AegisError::Unsupported("tensor element count overflow".into()))
    })
}
