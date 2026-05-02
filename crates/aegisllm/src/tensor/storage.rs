use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};

use crate::error::Result;
use crate::graph::{ModelGraph, RegionId, TensorRole};
use crate::planning::placement::{
    ComputePlacement, ResolvedPlacement, StoragePlacement, TransferPolicy,
};
use crate::tensor::TensorInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoragePlan {
    pub tensors: Vec<TensorStoragePlan>,
    pub totals: StorageTotals,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorStoragePlan {
    pub name: String,
    pub region_id: RegionId,
    pub role: TensorRole,
    pub bytes: u64,
    pub store: StoragePlacement,
    pub compute: ComputePlacement,
    pub residency: TensorResidencyPlan,
    pub transfer: TransferPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorResidencyPlan {
    RamResident,
    FileBackedMmap,
    VramResident {
        device: usize,
    },
    StagedHostToDevice {
        device: usize,
    },
    StagedDeviceToHost {
        device: usize,
    },
    CrossDevice {
        store_device: usize,
        compute_device: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StorageTotals {
    pub ram_resident_bytes: u64,
    pub mmap_file_backed_bytes: u64,
    pub vram_resident_bytes: Vec<(usize, u64)>,
    pub staged_host_to_device_peak_bytes: Vec<(usize, u64)>,
}

#[derive(Debug, Default)]
pub struct TensorStorageLoader {
    mmaps: BTreeMap<PathBuf, Arc<Mmap>>,
}

#[derive(Debug, Clone)]
pub struct LoadedHostTensor {
    pub name: String,
    pub storage: HostTensorStorage,
}

#[derive(Debug, Clone)]
pub enum HostTensorStorage {
    Ram(Vec<u8>),
    Mmap {
        map: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostLoadMode {
    RamResident,
    MmapFileBacked,
}

impl StoragePlan {
    pub fn from_graph_and_placement(graph: &ModelGraph, placement: &ResolvedPlacement) -> Self {
        let region_placements = placement.region_map();
        let mut tensors = Vec::new();
        for region in &graph.regions {
            let Some(region_placement) = region_placements.get(&region.id) else {
                continue;
            };
            for tensor in &region.tensors {
                let residency = residency_for(region_placement.store, region_placement.compute);
                tensors.push(TensorStoragePlan {
                    name: tensor.info.name.clone(),
                    region_id: region.id.clone(),
                    role: tensor.role,
                    bytes: tensor.info.data_len_bytes(),
                    store: region_placement.store,
                    compute: region_placement.compute,
                    residency,
                    transfer: region_placement.transfer,
                });
            }
        }
        let totals = StorageTotals::from_tensors(&tensors);
        Self { tensors, totals }
    }

    pub fn tensors_in_region<'a>(
        &'a self,
        region_id: &'a RegionId,
    ) -> impl Iterator<Item = &'a TensorStoragePlan> + 'a {
        self.tensors
            .iter()
            .filter(move |tensor| &tensor.region_id == region_id)
    }
}

impl TensorStorageLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_for_store(
        &mut self,
        tensor: &TensorInfo,
        store: StoragePlacement,
    ) -> Result<LoadedHostTensor> {
        let mode = match store {
            StoragePlacement::Ram => HostLoadMode::RamResident,
            StoragePlacement::Mmap | StoragePlacement::Vram { .. } => HostLoadMode::MmapFileBacked,
        };
        self.load(tensor, mode)
    }

    pub fn load(&mut self, tensor: &TensorInfo, mode: HostLoadMode) -> Result<LoadedHostTensor> {
        let storage = match mode {
            HostLoadMode::RamResident => HostTensorStorage::Ram(read_tensor_bytes(tensor)?),
            HostLoadMode::MmapFileBacked => {
                let map = self.mmap_shard(&tensor.shard_path)?;
                HostTensorStorage::Mmap {
                    map,
                    offset: tensor.file_offsets.0 as usize,
                    len: tensor.data_len_bytes() as usize,
                }
            }
        };
        Ok(LoadedHostTensor {
            name: tensor.name.clone(),
            storage,
        })
    }

    fn mmap_shard(&mut self, path: &PathBuf) -> Result<Arc<Mmap>> {
        if let Some(map) = self.mmaps.get(path) {
            return Ok(map.clone());
        }
        let file = File::open(path)?;
        let map = Arc::new(unsafe { MmapOptions::new().map(&file)? });
        self.mmaps.insert(path.clone(), map.clone());
        Ok(map)
    }
}

impl LoadedHostTensor {
    pub fn as_bytes(&self) -> &[u8] {
        self.storage.as_bytes()
    }

    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl HostTensorStorage {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Ram(bytes) => bytes,
            Self::Mmap { map, offset, len } => &map[*offset..*offset + *len],
        }
    }
}

impl StorageTotals {
    fn from_tensors(tensors: &[TensorStoragePlan]) -> Self {
        let mut totals = Self::default();
        for tensor in tensors {
            match tensor.store {
                StoragePlacement::Ram => {
                    totals.ram_resident_bytes += tensor.bytes;
                }
                StoragePlacement::Mmap => {
                    totals.mmap_file_backed_bytes += tensor.bytes;
                }
                StoragePlacement::Vram { device } => {
                    add_bytes(&mut totals.vram_resident_bytes, device, tensor.bytes);
                }
            }
            match tensor.residency {
                TensorResidencyPlan::StagedHostToDevice { device } => {
                    set_peak(
                        &mut totals.staged_host_to_device_peak_bytes,
                        device,
                        tensor.bytes,
                    );
                }
                TensorResidencyPlan::RamResident
                | TensorResidencyPlan::FileBackedMmap
                | TensorResidencyPlan::VramResident { .. }
                | TensorResidencyPlan::StagedDeviceToHost { .. }
                | TensorResidencyPlan::CrossDevice { .. } => {}
            }
        }
        totals
    }
}

fn residency_for(store: StoragePlacement, compute: ComputePlacement) -> TensorResidencyPlan {
    match (store, compute) {
        (StoragePlacement::Ram, ComputePlacement::Cpu) => TensorResidencyPlan::RamResident,
        (StoragePlacement::Mmap, ComputePlacement::Cpu) => TensorResidencyPlan::FileBackedMmap,
        (
            StoragePlacement::Vram {
                device: store_device,
            },
            ComputePlacement::Cuda { device },
        ) if store_device == device => TensorResidencyPlan::VramResident { device },
        (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cuda { device }) => {
            TensorResidencyPlan::StagedHostToDevice { device }
        }
        (StoragePlacement::Vram { device }, ComputePlacement::Cpu) => {
            TensorResidencyPlan::StagedDeviceToHost { device }
        }
        (
            StoragePlacement::Vram {
                device: store_device,
            },
            ComputePlacement::Cuda {
                device: compute_device,
            },
        ) => TensorResidencyPlan::CrossDevice {
            store_device,
            compute_device,
        },
    }
}

fn add_bytes(values: &mut Vec<(usize, u64)>, device: usize, bytes: u64) {
    if let Some((_, total)) = values.iter_mut().find(|(entry, _)| *entry == device) {
        *total += bytes;
    } else {
        values.push((device, bytes));
    }
}

fn set_peak(values: &mut Vec<(usize, u64)>, device: usize, bytes: u64) {
    if let Some((_, peak)) = values.iter_mut().find(|(entry, _)| *entry == device) {
        *peak = (*peak).max(bytes);
    } else {
        values.push((device, bytes));
    }
}

fn read_tensor_bytes(tensor: &TensorInfo) -> Result<Vec<u8>> {
    let len = tensor.data_len_bytes() as usize;
    let mut file = File::open(&tensor.shard_path)?;
    file.seek(SeekFrom::Start(tensor.file_offsets.0))?;
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}
