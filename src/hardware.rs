use std::fs;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareInventory {
    pub cpu: CpuInfo,
    pub gpus: Vec<GpuInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuInfo {
    pub model_name: String,
    pub physical_cores: usize,
    pub logical_threads: usize,
    pub ram_total_bytes: u64,
    pub ram_available_bytes: Option<u64>,
    pub avx2: bool,
    pub avx512: bool,
    pub bf16: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    pub driver_version: String,
    pub compute_capability: Option<String>,
    pub vram_total_bytes: u64,
    pub vram_free_bytes: Option<u64>,
    pub architecture: GpuArchitecture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArchitecture {
    Blackwell,
    Hopper,
    Ada,
    Ampere,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ComputeDevice {
    Cpu,
    Cuda { index: usize },
}

impl HardwareInventory {
    pub fn detect() -> Self {
        Self {
            cpu: CpuInfo::detect(),
            gpus: detect_nvidia_gpus(),
        }
    }

    pub fn preferred_cuda(&self) -> Option<&GpuInfo> {
        self.gpus
            .iter()
            .max_by_key(|gpu| gpu.vram_free_bytes.unwrap_or(gpu.vram_total_bytes))
    }

    pub fn device_label(&self, device: ComputeDevice) -> String {
        match device {
            ComputeDevice::Cpu => format!(
                "cpu:{}c/{}t",
                self.cpu.physical_cores, self.cpu.logical_threads
            ),
            ComputeDevice::Cuda { index } => self
                .gpus
                .iter()
                .find(|gpu| gpu.index == index)
                .map(|gpu| format!("cuda:{index}:{}", gpu.name))
                .unwrap_or_else(|| format!("cuda:{index}")),
        }
    }
}

impl CpuInfo {
    pub fn detect() -> Self {
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let model_name = cpuinfo
            .lines()
            .find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|v| v.split_once(':'))
            })
            .map(|(_, value)| value.trim().to_string())
            .unwrap_or_else(|| std::env::consts::ARCH.to_string());
        let logical_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let physical_cores = detect_physical_cores(&cpuinfo).unwrap_or(logical_threads);
        let flags_line = cpuinfo
            .lines()
            .find(|line| line.starts_with("flags"))
            .unwrap_or_default();
        let (ram_total_bytes, ram_available_bytes) = detect_meminfo();

        Self {
            model_name,
            physical_cores,
            logical_threads,
            ram_total_bytes,
            ram_available_bytes,
            avx2: flags_line.contains(" avx2"),
            avx512: flags_line.contains(" avx512f"),
            bf16: flags_line.contains(" avx512_bf16"),
        }
    }
}

impl GpuInfo {
    pub fn supports_fp4(&self) -> bool {
        matches!(self.architecture, GpuArchitecture::Blackwell)
    }

    pub fn supports_fp8(&self) -> bool {
        matches!(
            self.architecture,
            GpuArchitecture::Blackwell | GpuArchitecture::Hopper
        )
    }
}

fn detect_physical_cores(cpuinfo: &str) -> Option<usize> {
    let count = cpuinfo
        .lines()
        .filter(|line| line.starts_with("core id"))
        .filter_map(|line| {
            line.split_once(':')
                .map(|(_, value)| value.trim().to_string())
        })
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    (count > 0).then_some(count)
}

fn detect_meminfo() -> (u64, Option<u64>) {
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let total = parse_meminfo_kib(&meminfo, "MemTotal:").unwrap_or(0) * 1024;
    let available = parse_meminfo_kib(&meminfo, "MemAvailable:").map(|value| value * 1024);
    (total, available)
}

fn parse_meminfo_kib(meminfo: &str, key: &str) -> Option<u64> {
    meminfo.lines().find_map(|line| {
        line.strip_prefix(key)
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|value| value.parse::<u64>().ok())
    })
}

fn detect_nvidia_gpus() -> Vec<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(parse_nvidia_smi_line)
        .collect()
}

fn parse_nvidia_smi_line(line: &str) -> Option<GpuInfo> {
    let parts = line.split(',').map(str::trim).collect::<Vec<_>>();
    if parts.len() != 6 {
        return None;
    }
    let index = parts[0].parse::<usize>().ok()?;
    let memory_total_mib = parts[3].parse::<u64>().ok()?;
    let memory_free_mib = parts[4].parse::<u64>().ok();
    let compute_capability =
        (!parts[5].is_empty() && parts[5] != "[Not Supported]").then(|| parts[5].to_string());
    let architecture = map_architecture(parts[1], compute_capability.as_deref());

    Some(GpuInfo {
        index,
        name: parts[1].to_string(),
        driver_version: parts[2].to_string(),
        compute_capability,
        vram_total_bytes: memory_total_mib * 1024 * 1024,
        vram_free_bytes: memory_free_mib.map(|mib| mib * 1024 * 1024),
        architecture,
    })
}

fn map_architecture(name: &str, compute_capability: Option<&str>) -> GpuArchitecture {
    let cc = compute_capability.unwrap_or_default();
    if cc.starts_with("12.") || name.contains("50") {
        GpuArchitecture::Blackwell
    } else if cc.starts_with("9.") {
        GpuArchitecture::Hopper
    } else if cc == "8.9" || name.contains("40") {
        GpuArchitecture::Ada
    } else if cc.starts_with("8.") || name.contains("30") {
        GpuArchitecture::Ampere
    } else {
        GpuArchitecture::Unknown
    }
}
