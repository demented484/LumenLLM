use aegisllm_base::hardware::HardwareInventory;

pub(crate) fn nvrtc_arch_for_device(device_index: usize) -> &'static str {
    let inventory = HardwareInventory::detect();
    let compute_capability = inventory
        .gpus
        .iter()
        .find(|gpu| gpu.index == device_index)
        .and_then(|gpu| gpu.compute_capability.as_deref())
        .unwrap_or_default();
    match compute_capability {
        value if value.starts_with("12.") => "compute_120a",
        value if value.starts_with("10.") => "compute_100",
        value if value.starts_with("9.") => "compute_90",
        "8.9" => "compute_89",
        value if value.starts_with("8.") => "compute_80",
        _ => "compute_80",
    }
}
