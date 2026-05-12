use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/cutlass_bridge.cu");
    println!("cargo:rerun-if-changed=src/cuda/cutlass_bridge_moe.cu");
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=AEGIS_CUTLASS_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=AEGIS_CUTLASS_NVFP4_GROUPED_BUILD");

    let cutlass_dir = env::var_os("CUTLASS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../../../cutlass"));
    let cuda_dir = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/cuda"));

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let object = out_dir.join("cutlass_bridge.o");
    let archive = out_dir.join("libaegis_cutlass_bridge.a");

    compile_cutlass_bridge(&cuda_dir, &cutlass_dir, "src/cuda/cutlass_bridge.cu", &object);

    // Optional: compile the CUTLASS NVFP4 grouped MoE GEMM TU. Heavy
    // template instantiation (~2-3 min standalone compile on sm_120f),
    // so it's gated behind AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1. When the
    // flag is unset, the Rust runtime gets a stub from a sibling module
    // that returns "not built" so the dispatcher falls back to the
    // existing home-rolled grouped kernel.
    let build_grouped =
        env::var("AEGIS_CUTLASS_NVFP4_GROUPED_BUILD").map(|v| v != "0").unwrap_or(false);
    let mut objects = vec![object.clone()];
    if build_grouped {
        let moe_object = out_dir.join("cutlass_bridge_moe.o");
        compile_cutlass_bridge(
            &cuda_dir,
            &cutlass_dir,
            "src/cuda/cutlass_bridge_moe.cu",
            &moe_object,
        );
        objects.push(moe_object);
        println!("cargo:rustc-cfg=aegis_cutlass_nvfp4_grouped");
    }

    archive_objects(&archive, &objects);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=aegis_cutlass_bridge");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_dir.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn compile_cutlass_bridge(cuda_dir: &Path, cutlass_dir: &Path, src: &str, object: &Path) {
    let nvcc = cuda_dir.join("bin/nvcc");
    let cuda_arch = env::var("AEGIS_CUTLASS_CUDA_ARCH").unwrap_or_else(|_| "sm_120f".into());
    let status = Command::new(&nvcc)
        .arg("-c")
        .arg(src)
        .arg("-o")
        .arg(object)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg(format!("-arch={cuda_arch}"))
        .arg("-I")
        .arg(cutlass_dir.join("include"))
        .arg("-I")
        .arg(cutlass_dir.join("tools/util/include"))
        .arg("-I")
        .arg(cuda_dir.join("include"))
        .status()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", nvcc.display()));

    if !status.success() {
        panic!("nvcc failed to compile {src} with status {status}");
    }
}

fn archive_objects(archive: &Path, objects: &[PathBuf]) {
    // `ar crus` creates/replaces; ensure a clean slate so a previous
    // build's stale objects don't linger when we toggle the optional TU.
    let _ = std::fs::remove_file(archive);
    let mut cmd = Command::new("ar");
    cmd.arg("crus").arg(archive);
    for obj in objects {
        cmd.arg(obj);
    }
    let status = cmd.status().expect("failed to run ar");
    if !status.success() {
        panic!("ar failed to create {}", archive.display());
    }
}
