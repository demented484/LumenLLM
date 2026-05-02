use std::process::Command;

fn aegis_bin() -> String {
    option_env!("CARGO_BIN_EXE_aegisllm")
        .unwrap_or("./target/debug/aegisllm")
        .to_string()
}

fn run_aegis(args: &[&str]) -> std::process::Output {
    Command::new(aegis_bin())
        .args(args)
        .output()
        .expect("run aegisllm binary")
}

fn smoke_model() -> Option<String> {
    std::env::var("AEGIS_SMOKE_MODEL").ok()
}

fn output_text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn inspect_hardware_reports_cpu() {
    let output = run_aegis(&["inspect-hardware"]);
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("cpu:"), "{text}");
}

#[test]
fn cpu_quality_smoke_runs_when_model_env_is_set() {
    let Some(model) = smoke_model() else {
        eprintln!("skipping: set AEGIS_SMOKE_MODEL to run model-backed CPU quality smoke");
        return;
    };

    let output = run_aegis(&[
        "quality-smoke",
        "--model",
        &model,
        "--weights-store",
        "mmap",
        "--weights-compute",
        "cpu",
        "--kv-store",
        "ram",
        "--kv-compute",
        "cpu",
        "--ctx-size",
        "128",
    ]);
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("quality-smoke: case=english_hello"), "{text}");
    assert!(
        text.contains("quality-smoke: case=russian_greeting"),
        "{text}"
    );
}

#[test]
fn cuda_quality_smoke_runs_when_model_and_cuda_env_are_set() {
    if std::env::var("AEGIS_SMOKE_CUDA").ok().as_deref() != Some("1") {
        eprintln!("skipping: set AEGIS_SMOKE_CUDA=1 to run CUDA quality smoke");
        return;
    }
    let Some(model) = smoke_model() else {
        eprintln!("skipping: set AEGIS_SMOKE_MODEL to run CUDA quality smoke");
        return;
    };

    let output = run_aegis(&[
        "quality-smoke",
        "--model",
        &model,
        "--weights-store",
        "vram",
        "--weights-compute",
        "cuda",
        "--kv-store",
        "vram",
        "--kv-compute",
        "cuda",
        "--ctx-size",
        "128",
    ]);
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("quality-smoke: case=english_hello"), "{text}");
    assert!(
        text.contains("quality-smoke: case=russian_greeting"),
        "{text}"
    );
}

#[test]
fn hybrid_mvp_check_builds_scheduler_when_model_env_is_set() {
    let Some(model) = smoke_model() else {
        eprintln!("skipping: set AEGIS_SMOKE_MODEL to run hybrid scheduler smoke");
        return;
    };

    let output = run_aegis(&[
        "mvp-check",
        "--model",
        &model,
        "--n-gpu-layers",
        "1",
        "--kv-store",
        "ram",
        "--kv-compute",
        "cpu",
        "--ctx-size",
        "128",
    ]);
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(
        text.contains("selected=hybrid") && text.contains("build=ok"),
        "{text}"
    );
}
