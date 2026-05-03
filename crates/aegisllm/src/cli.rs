mod command;
mod flags;
mod generate;
pub mod gates;
mod helpers;
mod parse;
mod run;
mod smoke;

pub use command::Command;
pub use parse::parse_args;
pub use run::run_env;

#[cfg(test)]
mod tests {
    use super::*;
    use aegisllm_base::planning::placement::{ComputePlacement, StoragePlacement};
    use std::path::PathBuf;

    #[test]
    fn bench_generate_repeats_prompt_and_preserves_sampling_flags() {
        let command = parse_args([
            "bench-generate".to_string(),
            "--model".to_string(),
            "/tmp/model".to_string(),
            "--prompt".to_string(),
            "hello".to_string(),
            "--prompt-repeat".to_string(),
            "3".to_string(),
            "--max-tokens".to_string(),
            "7".to_string(),
            "--temperature".to_string(),
            "0".to_string(),
        ])
        .expect("bench-generate should parse");

        let Command::BenchGenerate(config, request, prompt_repeat, format) = command else {
            panic!("expected bench-generate command");
        };
        assert_eq!(config.model_path, PathBuf::from("/tmp/model"));
        assert_eq!(prompt_repeat, 3);
        assert_eq!(format, command::BenchOutputFormat::Text);
        assert_eq!(request.generate.prompt, "hello\nhello\nhello");
        assert_eq!(request.generate.max_tokens, 7);
        assert_eq!(request.generate.sampling.temperature, 0.0);
    }

    #[test]
    fn bench_generate_parses_json_format() {
        let command = parse_args([
            "bench-generate".to_string(),
            "--model".to_string(),
            "/tmp/model".to_string(),
            "--prompt".to_string(),
            "hello".to_string(),
            "--format".to_string(),
            "json".to_string(),
        ])
        .expect("bench-generate should parse");

        let Command::BenchGenerate(_, _, _, format) = command else {
            panic!("expected bench-generate command");
        };
        assert_eq!(format, command::BenchOutputFormat::Json);
    }

    #[test]
    fn bench_generate_parses_prompt_and_chunk_sweep() {
        let command = parse_args([
            "bench-generate".to_string(),
            "--model".to_string(),
            "/tmp/model".to_string(),
            "--prompt".to_string(),
            "hello".to_string(),
            "--prompt-repeats".to_string(),
            "1,4,16".to_string(),
            "--chunk-sizes".to_string(),
            "1,128".to_string(),
            "--format".to_string(),
            "csv".to_string(),
        ])
        .expect("bench-generate sweep should parse");

        let Command::BenchGenerateSweep(
            config,
            request,
            prompt_repeats,
            chunk_sizes,
            _warmup,
            _runs,
            format,
        ) = command
        else {
            panic!("expected bench-generate sweep command");
        };
        assert_eq!(config.model_path, PathBuf::from("/tmp/model"));
        assert_eq!(request.prompt, "hello");
        assert_eq!(prompt_repeats, [1, 4, 16]);
        assert_eq!(chunk_sizes, [1, 128]);
        assert_eq!(format, command::BenchOutputFormat::Csv);
    }

    #[test]
    fn cuda_device_flag_retargets_auto_policy() {
        let command = parse_args([
            "show-plan".to_string(),
            "--model".to_string(),
            "/tmp/model".to_string(),
            "--cuda-device".to_string(),
            "2".to_string(),
        ])
        .expect("show-plan should parse");

        let Command::ShowPlan(config) = command else {
            panic!("expected show-plan command");
        };
        assert_eq!(
            config.policy.weights_store,
            StoragePlacement::Vram { device: 2 }
        );
        assert_eq!(
            config.policy.weights_compute,
            ComputePlacement::Cuda { device: 2 }
        );
        assert_eq!(config.policy.kv_store, StoragePlacement::Vram { device: 2 });
        assert_eq!(
            config.policy.kv_compute,
            ComputePlacement::Cuda { device: 2 }
        );
    }

    #[test]
    fn quality_smoke_parses_config_command() {
        let command = parse_args([
            "quality-smoke".to_string(),
            "--model".to_string(),
            "/tmp/model".to_string(),
        ])
        .expect("quality-smoke should parse");

        let Command::QualitySmoke(config) = command else {
            panic!("expected quality-smoke command");
        };
        assert_eq!(config.model_path, PathBuf::from("/tmp/model"));
        assert!(config.enable_executor);
    }

    #[test]
    fn config_command_accepts_trailing_cuda_runtime_overrides() {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().expect("temp config");
        write!(
            file,
            r#"{{
                "model": {{
                    "path": "/tmp/model",
                    "store": "vram",
                    "compute": "cuda:0"
                }},
                "cuda": {{
                    "native-mxfp4-repack": false,
                    "native-mxfp4-inference": false
                }}
            }}"#
        )
        .expect("write config");

        let command = parse_args([
            "cuda-compare".to_string(),
            "--config".to_string(),
            file.path().to_string_lossy().into_owned(),
            "--native-mxfp4-repack".to_string(),
            "--cutlass-nvfp4-repack".to_string(),
            "--native-mxfp4-inference".to_string(),
        ])
        .expect("cuda-compare should parse config plus overrides");

        let Command::CudaCompare(config) = command else {
            panic!("expected cuda-compare command");
        };
        assert!(config.cuda.native_mxfp4_repack);
        assert!(config.cuda.cutlass_nvfp4_repack);
        assert!(config.cuda.native_mxfp4_inference);
    }

    #[test]
    fn gates_parses_backend_and_mode_flags() {
        use crate::cli::gates::{GatesBackend, GatesMode};

        let command = parse_args([
            "gates".to_string(),
            "--model".to_string(),
            "/tmp/model".to_string(),
            "--backend".to_string(),
            "cpu".to_string(),
            "--full".to_string(),
        ])
        .expect("gates should parse");

        let Command::Gates(config, gates) = command else {
            panic!("expected gates command");
        };
        assert_eq!(config.model_path, PathBuf::from("/tmp/model"));
        assert_eq!(gates.backend, GatesBackend::Cpu);
        assert_eq!(gates.mode, GatesMode::Full);
    }

    #[test]
    fn gates_defaults_to_cuda_quick() {
        use crate::cli::gates::{GatesBackend, GatesMode};

        let command = parse_args([
            "gates".to_string(),
            "--model".to_string(),
            "/tmp/model".to_string(),
        ])
        .expect("gates should parse");

        let Command::Gates(_, gates) = command else {
            panic!("expected gates command");
        };
        assert_eq!(gates.backend, GatesBackend::Cuda);
        assert_eq!(gates.mode, GatesMode::Quick);
    }

    #[test]
    fn generate_uses_sampling_defaults_from_config() {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().expect("temp config");
        write!(
            file,
            r#"{{
                "model": {{
                    "path": "/tmp/model"
                }},
                "other-parameters": {{
                    "temperature": 0.42,
                    "top-p": 0.7,
                    "top-k": 9
                }}
            }}"#
        )
        .expect("write config");

        let command = parse_args([
            "generate".to_string(),
            "--config".to_string(),
            file.path().to_string_lossy().into_owned(),
            "--prompt".to_string(),
            "hello".to_string(),
            "--top-k".to_string(),
            "1".to_string(),
        ])
        .expect("generate should parse config sampling defaults");

        let Command::Generate(_, request) = command else {
            panic!("expected generate command");
        };
        assert_eq!(request.sampling.temperature, 0.42);
        assert_eq!(request.sampling.top_p, 0.7);
        assert_eq!(request.sampling.top_k, 1);
    }
}
