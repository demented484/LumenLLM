    use super::*;
    use aegisllm_cuda::cuda::{AttentionComputeQuant, CudaPrefillAttentionKernel};
    use aegisllm_base::hardware::HardwareInventory;
    use aegisllm_base::planning::placement::{ComputePlacement, PlacementPolicy, StoragePlacement};
    use aegisllm_base::tensor::quant::KvCacheQuantization;

    #[test]
    fn cuda_runtime_flags_come_from_parameters() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model",
                "store": "vram",
                "compute": "cuda"
            },
            "cuda": {
                "device": 2,
                "native-mxfp4-repack": true,
                "native-mxfp4-inference": false,
                "prefill-attention": "warp-flash"
            }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("parameters should become an engine fragment");

        assert_eq!(
            fragment.policy.weights_store,
            StoragePlacement::Vram { device: 2 }
        );
        assert_eq!(
            fragment.policy.weights_compute,
            ComputePlacement::Cuda { device: 2 }
        );
        assert!(fragment.cuda.native_mxfp4_repack);
        assert!(!fragment.cuda.native_mxfp4_inference);
        assert_eq!(
            fragment.cuda.prefill_attention,
            CudaPrefillAttentionKernel::WarpFlash
        );
    }

    #[test]
    fn legacy_flash_attention_flag_controls_cuda_prefill_attention() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model"
            },
            "other-parameters": {
                "flash-attention": false
            }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("parameters should become an engine fragment");

        assert_eq!(
            fragment.cuda.prefill_attention,
            CudaPrefillAttentionKernel::Reference
        );
    }

    #[test]
    fn explicit_cuda_prefill_attention_wins_over_legacy_flash_attention_flag() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model"
            },
            "cuda": {
                "prefill-attention": "reference"
            },
            "other-parameters": {
                "flash-attention": true
            }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("parameters should become an engine fragment");

        assert_eq!(
            fragment.cuda.prefill_attention,
            CudaPrefillAttentionKernel::Reference
        );
    }

    #[test]
    fn model_mmap_false_uses_ram_when_store_is_not_explicit() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model",
                "mmap": false
            }
        }))
        .expect("parameters should parse");

        let mut policy = PlacementPolicy::auto_for(&HardwareInventory {
            cpu: crate::hardware::CpuInfo {
                model_name: "test-cpu".into(),
                physical_cores: 1,
                logical_threads: 1,
                ram_total_bytes: 8 * 1024 * 1024 * 1024,
                ram_available_bytes: Some(8 * 1024 * 1024 * 1024),
                avx2: false,
                avx512: false,
                bf16: false,
            },
            gpus: Vec::new(),
        });
        policy.weights_store = StoragePlacement::Mmap;

        let fragment = params
            .into_engine_fragment(policy)
            .expect("parameters should become an engine fragment");

        assert_eq!(fragment.policy.weights_store, StoragePlacement::Ram);
        assert_eq!(fragment.policy.spill_store, StoragePlacement::Ram);
    }

    #[test]
    fn attention_compute_quant_defaults_to_default_when_unset() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": { "path": "/tmp/model" }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("parameters should become an engine fragment");

        // Default behavior: bit-equivalent to main (env gates only).
        assert_eq!(
            fragment.cuda.attention_compute_quant,
            AttentionComputeQuant::Default
        );
    }

    #[test]
    fn attention_compute_quant_bf16_fa2_parses() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": { "path": "/tmp/model" },
            "attention": { "compute-quantization": "bf16-fa2" }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("parameters should become an engine fragment");

        assert_eq!(
            fragment.cuda.attention_compute_quant,
            AttentionComputeQuant::Bf16Fa2
        );
    }

    #[test]
    fn attention_compute_quant_bf16_distinct_from_weight_quant() {
        // `compute-quantization` and `attention-quantization` are independent
        // knobs: one selects the kernel precision, the other the weight quant.
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": { "path": "/tmp/model" },
            "attention": {
                "compute-quantization": "bf16",
                "attention-quantization": "fp8"
            }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("parameters should become an engine fragment");

        assert_eq!(
            fragment.cuda.attention_compute_quant,
            AttentionComputeQuant::Bf16
        );
        // weight quant stays on its own field, untouched.
        use aegisllm_base::planning::placement::WeightQuantOverride;
        assert_eq!(
            fragment.policy.attention_quantization,
            WeightQuantOverride::Fp8
        );
    }

    #[test]
    fn attention_compute_quant_fp8_requires_fp8_kv_cache() {
        // compute-quantization=fp8 with a non-FP8 KV cache is a genuine
        // incompatibility — the parser must reject it with a clear message.
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": { "path": "/tmp/model" },
            "attention": { "compute-quantization": "fp8" }
        }))
        .expect("parameters should parse");

        let err = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect_err("fp8 attention without fp8 KV must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("compute-quantization=fp8"), "msg: {msg}");
        assert!(msg.contains("FP8 KV cache"), "msg: {msg}");
    }

    #[test]
    fn attention_compute_quant_fp8_accepted_with_fp8_kv_cache() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": { "path": "/tmp/model" },
            "attention": { "compute-quantization": "fp8" },
            "hidden-layers": {
                "kv-cache": { "type-k": "fp8", "type-v": "fp8" }
            }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("fp8 attention + fp8 KV must be accepted");

        assert_eq!(
            fragment.cuda.attention_compute_quant,
            AttentionComputeQuant::Fp8
        );
        assert_eq!(fragment.policy.kv_quantization, KvCacheQuantization::Fp8);
    }

    #[test]
    fn attention_compute_quant_rejects_unknown_value() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": { "path": "/tmp/model" },
            "attention": { "compute-quantization": "int3" }
        }))
        .expect("parameters should parse");

        let err = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect_err("unknown compute-quantization must be rejected");
        assert!(format!("{err}").contains("compute-quantization"));
    }

    #[test]
    fn fp8_kv_cache_with_default_attention_compute_is_legal() {
        // An FP8 KV cache with a non-FP8 (default) attention kernel is legal:
        // the kernel dequantizes on read. The parser must NOT reject this.
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": { "path": "/tmp/model" },
            "hidden-layers": {
                "kv-cache": { "type-k": "fp8" }
            }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("fp8 KV with default attention compute must be accepted");

        assert_eq!(fragment.policy.kv_quantization, KvCacheQuantization::Fp8);
        assert_eq!(
            fragment.cuda.attention_compute_quant,
            AttentionComputeQuant::Default
        );
    }

    #[test]
    fn cuda_device_retargets_auto_cuda_policy() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model"
            },
            "cuda": {
                "device": 2
            }
        }))
        .expect("parameters should parse");

        let mut policy = PlacementPolicy::auto_for(&HardwareInventory {
            cpu: crate::hardware::CpuInfo {
                model_name: "test-cpu".into(),
                physical_cores: 1,
                logical_threads: 1,
                ram_total_bytes: 8 * 1024 * 1024 * 1024,
                ram_available_bytes: Some(8 * 1024 * 1024 * 1024),
                avx2: false,
                avx512: false,
                bf16: false,
            },
            gpus: Vec::new(),
        });
        policy.weights_store = StoragePlacement::Vram { device: 0 };
        policy.weights_compute = ComputePlacement::Cuda { device: 0 };
        policy.kv_store = StoragePlacement::Vram { device: 0 };
        policy.kv_compute = ComputePlacement::Cuda { device: 0 };

        let fragment = params
            .into_engine_fragment(policy)
            .expect("parameters should become an engine fragment");

        assert_eq!(
            fragment.policy.weights_store,
            StoragePlacement::Vram { device: 2 }
        );
        assert_eq!(
            fragment.policy.weights_compute,
            ComputePlacement::Cuda { device: 2 }
        );
        assert_eq!(
            fragment.policy.kv_store,
            StoragePlacement::Vram { device: 2 }
        );
        assert_eq!(
            fragment.policy.kv_compute,
            ComputePlacement::Cuda { device: 2 }
        );
    }
