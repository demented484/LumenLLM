    use super::*;
    use crate::cuda::CudaPrefillAttentionKernel;
    use crate::hardware::HardwareInventory;
    use crate::planning::placement::{ComputePlacement, PlacementPolicy, StoragePlacement};

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
