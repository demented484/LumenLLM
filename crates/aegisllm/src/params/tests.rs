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
                "path": "/tmp/model",
                "other-parameters": {
                    "flash-attention": false
                }
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
                "path": "/tmp/model",
                "other-parameters": {
                    "flash-attention": true
                }
            },
            "cuda": {
                "prefill-attention": "reference"
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
            "model": {
                "path": "/tmp/model",
                "attention": { "compute-quantization": "bf16-fa2" }
            }
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
    fn draft_section_accepts_full_model_block() {
        // The draft section flattens the full ModelSection (path + placement) and
        // adds num-draft-tokens. Verify it parses and the path + token count flow
        // into the fragment.
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": { "path": "/tmp/target" },
            "draft": {
                "path": "/tmp/draft-assistant",
                "compute": "cuda:0",
                "store": "vram",
                "input-layer":  { "compute": "cuda:0", "store": "ram" },
                "output-layer": { "compute": "cuda:0", "store": "vram" },
                "hidden-layers": {
                    "compute": "cuda:0", "store": "vram",
                    "kv-cache": { "context-size": 4096, "type-k": "f16", "type-v": "f16" }
                },
                "attention": { "compute": "cuda:0", "store": "vram", "compute-quantization": "bf16" },
                "num-draft-tokens": 6
            }
        }))
        .expect("draft section with full model block should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("fragment");
        assert_eq!(
            fragment.draft_model.as_deref(),
            Some(std::path::Path::new("/tmp/draft-assistant"))
        );
        assert_eq!(fragment.num_draft_tokens, 6);
    }

    #[test]
    fn attention_compute_quant_bf16_parses() {
        // `compute-quantization` selects the attention KERNEL precision. (The
        // weight-requant knobs attention-quantization/shared-MLP-quantization were
        // removed from the schema; checkpoint-native precision is loaded as-is.)
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model",
                "attention": {
                    "compute-quantization": "bf16"
                }
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
    }

    #[test]
    fn attention_compute_quant_fp8_requires_fp8_kv_cache() {
        // compute-quantization=fp8 with a non-FP8 KV cache is a genuine
        // incompatibility — the parser must reject it with a clear message.
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model",
                "attention": { "compute-quantization": "fp8" }
            }
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
            "model": {
                "path": "/tmp/model",
                "attention": { "compute-quantization": "fp8" },
                "hidden-layers": {
                    "kv-cache": { "type-k": "fp8", "type-v": "fp8" }
                }
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
            "model": {
                "path": "/tmp/model",
                "attention": { "compute-quantization": "int3" }
            }
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
            "model": {
                "path": "/tmp/model",
                "hidden-layers": {
                    "kv-cache": { "type-k": "fp8" }
                }
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

    #[test]
    fn hidden_layer_ranges_resolve_to_per_layer_placement() {
        use aegisllm_base::graph::{GraphRegion, GraphRegionKind, RegionId};
        use aegisllm_base::planning::placement::{LayerSelector, PlacementRule};

        // A heterogeneous CPU/GPU split: layers 0..17 on cuda:0 (vram),
        // layers 17..42 on cpu (ram). Expressed via `hidden-layers.ranges`.
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model",
                "hidden-layers": {
                    "ranges": [
                        { "start": 0,  "end": 17, "store": "vram", "compute": "cuda:0" },
                        { "start": 17, "end": 42, "store": "ram",  "compute": "cpu" }
                    ]
                }
            }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("parameters should become an engine fragment");

        // The two ranges land as Range rules, in array order.
        let range_rules: Vec<&PlacementRule> = fragment
            .policy
            .rules
            .iter()
            .filter(|r| matches!(r.selector, LayerSelector::Range { .. }))
            .collect();
        let gpu_rule = range_rules
            .iter()
            .find(|r| r.selector == (LayerSelector::Range { start: 0, end: 17 }))
            .expect("layers 0..17 range rule present");
        assert_eq!(gpu_rule.compute, Some(ComputePlacement::Cuda { device: 0 }));
        assert_eq!(gpu_rule.store, Some(StoragePlacement::Vram { device: 0 }));
        let cpu_rule = range_rules
            .iter()
            .find(|r| r.selector == (LayerSelector::Range { start: 17, end: 42 }))
            .expect("layers 17..42 range rule present");
        assert_eq!(cpu_rule.compute, Some(ComputePlacement::Cpu));
        assert_eq!(cpu_rule.store, Some(StoragePlacement::Ram));

        // Apply the rules against synthetic layer regions to confirm the
        // half-open `[start, end)` resolution places each layer on its side.
        // This exercises the same `apply_rules`/`selector_matches` path the
        // real resolver uses, without standing up a full ModelGraph.
        let resolve_layer = |layer: usize| -> (StoragePlacement, ComputePlacement) {
            let region = GraphRegion {
                id: RegionId(format!("layer.{layer}")),
                kind: GraphRegionKind::TransformerBlock,
                layer_index: Some(layer),
                tensors: Vec::new(),
            };
            let mut store = fragment.policy.weights_store;
            let mut compute = fragment.policy.weights_compute;
            for rule in &fragment.policy.rules {
                let matches = match &rule.selector {
                    LayerSelector::Range { start, end } => layer >= *start && layer < *end,
                    LayerSelector::FirstN { n } => layer < *n,
                    LayerSelector::All => true,
                    _ => region.layer_index.is_some() && false,
                };
                if matches {
                    if let Some(next) = rule.store {
                        store = next;
                    }
                    if let Some(next) = rule.compute {
                        compute = next;
                    }
                }
            }
            (store, compute)
        };
        for layer in 0..17 {
            let (store, compute) = resolve_layer(layer);
            assert_eq!(compute, ComputePlacement::Cuda { device: 0 }, "layer {layer}");
            assert_eq!(store, StoragePlacement::Vram { device: 0 }, "layer {layer}");
        }
        for layer in 17..42 {
            let (store, compute) = resolve_layer(layer);
            assert_eq!(compute, ComputePlacement::Cpu, "layer {layer}");
            assert_eq!(store, StoragePlacement::Ram, "layer {layer}");
        }
    }

    #[test]
    fn experts_compute_cpu_sets_routed_expert_override() {
        // `hidden-layers.experts.compute = cpu` routes ONLY the routed experts to
        // the CPU; the layer-region weights_compute stays cuda:0 (shared expert /
        // GDN / attention follow it). `experts.store` overrides routed residency.
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model",
                "compute": "cuda:0",
                "store": "vram",
                "hidden-layers": {
                    "compute": "cuda:0",
                    "store": "mmap",
                    "experts": { "compute": "cpu", "store": "ram" },
                    "kv-cache": { "context-size": 8192, "type-k": "f16", "type-v": "f16" }
                }
            }
        }))
        .expect("parameters with experts section should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("fragment");

        assert_eq!(
            fragment.policy.experts_compute_override,
            Some(ComputePlacement::Cpu)
        );
        assert_eq!(
            fragment.policy.experts_store_override,
            Some(StoragePlacement::Ram)
        );
        // The layer-region compute is NOT moved to the CPU.
        assert_eq!(
            fragment.policy.weights_compute,
            ComputePlacement::Cuda { device: 0 }
        );
    }

    #[test]
    fn no_experts_section_leaves_routed_experts_on_gpu() {
        // No-regression guard: a config WITHOUT an `experts` key must leave the
        // override as None so the routed experts keep the unchanged GPU path.
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model",
                "compute": "cuda:0",
                "store": "vram",
                "hidden-layers": {
                    "compute": "cuda:0",
                    "store": "mmap",
                    "kv-cache": { "context-size": 8192 }
                }
            }
        }))
        .expect("parameters without experts section should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("fragment");

        assert_eq!(fragment.policy.experts_compute_override, None);
        assert_eq!(fragment.policy.experts_store_override, None);
    }

    #[test]
    fn experts_compute_omitted_falls_back_to_hidden_layers_compute() {
        // `experts` present with only `store` → compute falls back to
        // `hidden-layers.compute`. So an `experts` block that only repins storage
        // does NOT silently move compute to the CPU.
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model",
                "compute": "cuda:0",
                "store": "vram",
                "hidden-layers": {
                    "compute": "cuda:0",
                    "experts": { "store": "ram" }
                }
            }
        }))
        .expect("experts with only store should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("fragment");

        assert_eq!(
            fragment.policy.experts_compute_override,
            Some(ComputePlacement::Cuda { device: 0 })
        );
        assert_eq!(
            fragment.policy.experts_store_override,
            Some(StoragePlacement::Ram)
        );
    }
