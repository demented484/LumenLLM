# LumenLLM

A from-scratch **Rust + CUDA** inference engine for NVIDIA **Blackwell (SM120)**, built to run modern LLMs on a single consumer **16 GB** GPU — including models mainstream engines can't fit.

> **Status: pre-alpha**, single-GPU (RTX 50-series), actively developed. Built and tuned on an RTX 5070 Ti.

## Core idea — you place everything, explicitly

LumenLLM never auto-decides anything. **You** specify, by hand in the config, the **compute device** for each component (`cuda:0` / `cpu`) **and** the **storage location** of the weights and everything else (`vram` / `ram`). The engine honors exactly what you write — no hidden heuristics, no automatic offload. That explicit, manual control is what lets you fit and tune a 35B MoE on a 16 GB card: stream the experts from RAM, pin attention in VRAM, compute on the GPU or the CPU — every region, your decision.

## Why

Run 35B-class MoE and 9B models on one 16 GB RTX 50-series card — including models vLLM OOMs on — with **native NVFP4 / FP8** inference, **hand-written CUDA kernels** (no PyTorch, cuDNN, or TensorRT), and **per-component compute/store placement** you control from a JSON config.

## Supported models

- **Qwen 3.5 / 3.6** — Gated-DeltaNet (linear attention) + periodic full-attention + MoE / dense MLP, with the Qwen3-VL native vision tower. (9B-FP8, 35B-A3B-NVFP4.)
- **Gemma 4** — 26B-A4B, E4B, E2B, 31B dense, with the SigLIP vision tower.
- **Text + vision** (image understanding), **reasoning** (thinking mode), and **tool-calling** (OpenAI + Anthropic chat APIs).

## Measured performance — RTX 5070 Ti, 16 GB, SM120

| Model | Decode | Prefill | Notes |
|---|---|---|---|
| **Qwen3.5-9B-FP8** | **61.8 tps** (vLLM eager 48.5) | **1350 tok/s** (vLLM 844) | beats vLLM both directions |
| **Qwen3.6-35B-A3B-NVFP4** | **~50 tps** greedy | **968–1640 tok/s** | runs where vLLM **OOMs**; vision + reasoning + tool-calling verified |
| **Gemma-4 26B-A4B** | ~40 tps | ~3000 tok/s | NVFP4 grouped-MoE prefill |

All numbers measured on-device. Sampling runs through an **on-device GPU sampler** (top-k / top-p / min-p), validated bit-faithful to the reference distribution. Vision verified by per-stage HF cross-dump (cos > 0.99) and real images (OCR-accurate descriptions).

## Features

- **Quantization:** native **NVFP4** (4-bit FP, block-scaled) + **FP8** (E4M3) + BF16, with hand-written SM120 kernels.
- **Placement:** config-driven per-component **compute** (`cuda:0` / `cpu`) and **store** (`vram` / `ram`) — e.g. stream MoE experts from host RAM, or compute them on the CPU.
- **Kernels:** CUDA-graph decode, FlashAttention-2 + native **FP8 MMA** attention, **CUTLASS NVFP4 grouped-MoE** prefill, cuBLASLt tensor-core GEMMs, warp-shuffle GEMVs, FP8 KV cache, Gated-DeltaNet linear attention.
- **Vision:** Qwen3-VL native ViT + interleaved M-RoPE, and Gemma SigLIP — HF cross-validated.
- **Backends:** CUDA (primary), CPU (AVX-512 / VNNI), wgpu / Vulkan (experimental).
- **Serving:** OpenAI- and Anthropic-compatible chat-completions server with tool-calling and reasoning extraction.

## Build

Requirements: Rust (stable), CUDA toolkit + CUTLASS, an SM120 GPU (RTX 50-series).

```
cargo build --release
```

## Run

```
# text
cargo run --release -- generate --config examples/parameters.qwen35-9b.json \
  --prompt "Explain Rayleigh scattering." --max-tokens 200

# vision (image understanding)
cargo run --release -- generate --config examples/parameters.qwen35-9b.json \
  --prompt "<|vision_start|><|image_pad|><|vision_end|>Describe this image." \
  --image path/to/image.png --max-tokens 200

# server (OpenAI / Anthropic endpoints)
cargo run --release -- serve --config examples/parameters.qwen36-35b.json
```

A config is a small JSON describing the model path and per-component `compute`/`store` placement and KV-cache settings (see `examples/`).

## Roadmap

- 35B MoE **prefill** to grouped-MoE parity with the 26B (~3k tok/s) — per-expert variable-grid NVFP4 GEMM + chunked GDN recurrence.
- **Speculative decode (MTP)** — works for dense models; on MoE it needs the grouped-MoE verify path above.
- True **file-mmap** weight storage (lower host RAM footprint).
- Additional architectures (Nemotron 3 Nano).

## Status & caveats

Pre-alpha. Single-GPU, SM120-tuned, specific architectures. Not production-ready. **License: TBD** — to be added before wider distribution. Contributions and issues welcome.
