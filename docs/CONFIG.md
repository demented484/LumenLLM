# LumenLLM — Configuration Guide

A config is a single JSON file. Its whole job is the **core idea**: *you* place everything, explicitly. There are no hidden heuristics — the engine loads and runs exactly what you write.

Two knobs appear almost everywhere:

| Knob | Values | Meaning |
|---|---|---|
| `compute` | `"cuda:0"` (or `cuda:N`), `"cpu"` | where the math runs |
| `store`   | `"vram"` (= `"gpu"`), `"ram"`, `"mmap"` | where the weights live |

Anything you omit **inherits from its parent**, falling back to the top-level `model.compute` / `model.store`. So you only spell out what differs.

> Note: `"mmap"` currently behaves like `"ram"` on the CUDA backend (both pin a host arena); true file-mmap is on the roadmap.

---

## Top-level structure

```jsonc
{
  "model":            { ... },   // REQUIRED — the main model + placement
  "vision":           { ... },   // optional — load the vision tower (omit = no vision)
  "audio":            { ... },   // optional — load the audio tower (omit = no audio)
  "draft":            { ... },   // optional — EAGLE/MTP speculative-decode draft model
  "cuda":             { ... },   // optional — low-level CUDA runtime knobs
  "server-parameters":{ ... }    // optional — host/port/api for `serve`
}
```

Only `model` is required. Everything else is opt-in.

---

## `model`

```jsonc
"model": {
  "path": "/path/to/Model-NVFP4",   // REQUIRED: a HF-format checkpoint dir
  "compute": "cuda:0",              // default compute for everything below
  "store": "vram",                  // default tier for dense weights (embed, lm_head, norms)

  "input-layer":   { "compute": "cuda:0", "store": "ram"  },   // embeddings
  "output-layer":  { "compute": "cuda:0", "store": "vram" },   // final norm + lm_head
  "attention":     { "mechanism": "aegis-varlen", "compute": "cuda:0", "store": "vram" },
  "hidden-layers": { ... },                                    // the transformer blocks
  "other-parameters": { ... }                                  // sampling defaults
}
```

### `attention`
- `mechanism`: `"aegis-varlen"` (the tuned FlashAttention-2 / paged path) or `"reference"`.
- `compute-quantization`: `"fp8"` runs the attention in native FP8 (E4M3) MMA — faster at long context. **Requires an FP8 KV cache** (`kv-cache.type-k`/`type-v` = `"fp8"`), otherwise the parser rejects it. Use `"default"` / `"bf16"` for BF16 attention.

### `hidden-layers` — the interesting one
```jsonc
"hidden-layers": {
  "compute": "cuda:0",
  "store": "ram",                       // e.g. stream MoE experts from host RAM (won't fit in VRAM)
  "experts": { "compute": "cpu", "store": "ram" },   // optional: compute MoE experts on the CPU
  "kv-cache": {
    "context-size": 262144,
    "store": "vram",
    "type-k": "f16",                    // or "fp8"
    "type-v": "f16"
  },
  "ranges": [                            // optional: per-layer-range overrides
    { "start": 0,  "end": 24, "store": "vram", "compute": "cuda:0" },
    { "start": 24, "end": 48, "store": "ram",  "compute": "cpu" }
  ]
}
```
- `store: "ram"` here = the experts live in host RAM and stream to the GPU per token (how a 35B MoE fits on 16 GB).
- `experts: { compute: "cpu" }` = compute the routed experts **on the CPU** instead of streaming them to the GPU (the `--cpu-moe` analogue — best on hosts with high memory bandwidth).
- `kv-cache.type-k`/`type-v: "fp8"` ≈ halves KV memory; pairs with `attention.compute-quantization: "fp8"`.
- `ranges` lets you split layers (e.g. early layers on GPU, late layers on CPU).

### `other-parameters` (sampling defaults)
```jsonc
"other-parameters": { "temperature": 1.0, "top-p": 0.95, "top-k": 50, "min-p": 0.05 }
```
Greedy = `temperature: 0` (or pass `--temp 0`). Sampling runs on-device (the GPU sampler), so these are nearly free.

---

## `vision`

```jsonc
"vision": { "compute": "cuda:0", "store": "vram" }
```
**Omit this section → the vision tower is not loaded.** Include it to enable image understanding (the tower is read from the same checkpoint). Then at runtime:

```
... generate --config cfg.json \
  --prompt "<|vision_start|><|image_pad|><|vision_end|>Describe this image." \
  --image photo.png --max-tokens 200
```

The tower is small (BF16); `store: "vram"` keeps it resident.

---

## `draft` — speculative decoding (EAGLE / MTP)

```jsonc
"draft": {
  "path": "/path/to/Draft-Or-EAGLE-Model",   // a compatible, smaller draft model
  "compute": "cuda:0",
  "store": "vram",
  "num-draft-tokens": 4                       // tokens proposed per round (default 4)
}
```
Present → speculative decode is enabled with this draft (the big model verifies K drafted tokens per pass). Omit → plain decode. A `--draft-model` CLI flag overrides the config.

The draft accepts a **full model block** (same `path` + `compute`/`store` placement as `model`), so you can, say, keep the draft in VRAM while the main model streams from RAM. Speculative decode is **lossless** — accepted tokens are exactly what the main model would have produced. It pays off most on dense / non-streaming models.

---

## `server-parameters`

```jsonc
"server-parameters": {
  "host": "127.0.0.1", "port": 8080,
  "server-api": "openai",                 // or "anthropic"
  "api-keys": ["sk-..."]                  // optional auth
}
```

---

## Worked example

[`examples/parameters.gemma4-26b-vision-draft.json`](../examples/parameters.gemma4-26b-vision-draft.json) — Gemma-4-26B with the vision tower **and** a Gemma-4-E2B draft for speculative decode: experts stream from RAM (`hidden-layers.store: ram`), KV in VRAM at 262k context, vision + draft both resident on the GPU. Edit the two `path` fields to your local model directories.

---

## Recipes — fitting a big model on 16 GB

- **35B MoE that won't fit in VRAM:** `hidden-layers.store: "ram"` (stream experts) + KV `type-k`/`type-v: "fp8"` to halve KV. Dense parts stay `vram`.
- **Trade PCIe for CPU compute:** `hidden-layers.experts: { compute: "cpu", store: "ram" }` — experts never touch PCIe.
- **Long context cheaply:** hybrid models (Qwen3.5/3.6 GDN) keep KV tiny — only the periodic full-attention layers grow, so 262k context costs a few GiB.
- **Max long-context speed:** `attention.compute-quantization: "fp8"` + `kv-cache.type-k`/`type-v: "fp8"`.
- **Faster decode on dense models:** add a `draft` section (speculative decode).
