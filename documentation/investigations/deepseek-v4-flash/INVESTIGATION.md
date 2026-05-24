# DeepSeek-V4-Flash on DGX Spark (sm_121 / GB10)

## Status: Investigation

**Goal:** Serve DeepSeek-V4-Flash (284B MoE, 13B active params) on a 4× DGX Spark cluster.

**Why vLLM v0.21.0:** DS4-Flash support was introduced in v0.21.0. This is the primary motivation for running this version rather than an older release.

## Model overview

- **Architecture:** 284B total params, 13B active per token, MoE with FP4 expert weights + FP8 everything else
- **Attention:** Novel compressed attention — `c4a` (4x compression) and `c128a` (128x compression) with shared KV and inverse RoPE
- **Context:** Up to 1M tokens
- **Quantization:** Mixed FP4 (MoE experts) + FP8 (dense layers), block-scaled with `weight_block_size: [128, 128]`

## Known blockers on sm_121 / GB10

vLLM's DS4-Flash implementation targets **Hopper (sm_90) and Blackwell (sm_100)** exclusively. The following components lack sm_121 support:

### 1. DeepGEMM — required for FP4 MoE expert GEMMs

DeepGEMM is the only backend that supports FP4 MoE weights. It is required for DS4-Flash.

- **`support_deep_gemm()`** in `vllm/platforms/cuda.py` only returns True for `is_device_capability(90) or is_device_capability_family(100)`
- **DeepGEMM 2.5.0 is installed** on all nodes but is non-functional on sm_121
- The compiled `_C.so` has sm_120 cubins but **not sm_121** — those cubins come from bundled CUTLASS, not DeepGEMM's own kernels
- DeepGEMM's own GEMM kernels are arch-specific:
  - `sm90_fp8_gemm_1d1d.cuh` — uses Hopper `WGMMA` instructions (`#if __CUDA_ARCH__ >= 900`)
  - `sm100_*.cuh` — uses Blackwell `UMMA` instructions (`#if __CUDA_ARCH__ >= 1000`)
  - Both contain runtime assertions: `"This kernel only support sm_90a"` / `"This kernel only support sm_100f"`
  - **No sm120/sm121 implementations exist** in DeepGEMM 2.5.0
- This is not just a compile guard issue — DeepGEMM fundamentally relies on Hopper WGMMA or Blackwell UMMA tensor core instructions that do not exist on sm_121 (GB10 uses a different tensor core ISA)
- The install script at `model_scripts/deepseek-v4-flash-prereq/deepgemm-install.sh` installed DeepGEMM but it cannot run on GB10

**Impact:** Without DeepGEMM, FP4 MoE expert GEMMs cannot run. This is a hard blocker — no other MoE backend in vLLM supports FP4 weights.

**Possible paths forward:**
- Wait for DeepGEMM upstream to add sm_120 family support (they bundle CUTLASS headers that include sm120 kernels, so the foundation exists)
- Use a different FP4 MoE kernel backend (e.g., CUTLASS sm120 GEMM kernels adapted for grouped MoE)
- Fall back to dequantizing FP4 weights to FP8 and using the existing `CutlassFp8BlockScaledMMKernel` / TRITON MoE path (would increase memory usage but might work)

### 2. FlashInfer — used for compressed attention kernels

DS4-Flash's c4a/c128a attention uses FlashInfer for optimized attention kernels. The FlashInfer CUTLASS MoE backend supports sm_120 family at the device level, but its block-scaled FP8 quant scheme (`kFp8Static128BlockSym` / `kFp8Dynamic128Sym`) is gated to sm_90 only.

**Impact:** Unknown severity — depends on which FlashInfer kernels DS4-Flash actually calls at runtime and whether they have sm_121 codepaths.

### 3. CUDA compilation flags

Same `CUDA_SUPPORTED_ARCHS` issue we fixed for MiniMax-M2.7 — our v0.21.0-sm121-fix branch already includes the patch for `12.1` in `CUDA_SUPPORTED_ARCHS` and `enable_sm120_family` guards.

**Impact:** Already fixed in our branch (`urbanspr1nter/vllm`, branch `v0.21.0-sm121-fix`).

### 4. Tuned MoE configs

No tuned MoE config files exist for `NVIDIA_GB10` in `vllm/model_executor/layers/fused_moe/configs/`. This causes a fallback to default configs with the TRITON MoE backend, which is suboptimal but not a blocker.

**Impact:** Performance only — the model will serve but with lower MoE throughput.

## What works (from MiniMax-M2.7 investigation)

- `CutlassFp8BlockScaledMMKernel` — sm_121 cubins compile and run correctly with our patches
- `TORCH_CUDA_ARCH_LIST="12.0 12.1"` — both archs needed (12.0 for NVFP4/MXFP4 instructions, 12.1 for native sm_121 FP8 block-scaled GEMM)
- CUDA graphs — work on sm_121, significant decode throughput boost
- Expert parallel with TP4 — works correctly with `CutlassFp8BlockScaledMMKernel`
- `VLLM_DISABLED_KERNELS` — must be unset before Ray starts

## Open questions

- Can DeepGEMM be adapted for sm_121, or does it fundamentally rely on Hopper-specific wgmma instructions?
- Does the FP4 MoE path in vLLM have a fallback that doesn't require DeepGEMM?
- Which FlashInfer kernels does DS4-Flash actually use at runtime on sm_121?
- Is there an FP4 MoE kernel path that works on sm_121 (e.g., via CUTLASS or Triton)?
- What is the minimum set of changes needed to get DS4-Flash loading and serving on GB10, even at reduced performance?

## jasl/vllm ds4-sm120 fork — existing SM12x implementation

There is an existing fork that implements DS4-Flash support for SM12x GPUs: [jasl/vllm](https://github.com/jasl/vllm/tree/ds4-sm120), branch `ds4-sm120`. This was submitted as [PR #40991](https://github.com/vllm-project/vllm/pull/40991) to vLLM on April 27, 2026.

### Key insight: DeepGEMM-free approach

The fork's critical innovation is that it is **"DeepGEMM free"** — it replaces all DeepGEMM-dependent paths with portable Triton fallbacks for SM12x. This means:

1. **Sparse MLA attention** — Replaces FlashMLA (which uses DeepGEMM's MQA logits kernels) with a portable Triton sparse MLA path (`VLLM_TRITON_MLA_SPARSE=1`). This is controlled by environment variables and auto-enables on SM12x where FlashMLA is unavailable.

2. **MQA logits (indexer)** — For the c4a compressed sparse attention indexer, the fork provides SM12x fallback kernels that use `torch.einsum` instead of DeepGEMM's `fp8_fp4_mqa_logits`. The `_sparse_indexer_requires_deep_gemm()` function returns `False` on SM12x, routing to the Triton path instead.

3. **Paged MQA logits** — Similar fallback for paged MQA with `fp8_fp4_paged_mqa_logits` / `fp8_fp4_paged_mqa_topk_indices`. SM12x gets local Triton-based fallbacks that don't initialize DeepGEMM.

4. **FP8 einsum** — A new `vllm::deepseek_v4_fp8_einsum` Triton kernel for the c4a/c128a compressor and inverse-RoPE operations that DeepGEMM's `einsum` handles on SM90/SM100.

5. **MoE** — The `VLLM_DEEPSEEK_V4_USE_MEGA_MOE` env var (default `False`) controls whether to use the DeepGEMM MegaMoE fused expert kernel. On SM12x, this defaults to off, falling back to the standard TRITON MoE backend.

### Environment variables

The fork adds several new environment variables for SM12x control:

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_TRITON_MLA_SPARSE` | auto | `1` forces Triton sparse MLA, `0` disables. Auto-enables on SM12x. |
| `VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE` | `512` | Top-k candidate chunk size for sparse MLA accumulation |
| `VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE` | `256` | Query chunk size for prefill sparse MLA fallback |
| `VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH` | context | Allows CUDA graphs for sparse MLA. Disables for speculative decoding. |
| `VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE` | auto | Decode head block override. Benchmarks used `4`. |
| `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE` | auto | Matmul-based sparse MLA decode toggle. Auto-enables on SM12x. |
| `VLLM_DEEPSEEK_V4_USE_MEGA_MOE` | `False` | Controls DeepGEMM MegaMoE. Off by default (needed for SM12x). |

### Benchmark results (2x RTX PRO 6000 Blackwell, SM120)

**No MTP, 128→512 context:**

| Concurrency | Output tok/s | Mean TPOT | Mean TTFT |
|---|---|---|---|
| 1 | 100.38 | 9.76 ms | 113.4 ms |
| 4 | 296.84 | 13.16 ms | 171.9 ms |
| 8 | 478.34 | 16.18 ms | 291.6 ms |

**No MTP, 8192→512 context:**

| Concurrency | Output tok/s | Mean TPOT | Mean TTFT |
|---|---|---|---|
| 1 | 58.61 | 10.94 ms | 3143.0 ms |
| 2 | 81.35 | 15.37 ms | 4732.0 ms |

**With MTP (preview branch), 128→512:**

| Concurrency | no-MTP tok/s | MTP tok/s | MTP delta |
|---|---|---|---|
| 1 | 103.03 | 161.14 | +56.4% |
| 4 | 303.20 | 326.51 | +7.7% |
| 8 | 473.53 | 525.08 | +10.9% |

### Serving command (from the PR)

```
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.94 \
  --tensor-parallel-size 2 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4
```

### Key differences from our DGX Spark setup

| Aspect | jasl/vllm (RTX PRO 6000) | DGX Spark (GB10) |
|---|---|---|
| GPU | SM120 (2x 96GB) | SM121 (4x 96GB) |
| CUDA arch list | `12.0a` / `120a` | `12.0 12.1` (already patched) |
| vLLM version | v0.20.1rc1 (base) | v0.21.0 (our fork) |
| DeepGEMM | Not needed (Triton fallbacks) | Installed but non-functional |
| `--enforce-eager` | Not used (CUDA graphs work) | Not needed (CUDA graphs work after our fix) |
| `--enable-expert-parallel` | Not used (TP=2) | Needed (TP=4) |
| `--compilation-config` | Full CUDA graphs + piecewise | Not used yet |
| Triton sparse MLA | Enabled via env vars | Not yet integrated |

### What we need to port from jasl/vllm

The jasl/vllm fork is based on an older vLLM (v0.20.1rc1), not v0.21.0. We need to cherry-pick the SM12x fallback logic rather than merge the entire branch. Key files to port:

1. **`vllm/v1/attention/backends/mla/sparse_mla_kernels.py`** (2,694 new lines) — Triton kernels for SM12x sparse MLA decode and prefill
2. **`vllm/v1/attention/backends/mla/sparse_mla_env.py`** (150 new lines) — Environment variable controls for Triton sparse MLA
3. **`vllm/v1/attention/backends/mla/sparse_mla_reference.py`** (242 new lines) — Reference implementation for correctness testing
4. **`vllm/v1/attention/backends/mla/sparse_swa.py`** (47 new lines) — Sink-aware SWA additions
5. **`vllm/v1/attention/backends/mla/flashmla_sparse.py`** (18 new lines) — SM12x guard for FlashMLA sparse
6. **`vllm/v1/attention/backends/mla/indexer.py`** (29 changed lines) — SM12x indexer fallback logic
7. **`vllm/model_executor/layers/sparse_attn_indexer.py`** (126 changed lines) — SM12x short-row top-k and logits width fallbacks
8. **`vllm/model_executor/layers/deepseek_v4_triton_kernels.py`** (1,282 new lines) — Triton kernels for c4a/c128a compressor and inverse-RoPE
9. **`vllm/model_executor/layers/deepseek_v4_attention.py`** (694 changed lines) — SM12x FP8 einsum path, Triton sparse MLA decode integration
10. **`vllm/utils/deep_gemm.py`** (518 changed lines) — SM12x fallbacks for MQA logits, paged MQA, DeepGEMM support detection changes
11. **`vllm/model_executor/models/deepseek_v4.py`** (93 changed lines) — `_use_deepseek_v4_mega_moe()` guard, MoE clamp limit forwarding
12. **`vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`** (45 new lines) — SM12x NVFP4 block-scaled MM kernel support
13. **`vllm/envs.py`** (41 new lines) — New environment variables
14. **`vllm/v1/attention/ops/deepseek_v4_ops/fp8_einsum.py`** (175 new lines) — SM12x FP8 einsum Triton kernel

### Risks and considerations

- The fork was tested on SM120 (RTX PRO 6000), not SM121 (GB10). The Triton kernels should work on both, but SM121-specific testing is needed.
- The fork uses `--compilation-config` with CUDA graphs and piecewise compilation. Our DGX Spark setup currently uses `--enforce-eager`. We should test whether CUDA graphs work with the DS4-Flash model on our hardware.
- The MoE path defaults to TRITON backend (no DeepGEMM MegaMoE). This is the same situation we already have with MiniMax-M2.7 — throughput will be suboptimal but functional.
- The `VLLM_DEEPSEEK_V4_USE_MEGA_MOE` flag allows forcing MegaMoE on if DeepGEMM is available, but it won't work on SM121.
- The fork's base is v0.20.1rc1, which is older than our v0.21.0-sm121-fix. Some changes may conflict with our existing patches.

## Current run script

`model_scripts/run-deepseek-v4-flash.sh` — currently has `--enforce-eager` and `VLLM_DISABLED_KERNELS` is NOT set (good). DeepGEMM install script is at `model_scripts/deepseek-v4-flash-prereq/deepgemm-install.sh`.