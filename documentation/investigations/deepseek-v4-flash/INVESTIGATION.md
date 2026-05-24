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

## Current run script

`model_scripts/run-deepseek-v4-flash.sh` — currently has `--enforce-eager` and `VLLM_DISABLED_KERNELS` is NOT set (good). DeepGEMM install script is at `model_scripts/deepseek-v4-flash-prereq/deepgemm-install.sh`.