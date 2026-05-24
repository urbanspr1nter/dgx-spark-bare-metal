# DeepSeek-V4-Flash on DGX Spark (sm_121 / GB10)

## Status: In Progress — Blockers #1–#3 fixed; Blocker #4 (FlashMLA) is next

**Goal:** Serve DeepSeek-V4-Flash (284B MoE, 13B active params) on a 4× DGX Spark cluster.

**Why vLLM v0.21.0:** DS4-Flash support was introduced in v0.21.0. This is the primary motivation for running this version rather than an older release.

## Model overview

- **Architecture:** 284B total params, 13B active per token, MoE with FP4 expert weights + FP8 everything else
- **Attention:** Novel compressed attention — `c4a` (4x compression) and `c128a` (128x compression) with shared KV and inverse RoPE
- **Context:** Up to 1M tokens
- **Quantization:** Mixed FP4 (MoE experts) + FP8 (dense layers), block-scaled with `weight_block_size: [128, 128]`

## Progress through blockers

Each blocker was discovered incrementally: fix one, run the model, see what crashes next.

### Blocker #1: mHC (Manifold-Constrained Hyper-Connection) — ✅ FIXED

- **Crash:** `tf32_hc_prenorm_gemm()` from DeepGEMM asserts on unsupported architecture
- **Location:** First thing called in every DS4-Flash transformer layer (`deepseek_v4.py:1213`)
- **Fix (commit `ffabca2e3`):**
  - SM12x dispatch check in `vllm/utils/deep_gemm.py` — routes to fallback on `is_device_capability_family(120)`
  - Triton split-K kernel (`tf32_hc_prenorm_gemm_triton` in `vllm/model_executor/layers/deepseek_v4_triton_kernels.py`) — primary path
  - Pure torch.matmul fallback (`_tf32_hc_prenorm_gemm_torch`) — writes full result to split-0
  - Post-GEMM fusion (`mhc_pre_big_fuse_tilelang`) already works on sm_121 via tilelang — no changes needed
- **Verified:** Full mHC pipeline produces correct Sinkhorn-normalized outputs on sm_121

### Blocker #2: UE8M0 ScalarType in cutlass_scaled_mm — ✅ FIXED

- **Crash:** `RuntimeError: Not yet supported ScalarType 44` — DS4-Flash uses `scale_fmt=ue8m0` (float8_e8m0fnu) for FP8 weight scales, but the C++ `cutlass_scaled_mm` op doesn't support this dtype
- **Location:** Attention Wq/Wk/Wv projection (`BlockScaledMMLinearKernel.apply_block_scaled_mm → cutlass_scaled_mm`)
- **Root cause:** MiniMax-M2.7 uses Float32 block scales with the same kernel, so ScalarType 44 was never hit before
- **Fix (commit `78bf5cda7`):**
  - Added `process_weights_after_loading` in `CutlassFp8BlockScaledMMKernel` that upcasts UE8M0 scales to float32 using existing `_upcast_e8m0_to_fp32` helper
  - Lossless conversion: UE8M0 values are always exact powers of 2, so float32 representation is exact
  - Simpler than jasl/vllm's approach which disables `CutlassFp8BlockScaledMMKernel` entirely on SM12x

### Blocker #3: DeepGEMM FP8 einsum (O-projection inverse-RoPE) — ✅ FIXED

- **Crash:** `RuntimeError: Assertion error (layout.hpp:39): t.dim() == N` — DeepGEMM's `fp8_einsum` layout assertions fail on SM12x
- **Location:** Attention O-projection (`deepseek_v4_attention.py:344 → deepseek_v4_fp8_einsum → fp8_einsum`)
- **Additional bug:** `_einsum_recipe` was `(1, 1, 128)` with `tma_aligned_scales=True` on SM12x (SM100 settings) — should be `(1, 128, 128)` with `False`
- **Fix (commit `648b521fe`):**
  - New Triton kernel `deepseek_v4_sm12x_fp8_einsum` in `vllm/v1/attention/ops/deepseek_v4_ops/fp8_einsum.py`
  - Dispatch in `deepseek_v4_fp8_einsum`: on SM12x, reshape 2D weights → 3D and call Triton kernel instead of DeepGEMM
  - Fixed `_einsum_recipe` and `_tma_aligned_scales` for SM12x in `DeepseekV4MLAAttention.__init__`
  - Handles UE8M0 → float32 scale upcasting within the kernel

### Blocker #4: FlashMLA C extension not compiled for sm_121 — 🔴 CURRENT

- **Crash:** `RuntimeError: vllm._flashmla_C is not available, likely was not compiled due to insufficient nvcc version or a supported arch was not in the list of target arches to compile for.`
- **Location:** Decode attention — `flash_mla_sparse_fwd` called from `deepseek_v4_attention.py:1098`
- **Root cause:** FlashMLA is a C extension that only compiles for sm_90/sm_100. The `is_flashmla_sparse_supported()` function explicitly rejects sm_121: `"FlashMLA Sparse is only supported on Hopper and Blackwell devices."`
- **Impact:** The model loaded, completed dummy forward pass, allocated KV caches, and crashed when processing the **first inference request** (decode attention path)
- **This is the largest remaining blocker.** The sparse MLA decode path is deeply integrated and has no fallback in stock v0.21.0.

### Blocker #5: MARLIN MoE backend — ❓ UNKNOWN

- The model selects `Using 'MARLIN' Mxfp4 MoE backend` — this may work on sm_121 since we compiled with `TORCH_CUDA_ARCH_LIST="12.0 12.1"` (arch 12.0 compiles NVFP4/MXFP4 kernels)
- **Cannot test** until blocker #4 is resolved

### Blocker #6: DeepGEMM MQA logits (sparse attention indexer) — ❓ UNKNOWN

- `fp8_fp4_mqa_logits` and `fp8_fp4_paged_mqa_logits` from DeepGEMM have no SM12x support
- May or may not be hit depending on the decode path taken — need to test
- jasl/vllm provides Triton fallbacks

## What works (confirmed by run logs)

| Component | Status | Notes |
|---|---|---|
| Model weight loading | ✅ | 37.84 GiB per GPU, ~46-119s |
| `CutlassFp8BlockScaledMMKernel` | ✅ | Selected for FP8 dense layers |
| UE8M0 scale handling | ✅ | Upcast to float32 at load time |
| mHC (hyper-connection) | ✅ | tilelang + Triton fallback |
| FP8 einsum (O-projection) | ✅ | Triton fallback |
| `_einsum_recipe` / `_tma_aligned_scales` | ✅ | Fixed for SM12x |
| MARLIN MoE backend selection | ✅ | `Using 'MARLIN' Mxfp4 MoE backend` |
| `fp8_ds_mla` KV cache format | ✅ | `Using DeepSeek's fp8_ds_mla KV cache format` |
| FP8 indexer cache | ✅ | `Using FP8 indexer cache for Lightning Indexer` |
| Expert parallelism | ✅ | 64 local / 256 global experts per rank |
| KV cache memory computation | ✅ | ~67.5 GiB per GPU |
| NCCL cross-node | ✅ | Working with custom all-reduce disabled |
| Dummy forward pass | ✅ | Completes successfully on all 4 nodes |
| Decode attention | ❌ | FlashMLA not compiled for sm_121 |
| MoE expert GEMM | ❓ | MARLIN selected but untested |

## GPU OOM note

The first successful dummy run hit GPU OOM during KV cache allocation with `--gpu-memory-utilization 0.90` and `--max-model-len 262144`. Reducing these parameters (e.g., `--gpu-memory-utilization 0.85` and `--max-model-len 131072`) allows KV caches to fit. This is a memory tuning issue, not a code blocker.

## jasl/vllm ds4-sm120 fork — reference for remaining blockers

[jasl/vllm](https://github.com/jasl/vllm/tree/ds4-sm120), branch `ds4-sm120`, PR [#40991](https://github.com/vllm-project/vllm/pull/40991). DeepGEMM-free DS4-Flash on SM12x with portable Triton fallbacks. Tested on 2× RTX PRO 6000 (SM120).

### Key approach: Triton sparse MLA replaces FlashMLA

The fork's most significant addition is a complete portable Triton sparse MLA decode path (`VLLM_TRITON_MLA_SPARSE=1`), auto-enabled on SM12x where FlashMLA is unavailable. This is what we need for blocker #4.

### Environment variables added

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_TRITON_MLA_SPARSE` | auto | `1` forces Triton sparse MLA; auto-enables on SM12x |
| `VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE` | `512` | Top-k candidate chunk size for sparse MLA accumulation |
| `VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE` | `256` | Query chunk size for prefill sparse MLA fallback |
| `VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH` | context | CUDA graphs for sparse MLA; disables for speculative |
| `VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE` | auto | Decode head block override |
| `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE` | auto | Matmul-based sparse MLA decode; auto on SM12x |
| `VLLM_DEEPSEEK_V4_USE_MEGA_MOE` | `False` | DeepGEMM MegaMoE; off by default (needed for SM12x) |

### Benchmark results (2× RTX PRO 6000, SM120)

| Concurrency | Context | Output tok/s | Mean TPOT | Mean TTFT |
|---|---|---|---|---|
| 1 | 128→512 | 100.38 | 9.76 ms | 113.4 ms |
| 4 | 128→512 | 296.84 | 13.16 ms | 171.9 ms |
| 8 | 128→512 | 478.34 | 16.18 ms | 291.6 ms |
| 1 | 8192→512 | 58.61 | 10.94 ms | 3143 ms |

### Files to port for remaining blockers

Sorted by priority:

**For blocker #4 (FlashMLA / sparse MLA decode):**
1. `vllm/v1/attention/backends/mla/sparse_mla_kernels.py` (2,694 new lines) — Triton kernels for sparse MLA decode
2. `vllm/v1/attention/backends/mla/sparse_mla_env.py` (150 new lines) — env var controls
3. `vllm/v1/attention/backends/mla/sparse_mla_reference.py` (242 new lines) — reference impl for testing
4. `vllm/v1/attention/backends/mla/sparse_swa.py` (47 new lines) — sink-aware SWA additions
5. `vllm/v1/attention/backends/mla/flashmla_sparse.py` (18 new lines) — SM12x guard
6. `vllm/v1/attention/backends/mla/indexer.py` (29 changed lines) — SM12x indexer fallback
7. `vllm/model_executor/layers/sparse_attn_indexer.py` (126 changed lines) — SM12x short-row top-k and logits fallbacks
8. `vllm/model_executor/layers/deepseek_v4_attention.py` (694 changed lines) — Triton sparse MLA decode integration, env var logic
9. `vllm/model_executor/layers/deepseek_v4_triton_kernels.py` (1,282 new lines total) — additional kernels for c4a/c128a compressor, MQA logits, etc.
10. `vllm/utils/deep_gemm.py` (518 changed lines) — SM12x fallbacks for MQA logits, paged MQA
11. `vllm/envs.py` (41 new lines) — new env vars

**For blocker #5 (MoE):**
12. `vllm/model_executor/models/deepseek_v4.py` (93 changed lines) — `_use_deepseek_v4_mega_moe()` guard

**Already ported (our work):**
- ✅ mHC Triton fallback — our own implementation, not from jasl
- ✅ UE8M0 → float32 upcast in CutlassFp8BlockScaledMMKernel — simpler than jasl's approach
- ✅ FP8 einsum Triton fallback — adapted from jasl's `fp8_einsum.py`
- ✅ `_einsum_recipe` / `_tma_aligned_scales` fix for SM12x

### Risks and considerations

- jasl's fork is based on v0.20.1rc1, not v0.21.0. Direct merge is not possible — must cherry-pick and adapt.
- The sparse MLA path is the largest and most complex port (3,000+ lines across multiple files).
- jasl's fork tested on SM120 (RTX PRO 6000), not SM121 (GB10). Triton kernels should work on both but SM121-specific testing is needed.
- The `deepseek_v4_attention.py` diff is 694 changed lines — many are intertwined with other changes. Will need careful review.

## tilelang on sm_121 — verified working

tilelang 0.1.9 is installed and works on sm_121. `mhc_pre_big_fuse_tilelang` JIT-compiles and produces correct Sinkhorn-normalized outputs. TVM targets `cuda -arch=sm_121` correctly.

## What works (from MiniMax-M2.7 investigation)

- `CutlassFp8BlockScaledMMKernel` — sm_121 cubins compile and run correctly with our patches
- `TORCH_CUDA_ARCH_LIST="12.0 12.1"` — both archs needed (12.0 for NVFP4/MXFP4, 12.1 for native sm_121 FP8 block-scaled GEMM)
- CUDA graphs — work on sm_121, significant decode throughput boost (9 tk/s → 22 tk/s for MiniMax-M2.7)
- Expert parallel with TP4 — works correctly with `CutlassFp8BlockScaledMMKernel`
- `VLLM_DISABLED_KERNELS` — must be unset before Ray starts

## Current run script

`model_scripts/run-deepseek-v4-flash.sh` — currently has `--enforce-eager` and `--enable-expert-parallel`. `VLLM_DISABLED_KERNELS` is NOT set (good).