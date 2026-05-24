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
  - Triton split-K kernel (`tf32_hc_prenorm_gemm_triton` in `deepseek_v4_triton_kernels.py`) — primary path
  - Pure torch.matmul fallback (`_tf32_hc_prenorm_gemm_torch`) — writes full result to split-0
  - Post-GEMM fusion (`mhc_pre_big_fuse_tilelang`) already works on sm_121 via tilelang
- **Verified:** Full mHC pipeline produces correct Sinkhorn-normalized outputs on sm_121

### Blocker #2: UE8M0 ScalarType in cutlass_scaled_mm — ✅ FIXED

- **Crash:** `RuntimeError: Not yet supported ScalarType 44` — DS4-Flash uses `scale_fmt=ue8m0` (float8_e8m0fnu) for FP8 weight scales
- **Location:** Attention Wq/Wk/Wv projection
- **Fix (commit `78bf5cda7`):** `process_weights_after_loading` in `CutlassFp8BlockScaledMMKernel` upcasts UE8M0 → float32 at load time using `_upcast_e8m0_to_fp32`

### Blocker #3: DeepGEMM FP8 einsum (O-projection inverse-RoPE) — ✅ FIXED

- **Crash:** `RuntimeError: Assertion error (layout.hpp:39): t.dim() == N` — DeepGEMM layout assertions fail on SM12x
- **Location:** Attention O-projection (`deepseek_v4_fp8_einsum → fp8_einsum`)
- **Additional bug fixed:** `_einsum_recipe` was `(1, 1, 128)` / `tma_aligned_scales=True` on SM12x (SM100 settings) → corrected to `(1, 128, 128)` / `False`
- **Fix (commit `648b521fe`):**
  - New Triton kernel `deepseek_v4_sm12x_fp8_einsum` in `vllm/v1/attention/ops/deepseek_v4_ops/fp8_einsum.py`
  - Dispatch in `deepseek_v4_fp8_einsum`: reshape 2D→3D weights on SM12x, call Triton kernel

### Blocker #4: FlashMLA C extension not compiled for sm_121 — 🔴 CURRENT

- **Crash:** `RuntimeError: vllm._flashmla_C is not available`
- **Location:** Decode attention (`flash_mla_with_kvcache`) and prefill attention (`flash_mla_sparse_fwd`)
- **Root cause:** FlashMLA is a C extension compiled only for sm_90/sm_100. `is_flashmla_sparse_supported()` explicitly rejects sm_121.
- **Impact:** Model loads, completes dummy forward pass, allocates KV caches, crashes on first inference request
- **This is the largest remaining blocker.** Requires porting ~3,500 lines of portable Triton sparse MLA from jasl/vllm.

### Blocker #5: MARLIN MoE backend — ❓ UNKNOWN (untestable until #4 fixed)

- Model selects `Using 'MARLIN' Mxfp4 MoE backend` — may work on sm_121 since arch 12.0 compiles NVFP4/MXFP4 kernels

### Blocker #6: DeepGEMM MQA logits (sparse attention indexer) — ❓ UNKNOWN

- May be hit depending on decode path after Triton sparse MLA is integrated
- jasl/vllm provides torch.einsum + Triton fallbacks

## What works (confirmed by run logs)

| Component | Status | Notes |
|---|---|---|
| Model weight loading | ✅ | 37.84 GiB per GPU |
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
| Dummy forward pass | ✅ | Completes successfully on all 4 nodes |
| Decode attention | ❌ | FlashMLA not compiled for sm_121 |
| MoE expert GEMM | ❓ | MARLIN selected but untested |

## GPU OOM note

With `--gpu-memory-utilization 0.90` and `--max-model-len 262144`, GPU OOM occurs during KV cache allocation. Reducing these parameters (e.g., `0.85` and `131072`) allows KV caches to fit. Memory tuning issue, not a code blocker.

## Plan: Porting Triton sparse MLA (Blocker #4)

The FlashMLA C extension provides two functions used by DS4-Flash attention:
1. **`flash_mla_with_kvcache`** — decode path (SWA-only and compressed c4a/c128a)
2. **`flash_mla_sparse_fwd`** — prefill path (c4a/c128a sparse attention over gathered KV)

Both crash on sm_121. jasl/vllm replaces them with portable Triton kernels on SM12x. The port is structured in 4 phases:

### Phase 1: Environment detection + guard infrastructure — ✅ DONE (commit `4b23c8309`)

**Goal:** Add SM12x detection and env var controls so the codebase knows when to use Triton sparse MLA instead of FlashMLA.

**Files created/modified:**
- **Created** `vllm/v1/attention/backends/mla/sparse_mla_env.py` (223 lines)
  - `is_triton_sparse_mla_enabled_for_platform()` — auto-enables on SM12x
  - `is_triton_sparse_mla_enabled(device)` — per-device check
  - `triton_sparse_mla_matmul_decode_enabled()` — matmul vs gather decode toggle (auto on SM12x)
  - `triton_sparse_mla_topk_chunk_size()` / `query_chunk_size()` / `head_block_size()` — tuning knobs
  - `triton_sparse_mla_cudagraphs_allowed()` — respects env var + speculative decoding
  - `disable_triton_sparse_mla_cudagraphs_if_enabled()` — disables CUDA graphs for Triton MLA when speculative decoding is configured
  - 6 env vars: `VLLM_TRITON_MLA_SPARSE` (auto), `TOPK_CHUNK_SIZE` (512), `QUERY_CHUNK_SIZE` (256), `ALLOW_CUDAGRAPH` (true), `HEAD_BLOCK_SIZE` (auto), `MATMUL_DECODE` (auto)
- **Modified** `vllm/envs.py` (+37 lines) — 6 dataclass fields + 6 environment_variables entries
- **Modified** `vllm/v1/attention/backends/mla/flashmla_sparse.py` (+18 lines) — `get_cudagraph_support()` override on `FlashMLASparseMetadataBuilder`: returns `NEVER` when DS4 + Triton MLA + speculative decoding

**Tests passed:** auto-detection on sm_121, force-enable/disable via env var, CUDA graph guard with/without speculative decoding, default tuning knobs.

### Phase 2: Triton sparse MLA kernels (~2,700 lines)

**Goal:** Port the portable Triton kernels that replace FlashMLA for decode and prefill attention.

**Files to create:**
- **Create** `vllm/v1/attention/backends/mla/sparse_mla_kernels.py` (2,694 lines) — from jasl
  - Decode kernels: `fp8ds_paged_sparse_mla_attention_with_sink_multihead`, `fp8ds_global_paged_sparse_mla_attention_with_sink_multihead`, `matmul_sparse_mla_attention_with_sink`
  - Prefill kernels: `accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead`, `accumulate_indexed_sparse_mla_attention_chunk`, `build_combined_sparse_mla_decode_valid_mask`, `finish_sparse_mla_attention_with_sink`, `finish_two_sparse_mla_attention_states_with_sink`
  - Helper: `sparse_mla_decode_head_block_size`, `dequantize_combined_sparse_mla_decode_kv`
- **Create** `vllm/v1/attention/backends/mla/sparse_mla_reference.py` (242 lines) — from jasl
  - Reference implementation for correctness testing

**Test:** Import kernels, run a small decode attention test with known inputs to verify numerical correctness.

### Phase 3: Attention dispatch integration (~700 lines diff)

**Goal:** Wire the Triton sparse MLA kernels into `deepseek_v4_attention.py` so that on SM12x, `_forward_decode` and `_forward_prefill` use the Triton path instead of FlashMLA.

**Files to modify:**
- **Modify** `vllm/model_executor/layers/deepseek_v4_attention.py` (~694 changed lines from jasl)
  - Import `sparse_mla_kernels` and `sparse_mla_env`
  - Add `_forward_sparse_mla_swa_decode_triton()` — SWA-only decode via Triton
  - Add `_forward_sparse_mla_compressed_decode_triton()` — c4a/c128a decode via Triton
  - Modify `_forward_decode()` — dispatch to Triton on SM12x before FlashMLA
  - Modify `_forward_prefill()` — add Triton chunked prefill path
  - Modify `__init__()` — call `disable_triton_sparse_mla_cudagraphs_if_enabled()`
  - Modify `get_kv_cache_spec()` — adjust alignment for Triton MLA (576 → variable)
- **Modify** `vllm/v1/attention/backends/mla/sparse_swa.py` (~47 changed lines) — sink-aware SWA additions
- **Modify** `vllm/v1/attention/backends/mla/flashmla_sparse.py` (~18 changed lines) — CUDA graph guard

**Test:** Run DS4-Flash with the Triton sparse MLA path. Decode attention should work. Verify output is not garbled.

### Phase 4: MQA logits fallbacks (~1,300 lines)

**Goal:** Replace DeepGEMM's MQA logits kernels with portable fallbacks for the c4a/c128a compressed attention indexer.

**Files to create/modify:**
- **Modify** `vllm/model_executor/layers/deepseek_v4_triton_kernels.py` (+1,130 lines)
  - `sparse_attention_triton()` — bf16 sparse attention (prefill)
  - `decode_sparse_attention_triton()` — fp8 decode sparse attention
  - `fp8_mqa_logits_triton()` — FP8 MQA logits
  - `fp8_paged_mqa_logits_triton()` — FP8 paged MQA logits
  - `fp8_paged_mqa_logits_rowwise_triton()` — rowwise variant
  - Helper functions: `_view_packed_fp8_paged_mqa_kv_cache`, `_e8m0_to_fp32`, `_unpack_int32_e8m0_scales`, `_normalize_deepseek_v4_fp8_einsum_inputs`
- **Modify** `vllm/utils/deep_gemm.py` (~518 changed lines from jasl)
  - SM12x dispatch for `fp8_fp4_mqa_logits` → `_fp8_mqa_logits_sm12x`
  - SM12x dispatch for `fp8_fp4_paged_mqa_logits` → torch/Triton fallback
  - SM12x dispatch for `fp8_fp4_mqa_topk_indices` → `_fp8_mqa_logits_topk_torch`
  - `_uses_deep_gemm_scheduler_metadata()` — returns False on SM12x
- **Modify** `vllm/v1/attention/backends/mla/indexer.py` (~29 changed lines)
  - `sparse_indexer_max_logits_bytes()` — reduced max logits on SM12x (256MB vs 512MB)
  - `_uses_deep_gemm_scheduler_metadata()` — SM12x guard for DeepGEMM metadata path
- **Modify** `vllm/model_executor/layers/sparse_attn_indexer.py` (~126 changed lines)
  - SM12x short-row top-k fallback
  - SM12x logits width constraints

**Test:** Full c4a/c128a compressed attention should work. Run with longer context to exercise indexer.

### Post-Phase 4: MoE validation

Once all four phases are complete and the model serves tokens, verify that the MARLIN MoE backend works for FP4 expert GEMMs. If it does, no additional work needed. If not, may need to investigate MoE fallbacks.

### Estimated line counts per phase

| Phase | New lines | Changed lines | Complexity |
|---|---|---|---|
| 1. Env detection | ~150 | ~50 | Low |
| 2. Triton kernels | ~2,940 | ~0 | High (Triton math) |
| 3. Dispatch integration | ~0 | ~700 | High (API coupling) |
| 4. MQA logits fallbacks | ~1,130 | ~700 | Medium |
| **Total** | ~4,220 | ~1,450 | — |

### Key risks

- **v0.20.1 → v0.21.0 API drift:** jasl's fork is based on v0.20.1rc1. The `deepseek_v4_attention.py` diff (694 lines) is the riskiest — many changes are intertwined with other v0.20.1→v0.21.0 differences.
- **SM120 vs SM121:** jasl tested on RTX PRO 6000 (SM120). Triton kernels should work on both, but SM121-specific testing is needed.
- **`sparse_mla_kernels.py` is the biggest file** (2,694 lines). Mostly pure Triton math with minimal vLLM API coupling, which makes it the safest large port.
- **Phase 3 (dispatch) is the hardest** despite being the smallest in lines — it touches `deepseek_v4_attention.py` which has diverged significantly between v0.20.1 and v0.21.0.

### Approach: write our own code, use jasl as reference

Following the same strategy that worked for blockers #1–#3: study jasl's fork for the approach, then write our own implementation on v0.21.0-sm121-fix. This avoids merge conflicts and ensures we understand every line.

## jasl/vllm ds4-sm120 fork — reference

[jasl/vllm](https://github.com/jasl/vllm/tree/ds4-sm120), branch `ds4-sm120`, PR [#40991](https://github.com/vllm-project/vllm/pull/40991). DeepGEMM-free DS4-Flash on SM12x with portable Triton fallbacks. Tested on 2× RTX PRO 6000 (SM120).

### Benchmark results (2× RTX PRO 6000, SM120)

| Concurrency | Context | Output tok/s | Mean TPOT | Mean TTFT |
|---|---|---|---|---|
| 1 | 128→512 | 100.38 | 9.76 ms | 113.4 ms |
| 4 | 128→512 | 296.84 | 13.16 ms | 171.9 ms |
| 8 | 128→512 | 478.34 | 16.18 ms | 291.6 ms |
| 1 | 8192→512 | 58.61 | 10.94 ms | 3143 ms |

### Already ported from jasl (our work, adapted for v0.21.0)

- ✅ mHC Triton fallback — our own implementation, inspired by jasl's approach
- ✅ UE8M0 → float32 upcast — simpler than jasl's approach (we keep CutlassFp8BlockScaledMMKernel, jasl disables it on SM12x)
- ✅ FP8 einsum Triton fallback — adapted from jasl's `fp8_einsum.py`
- ✅ `_einsum_recipe` / `_tma_aligned_scales` fix — from jasl's approach

### What works (from MiniMax-M2.7 investigation)

- `CutlassFp8BlockScaledMMKernel` — sm_121 cubins compile and run correctly
- `TORCH_CUDA_ARCH_LIST="12.0 12.1"` — both archs needed
- CUDA graphs — work on sm_121 (9 tk/s → 22 tk/s for MiniMax-M2.7)
- Expert parallel with TP4 — works correctly
- `VLLM_DISABLED_KERNELS` — must be unset before Ray starts

## Current run script

`model_scripts/run-deepseek-v4-flash.sh` — currently has `--enforce-eager` and `--enable-expert-parallel`.