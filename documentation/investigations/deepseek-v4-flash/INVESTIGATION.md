# DeepSeek-V4-Flash on DGX Spark (sm_121 / GB10)

## Status: In Progress — Blockers #1, #2, #3 fixed; testing Blocker #4

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

## First run attempt on v0.21.0-sm121-fix (2025-05-24)

Attempted to serve DS4-Flash on our 4× DGX Spark cluster with the current `v0.21.0-sm121-fix` branch (without any jasl/vllm patches). The model loaded successfully across all 4 workers (37.84 GiB per GPU), but crashed during the profile/dummy run with a single fatal error.

### Crash: `mhc_pre` → `tf32_hc_prenorm_gemm` → DeepGEMM "Unsupported architecture"

```
File "vllm/model_executor/models/deepseek_v4.py", line 1213, in forward
    x, post_mix, res_mix = self.hc_pre(
File "vllm/model_executor/models/deepseek_v4.py", line 1179, in hc_pre
    post_mix, res_mix, layer_input = torch.ops.vllm.mhc_pre(
File "vllm/model_executor/layers/mhc.py", line 310, in mhc_pre
    tf32_hc_prenorm_gemm(
File "vllm/utils/deep_gemm.py", line 477, in tf32_hc_prenorm_gemm
    return _tf32_hc_prenorm_gemm_impl(
RuntimeError: Assertion error (csrc/apis/hyperconnection.hpp:56): Unsupported architecture
```

**Root cause:** The **Manifold-Constrained Hyper-Connection (mHC)** layer calls `torch.ops.vllm.mhc_pre()` which internally calls `tf32_hc_prenorm_gemm()` from DeepGEMM. This is a DeepGEMM C++ extension function (`_tf32_hc_prenorm_gemm_impl`) that asserts the GPU architecture is supported. On sm_121 (GB10), DeepGEMM's hyperconnection kernel has no implementation — it only supports sm_90 and sm_100.

**This is the FIRST blocker encountered — not the MoE, not the attention, but mHC.** The model fails at the very first transformer layer's hyper-connection pre-norm, before it ever reaches attention or MoE.

### What loaded successfully before the crash

| Component | Status | Notes |
|---|---|---|
| Model weights | ✅ Loaded | 37.84 GiB per GPU, ~86-171s load time |
| `CutlassFp8BlockScaledMMKernel` | ✅ Selected | For FP8 dense layers |
| `deepseek_v4_fp8` quantization | ✅ Detected | `scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM` |
| Expert parallelism | ✅ Configured | 64 local / 256 global experts per rank, linear placement |
| MXFP4 MoE backend | ✅ MARLIN selected | `Using 'MARLIN' Mxfp4 MoE backend` |
| `fp8_ds_mla` KV cache | ✅ Activated | `Using DeepSeek's fp8_ds_mla KV cache format` |
| FP8 indexer cache | ✅ Activated | `Using FP8 indexer cache for Lightning Indexer` |
| NCCL | ✅ Working | Cross-node TP4 with custom all-reduce disabled |
| SymmMemCommunicator | ⚠️ Not supported | `Device capability 12.1 not supported` (expected, non-fatal) |

### The full list of sm_121 blockers (updated from this run)

Now that we've seen the model actually attempt to run, the blockers are clear and ordered:

**Blocker #1: mHC (Manifold-Constrained Hyper-Connection) — ✅ FIXED**
- `tf32_hc_prenorm_gemm()` from DeepGEMM calls a C++ extension that asserts on architecture
- This is the **first** thing called in every DS4-Flash transformer layer
- DeepGEMM's `csrc/apis/hyperconnection.hpp:56` has `assert(arch == 90 || arch == 100)` (or similar)
- **Fix applied** (commit `ffabca2e3` on `v0.21.0-sm121-fix`):
  - Added `is_device_capability_family(120)` check in `tf32_hc_prenorm_gemm()` to route SM12x to fallback
  - Triton split-K kernel (`tf32_hc_prenorm_gemm_triton` in `deepseek_v4_triton_kernels.py`) — primary path
  - Pure torch.matmul fallback (`_tf32_hc_prenorm_gemm_torch`) — writes full result to split-0
  - The post-GEMM fusion (`mhc_pre_big_fuse_tilelang`) already works on sm_121 via tilelang — no changes needed
  - Verified: full mHC pipeline produces correct outputs on sm_121

**Blocker #2: UE8M0 ScalarType in cutlass_scaled_mm — FIXED** ✅
- After mHC fix, the model proceeded to the attention layer's fused Wq/Wk/Wv projection
- DS4-Flash uses `quantization_config.scale_fmt=ue8m0` — 8-bit exponent-only scales (ScalarType 44)
- `CutlassFp8BlockScaledMMKernel` was selected but the C++ `cutlass_scaled_mm` op doesn't support ScalarType 44
- **Fix:** Added `process_weights_after_loading` in `CutlassFp8BlockScaledMMKernel` that upcasts UE8M0 scales to float32 using existing `_upcast_e8m0_to_fp32` helper. This is lossless since UE8M0 values are always exact powers of 2. Commit `78bf5cda7`.

**Blocker #3: DeepGEMM FP8 einsum — FIXED** ✅
- After UE8M0 fix, model proceeds further into attention and crashes in `fp8_einsum` (DeepGEMM's einsum for the `wo_a` O-projection inverse-RoPE)
- Error: `RuntimeError: Assertion error (layout.hpp:39): t.dim() == N` — DeepGEMM's layout assertions fail on SM12x
- Also fixed `_einsum_recipe` for SM12x: was `(1, 1, 128)` with `tma_aligned_scales=True` (SM100 path); now `(1, 128, 128)` with `tma_aligned_scales=False`
- **Fix:** New Triton kernel `deepseek_v4_sm12x_fp8_einsum` in `vllm/v1/attention/ops/deepseek_v4_ops/fp8_einsum.py`. Dispatch logic in `deepseek_v4_fp8_einsum` reshapes 2D weights to 3D and calls Triton kernel on SM12x. Commit `648b521fe`.

**Blocker #4: DeepGEMM MQA logits (sparse attention indexer) — NOT YET HIT**
- `fp8_fp4_mqa_logits` and `fp8_fp4_paged_mqa_logits` from DeepGEMM have no SM12x support
- **Fix needed:** SM12x fallback (torch.einsum or Triton) — jasl/vllm provides this

**Blocker #5: DeepGEMM MegaMoE (FP4 grouped GEMM) — NOT YET HIT** (renumbered from #5)
- `_grouped_fp4_impl` from DeepGEMM for fused expert computation
- The model currently selects MARLIN as the MoE backend, which may or may not work
- **Note:** The log shows `Using 'MARLIN' Mxfp4 MoE backend` — this is interesting. If MARLIN actually works for FP4 MoE on sm_121, this might not be a blocker at all.

**Blocker #6: Triton sparse MLA (decode attention) — NOT YET HIT** (renumbered from #6)
- FlashMLA sparse uses DeepGEMM under the hood
- **Fix needed:** Portable Triton sparse MLA decode path — jasl/vllm provides this

### Important observation: MARLIN MoE backend selected

The log shows:
```
Using 'MARLIN' Mxfp4 MoE backend.
```

This is notable because it means v0.21.0's MoE weight loading path chose MARLIN over DeepGEMM for the FP4 expert weights. If MARLIN actually has sm_121 support for MXFP4 (NVFP4) GEMMs, this could mean the MoE expert path works without DeepGEMM — we just haven't gotten far enough to test it. This aligns with the `TORCH_CUDA_ARCH_LIST="12.0 12.1"` requirement: arch 12.0 compiles NVFP4/MXFP4 kernels which MARLIN may use.

### What this tells us about the porting strategy

The crash happens at **layer 0, step 1** (mHC pre-norm). We don't even get to test attention or MoE. The porting priority is:

1. **mHC fallback** — Must fix first. Without this, the model cannot take a single forward step.
2. **DeepGEMM MQA logits fallback** — Needed for the sparse attention indexer.
3. **DeepGEMM FP8 einsum fallback** — Needed for c4a/c128a compressed attention.
4. **Triton sparse MLA** — Needed for decode-time attention.
5. **MoE path** — May already work via MARLIN; needs testing once blockers 1-4 are cleared.

This confirms that cherry-picking from jasl/vllm is the right approach. The question is how to do it cleanly onto v0.21.0.

## tilelang on sm_121 — verified working (2025-05-24)

tilelang 0.1.9 is installed in the vLLM venv and **works on sm_121**. Tested the actual `mhc_pre_big_fuse_tilelang` kernel from `vllm/model_executor/layers/mhc.py` with real tensor shapes (hidden_size=7168, hc_mult=4) and it:

1. **JIT-compiled successfully** on sm_121 (took ~7 seconds for first compilation)
2. **Ran without errors** — no architecture assertion failures
3. **Produced numerically correct results** — Sinkhorn-normalized doubly-stochastic matrices with row/col sums ≈ 0.99 (the epsilon offset from 1.0 is expected)

This means the **post-GEMM fusion step** of mHC (`mhc_pre_big_fuse_tilelang`) already works on sm_121. Only the **GEMM step** (`tf32_hc_prenorm_gemm`) needs a fallback.

Also confirmed: TVM (which tilelang uses under the hood) correctly targets `cuda -arch=sm_121`.

## tf32_hc_prenorm_gemm SM12x fallback — jasl/vllm approach (2025-05-24)

jasl/vllm's `ds4-sm120` branch adds SM12x fallbacks in `vllm/utils/deep_gemm.py`. The approach is clean:

### Architecture check at dispatch

```python
def tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_split):
    if current_platform.is_device_capability_family(120):
        return _tf32_hc_prenorm_gemm_sm12x(x, fn, out, sqrsum, num_split)
    _lazy_init()  # DeepGEMM path for sm_90/sm_100
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(x, fn, out, sqrsum, num_split)
```

### Two-level SM12x fallback

**Primary: Triton kernel** (`_tf32_hc_prenorm_gemm_sm12x`)
- If `out.dim() == 3 and sqrsum.dim() == 2`, uses `tf32_hc_prenorm_gemm_triton` from `deepseek_v4_triton_kernels.py`
- The Triton kernel does split-K GEMM with `tl.dot(x, fn, input_precision="tf32")`, accumulating partial products across splits
- This matches DeepGEMM's split-K output format — downstream `mhc_pre_big_fuse_tilelang` sums across the split dimension

**Fallback: Pure PyTorch** (`_tf32_hc_prenorm_gemm_torch`)
- If the Triton kernel is unavailable or tensor dims don't match, falls back to pure `torch.matmul`
- Computes `product = x.float() @ fn.float().T` and `norm = x.float().square().sum(dim=-1)`
- Writes full result to split-0, zeros other splits — downstream fusion sums across splits and gets the correct result
- Simpler but slower than the Triton kernel

### The Triton kernel (from `deepseek_v4_triton_kernels.py`)

The kernel `_tf32_hc_prenorm_gemm_kernel` is a straightforward split-K matmul:
- Each program handles one (M-block, N-block, split) tile
- Accumulates `acc += tl.dot(x, fn, input_precision="tf32")` and `sq += tl.sum(x * x, axis=1)` over K within the split range
- Block sizes: `BLOCK_M=16`, `BLOCK_N=power_of_2(N)` (clamped 16-32), `BLOCK_K=64`, `num_warps=4`
- Very portable — uses only `tl.dot` with tf32 precision, no arch-specific instructions

### What this means for our port

The `tf32_hc_prenorm_gemm` fix is **self-contained and small**:
1. Add the SM12x dispatch check to `vllm/utils/deep_gemm.py`
2. Add `_tf32_hc_prenorm_gemm_torch` (7 lines of logic)
3. Add `_tf32_hc_prenorm_gemm_sm12x` (10 lines)
4. Add the Triton kernel `tf32_hc_prenorm_gemm_triton` to a new or existing Triton kernels file

This is a good first PR — small, well-understood, and it unblocks the very first forward pass.

The `mhc_pre_big_fuse_tilelang` kernel (the fusion step) already works on sm_121, so **no changes needed there**.

## Current run script

`model_scripts/run-deepseek-v4-flash.sh` — currently has `--enforce-eager` and `VLLM_DISABLED_KERNELS` is NOT set (good). DeepGEMM install script is at `model_scripts/deepseek-v4-flash-prereq/deepgemm-install.sh`.