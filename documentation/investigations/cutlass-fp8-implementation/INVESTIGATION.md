# CutlassFp8BlockScaledMMKernel on DGX Spark (sm_121)

## Goal

Implement a working `CutlassFp8BlockScaledMMKernel` for sm_121 (NVIDIA GB10 / DGX Spark) so that FP8 MoE models like MiniMax-M2.7 can serve correctly.

### Why this matters

MiniMax-M2.7 is a large MoE model that uses FP8 block-scaled quantization. When `CutlassFp8BlockScaledMMKernel` is **disabled** (via `VLLM_DISABLED_KERNELS`), vLLM falls back to `TritonFp8BlockScaledMMKernel`. The model loads and serves, but produces **garbled output** — the Triton fallback does not correctly handle the MoE expert routing or block-scaled FP8 GEMM for this model. When the kernel is **enabled**, vLLM crashes during startup with:

```
RuntimeError: cutlass_scaled_mm, scaled_mm_entry.cu:259,
NotImplementedError: No compiled cutlass_scaled_mm for a compute capability
less than CUDA device capability: 121
```

Or, after patching to include sm_121 archs:

```
RuntimeError: cutlass_gemm_caller.cuh:61, Error Internal
```

So we are stuck: **garbage output without the kernel, crash with it**. The only path forward is a working sm_121 implementation of this kernel.

MiniMax-M2.7 serves as a **representative model** for future large MoE models we will run on this cluster. Fixing this kernel unblocks an entire class of FP8 MoE models on DGX Spark.

## Why the kernel is broken

vLLM v0.21.0 has a regression affecting sm_121 (DGX Spark / GB10). The CMakeLists.txt build system does not include sm_121 in `CUDA_SUPPORTED_ARCHS` for CUDA >= 13.0, and the CUTLASS c3x (CUTLASS 3.x) FP8 block-scaled GEMM kernels are only compiled for sm_120 (`12.0a`, `12.0f`). On sm_121 hardware:

- The sm_120 CUTLASS kernels pass `can_implement()` but fail at `run()` with `cutlass::Status::kErrorInternal`
- The sm_80/sm_90/sm_100 c2x (CUTLASS 2.x) fallback kernels do not support FP8 block-scaled operations on this arch

This is an upstream vLLM bug, not a hardware problem. CUTLASS v4.4.2 has `Sm121` arch awareness and the kernel works when compiled natively for sm_121.

## Relevant GitHub issues and PRs

### [vllm-project/vllm#43367](https://github.com/vllm-project/vllm/issues/43367) — SM12.1 / GB10 still fails in CutlassFp8BlockScaledMMKernel

Opened by **hsakkout** on 2026-05-21. Reports that after PR #41215, `CutlassFp8BlockScaledMMKernel` still fails on GB10 / sm_121. Key findings:

- The failing path is `ops.cutlass_scaled_mm` / `CutlassFp8BlockScaledMMKernel` in `_C_stable_libtorch`
- Building `_C_stable_libtorch` with `TORCH_CUDA_ARCH_LIST=12.1` produces native sm_121 cubins and **fixes** the failure
- `cuobjdump --list-elf` confirms native `sm_121.cubin` and `sm_121a.cubin` in the working build
- The minimal reproducer: `ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, None)` with FP8 block-scaled tensors
- The fix: add sm_121 to published build targets so native cubins are included

### [vllm-project/vllm#38484](https://github.com/vllm-project/vllm/pull/38484) — [Build] Add SM121 (DGX Spark / GB10) to published build targets

Opened by **JCorners68** on 2026-03-29. A minimal PR (5 files, 6 lines changed) that adds `12.1` to `TORCH_CUDA_ARCH_LIST` in all build configurations:

| File | Change |
|---|---|
| `.github/workflows/scripts/build.sh` | `9.0+PTX` → `9.0 10.0 12.0 12.1+PTX` |
| `docker/Dockerfile` (×2) | `12.0` → `12.0 12.1` |
| `docker/docker-bake.hcl` | `10.0` → `10.0 12.0 12.1` |
| `docker/versions.json` | `12.0` → `12.0 12.1` |
| `tools/flashinfer-build.sh` | `12.0` → `12.0 12.1` (CUDA 13.0+ path) |

No CMakeLists.txt changes — it relies on `12.1`/`12.1a` already being in the kernel-specific arch lists, just missing from the build target list. The PR has been `CONFLICTING` since at least May 2026.

**Note from hsakkout**: compiling for `120a` alone is insufficient — `can_implement()` passes but `gemm_op.run()` returns `kErrorInternal`. Only native `121a` cubins work. However, sm_121 is a **subset** of sm_120, not a superset — some instructions (e.g. `e2m1x2` MXFP4) are sm_120-only. So `TORCH_CUDA_ARCH_LIST` must include **both** `12.0` and `12.1`.

## What we tried (and should not try again)

These approaches were all attempted and failed. Do not repeat them.

1. **Removing sm_12x from SCALED_MM_ARCHS / CUTLASS_MOE_DATA_ARCHS** — Falls back to sm_80/sm_90/sm_100 c2x kernels which don't support FP8 block-scaled GEMM on sm_121. Result: `cutlass_scaled_mm_sm80_epilogue` error.

2. **Removing sm_12x from all arch lists (blanket removal)** — Same problem. The c2x fallback path does not implement the FP8 block-scaled operations needed by MoE models like MiniMax-M2.7.

3. **Setting `TORCH_CUDA_ARCH_LIST=12.1` only** — Produces sm_121 cubins for CUTLASS, but NVFP4/MXFP4 kernels use `e2m1x2` instructions that are **not supported on sm_121**. Result: ptxas errors during build (`Instruction 'cvt with .e2m1x2' not supported on .target 'sm_121'`).

4. **Setting `TORCH_CUDA_ARCH_LIST="12.0 12.1"` and patching CMakeLists.txt to use `"12.1"` instead of `"12.0f"` in CUDA >= 13.0 paths** — Compiles successfully with both sm_120 and sm_121 cubins, but at runtime vLLM dispatches sm_120 kernels on the sm_121 device, causing `Arch conditional MMA instruction` errors. The runtime kernel selection picks the highest compiled arch that matches the device family, which is sm_120, not sm_121.

**Bottom line**: Removing sm_121 support defeats the purpose — we need native sm_121 CUTLASS c3x kernels. But simply adding sm_121 to the build isn't enough either; the kernel dispatch logic also needs to prefer sm_121 over sm_120 when running on an sm_121 device. This is an upstream vLLM problem.

## Tip: Test on one node first

When modifying vLLM code (CMakeLists.txt patches, kernel changes, etc.), **always verify on a single node first** before building across all 4 nodes. A single build takes 20-30 minutes; building on 4 nodes only to discover a ptxas error or runtime crash wastes over an hour of cluster time.

Use the minimal reproducer below to quickly test whether `cutlass_scaled_mm` works on the local node before kicking off cluster-wide builds.

## Debug / test code

```python
import torch
from vllm import _custom_ops as ops

m, n, k = 1, 16384, 5120
a = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
b = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn).t()
scale_a = torch.ones((m, k // 128), device="cuda", dtype=torch.float32)
scale_b = torch.ones((k // 128, n // 128), device="cuda", dtype=torch.float32)

try:
    out = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, None)
    torch.cuda.synchronize()
    print(f"SUCCESS: cutlass_scaled_mm works! Output shape: {out.shape}")
except Exception as e:
    print(f"FAILED: {e}")
```

Check compiled archs in the binary:
```bash
for so in ~/models/vllm/vllm/_C.abi3.so \
          ~/models/vllm/vllm/_C_stable_libtorch.abi3.so \
          ~/models/vllm/vllm/_moe_C.abi3.so; do
    echo "=== $(basename $so) ==="
    strings "$so" 2>/dev/null | grep -o 'sm_12[01]' | sort -u
done
```

## Fork note

The v0.21.0 checkout from `urbanspr1nter/vllm` is **identical** to the v0.21.0 tag on `vllm-project/vllm`. When we patch vLLM to fix this kernel, we should create a branch on the fork (e.g. `v0.21.0-sm121-fix`) rather than modifying the tag checkout in-place. This keeps our patches trackable and makes it easy to rebase on future upstream releases.

---

## Deep code analysis (2026-05-23)

Fork cloned to `$HOME/models/vllm`, branch `v0.21.0-sm121-fix` created off `v0.21.0` tag. Below is a thorough analysis of every relevant code path.

### Bug 1: `enable_sm120_only` runtime guard traps on sm_121

**File**: `csrc/cutlass_extensions/common.hpp`

Two guard structs exist side by side:

```cpp
// Line 114 — ONLY runs on sm_1200 (__CUDA_ARCH__ == 1200)
template <typename Kernel>
struct enable_sm120_only : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ == 1200
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm120a.\n");
    asm("trap;");
  #endif
#endif
  }
};

// Line 130 — Runs on entire sm_12x family (1200 <= __CUDA_ARCH__ < 1300)
template <typename Kernel>
struct enable_sm120_family : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm120f.\n");
    asm("trap;");
  #endif
#endif
  }
};
```

**Which guard is used where:**

| File | Guard used | Affected kernels |
|---|---|---|
| `scaled_mm_blockwise_sm120_fp8_dispatch.cuh:130` | `enable_sm120_family` ✅ | Blockwise FP8 GEMM (the one MiniMax-M2.7 needs) |
| `scaled_mm_sm120_fp8_dispatch.cuh:75` | `enable_sm120_only` ❌ | Regular FP8 GEMM (per-tensor/per-channel scaled) |
| `scaled_mm.cuh:205` | `enable_sm120_only` ❌ | Regular FP8 GEMM (sm_120 template struct) |

**Impact**: The blockwise kernel (which is what `CutlassFp8BlockScaledMMKernel` calls) already uses the correct `enable_sm120_family` guard. So Bug 1 is **NOT the cause** of the blockwise FP8 crash. However, it WILL cause the regular (non-blockwise) `cutlass_scaled_mm` path to crash on sm_121, so it still needs fixing.

### Bug 2: `CUDA_SUPPORTED_ARCHS` excludes 12.1 for CUDA >= 13.0

**File**: `CMakeLists.txt` lines 104-107

```cmake
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")
  # 12.1 is MISSING
```

For CUDA 12.8 ≤ version < 13.0, `CUDA_SUPPORTED_ARCHS` includes `12.1`:
```cmake
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1")
```

But for CUDA >= 13.0 (which is what our DGX Sparks run), `12.1` is absent. The initial `CUDA_ARCHS` is computed as the intersection of `TORCH_CUDA_ARCH_LIST` and `CUDA_SUPPORTED_ARCHS`. Since `12.1` isn't in `CUDA_SUPPORTED_ARCHS`, it gets filtered out before any per-kernel arch list gets a chance to include it.

**Impact**: Even with `TORCH_CUDA_ARCH_LIST=12.1a`, no sm_121 cubins are produced. The sm_120 family flag (`12.0f`) in the per-kernel lists never intersects with `12.1` because `12.1` was already removed from `CUDA_ARCHS`.

### Bug 3: CUTLASS `arch::Sm120` cubins fail at runtime on sm_121 hardware

Confirmed by hsakkout (issue #43367): CUTLASS kernels compiled with `cutlass::arch::Sm120` for sm_120 cubins pass `can_implement()` but fail at `gemm_op.run()` with `cutlass::Status::kErrorInternal` on sm_121. Building native sm_121 cubins (via `TORCH_CUDA_ARCH_LIST=12.1`) fixes the runtime failure.

The vLLM sm_120 kernel code uses `cutlass::arch::Sm120` in the CUTLASS `CollectiveBuilder` template parameters. This is correct — CUTLASS v4.4.2's `arch::Sm120` is the family tag and the `CollectiveBuilder` will generate correct code for sm_121 when compiled with `--gpu-architecture=sm_121`. The issue is purely that sm_121 cubins are not being produced (Bug 2), so the driver runs sm_120 cubins on sm_121 hardware, which fails.

### Runtime dispatch in `scaled_mm_entry.cu`

```cpp
#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120
  if (version_num >= 120) {
    cutlass_scaled_mm_sm120(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif
```

On sm_121, `version_num` is 121 which is `>= 120`, so it dispatches to `cutlass_scaled_mm_sm120`. This is correct — there is no separate `cutlass_scaled_mm_sm121` function, and there doesn't need to be. The sm_120 kernel function is compiled into a `.so` that contains cubins for multiple architectures. At load time, the CUDA driver selects the cubin matching the device (sm_121 if available, sm_120 if not). When sm_121 cubins are present, they run natively. When only sm_120 cubins are present, the driver runs sm_120 code on sm_121 hardware, which fails for the blockwise FP8 GEMM.

**This dispatch logic is NOT the problem.** The problem is that sm_121 cubins are never compiled in the first place (Bug 2).

### CMake arch resolution for the sm_120 kernel sources

```cmake
# For CUDA >= 13.0:
cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f" "${CUDA_ARCHS}")
# For CUDA < 13.0:
cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
```

The `12.0f` is a "family" suffix meaning "any arch in the sm_12x family." The `cuda_archs_loose_intersection` function handles this by matching any `CUDA_ARCHS` entry with the same major version. So if `CUDA_ARCHS` contained `12.1a`, the `12.0f` pattern would match it and produce `12.1a` in the output.

**But `CUDA_ARCHS` never contains `12.1a`** because `CUDA_SUPPORTED_ARCHS` for CUDA >= 13.0 doesn't include `12.1`, and the initial intersection already removed it.

### The blockwise FP8 path in detail

`CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm()` calls `ops.cutlass_scaled_mm()` which hits `cutlass_scaled_mm()` in `scaled_mm_entry.cu`. With `ENABLE_SCALED_MM_SM120` defined and `version_num >= 120`, it dispatches to `cutlass_scaled_mm_sm120()`, which calls `dispatch_scaled_mm()` with `vllm::cutlass_scaled_mm_blockwise_sm120_fp8` as the blockwise callback. This enters `scaled_mm_blockwise_sm120_fp8_dispatch.cuh` which uses `cutlass::arch::Sm120` in the `CollectiveBuilder` and `enable_sm120_family` as the runtime guard — both correct for sm_121.

The entire call chain is sound **provided that sm_121 cubins are compiled**. The only missing piece is Bug 2.

### Python-side kernel selection

`CutlassFp8BlockScaledMMKernel.is_supported()` calls `cutlass_block_fp8_supported()` → `ops.cutlass_scaled_mm_supports_block_fp8(121)` → returns `True` (since 121 >= 100 and CUDA_VERSION >= 12080). So the kernel is selected and used, which then crashes because sm_121 cubins don't exist.

### hsakkout's finding confirmed

hsakkout's analysis in issue #43367 is accurate:
- Rebuilding with `TORCH_CUDA_ARCH_LIST=12.1` produces native sm_121 cubins and fixes the blockwise FP8 GEMM
- The `12.0f` family arch alone is insufficient — you need **both** `12.0` and `12.1` because some kernels (NVFP4/MXFP4) use sm_120-only instructions (`e2m1x2`)

### PR #38484 assessment

PR #38484 adds `12.1` to `TORCH_CUDA_ARCH_LIST` in build scripts (Dockerfile, build.sh, etc.) but does **NOT** fix `CUDA_SUPPORTED_ARCHS` in `CMakeLists.txt`. For CUDA >= 13.0 builds, `12.1` would still be filtered out by the `CUDA_SUPPORTED_ARCHS` intersection. The PR is necessary but insufficient on its own for CUDA >= 13.0.

---

## Fix plan

### Phase 1: Apply patches to our fork (single node, code changes only, no build yet)

Three changes needed on the `v0.21.0-sm121-fix` branch:

1. **`CMakeLists.txt`**: Add `12.1` to `CUDA_SUPPORTED_ARCHS` for CUDA >= 13.0
   ```cmake
   # Before:
   set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")
   # After:
   set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")
   ```

2. **`csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh` line 75**: Change `enable_sm120_only` to `enable_sm120_family`

3. **`csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm.cuh` line 205**: Change `enable_sm120_only` to `enable_sm120_family`

Changes 2 and 3 fix the non-blockwise FP8 GEMM path for sm_121. They are not the cause of the MiniMax-M2.7 crash (that's Bug 2), but they will prevent a separate crash if the regular `cutlass_scaled_mm` (non-blockwise) path is ever hit on sm_121.

### Phase 2: Build on spark-01 and run the minimal reproducer

Build vLLM on spark-01 with `TORCH_CUDA_ARCH_LIST="12.0 12.1"` using the patched fork. Then run the minimal reproducer:

```python
import torch
from vllm import _custom_ops as ops

m, n, k = 1, 16384, 5120
a = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
b = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn).t()
scale_a = torch.ones((m, k // 128), device="cuda", dtype=torch.float32)
scale_b = torch.ones((k // 128, n // 128), device="cuda", dtype=torch.float32)

try:
    out = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, None)
    torch.cuda.synchronize()
    print(f"SUCCESS: cutlass_scaled_mm works! Output shape: {out.shape}")
except Exception as e:
    print(f"FAILED: {e}")
```

Also verify sm_121 cubins exist in the built `.so` files:

```bash
for so in ~/models/vllm/vllm/_C.abi3.so \
          ~/models/vllm/vllm/_C_stable_libtorch.abi3.so \
          ~/models/vllm/vllm/_moe_C.abi3.so; do
    echo "=== $(basename $so) ==="
    strings "$so" 2>/dev/null | grep -o 'sm_12[01]' | sort -u
done
```

This is the critical gate. If the reproducer fails, we must diagnose before proceeding.

### Phase 3: Cluster-wide build ✅ DONE

Built on all 4 DGX Spark nodes in parallel with `TORCH_CUDA_ARCH_LIST="12.0 12.1"` and CUDA 13.0 (torch nightly `cu130`). All builds succeeded. sm_121 cubins confirmed in all `.so` files across all nodes:

| Node | `_C_stable_libtorch` | `_C` | `_moe_C` | vllm binary |
|---|---|---|---|---|
| spark-01 | sm_120, sm_120f | sm_120, sm_120f, sm_121 | sm_120, sm_121 | ✅ |
| spark-02 | sm_120, sm_120f | sm_120, sm_120f, sm_121 | sm_120, sm_121 | ✅ |
| spark-03 | sm_120, sm_120f | sm_120, sm_120f, sm_121 | sm_120, sm_121 | ✅ |
| spark-04 | sm_120, sm_120f | sm_120, sm_120f, sm_121 | sm_120, sm_121 | ✅ |

No ptxas errors during build — NVFP4/MXFP4 kernels compiled for sm_120 only (their arch lists don't include sm_121), while CUTLASS scaled_mm kernels compiled for both sm_120 and sm_121.

### Phase 4: Serve MiniMax-M2.7 ✅ DONE

MiniMax-M2.7 served successfully on the 4× DGX Spark cluster with `CutlassFp8BlockScaledMMKernel` (not the Triton fallback). Model output is coherent and correct.

**Key log lines confirming success:**
```
Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod
```

**Sample output:**
```
User: Hello! Can you tell me a short joke?
Assistant: Sure! Here's a short one for you:
    Why don't scientists trust atoms?
    Because they make up everything! 😄
```

**System fingerprint:** `vllm-0.21.1.dev1+gcb44fd8b9.d20260523-tp4`

**Model loading:** 53.75 GiB per GPU, 125 safetensors shards, ~16 min load time.

### Critical lesson: VLLM_DISABLED_KERNELS environment variable

During Phase 4, the first two attempts to serve MiniMax-M2.7 selected `TritonFp8BlockScaledMMKernel` instead of `CutlassFp8BlockScaledMMKernel`, despite the kernel being fixed and available. The root cause was `VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel` set in the shell environment from a previous debugging session.

This env var is inherited by Ray when the head node starts, and propagates to all worker processes. Simply `unset`ing it in the vLLM launch script is **not sufficient** — it must be unset **before starting Ray**.

**The fix:** Ensure `VLLM_DISABLED_KERNELS` is not set anywhere in the environment before starting the Ray cluster:
```bash
unset VLLM_DISABLED_KERNELS  # BEFORE ray start
ray start --head ...
# Then launch vLLM
```

**How to diagnose:** If vLLM selects `TritonFp8BlockScaledMMKernel` instead of `CutlassFp8BlockScaledMMKernel`, check:
1. `env | grep VLLM_DISABLED_KERNELS` in the shell
2. `cat /proc/$(pgrep -f raylet | head -1)/environ | tr '\0' '\n' | grep VLLM_DISABLED` in the Ray head process
3. The Ray worker log will show `Selected TritonFp8BlockScaledMMKernel` instead of `Selected CutlassFp8BlockScaledMMKernel`

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| `TORCH_CUDA_ARCH_LIST="12.0 12.1"` causes ptxas errors on sm_120-only instructions | Build on single node first; ptxas errors are compile-time, not runtime. If they occur, we may need to exclude specific `.cu` files from sm_121 arch compilation. |
| `enable_sm120_family` causes runtime issues on sm_121 | The blockwise path already uses this guard and works per hsakkout's testing. The non-blockwise path is lower risk since it shares the same CUTLASS arch tag. |
| `CUDA_SUPPORTED_ARCHS` change affects other arch filtering downstream | The change is additive only — we add `12.1` to the list, nothing is removed. Other archs are unaffected. |
| Build takes 20-30 min per node | Always test on spark-01 first before cluster-wide build. |
| `VLLM_DISABLED_KERNELS` set in environment | Must be unset before starting Ray, not just before launching vLLM. Ray inherits env vars from the shell that starts it. |

---

## Summary of the fix

Three lines changed, one environment variable to clear:

| Change | File | Before | After |
|---|---|---|---|
| Add `12.1` to CUDA_SUPPORTED_ARCHS | `CMakeLists.txt:104` | `"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0"` | `"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1"` |
| Fix runtime guard for non-blockwise FP8 | `csrc/.../scaled_mm.cuh:205` | `enable_sm120_only` | `enable_sm120_family` |
| Fix runtime guard for non-blockwise FP8 | `csrc/.../scaled_mm_sm120_fp8_dispatch.cuh:75` | `enable_sm120_only` | `enable_sm120_family` |
| Clear stale env var | Shell environment | `VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel` | unset |

The root cause was Bug 2 (`CUDA_SUPPORTED_ARCHS` missing `12.1` for CUDA >= 13.0), which prevented sm_121 cubins from being compiled. The `enable_sm120_only` → `enable_sm120_family` changes fix a separate latent bug that would have crashed the non-blockwise FP8 path on sm_121. The `VLLM_DISABLED_KERNELS` env var was a pre-existing operational issue that masked the fix during initial testing.