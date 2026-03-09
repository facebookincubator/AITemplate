# Adding a CuTeDSL Backend to AITemplate: Execution Plan & Reusable Skill

This document captures the end-to-end process of converting an AITemplate (AIT)
operator from CUTLASS C++ template-based code generation to CuTeDSL (CUTLASS
Python DSL) with AOT compilation. It is generalized as a reusable pattern
applicable to any AIT operator.

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        CUTLASS C++ Backend (Original)                        │
│                                                                              │
│  Python (AIT codegen)          C++ (generated)               Binary          │
│  ┌─────────────────┐      ┌──────────────────┐        ┌──────────────────┐   │
│  │ Jinja2 template  │─────▶│ <op>.cu           │─nvcc──▶│ <op>.obj         │   │
│  │ (FUNC_TEMPLATE)  │      │ (CUTLASS C++      │        │                  │   │
│  │                  │      │  template code)   │        │                  │   │
│  └─────────────────┘      └──────────────────┘        └───────┬──────────┘   │
│                                                               │ link         │
│                                                        ┌──────▼──────────┐   │
│                                                        │ test.so          │   │
│                                                        └─────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                        CuTeDSL Backend (New)                                 │
│                                                                              │
│  Python (AIT codegen + CuTeDSL AOT)        C++ (thin wrapper)   Binary      │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  ┌────────────┐   │
│  │ CuTeDSL kernel   │  │ cute.compile()│  │<op>.cu        │  │<op>.obj    │   │
│  │ (@cute.jit +     │──▶│ export_to_c()│──▶│(struct init + │──▶│(from nvcc) │   │
│  │  @cute.kernel)   │  │              │  │ metadata load)│  │            │   │
│  └─────────────────┘  └──────┬───────┘  └───────────────┘  └────┬───────┘   │
│                              │                                    │ link     │
│                        ┌─────▼────────┐                    ┌─────▼───────┐   │
│                        │<op>_cutedsl.h │                    │ test.so      │   │
│                        │<op>_cutedsl.o │────────────────────▶│ (+libcuda)   │   │
│                        │(pre-compiled) │                    └─────────────┘   │
│                        └──────────────┘                                       │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Key Difference

- **CUTLASS C++ backend**: Jinja2 template IS the kernel. It generates C++ source
  that instantiates CUTLASS template classes. `nvcc` compiles everything.
- **CuTeDSL backend**: Kernel is written in Python using CuTeDSL decorators
  (`@cute.jit`, `@cute.kernel`). `cute.compile()` produces MLIR+cubin at
  codegen time, `export_to_c()` emits `.h`+`.o`. Only a thin C++ wrapper
  needs nvcc compilation.

---

## Step-by-Step Integration Guide

### Step 1: Write the CuTeDSL Kernel

**Create**: `backend/cuda/<op_dir>/cutedsl_<op>_sm80.py`

This file contains the pure CuTeDSL kernel implementation:

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp

class MyOpSm80Kernel:
    def __init__(self, ...compile_time_params...):
        self.param = param
        self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=128)

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, ..., stream: cuda.CUstream):
        """Host-side: create SMEM layouts, copy atoms, MMA config, launch kernel."""
        # 1. Compute swizzled SMEM layouts
        sA_layout = cute.tile_to_shape(get_smem_layout_atom(...), ...)
        # 2. Create copy atoms (cpasync for GMEM->SMEM, universal for stores)
        gmem_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cpasync.CopyG2SOp(), ...), ...)
        # 3. Create tiled MMA
        tiled_mma = cute.make_tiled_mma(warp.MmaF16BF16Op(...), ...)
        # 4. Compute grid dimensions
        grid_dim = (m_blocks, batch_size, 1)
        # 5. Launch kernel
        self.kernel(...).launch(grid=grid_dim, block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, ...all_params...):
        """Device-side: actual CUDA kernel body."""
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        # ... GEMM mainloop, epilogue, store ...
```

**Key patterns**:

1. **Swizzled SMEM layouts** (for bank-conflict-free access):
   ```python
   def get_smem_layout_atom(dtype, k_dim):
       dtype_byte = cutlass.const_expr(dtype.width // 8)
       bytes_per_row = cutlass.const_expr(k_dim * dtype_byte)
       smem_k_block_size = (128 if bytes_per_row % 128 == 0 else ...) // dtype_byte
       swizzle_bits = 4 if smem_k_block_size == 128 else ...
       return cute.make_composed_layout(
           cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base), 0,
           cute.make_ordered_layout((8, smem_k_block_size), order=(1, 0)),
       )
   ```

2. **SM80 warp-level MMA** (`warp.MmaF16BF16Op`):
   ```python
   tiled_mma = cute.make_tiled_mma(
       warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)),
       (num_warps, 1, 1),
       permutation_mnk=(num_warps * 16, 16, 16),
   )
   ```

3. **cp.async for GMEM->SMEM**:
   ```python
   atom_async = cute.make_copy_atom(
       cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
       dtype, num_bits_per_copy=128)
   gmem_tiled_copy = cute.make_tiled_copy_tv(atom_async, thread_layout, value_layout)
   ```

4. **SMEM->register via ldmatrix**:
   ```python
   smem_copy_atom = cute.make_copy_atom(
       warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), dtype)
   smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma)
   ```

5. **Activation dispatch table**:
   ```python
   ACTIVATION_MAP = {
       "Sigmoid": lambda x: 1.0 / (1.0 + cute.math.exp(-x)),
       "ReLu":    lambda x: cute.arch.fmax(x, 0.0),
       "Identity": lambda x: x,
   }
   ```

### Step 2: Create the Backend Codegen Module

**Create**: `backend/cuda/<op_dir>/<op>_cutedsl.py`

This file registers AIT backend functions with `_cutedsl` suffix:

```python
import functools, os, jinja2
from aitemplate.backend import registry
from aitemplate.backend.target import Target

# C++ wrapper template - constructs CuTeDSL typed tensor structs from AIT void* pointers
CUTEDSL_WRAPPER_TEMPLATE = jinja2.Template("""
#include <cuda.h>
#include "{{cutedsl_header}}"

static {{func_name}}_cutedsl_Kernel_Metadata_t g_metadata;
static bool g_metadata_loaded = false;

static void ensure_metadata_loaded() {
    if (!g_metadata_loaded) {
        cuInit(0);
        {{func_name}}_cutedsl_Kernel_Metadata_Load(&g_metadata);
        g_metadata_loaded = true;
    }
}

void {{func_name}}(...AIT signature...) {
    ensure_metadata_loaded();

    // Construct CuTeDSL typed tensor structs:
    {{func_name}}_cutedsl_Tensor_mA_t t_mA;
    t_mA.data = a_ptr;
    t_mA.dynamic_shapes[0] = batch;       // first dynamic dim
    t_mA.dynamic_shapes[1] = seq_len;     // second dynamic dim
    t_mA.dynamic_strides[0] = seq_len * K; // batch stride

    // ... same for all tensors ...

    CUstream cu_stream = reinterpret_cast<CUstream>(stream);
    {{func_name}}_cutedsl_wrapper(&g_metadata, &t_mA, ..., cu_stream);
}
""")

@functools.lru_cache(maxsize=32)
def _aot_compile_cutedsl_kernel(n0, n1, ..., output_dir, func_name, arch):
    """AOT compile kernel and export .h + .o."""
    from my_kernel import MySm80Kernel
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.export import export_to_c

    kernel = MySm80Kernel(n0=n0, n1=n1, ...)

    # Create representative tensors (shapes don't matter for dynamic dims)
    q_pt = torch.zeros(4, 128, n1, device="cuda", dtype=torch.float16)

    # Mark dynamic modes with per-mode divisibility
    q_cute = from_dlpack(q_pt, assumed_align=16)
    q_cute = q_cute.mark_compact_shape_dynamic(
        mode=0, stride_order=q_pt.dim_order(), divisibility=1)
    q_cute = q_cute.mark_compact_shape_dynamic(
        mode=1, stride_order=q_pt.dim_order(), divisibility=8)

    # Compile and export
    compiled = cute.compile(kernel, q_cute, ..., cu_stream)
    export_to_c(compiled, file_path=output_dir, file_name=f"{func_name}_cutedsl")
    return h_path, o_path

@registry.reg("cuda.<op>.gen_function_cutedsl")
def gen_function_cutedsl(func_attrs):
    h_path, o_path = _aot_compile_cutedsl_kernel(...)
    func_attrs["cutedsl_obj_path"] = o_path  # For builder to pick up
    return CUTEDSL_WRAPPER_TEMPLATE.render(...)

@registry.reg("cuda.<op>.func_decl_cutedsl")
def func_decl_cutedsl(func_attrs): ...

@registry.reg("cuda.<op>.func_call_cutedsl")
def func_call_cutedsl(func_attrs, indent="  "): ...
```

**Critical detail**: The C++ wrapper MUST construct CuTeDSL typed tensor structs
(`*_Tensor_mX_t`) with proper `dynamic_shapes[]` and `dynamic_strides[]` fields.
The `export_to_c()` API generates these structs -- they are NOT raw `void*`.

### Step 3: Add Backend Dispatch in the Op Class

**Modify**: `compiler/ops/<op_dir>/<op>.py`

Add methods to detect CuTeDSL mode and dispatch with suffix:

```python
def _use_cutedsl(self) -> bool:
    current_target = target.Target.current()
    return current_target._kwargs.get("use_cutedsl_<op>", False)

def _backend_suffix(self) -> str:
    return "_cutedsl" if self._use_cutedsl() else ""

def gen_function(self) -> str:
    suffix = self._backend_suffix()
    self._attrs["backend_suffix"] = suffix  # For codegen.py to pick up
    func_key = f"{target_name}.{op_name}.gen_function{suffix}"
    return registry.get(func_key)(self._attrs)

def gen_function_decl(self, func_attrs=None) -> str:
    suffix = self._backend_suffix()
    func_key = f"{target_name}.{op_name}.func_decl{suffix}"
    return registry.get(func_key)(self._attrs)

def gen_function_call(self, func_attrs=None, indent="  ") -> str:
    suffix = self._backend_suffix()
    func_key = f"{target_name}.{op_name}.func_call{suffix}"
    return registry.get(func_key)(self._attrs, indent)
```

### Step 4: Modify the Build System

#### 4.1 `backend/codegen.py` -- `gen_function_src()`

Pass `workdir` to `func_attrs` so the AOT compiler knows where to write:
```python
func._attrs["workdir"] = prefix
```

After writing the `.cu` wrapper, add the pre-compiled `.o` to `file_pairs`:
```python
cutedsl_obj = func._attrs.get("cutedsl_obj_path")
if cutedsl_obj and os.path.exists(cutedsl_obj):
    file_pairs.append((cutedsl_obj, cutedsl_obj))
```

In `_process_src_ops()`, read `backend_suffix` for `func_decl` / `func_call` dispatch:
```python
backend_suffix = func._attrs.get("backend_suffix", "")
f_func_decl = registry.get(f"{target}.{op}.func_decl{backend_suffix}")
f_func_call = registry.get(f"{target}.{op}.func_call{backend_suffix}")
```

#### 4.2 `backend/builder.py` -- Handle pre-compiled `.o` files

In `build_objs()`: skip compilation for `.o` source files:
```python
if src.endswith(".o"):
    _LOGGER.info(f"Skipping compilation for pre-compiled object: {src}")
    continue
```

In `build_so()`: add `-lcuda` when CuTeDSL objects are present:
```python
has_cutedsl = any(obj.endswith("_cutedsl.o") for obj in objs)
extra_libs = " -lcuda" if has_cutedsl else ""
```

In `gen_makefile()`: use `os.path.basename()` for `.o` paths (Make runs from workdir):
```python
if pair[0].endswith(".o"):
    obj_files.append(os.path.basename(pair[0]))  # NOT full path
    has_cutedsl = True
```

### Step 5: Update Target Configuration

**Modify**: `backend/target.py`

Add `use_cutedsl_<op>` kwarg support to the Target class (already passed through
`_kwargs`; no explicit changes needed unless you want validation).

### Step 6: Update `__init__.py` and Examples

**Modify**: `backend/cuda/<op_dir>/__init__.py`

Add the new module import for auto-registration:
```python
from aitemplate.backend.cuda.<op_dir> import <op>_cutedsl
```

**Modify**: Example/test scripts

Add CLI arg to enable CuTeDSL:
```python
target = Target(..., use_cutedsl_<op>=True)
```

### Step 7: Add BUCK Dependencies

Add `fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl` to deps
in `examples/BUCK` for any target that uses the CuTeDSL backend.

---

## Common Pitfalls and Debug Checklist

| # | Error | Root Cause | Fix |
|---|-------|------------|-----|
| 1 | `ModuleNotFoundError: No module named 'cuda'` | Missing `nvidia-cutlass-dsl` BUCK dep | Add `fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl` to deps |
| 2 | `ValueError: Invalid leading dimension: 2` | `LayoutEnum.from_tensor()` only accepts 2D tensors (leading_dim 0 or 1) | Hardcode `utils.LayoutEnum.ROW_MAJOR` for 3D row-major tensors |
| 3 | `_ScaledBasis TypeError` in SM90 TMA | CuTeDSL TMA layout algebra doesn't yet handle 3D tensors well | Use SM80 kernel (works on SM90+ hardware too) |
| 4 | `coord and shape of view are weakly congruent: N modes vs M coords` | After `partition_S(local_tile(..., (tile, None)))`, the tile mode creates an extra dimension | Add extra `None` to indexing: `tQgQ[None, None, None, k_tile]` instead of `tQgQ[None, None, k_tile]` |
| 5 | `ptr alignment does not meet requirement` for 128-bit cp.async | `mark_layout_dynamic()` loses alignment info | Use `mark_compact_shape_dynamic(mode=M, divisibility=8)` to preserve alignment |
| 6 | `cute.copy expects src and dst to have static shape` | `mark_layout_dynamic()` makes ALL dims dynamic, but inner dims must be static constants | Use per-mode `mark_compact_shape_dynamic()` only on truly dynamic modes (batch, seq_len); leave inner dims (N0, N1, K) static |
| 7 | `shape(X) of mode(0) is not divisible by divisibility(8)` | Batch dim size not divisible by the divisibility requirement | Use per-mode divisibility: batch `div=1`, M/seq_len `div=8` |
| 8 | `argument of type "void *" incompatible with "..._Tensor_mX_t *"` | CuTeDSL `export_to_c` generates typed tensor structs, not raw pointer APIs | Construct typed tensor structs in C++ wrapper: set `.data`, `.dynamic_shapes[]`, `.dynamic_strides[]` |
| 9 | `No rule to make target 'tmp/.../<op>_cutedsl.o'` | Makefile has full path but Make runs from workdir | Use `os.path.basename(pair[0])` in `gen_makefile()` |
| 10 | `DT_TEXTREL` linker warning | CuTeDSL `.o` contains text relocations | Non-fatal warning; can be ignored |

---

## Dynamic Tensor Marking Strategy

This is the most critical part for correctness. CuTeDSL needs to know which
tensor dimensions are compile-time constants vs runtime-dynamic.

### DO NOT use `mark_layout_dynamic()`

`mark_layout_dynamic(leading_dim=N)` makes ALL shapes and strides dynamic.
This breaks:
- `cute.copy` which requires static shapes for src/dst
- cp.async alignment checks (loses divisibility info)
- Optimal code generation (no compile-time constants)

### DO use `mark_compact_shape_dynamic()` per mode

```python
def make_cute_tensor(torch_tensor, dynamic_modes_div):
    """
    dynamic_modes_div: list of (mode_index, divisibility) tuples.
    Only the specified modes become dynamic; all other dims stay static.
    """
    ct = from_dlpack(torch_tensor, assumed_align=16)
    for mode, div in dynamic_modes_div:
        ct = ct.mark_compact_shape_dynamic(
            mode=mode,
            stride_order=torch_tensor.dim_order(),
            divisibility=div,
        )
    return ct
```

**Typical b2b_bmm pattern**:
| Tensor | Shape | Dynamic Modes |
|--------|-------|---------------|
| Q      | [batch, M, K] | batch (div=1), M (div=8) |
| K      | [batch, N0, K] | batch (div=1) only |
| V      | [batch, N0, N1] | batch (div=1) only |
| Bias   | [batch, M, N0] | batch (div=1), M (div=8) |
| Out    | [batch, M, N1] | batch (div=1), M (div=8) |

### Why div=8 for M?

128-bit cp.async copies 8 fp16 elements at once. The M dimension appears in
strides that feed cp.async, so CuTeDSL needs `divisibility=8` to guarantee
alignment for 128-bit vectorized copies.

---

## Generated Artifact Reference

After successful compilation, the workdir contains:

| File | Size (typical) | Description |
|------|----------------|-------------|
| `<op>_cutedsl.h` | ~4 KB | CuTeDSL header: typed tensor structs, `Metadata_Load/Unload`, `wrapper()` |
| `<op>_cutedsl.o` | ~50 KB | Pre-compiled object with embedded cubin(s) |
| `<op>.cu` | ~4 KB | Thin C++ wrapper: `cuInit`, metadata load, struct construction, launch |
| `<op>.obj` | ~16 KB | Compiled wrapper object |
| `test.so` | ~1.4 MB | Final shared library (linked with `-lcuda`) |

### Header structure (`<op>_cutedsl.h`)

```c
// Metadata struct (holds CUlibrary handle)
typedef struct {
    CUlibrary module;
} <op>_cutedsl_Kernel_Metadata_t;

// Typed tensor structs (one per kernel argument)
typedef struct {
    void *data;
    int32_t dynamic_shapes[N_DYNAMIC_SHAPES];   // runtime batch, seq_len, etc.
    int64_t dynamic_strides[N_DYNAMIC_STRIDES];  // runtime strides
} <op>_cutedsl_Tensor_mX_t;

// Load/Unload/Wrapper functions
void <op>_cutedsl_Kernel_Metadata_Load(<op>_cutedsl_Kernel_Metadata_t *metadata);
void <op>_cutedsl_Kernel_Metadata_Unload(<op>_cutedsl_Kernel_Metadata_t *metadata);
int32_t <op>_cutedsl_wrapper(<op>_cutedsl_Kernel_Metadata_t *metadata,
                              <op>_cutedsl_Tensor_mX_t *mX, ..., CUstream stream);
```

---

## File Checklist (for classic_b2b_bmm)

### Files Created

| File | Path (relative to aitemplate/) |
|------|-------------------------------|
| SM80 kernel | `backend/cuda/b2b_bmm/cutedsl_b2b_bmm_sm80.py` |
| SM90 kernel (WIP) | `backend/cuda/b2b_bmm/cutedsl_b2b_bmm_sm90.py` |
| Backend codegen | `backend/cuda/b2b_bmm/classic_b2b_bmm_cutedsl.py` |
| AOT example | `examples/07_how_to_run_pt_model/cutedsl_b2b_bmm_aot_example.py` |

### Files Modified

| File | Change |
|------|--------|
| `compiler/ops/b2b_bmm/classic_b2b_bmm.py` | Added `_use_cutedsl()`, `_backend_suffix()`, dispatch in `gen_function/decl/call` |
| `backend/codegen.py` | Pass `workdir`, handle `.o` in `file_pairs`, read `backend_suffix` |
| `backend/builder.py` | Skip `.o` compilation, add `-lcuda`, use `basename()` in Makefile |
| `backend/cuda/b2b_bmm/__init__.py` | Added `classic_b2b_bmm_cutedsl` import |
| `examples/07_how_to_run_pt_model/classic_b2b_bmm_example.py` | Added `--use-cutedsl` and `--both` CLI args |
| `examples/BUCK` | Added `nvidia-cutlass-dsl` dependency |

---

## Reference Codebases

| Reference | Path | What to reuse |
|-----------|------|---------------|
| FA4 SM80 helpers | `fbcode/ads_mkl/ops/cute_dsl/fa4/src/ampere_helpers.py` | SM80 warp MMA, SMEM layout, gemm/gemm_rs patterns |
| FA4 SM90 forward | `fbcode/ads_mkl/ops/cute_dsl/fa4/src/flash_fwd.py` | SM90 TMA+WGMMA, pipeline patterns |
| Hopper GEMM example | `fbcode/ai_acceleration/cute_dsl/examples/hopper/dense_gemm.py` | Complete CuTeDSL GEMM lifecycle |
| CuTeDSL AOT export | `third-party/cutlass/4.3.5/python/CuTeDSL/cutlass/cute/export/` | `export_to_c()`, `dump_to_object()` API |
| Quack PT2 compat | `fbcode/ads_mkl/ops/cute_dsl/quack/quack_rmsnorm_pt2.py` | `@functools.cache` + `cute.compile()` caching |
| Inductor CuTeDSL | `fbcode/caffe2/torch/_inductor/codegen/cutedsl/` | Framework integration pattern |
| Current CUTLASS backend | `backend/cuda/b2b_bmm/classic_b2b_bmm.py` | Interface contract (signature, registry keys) |
| CUTLASS C++ kernels | `static/include/kernels/classic_b2b_bmm/` | Algorithm reference for register-passing b2b GEMM |

---

## Claude Skill: `add-cutedsl-backend`

Below is a reusable Claude skill specification for adding a CuTeDSL backend
to any AIT operator.

### Skill Description

```
When adding a CuTeDSL backend for an AITemplate operator, follow these steps:

1. RESEARCH: Read the existing CUTLASS C++ backend for the target op:
   - `backend/cuda/<op_dir>/<op>.py` -- understand FUNC_TEMPLATE, registry keys,
     tensor shapes, tiling parameters
   - `compiler/ops/<op_dir>/<op>.py` -- understand gen_function/decl/call dispatch

2. KERNEL: Write the CuTeDSL kernel in `backend/cuda/<op_dir>/cutedsl_<op>_sm80.py`:
   - Class with __init__ (compile-time params), @cute.jit __call__ (host setup),
     @cute.kernel kernel (device code)
   - SM80: use warp.MmaF16BF16Op, cpasync.CopyG2SOp, warp.LdMatrix8x8x16bOp
   - Swizzled SMEM layouts via get_smem_layout_atom()
   - Named barriers for CTA sync

3. CODEGEN: Create `backend/cuda/<op_dir>/<op>_cutedsl.py`:
   - @functools.lru_cache AOT compile function
   - C++ wrapper template with typed tensor struct construction
   - Registry functions with _cutedsl suffix
   - CRITICAL: Use mark_compact_shape_dynamic() per-mode, NOT mark_layout_dynamic()
   - CRITICAL: C++ wrapper must construct *_Tensor_mX_t structs, NOT pass void*

4. DISPATCH: Modify `compiler/ops/<op_dir>/<op>.py`:
   - Add _use_cutedsl(), _backend_suffix()
   - Add suffix dispatch in gen_function(), gen_function_decl(), gen_function_call()
   - Store backend_suffix in self._attrs

5. BUILD: Modify `backend/codegen.py` and `backend/builder.py`:
   - codegen.py: pass workdir, handle .o in file_pairs, read backend_suffix
   - builder.py: skip .o compilation, add -lcuda, use basename() for .o in Makefile

6. REGISTER: Add import in `backend/cuda/<op_dir>/__init__.py`

7. TEST: Run with Target(..., use_cutedsl_<op>=True) and validate against
   PyTorch reference (atol=1e-2 for fp16)

Common pitfalls to watch for:
- mark_layout_dynamic makes ALL dims dynamic -> breaks static copy requirements
- partition_S(local_tile(..., (tile, None))) adds extra modes -> count None indices
- CuTeDSL export_to_c generates typed structs, not raw void* -> read .h header
- Makefile .o paths must use basename (Make runs from workdir)
- Need -lcuda for CUlibrary-based metadata loading
```

---

## Execution Order for New Operators

1. **Read existing backend** -- understand tensor shapes, tiling, registry keys
2. **Write SM80 kernel** -- implement, test standalone with `cute.compile()`
3. **AOT export test** -- verify `export_to_c()` produces valid `.h` + `.o`
4. **Create backend codegen** -- registry functions, C++ wrapper template
5. **Add dispatch** -- op class modifications, `__init__.py` import
6. **Modify build system** -- codegen.py, builder.py changes
7. **End-to-end test** -- run full AIT pipeline, compare outputs
8. **(Optional) SM90 kernel** -- TMA + WGMMA version for Hopper

---

## Verification Checklist

- [ ] CuTeDSL kernel compiles: `cute.compile()` succeeds
- [ ] AOT export produces `.h` and `.o` files
- [ ] C++ wrapper compiles with nvcc (no type mismatches)
- [ ] `.o` links into `.so` (no undefined symbols, `-lcuda` present)
- [ ] Runtime loads metadata: `Kernel_Metadata_Load()` succeeds
- [ ] Numerical correctness: output matches PyTorch reference (atol<=0.02 for fp16)
- [ ] Dynamic shapes work: different batch sizes / sequence lengths
- [ ] Original CUTLASS backend still works (no regression)
