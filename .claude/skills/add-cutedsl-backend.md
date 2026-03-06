# Skill: Add CuTeDSL Backend to an AITemplate Operator

## Description

This skill guides adding a CuTeDSL (CUTLASS Python DSL) backend to an existing
AITemplate operator, replacing the Jinja2/CUTLASS C++ code generation with
Python-based AOT compilation via `cute.compile()` + `export_to_c()`.

## When to Use

- Converting an existing AIT operator from CUTLASS C++ templates to CuTeDSL
- Adding an alternative CuTeDSL backend alongside the existing CUTLASS backend
- Creating a new GPU kernel using CuTeDSL within the AIT framework

## Prerequisites

- The operator already has a working CUTLASS C++ backend
- `nvidia-cutlass-dsl` package is available in the build system
- Target GPU is SM80+ (Ampere or newer)

## Steps

### 1. RESEARCH the existing backend

Read these files to understand the operator interface:

```
backend/cuda/<op_dir>/<op>.py       # FUNC_TEMPLATE, registry keys, tensor shapes
compiler/ops/<op_dir>/<op>.py       # gen_function/decl/call dispatch
```

Key things to note:
- Function signature (what void* params, what scalar dims)
- Tensor shapes and which dims are static vs dynamic
- Tiling parameters (threadblock M/N/K, warp M/N/K)
- Registry key pattern (e.g., `cuda.classic_b2b_bmm.gen_function`)

### 2. WRITE the CuTeDSL kernel

Create: `backend/cuda/<op_dir>/cutedsl_<op>_sm80.py`

Structure:
```python
class MyOpSm80Kernel:
    def __init__(self, ...compile_time_params...):
        # Store tile sizes, compile-time constants

    @cute.jit
    def __call__(self, mA: cute.Tensor, ..., stream: cuda.CUstream):
        # Host-side: SMEM layouts, copy atoms, MMA config, kernel launch

    @cute.kernel
    def kernel(self, ...):
        # Device-side: actual CUDA kernel body
```

Key SM80 patterns:
- `warp.MmaF16BF16Op` for MMA atoms
- `cpasync.CopyG2SOp()` for GMEM->SMEM async copy
- `warp.LdMatrix8x8x16bOp` for SMEM->register loads
- Swizzled SMEM layouts via `cute.make_composed_layout(cute.make_swizzle(...))`
- `pipeline.NamedBarrier` for CTA synchronization

### 3. CREATE the backend codegen module

Create: `backend/cuda/<op_dir>/<op>_cutedsl.py`

Must contain:
- `@functools.lru_cache` AOT compilation function
- C++ wrapper template (Jinja2) that constructs typed tensor structs
- Three registry functions with `_cutedsl` suffix:
  - `cuda.<op>.gen_function_cutedsl`
  - `cuda.<op>.func_decl_cutedsl`
  - `cuda.<op>.func_call_cutedsl`

CRITICAL: The C++ wrapper must construct `*_Tensor_mX_t` typed structs
(not pass raw void*). Read the generated `.h` header to see struct fields.

CRITICAL: Use `mark_compact_shape_dynamic()` per-mode, NOT `mark_layout_dynamic()`.
Only mark truly dynamic dims (batch, seq_len). Keep inner dims static.

### 4. ADD backend dispatch

Modify: `compiler/ops/<op_dir>/<op>.py`

```python
def _use_cutedsl(self) -> bool:
    return target.Target.current()._kwargs.get("use_cutedsl_<op>", False)

def _backend_suffix(self) -> str:
    return "_cutedsl" if self._use_cutedsl() else ""
```

Add suffix dispatch in `gen_function()`, `gen_function_decl()`, `gen_function_call()`.
Store `backend_suffix` in `self._attrs` for codegen.py to pick up.

### 5. MODIFY the build system

`backend/codegen.py`:
- Pass `workdir` to `func_attrs`
- Add CuTeDSL `.o` to `file_pairs`
- Read `backend_suffix` for func_decl/func_call dispatch

`backend/builder.py`:
- Skip compilation for pre-compiled `.o` files
- Add `-lcuda` when CuTeDSL objects are present
- Use `os.path.basename()` for `.o` paths in Makefile

### 6. REGISTER the new module

Add import in `backend/cuda/<op_dir>/__init__.py`:
```python
from aitemplate.backend.cuda.<op_dir> import <op>_cutedsl
```

### 7. TEST

Run with `Target(..., use_cutedsl_<op>=True)`.
Validate against PyTorch reference (atol=1e-2 for fp16).

## Common Pitfalls

1. `mark_layout_dynamic()` makes ALL dims dynamic -> breaks static copy reqs
2. `partition_S(local_tile(..., (tile, None)))` adds extra modes -> count None indices
3. `export_to_c` generates typed structs, not raw void* -> read .h header
4. Makefile .o paths must use basename (Make runs from workdir)
5. Need `-lcuda` for CUlibrary-based metadata loading
6. Batch dim divisibility: use div=1 for batch, div=8 for seq_len
7. `LayoutEnum.from_tensor()` only handles 2D -> hardcode ROW_MAJOR for 3D

## References

| Reference | Path |
|-----------|------|
| FA4 SM80 helpers | `fbcode/ads_mkl/ops/cute_dsl/fa4/src/ampere_helpers.py` |
| FA4 SM90 forward | `fbcode/ads_mkl/ops/cute_dsl/fa4/src/flash_fwd.py` |
| Hopper GEMM example | `fbcode/ai_acceleration/cute_dsl/examples/hopper/dense_gemm.py` |
| CuTeDSL AOT export | `third-party/cutlass/4.3.5/python/CuTeDSL/cutlass/cute/export/` |
| classic_b2b_bmm impl | `backend/cuda/b2b_bmm/classic_b2b_bmm_cutedsl.py` |
| classic_b2b_bmm SM80 | `backend/cuda/b2b_bmm/cutedsl_b2b_bmm_sm80.py` |
