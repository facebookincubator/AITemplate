## Migrating GEMM Operators from CUTLASS C++ to CuTeDSL

### Overview

This skill documents the pattern for migrating any GEMM variant (gemm_rcr_bias, gemm_rrr, gemm_ccr, etc.) from CUTLASS C++ template-based code generation to CuTeDSL (CUTLASS Python DSL), using AOT-style deployment.

### Layout Reference

CUTLASS GEMM naming convention: `gemm_<A-layout><B-layout><C-layout>` where R=Row-major, C=Column-major.

| Layout | A shape/order | B shape/order | MMA Atom | Key property |
|--------|--------------|--------------|----------|-------------|
| RCR | [M,K] K-major | [N,K] K-major | TN | Both K-contiguous |
| RRR | [M,K] K-major | [K,N] N-major | TN/NT | A=K-cont, B=N-cont |
| CCR | [K,M] M-major | [N,K] K-major | NN | A=M-cont, B=K-cont |

For SM80 with fp16: `warp.MmaF16BF16Op(Float16, Float32, (16,8,16))` with TN layout.

### Step-by-Step Migration

#### 1. Identify the operation

Read the CUTLASS backend file (e.g., `backend/cuda/gemm_universal/gemm_rcr_bias.py`) to understand:
- The GEMM layout (RCR, RRR, etc.)
- The epilogue (bias, relu, bias+relu, etc.)
- The function signature from `common_bias.py` or `common.py`
- Whether it supports split-K

#### 2. Create the CuTeDSL kernel

Create `backend/cuda/gemm_universal/cutedsl_<op_name>_sm80.py`.

Key components to implement:

```python
class GemmXxxSm80Kernel:
    def __init__(self, tile_m=128, tile_n=128, tile_k=32):
        self.tile_m, self.tile_n, self.tile_k = tile_m, tile_n, tile_k
        self.num_threads = 128  # 4 warps

    @cute.jit
    def __call__(self, mA, mB, mC, ...):
        # 1. Set up SMEM layouts using get_smem_layout_atom()
        # 2. Define SharedStorage struct
        # 3. Create copy atoms (async for load, universal for store)
        # 4. Create tiled_mma with SM80 16x8x16 atom
        # 5. Compute grid dims and launch kernel

    @cute.kernel
    def kernel(self, ...):
        # 1. Allocate SMEM, partition GMEM tiles
        # 2. GEMM mainloop: K-loop with cp.async + warp MMA
        # 3. Epilogue: apply bias/activation on FP32 accumulators
        # 4. Store: Reg -> SMEM -> GMEM
```

**SMEM layout pattern** (from `cutedsl_b2b_bmm_sm80.py`):
```python
def get_smem_layout_atom(dtype, k_dim):
    dtype_byte = cutlass.const_expr(dtype.width // 8)
    bytes_per_row = cutlass.const_expr(k_dim * dtype_byte)
    smem_k_block_size = cutlass.const_expr(
        128 if bytes_per_row % 128 == 0
        else (64 if bytes_per_row % 64 == 0
              else (32 if bytes_per_row % 32 == 0 else 16))
    ) // dtype_byte
    # ... swizzle computation
    return cute.make_composed_layout(swizzle, 0, ordered_layout)
```

**GEMM mainloop pattern**:
```python
for k_tile in range(k_tiles):
    cute.copy(gmem_tiled_copy, src_gmem, dst_smem)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    barrier.arrive_and_wait()

    # Load first k-block from SMEM to registers
    cute.copy(smem_tiled_copy_A, tSsA[None, None, 0], tSrA_view[None, None, 0])
    cute.copy(smem_tiled_copy_B, tSsB[None, None, 0], tSrB_view[None, None, 0])

    for k_block in cutlass.range_constexpr(num_k_blocks):
        k_next = (k_block + 1) % num_k_blocks
        cute.copy(smem_tiled_copy_A, tSsA[None, None, k_next], ...)
        cute.copy(smem_tiled_copy_B, tSsB[None, None, k_next], ...)
        cute.gemm(tiled_mma, acc, tSrA[None, None, k_block], tSrB[None, None, k_block], acc)

    barrier.arrive_and_wait()
```

**Store pattern** (Reg -> SMEM -> GMEM):
```python
rOut = cute.make_fragment_like(acc, dtype)
rOut.store(acc.load().to(dtype))
sOut = cute.make_tensor(sA.iterator, sOut_layout)  # reuse SMEM

smem_copy_atom_C = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)
smem_tiled_copy_C = cute.make_tiled_copy_C(smem_copy_atom_C, tiled_mma)
# ... copy from reg to SMEM, sync, then SMEM to GMEM
```

#### 3. Handle the epilogue

**1D bias (broadcast)**: Load bias into SMEM as a 2D view with stride-0 in M, use `thr_mma.partition_C()` to align with accumulator layout:
```python
# In __call__: create 2D broadcast view
mBias_2d = cute.make_tensor(mBias.iterator,
    cute.make_layout((M_size, N_size), stride=(0, bias_stride)))

# In kernel: tile and load
gBias = cute.local_tile(mBias_2d, (tile_m, tile_n), (m_block, n_block))
# ... cp.async load to sBias ...
tCsBias = thr_mma.partition_C(sBias)
# element-wise add in FP32
```

**2D bias**: Direct tile and load (like b2b_bmm does).

**Activation**: Apply element-wise in FP32 accumulator space before store.

#### 4. Create AOT example

Create `examples/07_how_to_run_pt_model/cutedsl_<op_name>_aot_example.py`:

```python
def make_cute_tensor(torch_tensor):
    ct = from_dlpack(torch_tensor, assumed_align=16)
    ct = ct.mark_layout_dynamic(leading_dim=len(torch_tensor.shape) - 1)
    ct = ct.mark_compact_shape_dynamic(
        mode=len(torch_tensor.shape) - 1,
        stride_order=torch_tensor.dim_order(),
        divisibility=(128 // cutlass.Float16.width),
    )
    return ct

def jit_compile_and_validate():
    # 1. Create PyTorch reference tensors
    # 2. Compute PyTorch reference output
    # 3. Create kernel instance
    # 4. cute.compile(kernel, *cute_tensors, *scalar_args, stream)
    # 5. Execute and compare (atol=1e-2, rtol=1e-2 for fp16)

def aot_export(compiled):
    # export_to_c(compiled, file_path=output_dir, file_name=name)
    # Generates: <name>.h + <name>.o
```

#### 5. Add BUCK target

```python
python_binary(
    # @autodeps-skip
    name = "cutedsl_<op_name>_aot_example",
    srcs = ["07_how_to_run_pt_model/cutedsl_<op_name>_aot_example.py"],
    main_module = "aitemplate.AITemplate.examples.07_how_to_run_pt_model.cutedsl_<op_name>_aot_example",
    deps = [
        "fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl",
        "//aitemplate/AITemplate/python/aitemplate:aitemplate",
        "//caffe2:torch",
    ],
)
```

### Reference Implementations

| File | What it demonstrates |
|------|---------------------|
| `backend/cuda/gemm_universal/cutedsl_gemm_rcr_bias_sm80.py` | GEMM RCR + 1D bias, 2D grid |
| `backend/cuda/b2b_bmm/cutedsl_b2b_bmm_sm80.py` | Fused B2B GEMM, 2D bias, activation |
| `examples/.../cutedsl_gemm_rcr_bias_aot_example.py` | AOT example for GEMM |
| `examples/.../cutedsl_b2b_bmm_aot_example.py` | AOT example for B2B BMM |
| `ads_mkl/ops/cute_dsl/fa4/src/ampere_helpers.py` | Reusable SM80 helpers |

### Common Pitfalls

1. **Alignment**: Use `mark_compact_shape_dynamic(divisibility=8)` for fp16 to satisfy 128-bit cp.async alignment
2. **Bias broadcast**: Create 2D view with stride-0 in broadcast dim at host level (`__call__`), not inside `@cute.kernel`
3. **SMEM reuse**: Output staging can reuse SMEM from input tiles (e.g., `sC = cute.make_tensor(sA.iterator, sC_layout)`)
4. **MN view**: Use `_make_acc_tensor_mn_view()` to iterate accumulator in logical (M, N) order for epilogue ops
5. **Python slicing**: CuTe tensors don't support Python slicing (`tensor[start:]`). Use `cute.local_tile()` or `cute.make_tensor()` with offset iterators
