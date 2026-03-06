# AITemplate classic_b2b_bmm: Graph Optimization & Code Generation Flow

## Overview

This document describes the end-to-end compilation flow when decomposed
attention ops are automatically fused into a single `classic_b2b_bmm` kernel.

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  User Code: build_decomposed_b2b_bmm_graph()                   │
│                                                                 │
│  Q ──► bmm_rcr(Q,K) ──► MUL(α₀) ──► ADD(bias) ──► SIGMOID     │
│                                                        │        │
│                                                  MUL(α₁)       │
│                                                        │        │
│                                           bmm_rrr(score,V) ──► Y│
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  compile_model(Y, target, workdir, test_name)                    │
│  [compiler.py]                                                   │
│                                                                  │
│  1. toposort(output_tensors)                                     │
│  2. name_graph(sorted_graph)                                     │
│  3. optimize_graph(sorted_graph)  ◄──────────────────────────┐   │
│     │                                                        │   │
│     ├─ constant_folding                                      │   │
│     ├─ fuse_ops (elementwise fusions, etc.)                  │   │
│     ├─ ★ fuse_b2b_bmm(sorted_graph) ◄───── PATTERN MATCH    │   │
│     │   │                                                    │   │
│     │   │  Matches chain:                                    │   │
│     │   │    bmm_rcr → MUL(const) → ADD(tensor)             │   │
│     │   │    → activation → [MUL(const)] → bmm_rrr          │   │
│     │   │                                                    │   │
│     │   │  Replaces with:                                    │   │
│     │   │    classic_b2b_bmm(Q, K, V, bias)                  │   │
│     │   │    α₀, α₁, epilogue baked into op attrs            │   │
│     │   │                                                    │   │
│     │   └─ Removes 6 intermediate ops, 4+ intermediate       │   │
│     │      tensors                                            │   │
│     │                                                        │   │
│     ├─ memory_planning(sorted_graph)                         │   │
│     └─ other passes...                                       │   │
│                                                                  │
│  4. codegen(sorted_graph, workdir)                               │
│     │                                                            │
│     ├─ gen_function_src()                                        │
│     │   For each op (including classic_b2b_bmm):                 │
│     │   ┌────────────────────────────────────────────────────┐   │
│     │   │ op.gen_function()                                  │   │
│     │   │   → registry.get("cuda.classic_b2b_bmm.gen_function")│ │
│     │   │   → Renders Jinja2 FUNC_TEMPLATE                   │  │
│     │   │   → Writes <func_name>.cu                          │  │
│     │   └────────────────────────────────────────────────────┘   │
│     │                                                            │
│     ├─ ModelContainerGenerator                                   │
│     │   → func_decl(): function declarations                     │
│     │   → func_call(): invocations in RunImpl()                  │
│     │   → Writes model.cu, model_container.cu                    │
│     │                                                            │
│     └─ copy_headers_and_csrc_to_workdir()                        │
│                                                                  │
│  5. build(file_pairs, workdir, test_name)                        │
│     │                                                            │
│     ├─ gen_makefile()                                             │
│     ├─ nvcc <func>.cu → <func>.obj                               │
│     ├─ nvcc model.cu → model.obj                                 │
│     └─ nvcc -shared *.obj → test.so                              │
│                                                                  │
│  6. Return Model(workdir)                                        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Runtime: module.run_with_tensors(inputs, outputs)               │
│  [model.py → Model class]                                        │
│                                                                  │
│  1. ctypes.CDLL loads test.so                                    │
│  2. Sets input pointers + dynamic dims                           │
│  3. Calls RunImpl(stream) in C++                                 │
│     → Invokes classic_b2b_bmm_func(output, Q, K, V, bias,       │
│         batch_size, num_heads, m0, k0, stream)                   │
│     → Inside: instantiates B2bGemmBatched<...>, runs on GPU      │
│  4. Returns output tensors                                       │
└──────────────────────────────────────────────────────────────────┘
```

## Generated CUDA Code Structure

The backend codegen (`backend/cuda/b2b_bmm/classic_b2b_bmm.py`) produces:

### `<func_name>.cu` — Kernel Source
```cpp
#include "cutlass/cutlass.h"
#include "classic_b2b_bmm/device/b2b_batched_gemm.h"

// Hardcoded tile sizes
constexpr int ThreadblockM = 64, ThreadblockK = 32;
constexpr int WarpM = 16, WarpK = 32;
constexpr int N0 = <seq_len>, N1 = <head_dim>;

void <func_name>(void* output, void* query, void* key, void* value,
                 void* bias, int64_t batch_size, int64_t num_heads,
                 int64_t m0, int64_t k0, cudaStream_t stream) {
    // Type aliases, epilogue ops, B2bGemmBatched instantiation
    // Argument construction with batched/multi-head strides
    // Initialize and execute
}
```

### `model.cu` — Container
```cpp
class Model : public ModelBase<Model> {
    void RunImpl(StreamType stream) {
        // ... sets up pointers ...
        <func_name>(output, Q, K, V, bias, batch, heads, m0, k0, stream);
    }
};
```

## Key Files

| Component | File |
|-----------|------|
| Pattern matching | `compiler/transform/fuse_b2b_bmm.py` |
| Op definition | `compiler/ops/b2b_bmm/classic_b2b_bmm.py` |
| Base class | `compiler/ops/b2b_bmm/b2b_bmm_base.py` |
| CUDA backend | `backend/cuda/b2b_bmm/classic_b2b_bmm.py` |
| CUTLASS headers | `static/include/kernels/classic_b2b_bmm/` |
| Compiler entry | `compiler/compiler.py` |
| Code generation | `backend/codegen.py` |
| Builder | `backend/builder.py` |
| Runtime | `compiler/model.py` |
