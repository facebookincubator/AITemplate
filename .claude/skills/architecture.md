## AITemplate Architecture

### Compilation Pipeline

AITemplate transforms a high-level neural network graph into a compiled GPU binary through these stages:

1. **Frontend (Python)**: Define or import a model using AIT operators (`python/aitemplate/compiler/ops/`)
2. **Graph Optimization**: Fuse operators (horizontal, vertical, memory fusion) in `python/aitemplate/compiler/transform/`
3. **Code Generation**: Emit C++ and CUDA/HIP source files via `python/aitemplate/backend/`
4. **Compilation**: Invoke nvcc/hipcc to produce a `.so` shared library
5. **Runtime**: Load the `.so` using the C++ runtime in `static/`

### Key Modules

- **`python/aitemplate/compiler/`**: Core compiler — IR, ops, graph transforms, type inference
  - `ops/`: Operator definitions (gemm, conv, elementwise, etc.)
  - `transform/`: Graph optimization passes (fusion, memory planning)
  - `base.py`: Base tensor and operator classes
- **`python/aitemplate/backend/`**: Code generation backends
  - `cuda/`: NVIDIA-specific codegen using CUTLASS
  - `rocm/`: AMD-specific codegen using Composable Kernel
  - `common/`: Shared codegen utilities
- **`python/aitemplate/frontend/`**: User-facing API for building models
- **`python/aitemplate/utils/`**: Utility functions (shape inference, profiling, etc.)
- **`static/`**: C++ runtime for loading and running compiled models

### Operator Fusion Types

- **Vertical fusion**: Fuse sequential ops (e.g., GEMM + bias + ReLU)
- **Horizontal fusion**: Batch independent ops of the same type
- **Memory fusion**: Eliminate intermediate allocations between fused ops

### Backend Codegen

Each backend (CUDA/ROCm) provides:
- Kernel templates (Jinja2-based C++ templates)
- Profiling to select optimal kernel configurations
- Header generation for the compiled model API
