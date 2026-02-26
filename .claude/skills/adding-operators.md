## Adding New Operators to AITemplate

### Overview

Adding a new operator involves defining it in Python, writing backend codegen, and adding tests.

### Steps

#### 1. Define the Operator (Python)

Create a new file in `python/aitemplate/compiler/ops/` (or the appropriate subdirectory):

```python
# python/aitemplate/compiler/ops/common/my_new_op.py

from aitemplate.compiler.base import Operator, Tensor

class my_new_op(Operator):
    """Description of what the op does."""

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "my_new_op"

    def _infer_shape(self, x: Tensor) -> List[IntVar]:
        """Compute output shape from input shapes."""
        return x._attrs["shape"]  # adjust as needed

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._set_depth()
        output = Tensor(shape=self._infer_shape(x), src_ops={self})
        self._attrs["outputs"] = [output]
        return output
```

#### 2. Write Backend Codegen

Add codegen templates in the appropriate backend directory:

- **CUDA**: `python/aitemplate/backend/cuda/`
- **ROCm**: `python/aitemplate/backend/rocm/`

Each backend file typically provides:
- `gen_function()` — generates the C++ kernel function
- `gen_function_decl()` — generates the function declaration
- `gen_function_call()` — generates the call site

#### 3. Register the Operator

Make sure the operator is importable from the ops package by adding it to the relevant `__init__.py`.

#### 4. Add Tests

Create a test file in the appropriate test directory following existing patterns. Tests typically:
- Build a small graph using the new op
- Compile it with `compile_model()`
- Compare outputs against a PyTorch reference implementation

### Tips

- Look at similar existing operators for patterns to follow
- Gemm ops are in `python/aitemplate/compiler/ops/gemm/`
- Elementwise ops are in `python/aitemplate/compiler/ops/common/`
- Use `_attrs` dict to store operator metadata
- Shape inference is critical — get it right or compilation will fail
