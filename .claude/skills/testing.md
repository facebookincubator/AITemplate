## Testing in AITemplate

### Running Tests

#### Python tests with pytest
```bash
# Run a specific test file
python -m pytest tests/unittest/test_my_op.py -v

# Run a specific test case
python -m pytest tests/unittest/test_my_op.py::TestMyOp::test_basic -v
```

#### Buck tests (Meta-internal)
```bash
# Run a specific Buck test target
buck2 test //aitemplate/AITemplate:test_target_name
```

### Test Patterns

AITemplate tests typically follow this structure:

```python
import unittest
import torch
from aitemplate.testing import detect_target
from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor, nn

class TestMyFeature(unittest.TestCase):
    def test_basic(self):
        # 1. Check for available GPU target
        target = detect_target()

        # 2. Build AIT graph
        X = Tensor(shape=[2, 3], name="X", is_input=True)
        Y = my_op()(X)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "Y"

        # 3. Compile
        module = compile_model(Y, target, "./tmp", "test_my_op")

        # 4. Run with real data
        x_pt = torch.randn(2, 3).cuda().half()
        y = torch.empty(2, 3).cuda().half()
        module.run_with_tensors([x_pt], [y])

        # 5. Compare against PyTorch reference
        y_ref = torch_reference_impl(x_pt)
        torch.testing.assert_close(y, y_ref)
```

### Key Testing Utilities

- `aitemplate.testing.detect_target()` — auto-detect CUDA or ROCm
- `compile_model()` — compile a graph to a loadable module
- `module.run_with_tensors()` — execute the compiled model
- Tests typically use `float16` precision and compare with tolerance

### FX2AIT Tests

For the PyTorch FX converter, tests are in `fx2ait/fx2ait/test/`:
```python
# These tests verify that PyTorch models convert correctly to AIT
# and produce numerically equivalent results
```
