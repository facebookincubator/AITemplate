import torch
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase

from parameterized import parameterized


class TestMatMulConverter(DispatchTestCase):
    @parameterized.expand(
        [
            [[2, 3], [3, 4], torch.ops.aten.mm.default],
            [[2, 3, 4], [4, 6], torch.ops.aten.mm.default],
            [[2, 3, 4], [2, 4, 6], torch.ops.aten.bmm.default],
            [[2, 2, 2, 3, 4], [4, 6], torch.ops.aten.mm.default],
        ]
    )
    def test_simple(self, lhs_shape, rhs_shape, op):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(x, y)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*lhs_shape).half().cuda(),
            torch.randn(*rhs_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={op})

    def test_mm(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.mm(x, y)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(2, 3).half().cuda(),
            torch.randn(3, 4).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={torch.ops.aten.mm.default})

    @parameterized.expand(
        [
            # Only M can be dynamic: https://github.com/fairinternal/AITemplate/blob/main/tests/unittest/ops/test_gemm.py
            [[[2, 3], [3, 3], [6, 6]], torch.ops.aten.mm.default],
            [[[2, 3], [2, 3], [3, 3], [6, 6]], torch.ops.aten.mm.default],
            [[[1, 3], [2, 3], [6, 8], [3, 3], [6, 6]], torch.ops.aten.mm.default],
            # FIXME: batch_size cannot be dynamic because the permutation of shape change the names: P544607056
            # b, m, k, n
            [[[2, 2], [6, 8], [3, 3], [6, 6]], torch.ops.aten.bmm.default, True],
        ]
    )
    def test_dynamic(self, shape, op, bmm=False):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(x, y)

        model = TestModule().cuda()

        input0_shape = []
        for i, s in enumerate(shape):
            if i == len(shape) - 1:
                break
            input0_shape.append(
                TensorSpec.gen_int_var_min_max(s[0], s[1], "dynamic" + str(i))
            )
        input1_shape = [input0_shape[-1]] + [
            TensorSpec.gen_int_var_min_max(shape[-1][0], shape[-1][1], "dynamic_last")
        ]
        if bmm:
            input1_shape = [input0_shape[0]] + input1_shape
        input_spec = TensorSpec.create_spec_from_int_vars(
            [input0_shape, input1_shape], dtype_list=[torch.float16] * 2
        )
        self.run_test_with_dynamic_shape(model, input_spec, expected_ops={op})
