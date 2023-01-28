import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase

from parameterized import parameterized


class TestMatMulConverter(AITTestCase):
    @parameterized.expand(
        [
            [[2, 3], [3, 4]],
            [[2, 3, 4], [2, 4, 6]],
            [[2, 3, 4], [4, 6]],
            [[3, 4], [5, 4, 6]],
            [[2, 2, 2, 3, 4], [4, 6]],
        ]
    )
    def test_simple(self, lhs_shape, rhs_shape):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(x, y)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*lhs_shape).half().cuda(),
            torch.randn(*rhs_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.matmul})

    def test_mm(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.mm(x, y)

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 3).half().cuda(),
            torch.randn(3, 4).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.matmul})

    @parameterized.expand(
        [
            [[1, 2, 3], [1, 3, 4]],
            [[3, 2, 3], [3, 3, 4]],
        ]
    )
    def test_bmm(self, lhs_shape, rhs_shape):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.bmm(x, y)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*lhs_shape).half().cuda(),
            torch.randn(*rhs_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.matmul})

    @parameterized.expand(
        [
            [[1, 1, 3, 4], [1, 1, 4, 6]],
            [[1, 2, 3, 4], [1, 2, 4, 6]],
            [[4, 1, 3, 4], [4, 1, 4, 6]],
            [[4, 2, 3, 4], [4, 2, 4, 6]],
        ]
    )
    def test_matmul_with_4d_tensors(self, lhs_shape, rhs_shape):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(x, y)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*lhs_shape).half().cuda(),
            torch.randn(*rhs_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.matmul})
