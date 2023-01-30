import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestLinalgConverter(AITTestCase):
    @parameterized.expand(
        [
            param(
                "l2_norm_dim_3",
                input_shape=[1, 100, 40, 40],
                ord=2,
                dim=3,
                keepdims=False,
            ),
            param(
                "l2_norm_dim_2",
                input_shape=[1, 100, 40, 40],
                ord=2,
                dim=2,
                keepdims=False,
            ),
            param(
                "l2_norm_dim_1",
                input_shape=[1, 100, 40, 40],
                ord=2,
                dim=1,
                keepdims=True,
            ),
        ]
    )
    def test_linalg_norm(
        self, test_name, input_shape, ord=None, dim=None, keepdims=False
    ):
        class TestModule(torch.nn.Module):
            def __init__(self, ord, dim, keepdims):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.linalg.norm(x, ord, dim, keepdims)

        model = TestModule(ord, dim, keepdims).cuda().half()
        inputs = [
            torch.randn(input_shape).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.linalg_norm})
