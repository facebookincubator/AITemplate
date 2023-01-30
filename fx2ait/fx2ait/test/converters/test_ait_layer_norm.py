import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized
from torch import nn


class TestLayernormConverter(AITTestCase):
    @parameterized.expand(
        [
            param("1d_normalized_shape", [10], [2, 10]),
            # Enable test case once layernorm support expand
            # param("2d_normalized_shape", [5, 10], [5, 10]),
        ]
    )
    def test_layer_norm(self, name, normalized_shape, input_shape):
        class TestModule(torch.nn.Module):
            def __init__(self, normalized_shape):
                super().__init__()
                # TODO remove hard code eps once layernorm api expose eps setting
                self.mod = nn.LayerNorm(normalized_shape, eps=1e-5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.mod(x)

        model = TestModule(normalized_shape).cuda().half()
        inputs = [
            torch.randn(input_shape).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.layer_norm})

    def test_layer_norm_IntImm_shape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                shape = x.shape
                normalized_shape = shape[1:]
                return torch.nn.functional.layer_norm(x, normalized_shape, eps=1e-5)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn([10, 10]).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.layer_norm})
