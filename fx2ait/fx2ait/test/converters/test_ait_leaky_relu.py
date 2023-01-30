# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase


class TestLeakyReluConverter(AITTestCase):
    def test_leaky_relu(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.leaky_relu(x, negative_slope=0.05)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.leaky_relu})
