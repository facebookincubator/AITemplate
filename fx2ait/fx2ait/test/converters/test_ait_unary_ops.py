import math
from typing import Callable

import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import parameterized


unary_ops = [
    (torch.abs, acc_ops.abs),
    (torch.sign, acc_ops.sign),
    (torch.log, acc_ops.log),
    (torch.relu, acc_ops.relu),
]


class TestUnaryOpsConverter(AITTestCase):
    @parameterized.expand([(op[0].__name__, op[0], op[1]) for op in unary_ops])
    def test_unary_ops(self, name, orig_op: Callable, expected_op):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return orig_op(x)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(
            model, inputs, expected_ops={expected_op} if expected_op is not None else {}
        )

    def test_sqrt(self):
        class TestModule(torch.nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def forward(self, y):
                return torch.div(y, math.sqrt(self.x))

        model = TestModule(x=64).cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={})
