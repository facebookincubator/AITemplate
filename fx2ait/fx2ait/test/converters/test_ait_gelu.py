import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import parameterized


class TestGeluConverter(AITTestCase):
    def test_gelu(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.gelu(x)

        inputs = [torch.randn(3, 10, 20).cuda().half()]
        model = TestModule().cuda().half()

        self.run_test(model, inputs, expected_ops={acc_ops.gelu})

    def test_fast_gelu(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.gelu(x, approximate="tanh")

        inputs = [torch.randn(3, 10, 20).cuda().half()]
        model = TestModule().cuda().half()

        self.run_test(model, inputs, expected_ops={acc_ops.gelu})

    @parameterized.expand(
        [
            ("none"),
            ("tanh"),
        ]
    )
    def test_gelu_module(self, name):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = torch.nn.GELU(approximate=name)

            def forward(self, x):
                return self.gelu(x)

        inputs = [torch.randn(3, 10, 20).cuda().half()]
        model = TestModule().cuda().half()

        self.run_test(model, inputs, expected_ops={acc_ops.gelu})
