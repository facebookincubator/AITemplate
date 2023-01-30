import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestClampConverter(AITTestCase):
    @parameterized.expand(
        [
            param("default", min=-1, max=0, use_clamp=True),
            param("min", min=0.5, use_clamp=False),
            param("max", max=0.5, use_clamp=True),
            param("minBiggerThanMax", min=1, max=0, use_clamp=False),
        ]
    )
    def test_clamp(self, name, min=None, max=None, use_clamp=True):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.op = torch.clamp if use_clamp else torch.clip

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.op(x, min=min, max=max)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.clamp})
