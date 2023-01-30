import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestSoftmaxConverter(AITTestCase):
    @parameterized.expand(
        [
            param("default", dim=1),
            param("neg", dim=-1),
        ]
    )
    def test_softmax(self, name, dim=None):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.softmax(x, dim=dim)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.softmax})
