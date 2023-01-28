import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase

from parameterized import parameterized


class TestPowConverter(AITTestCase):
    @parameterized.expand([("int", 3), ("float", 0.25)])
    def test_pow(self, _, exp):
        class Pow(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.pow(x, exp)

        model = Pow().half().cuda()
        input = [torch.randn(3, 3).half().cuda()]
        self.run_test(model, input, expected_ops={acc_ops.pow})
