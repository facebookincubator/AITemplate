import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase


class TestContiguousConverter(AITTestCase):
    def test_contigupus(self):
        class TestModule(torch.nn.Module):
            def forward(self, x) -> torch.Tensor:
                x = x.contiguous()
                return x + x

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.contiguous})
