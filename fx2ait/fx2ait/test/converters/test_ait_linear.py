import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase


class TestLinearConverter(AITTestCase):
    def test_linear(self):
        M = 2
        N = 4
        K = 8

        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                w = torch.randn(N, K).half().cuda()
                b = torch.randn(N).half().cuda()
                return torch.nn.functional.linear(x, w, b)

        model = TestModule().cuda()
        inputs = [torch.randn(M, K).half().cuda()]
        self.run_test(model, inputs, expected_ops={acc_ops.linear})

    def test_linear_no_bias(self):
        M = 2
        N = 4
        K = 8

        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                w = torch.randn(N, K).half().cuda()
                return torch.nn.functional.linear(x, w, bias=None)

        model = TestModule().cuda()
        inputs = [torch.randn(M, K).half().cuda()]
        self.run_test(model, inputs, expected_ops={acc_ops.linear})
