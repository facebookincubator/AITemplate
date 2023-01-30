import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from torch import nn


class TestSigmoidConverter(AITTestCase):
    def test_sigmoid(self):
        class Sigmoid(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        model = Sigmoid().cuda()
        inputs = [torch.randn(1, 2, 3).half().cuda()]
        self.run_test(model, inputs, expected_ops={acc_ops.sigmoid})
