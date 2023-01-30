# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase


class TestAdaptiveAvgPool2dConverter(AITTestCase):
    def test_batch_norm(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(x)

        model = TestModule().half().cuda()
        inputs = [torch.randn(1, 3, 244, 244).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.batch_norm},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )
