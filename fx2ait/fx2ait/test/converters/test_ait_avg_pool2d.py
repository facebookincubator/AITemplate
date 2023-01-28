# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import parameterized


class TestAvgPool2dConverter(AITTestCase):
    @parameterized.expand(
        [
            (1, 1, 0),
            ((2, 2), 2, 1),
            ((4, 4), (4, 4), 0),
        ]
    )
    def test_avgpool2d(self, kernel_size, stride, padding):
        class TestModule(torch.nn.Module):
            def __init__(self, kernel_size, stride, padding):
                super().__init__()
                self.pool = torch.nn.AvgPool2d(kernel_size, stride, padding)

            def forward(self, x):
                return self.pool(x)

        model = TestModule(kernel_size, stride, padding).half().cuda()
        inputs = [torch.randn(1, 4, 256, 256).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.avg_pool2d},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )
