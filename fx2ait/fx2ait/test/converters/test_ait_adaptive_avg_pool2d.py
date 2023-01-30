#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import parameterized


class TestAdaptiveAvgPool2dConverter(AITTestCase):
    @parameterized.expand(
        [
            ((64, 64),),
            ((128, 128),),
            (64,),
        ]
    )
    def test_adaptive_avgpool2d(
        self,
        output_size,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d(output_size)

            def forward(self, x):
                return self.pool(x)

        model = TestModule().half().cuda()
        inputs = [torch.randn(1, 32, 256, 256).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.adaptive_avg_pool2d},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )
