# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase


class TestAdaptiveAvgPool2dConverter(DispatchTestCase):
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
            expected_ops={torch.ops.aten.batch_norm},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )

    def test_dynamic_batch_norm(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(x)

        model = TestModule().half().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [1, 3, 244, 244],
            ],
            inputs_max=[
                [10, 3, 256, 256],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten.batch_norm},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )
