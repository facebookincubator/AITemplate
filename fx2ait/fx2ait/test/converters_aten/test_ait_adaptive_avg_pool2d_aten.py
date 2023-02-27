#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import parameterized


class TestAdaptiveAvgPool2dConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((64, 64), torch.ops.aten._adaptive_avg_pool2d.default),
            ((128, 128), torch.ops.aten._adaptive_avg_pool2d.default),
            (64, torch.ops.aten._adaptive_avg_pool2d.default),
            (
                (1, 1),
                torch.ops.aten.mean.dim,
            ),
        ]
    )
    def test_adaptive_avgpool2d(self, output_size, op_check):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d(output_size)

            def forward(self, x):
                return self.pool(x)

        model = TestModule().half().cuda()
        inputs = [torch.randn(1, 32, 256, 256).cuda().half()]
        if op_check == torch.ops.aten.mean.dim:
            permute_inputs = None
            permute_outputs = None
        else:
            permute_inputs = [0, 2, 3, 1]
            permute_outputs = [0, 3, 1, 2]
        self.run_test(
            model,
            inputs,
            expected_ops={op_check},
            permute_inputs=permute_inputs,
            permute_outputs=permute_outputs,
        )

    @parameterized.expand(
        [
            ((64, 64),),
            ((128, 128),),
            (64,),
        ]
    )
    def test_dynamic_adaptive_avgpool2d(
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
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 32, 256, 256],
            ],
            inputs_max=[
                [10, 32, 256, 256],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten._adaptive_avg_pool2d.default},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )
