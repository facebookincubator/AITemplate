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
import torch
from fx2ait.acc_tracer import acc_ops

from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_fx2ait import AITTestCase

from parameterized import parameterized


class TestUnsqueezeConverter(AITTestCase):
    @parameterized.expand(
        [
            ["default", 1],
            ["negative_dim", -1],
        ]
    )
    def test_simple(self, name: str, dim: int):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.unsqueeze(x, dim)

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 3, 4).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.unsqueeze})

    def test_simple_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.unsqueeze(x, 1)

        model = TestModule().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 4],
            ],
            inputs_max=[
                [20, 3, 4],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={acc_ops.unsqueeze}
        )
