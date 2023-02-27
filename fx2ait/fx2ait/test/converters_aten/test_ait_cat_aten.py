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
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import parameterized


class TestCatConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ["default", 0],
            ["positive_dim", 1],
            ["negative_dim", -1],
        ]
    )
    def test_cat(self, name: str, dim: int):
        class TestModule(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
            ) -> torch.Tensor:
                return torch.cat([x, y, z], dim=dim)

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 3, 4).half().cuda(),
            torch.randn(2, 3, 4).half().cuda(),
            torch.randn(2, 3, 4).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={torch.ops.aten.cat.default})

    @parameterized.expand(
        [
            ["default", 0],
            ["positive_dim", 1],
            ["negative_dim", -1],
        ]
    )
    def test_cat_dynamic_shape(self, name: str, dim: int):
        class TestModule(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
            ) -> torch.Tensor:
                return torch.cat([x, y, z], dim=dim)

        model = TestModule().cuda()

        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 4],
                [2, 3, 4],
                [2, 3, 4],
            ],
            inputs_max=[
                [20, 3, 4],
                [20, 3, 4],
                [20, 3, 4],
            ],
            dtype_list=[
                torch.float16,
                torch.float16,
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={torch.ops.aten.cat.default}
        )
