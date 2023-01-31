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
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import param, parameterized


class TestSplitConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param(
                "dim1",
                dim=1,
                split_size=3,
                expected_ops={torch.ops.aten.split.Tensor},
            ),
            param(
                "dim0",
                dim=0,
                split_size=3,
                expected_ops={torch.ops.aten.split.Tensor},
            ),
        ]
    )
    def test_split(self, name, dim, split_size, expected_ops):
        class TestModule(torch.nn.Module):
            def forward(self, y: torch.Tensor) -> torch.Tensor:
                res = torch.split(y, split_size, dim)
                return res[0]

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(20, 10, 8).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops=expected_ops)

    def test_split_dynamic(self):
        class TestModule(torch.nn.Module):
            def forward(self, y: torch.Tensor) -> torch.Tensor:
                res = torch.split(y, 4, 1)
                return res[0]

        model = TestModule().cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[[20, 10, 8]],
            inputs_max=[[50, 10, 8]],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={torch.ops.aten.split.Tensor}
        )

    # TODO low priority. May need to support it in future.
    # def test_split_imm(self):
    #     class TestModule(torch.nn.Module):
    #         def forward(self, y: torch.Tensor) -> torch.Tensor:
    #             dim1 = y.size(1)
    #             split_size = dim1 // 2
    #             return torch.split(y, split_size, 1)

    #     model = TestModule().cuda().half()
    #     inputs = [
    #         torch.randn(2, 10, 20).half().cuda(),
    #     ]
    #     self.run_test(model, inputs, expected_ops={torch.ops.aten.split.Tensor})
