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


class TestPowConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param("exp", size=[10], exp=5),
            param("3d_exp", size=[2, 5, 32], exp=5),
            param("4d_float_exp", size=[2, 5, 32, 128], exp=2.2),
        ]
    )
    def test_pow(self, name, size, exp):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.pow(x, exponent=exp)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn([2, 5, 32, 128]).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={torch.ops.aten.pow.Tensor_Scalar})

    @parameterized.expand(
        [
            param("exp", inputs_min=[10], inputs_max=[15], exp=5),
            param("3d_exp", inputs_min=[2, 5, 32], inputs_max=[3, 7, 64], exp=5),
            param(
                "4d_float_exp",
                inputs_min=[2, 5, 32, 128],
                inputs_max=[20, 7, 35, 140],
                exp=2.2,
            ),
        ]
    )
    def test_dynamic_pow(self, name, inputs_min, inputs_max, exp):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.pow(x, exponent=exp)

        model = TestModule().cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 4],
            ],
            inputs_max=[
                [3, 8, 10],
            ],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={torch.ops.aten.pow.Tensor_Scalar}
        )
