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


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class TestATenReluConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param("small", size=(2, 3)),
            param("large", size=(1024, 4096, 8)),
        ]
    )
    def test_relu(self, name, size):
        model = TestModule().cuda().half()
        inputs = (torch.randn(size).half().cuda(),)

        self.run_test(model, inputs, expected_ops={torch.ops.aten.relu.default})

    def test_relu_with_dynamic_shape(self):
        model = TestModule().cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [1, 3, 4],
            ],
            inputs_max=[
                [32, 1024, 2048],
            ],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten.relu.default},
        )
