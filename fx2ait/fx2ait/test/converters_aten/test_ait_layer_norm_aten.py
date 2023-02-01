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
from torch import nn


class TestLayernormConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param("1d_normalized_shape", [10], [2, 10]),
            param("1d_normalized_shape_3d_input", [10], [2, 6, 10]),
            param("2d_normalized_shape", [6, 10], [2, 6, 10]),
            # FIXME: Enable test case once layernorm support expand
            # param("2d_normalized_shape", [5, 10], [5, 10]),
        ]
    )
    def test_layer_norm(self, name, normalized_shape, input_shape):
        class TestModule(torch.nn.Module):
            def __init__(self, normalized_shape):
                super().__init__()
                # TODO remove hard code eps once layernorm api expose eps setting
                self.mod = nn.LayerNorm(normalized_shape, eps=1e-5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.mod(x)

        model = TestModule(normalized_shape).cuda().half()
        inputs = [
            torch.randn(input_shape).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={torch.ops.aten.layer_norm.default})

    def test_layer_norm_IntImm_shape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                shape = x.shape
                normalized_shape = shape[1:]
                return torch.nn.functional.layer_norm(x, normalized_shape, eps=1e-5)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn([10, 10]).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={torch.ops.aten.layer_norm.default})

    @parameterized.expand(
        [
            param("1d_normalized_shape", [10], [[2, 10], [12, 10]]),
            param("1d_normalized_shape_3d_input", [10], [[2, 6, 10], [12, 20, 10]]),
            param("2d_normalized_shape", [6, 10], [[2, 6, 10], [12, 6, 10]]),
        ]
    )
    def test_dynamic_layer_norm(self, name, normalized_shape, input_shape):
        class TestModule(torch.nn.Module):
            def __init__(self, normalized_shape):
                super().__init__()
                # TODO remove hard code eps once layernorm api expose eps setting
                self.mod = nn.LayerNorm(normalized_shape, eps=1e-5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.mod(x)

        model = TestModule(normalized_shape).cuda().half()

        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                input_shape[0],
            ],
            inputs_max=[
                input_shape[1],
            ],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={torch.ops.aten.layer_norm.default}
        )

    def test_dynamic_layer_norm_IntImm_shape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                shape = x.shape
                normalized_shape = shape[1:]
                return torch.nn.functional.layer_norm(x, normalized_shape, eps=1e-5)

        model = TestModule().cuda().half()
        inputs = TensorSpec.create_spec_from_shapes(
            inputs_min=[[10, 30]],
            inputs_max=[[20, 30]],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model, inputs, expected_ops={torch.ops.aten.layer_norm.default}
        )
