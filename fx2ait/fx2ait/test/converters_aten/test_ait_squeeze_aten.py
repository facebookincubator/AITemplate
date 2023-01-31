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


class TestSqueezeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param(
                "default",
                dim=None,
                shape=[2, 1, 1, 3],
                expected_ops={torch.ops.aten.squeeze.default},
            ),
            param(
                "1",
                dim=1,
                shape=[2, 1, 1, 3],
                expected_ops={torch.ops.aten.squeeze.dim},
            ),
            param(
                "-1",
                dim=-1,
                shape=[2, 1, 3, 1],
                expected_ops={torch.ops.aten.squeeze.dim},
            ),
        ]
    )
    def test_squeeze(self, name, dim, shape, expected_ops):
        class TestModule(torch.nn.Module):
            def forward(self, y: torch.Tensor) -> torch.Tensor:
                squeeze = (
                    torch.squeeze(y, dim=dim) if dim is not None else torch.squeeze(y)
                )
                return squeeze

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(shape).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops=expected_ops)

    @parameterized.expand(
        [
            param(
                "default",
                dim=None,
                shape1=[[2, 1, 3, 1]],
                shape2=[[4, 1, 10, 1]],
                expected_ops={torch.ops.aten.squeeze.default},
            ),
            param(
                "1",
                dim=1,
                shape1=[[2, 1, 3, 1]],
                shape2=[[4, 1, 10, 1]],
                expected_ops={torch.ops.aten.squeeze.dim},
            ),
            param(
                "-1",
                dim=-1,
                shape1=[[2, 1, 3, 1]],
                shape2=[[4, 1, 10, 1]],
                expected_ops={torch.ops.aten.squeeze.dim},
            ),
        ]
    )
    def test_dynamic_squeeze(self, name, dim, shape1, shape2, expected_ops):
        class TestModule(torch.nn.Module):
            def forward(self, y: torch.Tensor) -> torch.Tensor:
                squeeze = (
                    torch.squeeze(y, dim=dim) if dim is not None else torch.squeeze(y)
                )
                return squeeze

        model = TestModule().cuda().half()

        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=shape1,
            inputs_max=shape2,
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(model, inputs_spec, expected_ops=expected_ops)


class TestUnSqueezeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param("1", dim=1, shape=[2, 1, 1, 3]),
            param("-1", dim=-1, shape=[2, 1, 3, 1]),
        ]
    )
    def test_unsqueeze(self, name, dim, shape):
        class TestModule(torch.nn.Module):
            def forward(self, y: torch.Tensor) -> torch.Tensor:
                unsqueeze = (
                    torch.unsqueeze(y, dim=dim)
                    if dim is not None
                    else torch.unsqueeze(y)
                )
                return unsqueeze

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(shape).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={torch.ops.aten.unsqueeze.default})

    @parameterized.expand(
        [
            param(
                "1",
                dim=1,
                shape1=[[2, 1, 3, 1]],
                shape2=[[4, 1, 10, 1]],
            ),
            param(
                "-1",
                dim=-1,
                shape1=[[2, 1, 3, 1]],
                shape2=[[4, 1, 10, 1]],
            ),
        ]
    )
    def test_dynamic_squeeze(self, name, dim, shape1, shape2):
        class TestModule(torch.nn.Module):
            def forward(self, y: torch.Tensor) -> torch.Tensor:
                unsqueeze = (
                    torch.unsqueeze(y, dim=dim)
                    if dim is not None
                    else torch.unsqueeze(y)
                )
                return unsqueeze

        model = TestModule().cuda().half()

        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=shape1,
            inputs_max=shape2,
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={torch.ops.aten.unsqueeze.default}
        )
