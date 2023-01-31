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
from typing import Callable, List, Union

import torch
from fx2ait.acc_tracer import acc_ops, ait_acc_ops

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


class TestPermuteConverter(AITTestCase):
    def test_permute021(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.permute(x, [0, 2, 1])

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 3, 4).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.permute})

    def test_permute021_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.permute(x, [0, 2, 1])

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
            model, inputs_spec, expected_ops={acc_ops.permute}
        )


class TestCatConverter(AITTestCase):
    combo = [
        ["default", 0, torch.cat],
        ["positive_dim", 1, torch.cat],
        ["negative_dim", -1, torch.cat],
        ["default", 0, torch.concat],
        ["positive_dim", 1, torch.concat],
        ["negative_dim", -1, torch.concat],
    ]

    @parameterized.expand([(name, dim, op) for name, dim, op in combo])
    def test_cat(self, name: str, dim: int, op: Callable):
        class TestModule(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
            ) -> torch.Tensor:
                return op([x, y, z], dim=dim)

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 3, 4).half().cuda(),
            torch.randn(2, 3, 4).half().cuda(),
            torch.randn(2, 3, 4).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.cat})

    @parameterized.expand([(name, dim, op) for name, dim, op in combo])
    def test_cat_dynamic_shape(self, name: str, dim: int, op: Callable):
        class TestModule(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
            ) -> torch.Tensor:
                return op([x, y, z], dim=dim)

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

        self.run_test_with_dynamic_shape(model, inputs_spec, expected_ops={acc_ops.cat})


class TestReshapeConverter(AITTestCase):
    @parameterized.expand(
        [
            [[2, 3, 4], [6, 4]],
            [[2, 3, 4], [2, 12]],
            [[2, 3, 4], [24]],
            [[2, 3, 4], [-1, 4]],
            [[2, 3, 4], [2, -1]],
            [[2, 3, 4], [-1]],
        ]
    )
    def test_simple(self, original_shape: List[int], final_shape: List[int]) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.reshape(x, final_shape)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*original_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.reshape})

    def test_with_getitem_size(self) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                d0 = y.size(dim=0)
                d1 = y.size(dim=1)
                return x.reshape(d0, d1)

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 3, 4).half().cuda(),
            torch.randn(6, 4).half().cuda(),
        ]
        self.run_test(
            model, inputs, expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem}
        )

    def test_with_getitem_reshape_dim0(self) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d0 = x.size(dim=0)
                d1 = x.size(dim=1)
                d2 = x.size(dim=2)
                d = d1 * d2
                return x.reshape(d0, d)

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 3, 4).half().cuda(),
        ]
        self.run_test(
            model, inputs, expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem}
        )

    def test_with_getitem_reshape_dim0_dynamic(self) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d0 = x.size(dim=0)
                d1 = x.size(dim=1)
                d2 = x.size(dim=2)
                d = d1 * d2
                return x.reshape(d0, d)

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
            model,
            inputs_spec,
            expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem},
        )

    ###TODO dim=0,1 dynamic has problem due to output size is not IntVar for dim1(P537903486).
    # def test_with_getitem_reshape_dim01_dynamic(self) -> None:
    #     class TestModule(torch.nn.Module):
    #         def forward(self, x: torch.Tensor) -> torch.Tensor:
    #             d0 = x.size(dim=0)
    #             d1 = x.size(dim=1)
    #             d2 = x.size(dim=2)
    #             d = d1 * d2
    #             return x.reshape(d0, d)

    #     model = TestModule().cuda()
    #     inputs = [
    #         [
    #             torch.randn(2, 3, 4).half().cuda(),
    #         ],
    #         [
    #             torch.randn(20, 30, 4).half().cuda(),
    #         ],
    #     ]
    #     self.run_test_with_dynamic_shape(
    #         model, inputs, expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem}
    #     )


class TestTopkConverter(AITTestCase):
    @parameterized.expand(
        [
            [[4], 1],
            [[6], 3],
            [[6], 6],
        ]
    )
    def test_simple(self, input: List[int], k: int) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                values, indices = torch.topk(x, k)
                return indices

        model = TestModule().cuda()
        inputs = [
            torch.randn(*input).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.topk})

    @parameterized.expand(
        [
            [[2, 4], 1],
            [[2, 4], 2],
            [[3, 3], 3],
        ]
    )
    def test_multi_dimensional(self, input: List[int], k: int) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                values, indices = torch.topk(x, k)
                return indices

        model = TestModule().cuda()
        inputs = [
            torch.randn(*input).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.topk})

    ##TODO results mismatch.(P537992074)
    # def test_multi_dimensional_dynamic_shape(self) -> None:
    #     class TestModule(torch.nn.Module):
    #         def forward(self, x: torch.Tensor) -> torch.Tensor:
    #             values, indices = torch.topk(x, 1)
    #             return indices

    #     model = TestModule().cuda()
    #     inputs = [
    #         [
    #             torch.randn((2, 4)).half().cuda(),
    #         ],
    #         [
    #             torch.randn((20, 4)).half().cuda(),
    #         ],
    #     ]
    #     self.run_test_with_dynamic_shape(model, inputs, expected_ops={acc_ops.topk})


class TestSplitConverter(AITTestCase):
    @parameterized.expand(
        [
            [[2, 10], [2, 3, 5]],
            [[2, 10], 2],
            [[2, 10], 3],
        ]
    )
    def test_with_dim(
        self, input_shape: List[int], split_size_or_sections: Union[int, List[int]]
    ) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.split(x, split_size_or_sections, dim=1)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*input_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={ait_acc_ops.split})

    @parameterized.expand(
        [
            [[10], [2, 3, 5]],
            [[10], 2],
            [[10], 3],
        ]
    )
    def test_without_dim(
        self, input_shape: List[int], split_size_or_sections: Union[int, List[int]]
    ) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.split(x, split_size_or_sections)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*input_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={ait_acc_ops.split})

    def test_with_dim_dynamic_shape(self) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.split(x, 2, dim=1)

        model = TestModule().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 10],
            ],
            inputs_max=[
                [20, 10],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={ait_acc_ops.split}
        )
