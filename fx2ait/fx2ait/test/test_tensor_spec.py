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
import unittest

import torch
from aitemplate.compiler.public import IntImm, IntVar
from fx2ait.tensor_spec import TensorSpec
from parameterized import parameterized


class TestTensorSpec(unittest.TestCase):
    def test_two_input_lists(self):
        inputs1 = [
            torch.empty([1, 3, 4], dtype=torch.float16),
            torch.empty([5, 6], dtype=torch.int32),
            torch.empty([7, 128, 9], dtype=torch.float16),
        ]
        inputs2 = [
            torch.empty([32, 3, 4], dtype=torch.float16),
            torch.empty([5, 6], dtype=torch.int32),
            torch.empty([7, 1, 9], dtype=torch.float16),
        ]

        specs = TensorSpec.from_two_input_lists(inputs1, inputs2)

        self.assertEqual(3, len(specs))
        self.assertEqual(
            TensorSpec(
                [IntVar([1, 32], "dynamic_dim_0"), IntImm(3), IntImm(4)], torch.float16
            ),
            specs[0],
        )
        self.assertEqual(TensorSpec([IntImm(5), IntImm(6)], torch.int32), specs[1])
        self.assertEqual(
            TensorSpec(
                [IntImm(7), IntVar([1, 128], "dynamic_dim_1"), IntImm(9)], torch.float16
            ),
            specs[2],
        )

    @parameterized.expand(
        [
            ("single", [([10, 3, 4], torch.float16)]),
            (
                "multi",
                [
                    ([10, 3, 4], torch.float16),
                    ([10, 6], torch.int32),
                    ([10, 8, 9], torch.float16),
                ],
            ),
            (
                "different_bs_dim",
                [
                    ([10, 3, 4], torch.float16),
                    ([10, 6], torch.int32),
                    ([4, 10, 9], torch.float16),
                ],
            ),
        ]
    )
    def test_input_list_with_batch_size(self, _, settings):
        inputs = [torch.empty(setting[0], dtype=setting[1]) for setting in settings]
        # Test case default batch_size = 10, avoid set other shape param with this value
        batch_size = 10

        specs = TensorSpec.from_input_list_with_batch_size(inputs, 32)
        self.assertEqual(len(settings), len(specs))
        for index, setting in enumerate(settings):
            expected_shape = setting[0]
            expected_spec = []
            for shape in expected_shape:
                if shape == batch_size:
                    expected_spec.append(IntVar([1, 32], "batch_size"))
                else:
                    expected_spec.append(IntImm(shape))

            self.assertEqual(
                TensorSpec(expected_spec, setting[1]),
                specs[index],
            )

    def test_input_list_with_batch_size_non_default_dim(self):
        inputs = [
            torch.empty([2, 10, 4], dtype=torch.float16),
            torch.empty([5, 10], dtype=torch.int32),
            torch.empty([7, 10, 9], dtype=torch.float16),
        ]

        specs = TensorSpec.from_input_list_with_batch_size(inputs, 32, 1)
        self.assertEqual(3, len(specs))
        self.assertEqual(
            TensorSpec(
                [IntImm(2), IntVar([1, 32], "batch_size"), IntImm(4)], torch.float16
            ),
            specs[0],
        )
        self.assertEqual(
            TensorSpec([IntImm(5), IntVar([1, 32], "batch_size")], torch.int32),
            specs[1],
        )
        self.assertEqual(
            TensorSpec(
                [IntImm(7), IntVar([1, 32], "batch_size"), IntImm(9)], torch.float16
            ),
            specs[2],
        )

    def test_input_with_no_bs_tensor(self):
        inputs = [
            torch.empty([2, 10, 4], dtype=torch.float16),
            torch.empty([20], dtype=torch.int32),
            torch.empty([7, 10, 9], dtype=torch.float16),
            torch.empty([20, 7, 10, 9], dtype=torch.float16),
        ]

        specs = TensorSpec.from_input_list_with_batch_size(inputs, 32, 1)
        self.assertEqual(4, len(specs))
        self.assertEqual(
            TensorSpec(
                [IntImm(2), IntVar([1, 32], "batch_size"), IntImm(4)], torch.float16
            ),
            specs[0],
        )
        self.assertEqual(
            TensorSpec([IntImm(20)], torch.int32),
            specs[1],
        )
        self.assertEqual(
            TensorSpec(
                [IntImm(7), IntVar([1, 32], "batch_size"), IntImm(9)], torch.float16
            ),
            specs[2],
        )
        self.assertEqual(
            TensorSpec(
                [IntImm(20), IntImm(7), IntVar([1, 32], "batch_size"), IntImm(9)],
                torch.float16,
            ),
            specs[3],
        )
