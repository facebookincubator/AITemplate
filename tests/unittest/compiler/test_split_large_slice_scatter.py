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

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


class SliceScatterLargeInputsTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceScatterLargeInputsTestCase, self).__init__(*args, **kwargs)
        self.test_count = 1

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_slice_scatter(
        self, input_shape, start_indices, end_indices, concat_dim, dtype
    ):
        num_slices = 140
        slice_outputs = [
            ops.dynamic_slice()(
                Tensor(
                    shape=input_shape, dtype=dtype, name=f"input{idx}", is_input=True
                ),
                start_indices=start_indices,
                end_indices=end_indices,
            )
            for idx in range(num_slices)
        ]

        Y = ops.concatenate()(slice_outputs, concat_dim)

        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        test_name = f"slice_scatter_large_inputs_{self.test_count}"

        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        Y_src_ops = list(Y._attrs["src_ops"])
        self.assertEqual(len(Y_src_ops), 5)
        self.assertTrue(all(op._attrs["op"] == "slice_scatter" for op in Y_src_ops))

        input_pt = [
            get_random_torch_tensor(input_shape, dtype) for _ in range(num_slices)
        ]
        slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
        slice_outputs_pt = [input_i[slice_indices] for input_i in input_pt]
        y_pt = torch.cat(slice_outputs_pt, concat_dim)

        inputs = {f"input{idx}": input_pt[idx] for idx in range(num_slices)}
        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

        self.test_count += 1

    def test_slice_scatter_float(self):
        self._test_slice_scatter(
            input_shape=[3, 7, 10],
            start_indices=[0, 0, 0],
            end_indices=[2, 1, 4],
            concat_dim=0,
            dtype="float",
        )
        self._test_slice_scatter(
            input_shape=[3, 7, 10],
            start_indices=[0, 0, 0],
            end_indices=[2, 1, 4],
            concat_dim=1,
            dtype="float",
        )
        self._test_slice_scatter(
            input_shape=[3, 7, 10],
            start_indices=[0, 0, 0],
            end_indices=[2, 1, 4],
            concat_dim=2,
            dtype="float",
        )
        self._test_slice_scatter(
            input_shape=[3, 7, 10],
            start_indices=[0, 0, 0],
            end_indices=[2, 1, 4],
            concat_dim=1,
            dtype="float16",
        )


if __name__ == "__main__":
    unittest.main()
