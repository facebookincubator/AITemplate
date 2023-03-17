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

    def _test_slice_scatter_reshape_float16(
        self,
        input0_shape,
        input1_shape,
        start_indices,
        end_indices,
    ):
        dtype = "float16"

        input0 = Tensor(shape=input0_shape, dtype=dtype, name="input0", is_input=True)
        input1 = Tensor(shape=input1_shape, dtype=dtype, name="input1", is_input=True)

        num_slices = 139
        slice_outputs = [
            ops.dynamic_slice()(
                input0, start_indices=start_indices, end_indices=end_indices
            )
            for _ in range(num_slices)
        ]

        concat_dim = 1
        concat_2 = ops.concatenate()(slice_outputs, concat_dim)
        reshape_to = [-1, num_slices, 2]
        reshape_3 = ops.reshape()(concat_2, reshape_to)

        Y = ops.concatenate()([reshape_3, input1], concat_dim)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        test_name = "slice_scatter_large_inputs"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        Y_src_ops = list(Y._attrs["src_ops"])
        self.assertEqual(len(Y_src_ops), 5)
        self.assertTrue(all(op._attrs["op"] == "concatenate" for op in Y_src_ops))

        input0_pt = get_random_torch_tensor(input0_shape, dtype)
        input1_pt = get_random_torch_tensor(input1_shape, dtype)
        slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]

        slice_outputs_pt = [input0_pt[slice_indices] for _ in range(num_slices)]
        concat_2_pt = torch.cat(slice_outputs_pt, concat_dim)
        reshape_3_pt = torch.reshape(concat_2_pt, reshape_to)
        y_pt = torch.cat([reshape_3_pt, input1_pt], concat_dim)

        inputs = {"input0": input0_pt, "input1": input1_pt}
        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

    def test_slice_scatter_reshape_float16(self):
        self._test_slice_scatter_reshape_float16(
            input0_shape=[6, 2],
            input1_shape=[2, 4, 2],
            start_indices=[1, 0],
            end_indices=[3, None],
        )
        self._test_slice_scatter_reshape_float16(
            input0_shape=[2, 6],
            input1_shape=[2, 4, 2],
            start_indices=[0, 0],
            end_indices=[None, 2],
        )


if __name__ == "__main__":
    unittest.main()
