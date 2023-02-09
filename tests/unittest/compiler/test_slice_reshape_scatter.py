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
import logging
import unittest

import numpy as np
import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


_LOGGER = logging.getLogger(__name__)


class SliceScatterReshapeCatTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceScatterReshapeCatTestCase, self).__init__(*args, **kwargs)
        self.test_count = 1

    def _run_one_test(
        self,
        *,
        input_shapes,
        input_start_indices,
        input_end_indices,
        reshape_to,
        input_x_shape,
        dim,
        add_tanh=False,
        dtype="float16",
    ):
        target = detect_target()

        input_X_pt = get_random_torch_tensor(input_x_shape, dtype)

        Ys_pt = []
        Xs_pt = []
        for input_shape, start_indices, end_indices in zip(
            input_shapes, input_start_indices, input_end_indices
        ):
            X_pt = get_random_torch_tensor(input_shape, dtype)
            Xs_pt.append(X_pt)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
            Y_pt = X_pt[slice_indices]
            Ys_pt.append(Y_pt)
        Y1_pt = torch.cat(Ys_pt, dim)
        Y2_pt = torch.reshape(Y1_pt, reshape_to)
        Y_pt = torch.cat([input_X_pt, Y2_pt, input_X_pt], dim=dim)
        if add_tanh:
            Y_pt = torch.tanh(Y_pt)

        input_X = Tensor(
            shape=input_x_shape, dtype=dtype, name="input_x", is_input=True
        )
        Ys = []
        for idx, (input_shape, start_indices, end_indices) in enumerate(
            zip(input_shapes, input_start_indices, input_end_indices)
        ):
            slice_op = ops.dynamic_slice()
            X_name = "input_{}".format(idx)
            X = Tensor(shape=input_shape, dtype=dtype, name=X_name, is_input=True)
            Y = slice_op(X, start_indices=start_indices, end_indices=end_indices)
            Ys.append(Y)
        concat_op = ops.concatenate()
        Y1 = concat_op(Ys, dim)
        Y2 = ops.reshape()(Y1, reshape_to)
        concat_op_2 = ops.concatenate()
        if add_tanh:
            concat_op_2 = ops.concatenate_tanh()
        Y = concat_op_2([input_X, Y2, input_X], dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info(
            "AITemplate output_0 shape: {}, pt shape: {}".format(y_shape, Y_pt.size())
        )
        np.testing.assert_equal(y_shape, Y_pt.size())

        dll_name = f"test_{self.test_count}.so"
        module = compile_model(
            Y, target, "./tmp", "slice_scatter_reshape_cat", dll_name=dll_name
        )
        Y_src_ops = Y._attrs["src_ops"]
        np.testing.assert_equal(len(Y_src_ops), 2)
        np.testing.assert_equal(concat_op_2 in Y_src_ops, True)
        np.testing.assert_equal(concat_op_2._attrs["input_masks"], [True, False, True])
        Y_src_ops_list = list(Y_src_ops)
        slice_reshape_scatter_op = (
            Y_src_ops_list[1] if concat_op_2 == Y_src_ops_list[0] else Y_src_ops_list[0]
        )
        np.testing.assert_equal(
            slice_reshape_scatter_op._attrs["op"], "slice_reshape_scatter"
        )

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(Xs_pt) + 1)]
        for i, X_pt in enumerate(Xs_pt):
            inputs[input_name_to_index[f"input_{i}"]] = X_pt
        inputs[input_name_to_index["input_x"]] = input_X_pt
        y = get_torch_empty_tensor(y_shape, dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by cuda sm<80",
    )
    def test_slice_scatter_reshape(self):
        self._run_one_test(
            input_shapes=[[1, 2], [1, 2]],
            input_start_indices=[[0, 0], [0, 0]],
            input_end_indices=[[1, 2], [1, 2]],
            reshape_to=[1, 2, 2],
            input_x_shape=[1, 1, 2],
            dim=1,
        )
        self._run_one_test(
            input_shapes=[[10, 20], [15, 44]],
            input_start_indices=[[1, 5], [2, 10]],
            input_end_indices=[[4, 15], [5, 22]],
            reshape_to=[3, 2, 11],
            input_x_shape=[3, 1, 11],
            dim=1,
        )
        self._run_one_test(
            input_shapes=[[8, 16], [20, 30]],
            input_start_indices=[[0, 4], [12, 2]],
            input_end_indices=[[4, 14], [16, 8]],
            reshape_to=[4, 2, 8],
            input_x_shape=[4, 5, 8],
            dim=1,
        )
        self._run_one_test(
            input_shapes=[[8, 16], [20, 30]],
            input_start_indices=[[0, 4], [12, 2]],
            input_end_indices=[[4, 14], [16, 8]],
            reshape_to=[4, 2, 8],
            input_x_shape=[4, 5, 8],
            dim=1,
            add_tanh=True,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_slice_scatter_reshape_float(self):
        self._run_one_test(
            input_shapes=[[8, 16], [20, 30]],
            input_start_indices=[[0, 4], [12, 2]],
            input_end_indices=[[4, 14], [16, 8]],
            reshape_to=[4, 2, 8],
            input_x_shape=[4, 5, 8],
            dim=1,
            dtype="float",
        )

    def test_slice_scatter_reshape_float16_2(self):
        dtype = "float16"
        input_shape = [2, 6]
        input0 = Tensor(shape=input_shape, dtype=dtype, name="input0", is_input=True)
        input1 = Tensor(shape=input_shape, dtype=dtype, name="input1", is_input=True)
        input2_shape = [2, 3, 2]
        input2 = Tensor(shape=input2_shape, dtype=dtype, name="input2", is_input=True)

        start_indices = [0, 0]
        end_indices = [None, 2]
        slice_0 = ops.dynamic_slice()(
            input0, start_indices=start_indices, end_indices=end_indices
        )
        slice_1 = ops.dynamic_slice()(
            input0, start_indices=start_indices, end_indices=end_indices
        )
        concat_dim = 1
        concat_2 = ops.concatenate()([slice_0, slice_1], concat_dim)
        reshape_to = [-1, 2, 2]
        reshape_3 = ops.reshape()(concat_2, reshape_to)

        slice_4 = ops.dynamic_slice()(
            input1, start_indices=start_indices, end_indices=end_indices
        )
        slice_5 = ops.dynamic_slice()(
            input1, start_indices=start_indices, end_indices=end_indices
        )
        concat_6 = ops.concatenate()([slice_4, slice_5], concat_dim)
        reshape_7 = ops.reshape()(concat_6, reshape_to)

        Y = ops.concatenate()([input2, reshape_3, reshape_7], concat_dim)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = "test.so"
        test_name = "slice_scatter_reshape_cat_float16_2"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        Y_src_ops = Y._attrs["src_ops"]
        self.assertEqual(len(Y_src_ops), 3)
        slice_reshape_scatter_cnt = 0
        for op in Y_src_ops:
            if op._attrs["op"] == "slice_reshape_scatter":
                slice_reshape_scatter_cnt += 1
        self.assertEqual(slice_reshape_scatter_cnt, 2)

        slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]

        input0_pt = get_random_torch_tensor(input_shape, dtype)
        slice_0_pt = input0_pt[slice_indices]
        slice_1_pt = input0_pt[slice_indices]
        concat_2_pt = torch.cat([slice_0_pt, slice_1_pt], concat_dim)
        reshape_3_pt = torch.reshape(concat_2_pt, reshape_to)

        input1_pt = get_random_torch_tensor(input_shape, dtype)
        slice_4_pt = input1_pt[slice_indices]
        slice_5_pt = input1_pt[slice_indices]
        concat_6_pt = torch.cat([slice_4_pt, slice_5_pt], concat_dim)
        reshape_7_pt = torch.reshape(concat_6_pt, reshape_to)

        input2_pt = get_random_torch_tensor(input2_shape, dtype)
        y_pt = torch.cat([input2_pt, reshape_3_pt, reshape_7_pt], concat_dim)

        inputs = {"input0": input0_pt, "input1": input1_pt, "input2": input2_pt}
        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
