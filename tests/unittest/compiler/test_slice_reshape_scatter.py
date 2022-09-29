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

logger = logging.getLogger(__name__)


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
    ):
        target = detect_target()

        input_X_pt = torch.randn(input_x_shape).cuda().half()

        Ys_pt = []
        Xs_pt = []
        for input_shape, start_indices, end_indices in zip(
            input_shapes, input_start_indices, input_end_indices
        ):
            X_pt = torch.randn(input_shape).cuda().half()
            Xs_pt.append(X_pt)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
            Y_pt = X_pt[slice_indices]
            Ys_pt.append(Y_pt)
        Y1_pt = torch.cat(Ys_pt, dim)
        Y2_pt = torch.reshape(Y1_pt, reshape_to)
        Y_pt = torch.cat([Y2_pt, input_X_pt], dim=dim)
        if add_tanh:
            Y_pt = torch.tanh(Y_pt)

        input_X = Tensor(
            shape=input_x_shape, dtype="float16", name="input_x", is_input=True
        )
        Ys = []
        for idx, (input_shape, start_indices, end_indices) in enumerate(
            zip(input_shapes, input_start_indices, input_end_indices)
        ):
            slice_op = ops.dynamic_slice()
            X_name = "input_{}".format(idx)
            X = Tensor(shape=input_shape, dtype="float16", name=X_name, is_input=True)
            Y = slice_op(X, start_indices=start_indices, end_indices=end_indices)
            Ys.append(Y)
        concat_op = ops.concatenate()
        Y1 = concat_op(Ys, dim)
        Y2 = ops.reshape()(Y1, reshape_to)
        concat_op_2 = ops.concatenate()
        if add_tanh:
            concat_op_2 = ops.concatenate_tanh()
        Y = concat_op_2([Y2, input_X], dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logger.info(
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
        np.testing.assert_equal(concat_op_2._attrs["input_masks"], [False, True])
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
        y = torch.empty(y_shape).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_slice_scatter_reshape(self):
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


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
