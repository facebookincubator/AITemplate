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
from aitemplate.compiler.ops.common.epilogue import FuncEnum

from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class SizeOpTestCase(unittest.TestCase):
    def _test_size_op(
        self,
        batch_size=(1, 3),
        X_shape=(16, 32, 64),
        Y_shape=(-1, 16, 16, 128),
        test_name="size_op",
    ):
        target = detect_target()
        b_dim = shape_utils.gen_int_var_min_max(batch_size, name="input_batch")
        X = Tensor(
            shape=[b_dim, *X_shape],
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        Y1 = ops.size()(X)
        Y2 = ops.getitem()(Y1, 0)
        Y = ops.reshape()(X, [Y2, -1, X_shape[-1]])

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = torch.randn(X_shape_pt).cuda().half()
            Y_pt = X_pt.reshape(b, -1, X_shape_pt[-1])

            y = torch.empty(Y_pt.size()).cuda().half()
            module.run_with_tensors([X_pt], [y])

            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def _test_size_op_2(
        self,
        batch_size=(1, 3),
        X_shape=(16, 32, 64),
        Y_shape=(-1, 16, 16, 128),
        test_name="tensor_size_op",
        copy_op=False,
    ):
        target = detect_target()
        X1 = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), *X_shape],
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        Y1_op = ops.flatten(1, -1)
        Y2_op = ops.flatten(1, -1)
        if copy_op:
            Y1_op = ops.flatten(**Y1_op._get_op_attributes())
            Y2_op = ops.flatten(**Y2_op._get_op_attributes())
        Y1 = Y1_op(ops.elementwise(FuncEnum.ADD)(X1, X1))
        Y2 = Y2_op(ops.elementwise(FuncEnum.MUL)(X1, X1))
        Y3 = ops.concatenate()([Y1, Y2], 0)
        dim = ops.size()(Y3, -4)  # test negative dim
        Y = ops.reshape()(Y2, [dim, -1])

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", test_name)

        self.assertEqual(len(module.debug_sorted_graph), 6)

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = torch.randn(X_shape_pt).cuda().half()
            Y2_pt = X_pt * X_pt
            Y_pt = Y2_pt.reshape(2 * b, -1)

            y = torch.empty(Y_pt.size()).cuda().half()
            module.run_with_tensors([X_pt], [y])

            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_size_op(self):
        self._test_size_op(test_name="size_op_0")
        self._test_size_op([1], (4, 8, 8), (-1,), "size_op_static")
        self._test_size_op([4, 2], (4, 8, 8), (-1,), "size_op_1")
        self._test_size_op([3, 1], (5, 4, 16), (-1, 8), "size_op_2")

        self._test_size_op_2(test_name="size_op_3")
        self._test_size_op_2(test_name="size_op_3_copy_op", copy_op=True)


if __name__ == "__main__":
    unittest.main()
