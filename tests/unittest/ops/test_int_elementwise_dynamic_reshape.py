#  Copyright (c) Meta Platform, Inc. and its affiliates.
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

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class IntElementwiseReshapeOpTestCase(unittest.TestCase):
    def test_int_elementwise_reshape_op(
        self,
        batch_size=(1, 3),
        x1_size=(2, 3),
        X_shape=(32, 64),
        test_name="elementwise_reshape_op",
    ):
        target = detect_target()
        b_dim = shape_utils.gen_int_var_min_max(batch_size, name="input_batch")
        x1_dim = shape_utils.gen_int_var_min_max(x1_size, name="input_size")
        X = Tensor(
            shape=[b_dim, x1_dim, *X_shape],
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        Y1 = ops.size()(X)
        Y2 = ops.getitem()(Y1, 0)
        Y3 = ops.getitem()(Y1, 1)
        Y4 = ops.getitem()(Y1, 2)
        Y5 = ops.getitem()(Y1, 3)
        Y6 = Y2 * Y3  # infer_shape intvar[2,9]
        Y = ops.reshape()(X, [Y6, Y4, Y5])
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        for b, x1 in zip(batch_size, x1_size):
            X_shape_pt = (b, x1, *X_shape)
            X_pt = torch.randn(X_shape_pt).cuda().half()
            Y_pt = X_pt.reshape(
                X_shape_pt[1] * X_shape_pt[0], X_shape_pt[2], X_shape_pt[3]
            )

            y = torch.empty(Y_pt.size()).cuda().half()
            module.run_with_tensors([X_pt], [y])

            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_int_elementwise_reshape_op2(
        self,
        batch_size=(1, 3),
        x1_size=(2, 3),
        x2_size=(10, 32),
        x3_size=(48, 64),
        test_name="elementwise_reshape_op2",
    ):
        target = detect_target()
        b_dim = shape_utils.gen_int_var_min_max(batch_size, name="input_batch")
        x1_dim = shape_utils.gen_int_var_min_max(x1_size, name="x1_size")
        x2_dim = shape_utils.gen_int_var_min_max(x2_size, name="x2_size")
        x3_dim = shape_utils.gen_int_var_min_max(x3_size, name="x3_size")
        X = Tensor(
            shape=[b_dim, x1_dim, x2_dim, x3_dim],
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        Y1 = ops.size()(X)
        Y2 = ops.getitem()(Y1, 0)
        Y3 = ops.getitem()(Y1, 1)
        Y4 = ops.getitem()(Y1, 2)
        Y5 = ops.getitem()(Y1, 3)
        f1 = ops.int_elementwise(FuncEnum.MUL)(Y4, Y5)

        Y = ops.reshape()(X, [Y2 * Y3 * f1 / Y5, Y5])
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", test_name)

        for b, x1, x2, x3 in zip(batch_size, x1_size, x2_size, x3_size):
            X_shape_pt = (b, x1, x2, x3)
            X_pt = torch.randn(X_shape_pt).cuda().half()
            Y_pt = X_pt.reshape(-1, X_shape_pt[3])

            y = torch.empty(Y_pt.size()).cuda().half()
            module.run_with_tensors([X_pt], [y])

            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
