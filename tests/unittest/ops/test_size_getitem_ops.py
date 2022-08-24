# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum

from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target, gen_execution_module
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

        module = gen_execution_module(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = torch.randn(X_shape_pt).cuda().half()
            Y_pt = X_pt.reshape(b, -1, X_shape_pt[-1])

            y = torch.empty(Y_pt.size()).cuda().half()
            module.RunWithTensors([X_pt], [y])

            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def _test_size_op_2(
        self,
        batch_size=(1, 3),
        X_shape=(16, 32, 64),
        Y_shape=(-1, 16, 16, 128),
        test_name="tensor_size_op",
    ):
        target = detect_target()
        X1 = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), *X_shape],
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        Y1 = ops.flatten(1, -1)(ops.elementwise(FuncEnum.ADD)(X1, X1))
        Y2 = ops.flatten(1, -1)(ops.elementwise(FuncEnum.MUL)(X1, X1))
        Y3 = ops.concatenate()([Y1, Y2], 0)
        dim = ops.size()(Y3, -4)  # test negative dim
        Y = ops.reshape()(Y2, [dim, -1])

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = gen_execution_module(Y, target, "./tmp", test_name)

        self.assertEqual(len(module.debug_sorted_graph), 6)

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = torch.randn(X_shape_pt).cuda().half()
            Y2_pt = X_pt * X_pt
            Y_pt = Y2_pt.reshape(2 * b, -1)

            y = torch.empty(Y_pt.size()).cuda().half()
            module.RunWithTensors([X_pt], [y])

            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_size_op(self):
        self._test_size_op(test_name="size_op_0")
        self._test_size_op([1], (4, 8, 8), (-1,), "size_op_static")
        self._test_size_op([4, 2], (4, 8, 8), (-1,), "size_op_1")
        self._test_size_op([3, 1], (5, 4, 16), (-1, 8), "size_op_2")

        self._test_size_op_2(test_name="size_op_3")


if __name__ == "__main__":
    unittest.main()
