# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import graph_utils, shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class RemoveUnusedOpsTestCase(unittest.TestCase):
    def _test_remove_unused_ops(
        self,
        batch_size=(1, 3),
        X_shape=(5, 10),
        test_name="test_remove_unused_ops",
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
        Y2 = ops.getitem()(Y1, 1)
        CONST_X = Tensor(
            shape=[],
            dtype="float16",
            name="input_1",
            is_input=True,
            value=Y2._attrs["int_var"].value(),
        )
        Y = ops.elementwise(FuncEnum.ADD)(X, CONST_X)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = gen_execution_module(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = torch.randn(X_shape_pt).cuda().half()
            Y_pt = X_pt + X_pt.size(1)

            y = torch.empty(Y_pt.size()).cuda().half()
            module.RunWithTensors([X_pt], [y])

            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
            self.assertEqual(len(module.debug_sorted_graph), 2)
            self.assertEqual(
                len(graph_utils.get_sorted_ops(module.debug_sorted_graph)), 1
            )

    def test_remove_unused_ops(self):
        self._test_remove_unused_ops()


if __name__ == "__main__":
    unittest.main()
