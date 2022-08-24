# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Unittests for argmax Operator.
"""
import unittest

import torch
from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


class argmaxTestCase(unittest.TestCase):
    def _test_argmax(self, batch_size=1, shape=(2, 6), dim=0, test_name="argmax"):

        o_shape = list(shape)[:-1]

        X1 = Tensor(
            shape=shape,
            dtype="float16",
            name="X",
            is_input=True,
        )
        X4 = ops.argmax(dim=dim)(X1)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        module = gen_execution_module(X4, target, "./tmp", test_name)

        scores = torch.rand(shape).cuda().half()
        y_pt = torch.argmax(scores, dim=dim)
        y = torch.empty_like(y_pt, dtype=torch.int64)

        module.RunWithTensors([scores], [y])
        y_reshape = y.reshape(o_shape)
        self.assertTrue(torch.allclose(y_pt, y_reshape, atol=1e-2, rtol=1e-2))

    def test_argmax(self):
        self._test_argmax(shape=(300, 80), dim=1, test_name="argmax")


if __name__ == "__main__":
    torch.manual_seed(1024)
    unittest.main()
