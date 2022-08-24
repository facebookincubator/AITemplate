# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from parameterized import param, parameterized


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class PermuteTest(unittest.TestCase):
    @parameterized.expand(
        [
            param((0, 2, 1), "permute_1"),
            param((1, 0, 2), "permute_2"),
            param((2, 1, 0), "permute_3"),
        ]
    )
    def test_static_shape_3d(self, dims, testname):
        NN = 80
        WW = 300
        CI = 2
        X = Tensor(shape=[NN, WW, CI], name="X", is_input=True)
        op = ops.permute()
        Y = op(X, dims)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = gen_execution_module(Y, target, "./tmp", testname)

        X_pt = torch.randn(NN, WW, CI).cuda().half()
        Y_pt = torch.permute(X_pt, dims)

        y = torch.empty(Y_pt.size()).cuda().half()
        module.RunWithTensors([X_pt], [y])
        self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
