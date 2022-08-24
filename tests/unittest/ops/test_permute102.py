# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class Permute102(unittest.TestCase):
    def test_static_shape_3d(self):
        NN = 80
        WW = 300
        CI = 2
        X = Tensor(shape=[NN, WW, CI], name="X", is_input=True)
        op = ops.permute102()
        Y = op(X)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = gen_execution_module(Y, target, "./tmp", "perm102")

        X_pt = torch.randn(NN, WW, CI).cuda().half()
        Y_pt = torch.permute(X_pt, [1, 0, 2])
        y = torch.empty([WW, NN, CI]).cuda().half()
        module.RunWithTensors([X_pt], [y])
        self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
