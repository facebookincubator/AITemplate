# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class PadLastDim(unittest.TestCase):
    def test_static_shape_4d(self):
        NN = 2
        HH = 7
        WW = 7
        CI = 262
        CO = 264
        X = Tensor(shape=[NN, HH, WW, CI], name="X", is_input=True)
        op = ops.pad_last_dim(4, CO)
        Y = op(X)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = gen_execution_module(Y, target, "./tmp", "pad_last_dim4d")

        X_pt = torch.randn(NN, HH, WW, CI).cuda().half()
        Pad_pt = torch.zeros(NN, HH, WW, CO - CI).cuda().half()
        Y_pt = torch.cat([X_pt, Pad_pt], dim=3)

        y = torch.empty([NN, HH, WW, CO]).cuda().half()
        module.RunWithTensors([X_pt], [y])
        self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))

    def test_static_shape_2d(self):
        NN = 32
        CI = 259
        CO = 264
        X = Tensor(shape=[NN, CI], name="X", is_input=True)
        op = ops.pad_last_dim(2, CO)
        Y = op(X)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = gen_execution_module(Y, target, "./tmp", "pad_last_dim2d")

        X_pt = torch.randn(NN, CI).cuda().half()
        Pad_pt = torch.zeros(NN, CO - CI).cuda().half()
        Y_pt = torch.cat([X_pt, Pad_pt], dim=1)

        y = torch.empty([NN, CO]).cuda().half()
        module.RunWithTensors([X_pt], [y])
        self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
