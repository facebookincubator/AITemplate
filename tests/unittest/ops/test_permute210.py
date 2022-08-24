# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import itertools
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


class Permute210(unittest.TestCase):
    def test_static_shape_3d(self):
        for NWC in itertools.product([2, 80, 300], [2, 80, 300], [2, 80, 300]):
            with self.subTest(NWC=NWC):
                NN, WW, CI = NWC
                X = Tensor(shape=[NN, WW, CI], name="X", is_input=True)
                op = ops.permute210()
                Y = op(X)
                Y._attrs["is_output"] = True
                Y._attrs["name"] = "output"
                target = detect_target()
                module = gen_execution_module(
                    Y, target, "./tmp", "perm210_{}_{}_{}".format(NN, WW, CI)
                )

                X_pt = torch.randn(NN, WW, CI).cuda().half()
                Y_pt = torch.permute(X_pt, [2, 1, 0])
                y = torch.empty([CI, WW, NN]).cuda().half()
                module.RunWithTensors([X_pt], [y])
                self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
