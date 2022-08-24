# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
# _3306 = _3305.permute(0, 2, 1)  # Transpose
# _3307 = _3306  # torch.reshape(_3306, (-1, 745))  # Reshape
# _3308 = torch.nn.functional.linear(_3307, self._1184, bias=self._1185)  # FC
"""


import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class BMMTestCase(unittest.TestCase):
    def test_ccr(self):
        B = 1024
        M = 128
        K = 745
        # K = 752
        N = 30
        target = detect_target()
        X = Tensor(shape=[B, K, M], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[1, N, K], dtype="float16", name="input_1", is_input=True)
        OP = ops.perm021fc_ccr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(Y, target, "./tmp", "perm021_fc")

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()

        XT = X_pt.permute(0, 2, 1)
        XT = torch.reshape(XT, (-1, K))
        Y_pt = torch.nn.functional.linear(XT, W_pt)
        Y_pt = torch.reshape(Y_pt, (B, M, N))
        y = torch.empty([B, M, N]).cuda().half()
        module.RunWithTensors({"input_0": X_pt, "input_1": W_pt.unsqueeze(0)}, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
