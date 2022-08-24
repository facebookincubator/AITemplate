# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
C[m, b, n](row) = bmm(A[m, b, k](row), B[b, k, n](row))
in torch it is
# _2905_2929 = _2904.view(B, 25, -1).permute(1, 0, 2)
# _2930_2954 = torch.baddbmm(
#      self._1085_1133, _2905_2929, self._1084_1132) # baddbmm(bias, X, W)
"""


import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class Perm102BMMTestCase(unittest.TestCase):
    def test_perm102_bmm_rrr(self):
        B = 25
        M = 128
        K = 256
        N = 100
        target = detect_target()
        X = Tensor(shape=[M, B, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[B, K, N], dtype="float16", name="input_1", is_input=True)
        OP = ops.perm102_bmm_rrr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(Y, target, "./tmp", "perm102_bmm_rrr")

        X_pt = torch.randn(M, B, K).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()

        XT = X_pt.permute(1, 0, 2)
        Y_pt = torch.bmm(XT, W_pt)
        Y_pt = Y_pt.permute(1, 0, 2)
        y = torch.empty([M, B, N]).cuda().half()
        module.RunWithTensors({"input_0": X_pt, "input_1": W_pt}, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class Perm102BMMBiasTestCase(unittest.TestCase):
    def test_perm102_bmm_rrr_bias(self):
        B = 25
        M = 128
        K = 256
        N = 100
        target = detect_target()
        X = Tensor(shape=[M, B, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[B, K, N], dtype="float16", name="input_1", is_input=True)
        BIAS = Tensor(shape=[B, N], dtype="float16", name="input_2", is_input=True)
        OP = ops.perm102_bmm_rrr_bias()
        Y = OP(X, W, BIAS)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(Y, target, "./tmp", "perm102_bmm_rrr_bias")

        X_pt = torch.randn(M, B, K).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()
        B_pt = torch.randn(B, N).cuda().half()

        XT = X_pt.permute(1, 0, 2)
        Bias = B_pt.unsqueeze(1)
        Y_pt = torch.baddbmm(Bias, XT, W_pt)
        Y_pt = Y_pt.permute(1, 0, 2)

        y = torch.empty([M, B, N]).cuda().half()
        module.RunWithTensors({"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
