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
class Perm021BMMTestCase(unittest.TestCase):
    def test_crc(self):
        B = 1024
        M = 128
        K = 742
        # K = 752
        N = 64
        target = detect_target()
        X = Tensor(shape=[1, K, N], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[B, K, M], dtype="float16", name="input_1", is_input=True)
        BIAS = Tensor(shape=[N], dtype="float16", name="input_2", is_input=True)
        OP = ops.perm021fc_crc_bias()
        Y = OP(X, W, BIAS)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(Y, target, "./tmp", "perm021_fc_crc_bias")

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()
        B_pt = torch.ones(N).cuda().half() * 0.5

        XT = X_pt.permute(0, 2, 1)
        XT = torch.reshape(XT, (-1, K))
        Y_pt = torch.nn.functional.linear(XT, W_pt, bias=B_pt)
        Y_pt = torch.reshape(Y_pt, (B, M, N)).contiguous()
        WT = W_pt.transpose(0, 1).contiguous()
        y = torch.empty([B, M, N]).cuda().half()
        module.RunWithTensors(
            {"input_0": WT.unsqueeze(0), "input_1": X_pt, "input_2": B_pt}, [y]
        )

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
