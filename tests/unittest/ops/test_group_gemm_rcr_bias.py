# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module

logger = logging.getLogger(__name__)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GEMMTestCase(unittest.TestCase):
    def test_rcr(self):
        M = 256
        K1 = 128
        N1 = 60
        K2 = 192
        N2 = 64
        target = detect_target()
        if int(target._arch) < 80:
            logger.warning("Group Gemm need SM80 HW")
            return
        X1 = Tensor(shape=[M, K1], dtype="float16", name="x1", is_input=True)
        X2 = Tensor(shape=[M, K2], dtype="float16", name="x2", is_input=True)
        W1 = Tensor(shape=[N1, K1], dtype="float16", name="w1", is_input=True)
        W2 = Tensor(shape=[N2, K2], dtype="float16", name="w2", is_input=True)
        B1 = Tensor(shape=[N1], dtype="float16", name="b1", is_input=True)
        B2 = Tensor(shape=[N2], dtype="float16", name="b2", is_input=True)
        OP = ops.group_gemm_rcr_bias()
        Y1, Y2 = OP(operand_groups=[[X1, W1, B1], [X2, W2, B2]])
        Y1._attrs["name"] = "y1"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "y2"
        Y2._attrs["is_output"] = True
        module = gen_execution_module([Y1, Y2], target, "./tmp", "group_gemm_rcr_bias")
        X1_pt = torch.randn(M, K1).cuda().half()
        X2_pt = torch.randn(M, K2).cuda().half()
        W1_pt = torch.randn(N1, K1).cuda().half()
        W2_pt = torch.randn(N2, K2).cuda().half()
        B1_pt = torch.randn(N1).cuda().half()
        B2_pt = torch.randn(N2).cuda().half()
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt, bias=B1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt, bias=B2_pt)

        inputs = {
            "x1": X1_pt,
            "w1": W1_pt,
            "b1": B1_pt,
            "x2": X2_pt,
            "w2": W2_pt,
            "b2": B2_pt,
        }
        y1 = torch.empty([M, N1]).cuda().half()
        y2 = torch.empty([M, N2]).cuda().half()
        module.RunWithTensors(inputs, {"y1": y1, "y2": y2})
        self.assertTrue(torch.allclose(Y1_pt, y1, atol=1e-1, rtol=1e-1))
        self.assertTrue(torch.allclose(Y2_pt, y2, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
