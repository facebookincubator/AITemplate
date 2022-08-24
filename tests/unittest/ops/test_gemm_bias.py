# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntImm
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import shape_utils


class GEMMTestCase(unittest.TestCase):
    def _test_rcr(self, Ms, N, K, test_name):
        target = detect_target()
        MDim = shape_utils.gen_int_var_min_max(Ms, name="m")
        X = Tensor(
            shape=[MDim, IntImm(K)], dtype="float16", name="input_0", is_input=True
        )
        W = Tensor(
            shape=[IntImm(N), IntImm(K)], dtype="float16", name="input_1", is_input=True
        )
        B = Tensor(shape=[IntImm(N)], dtype="float16", name="input_2", is_input=True)
        OP = ops.gemm_rcr_bias()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(Y, target, "./tmp", f"gemm_rcr_bias_{test_name}")

        for M in Ms:
            logging.info(f"Testing {M=}")

            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)

            y = torch.empty([M, N]).half().cuda()
            module.RunWithTensors(
                {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt},
                [y],
            )
            if X_pt.nelement() == 0 or W_pt.nelement() == 0:
                pass
            else:
                self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rcr(self):
        target = detect_target()
        self._test_rcr([128], N=64, K=1024, test_name="static")
        if target.name() == "cuda":
            self._test_rcr([1, 7, 64, 127], N=64, K=1024, test_name="dynamic_m")
            # This test triggered a c10 assertion failure internally
            # caffe2/c10/util/SmallVector.h:338:
            # Assertion `idx < size()' failed
            if type(target).__name__ != "FBCUDA":
                self._test_rcr([2], N=64, K=0, test_name="zero_k")
            self._test_rcr([2], N=0, K=4, test_name="zero_n")
            self._test_rcr([0], N=4, K=4, test_name="zero_m")


if __name__ == "__main__":
    unittest.main()
