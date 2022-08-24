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
    def _test_rcr(self, Ms, test_name):
        K = 1024
        N = 64
        target = detect_target()
        MDim = shape_utils.gen_int_var_min_max(Ms, name="m")
        X = Tensor(
            shape=[MDim, IntImm(K)], dtype="float16", name="input_0", is_input=True
        )
        W = Tensor(
            shape=[IntImm(N), IntImm(K)], dtype="float16", name="input_1", is_input=True
        )
        B = Tensor(shape=[IntImm(N)], dtype="float16", name="input_2", is_input=True)
        OP = ops.gemm_rcr_bias_fast_gelu()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(
            Y, target, "./tmp", f"gemm_rcr_bias_fast_gelu_{test_name}"
        )

        for M in Ms:
            logging.info(f"Testing {M=}")

            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            Y_pt = torch.nn.GELU()(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt))
            y = torch.empty([M, N]).cuda().half()
            module.RunWithTensors(
                {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt},
                [y],
            )
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rcr(self):
        self._test_rcr([128], "static")
        if detect_target().name() == "cuda":
            self._test_rcr([1, 7, 64, 127], "dynamic_m")


if __name__ == "__main__":
    unittest.main()
