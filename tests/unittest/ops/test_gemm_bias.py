#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
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
        module = compile_model(Y, target, "./tmp", f"gemm_rcr_bias_{test_name}")

        for M in Ms:
            logging.info(f"Testing {M=}")

            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)

            y = torch.empty([M, N]).half().cuda()
            module.run_with_tensors(
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
        self._test_rcr([4096], N=4, K=4, test_name="static")
        self._test_rcr([1000], N=81, K=1024, test_name="static")
        self._test_rcr([67200], N=3, K=256, test_name="static")
        if target.name() == "cuda":
            self._test_rcr([1, 7, 64, 127], N=64, K=1024, test_name="dynamic_m")
            # This test triggered a c10 assertion failure internally
            # caffe2/c10/util/SmallVector.h:338:
            # Assertion `idx < size()' failed

            self._test_rcr([2], N=0, K=4, test_name="zero_n")
            self._test_rcr([0], N=4, K=4, test_name="zero_m")


if __name__ == "__main__":
    unittest.main()
