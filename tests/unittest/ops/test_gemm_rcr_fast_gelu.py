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
import math
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils


class NewGELUActivation(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


class GEMMRcrFastGeluTestCase(unittest.TestCase):
    def _test_rcr(self, Ms, test_name, use_fast_gelu=True):
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

        OP = ops.gemm_rcr_fast_gelu()

        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", f"gemm_rcr_fast_gelu_{test_name}")

        for M in Ms:
            logging.info(f"Testing {M=}")

            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            Y_pt = NewGELUActivation()(torch.nn.functional.linear(X_pt, W_pt))
            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(
                {"input_0": X_pt, "input_1": W_pt},
                [y],
            )
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rcr(self):
        self._test_rcr([128], "static", use_fast_gelu=True)
        if detect_target().name() == "cuda":
            self._test_rcr([1, 7, 64, 127], "dynamic_m", use_fast_gelu=True)
            self._test_rcr([128], "static", use_fast_gelu=False)
            self._test_rcr([1, 7, 64, 127], "dynamic_m", use_fast_gelu=False)


if __name__ == "__main__":
    unittest.main()
