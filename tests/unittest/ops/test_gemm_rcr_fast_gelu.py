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
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
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
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(10)

    def _test_rcr(
        self, Ms, test_name, use_fast_gelu=True, atol=1e-1, rtol=1e-1, dtype="float16"
    ):
        K = 1024
        N = 64
        target = detect_target()
        MDim = shape_utils.gen_int_var_min_max(Ms, name="m")
        X = Tensor(shape=[MDim, IntImm(K)], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(
            shape=[IntImm(N), IntImm(K)], dtype=dtype, name="input_1", is_input=True
        )

        OP = ops.gemm_rcr_fast_gelu()

        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", f"gemm_rcr_fast_gelu_{test_name}")

        for M in Ms:
            logging.info(f"Testing {M=}")

            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            Y_pt = NewGELUActivation()(torch.nn.functional.linear(X_pt, W_pt))
            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(
                {"input_0": X_pt, "input_1": W_pt},
                [y],
            )
            torch.testing.assert_close(Y_pt, y, atol=atol, rtol=rtol)

    def test_rcr(self):
        self._test_rcr([128], "static", use_fast_gelu=True)
        if detect_target().name() == "cuda":
            self._test_rcr([1, 7, 64, 127], "dynamic_m", use_fast_gelu=True)
            self._test_rcr([128], "static", use_fast_gelu=False)
            self._test_rcr([1, 7, 64, 127], "dynamic_m", use_fast_gelu=False)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_gemm_rcr_fast_gelu_float_sm80(self):
        self._test_rcr([128], "static_float", use_fast_gelu=True, dtype="float")
        self._test_rcr(
            [1, 7, 64, 127], "dynamic_m_float", use_fast_gelu=True, dtype="float"
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_gemm_rcr_fast_gelu_bfloat16_sm80(self):
        self._test_rcr(
            [128],
            "static_float",
            use_fast_gelu=True,
            atol=3e-1,
            rtol=3e-1,
            dtype="bfloat16",
        )
        self._test_rcr(
            [1, 7, 64, 127], "dynamic_m_float", use_fast_gelu=True, dtype="bfloat16"
        )


filter_test_cases_by_test_env(GEMMRcrFastGeluTestCase)

if __name__ == "__main__":
    unittest.main()
