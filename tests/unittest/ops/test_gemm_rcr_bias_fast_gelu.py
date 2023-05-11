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
from aitemplate.testing.test_utils import (
    env_variables,
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import shape_utils


_TOLERANCE_LIMITS = {
    "float16": {"atol": 1e-1, "rtol": 1e-1},
    "float32": {"atol": 1e-1, "rtol": 1e-1},
    "bfloat16": {"atol": 3e-1, "rtol": 3e-1},
}


class GEMMRcrBiasFastGeluTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_gemm_rcr_bias_fast_gelu(
        self,
        Ms,
        test_name,
        K=1024,
        N=64,
        use_fast_gelu=True,
        dtype="float16",
    ):
        MDim = shape_utils.gen_int_var_min_max(Ms, name="m")
        X = Tensor(
            shape=[MDim, IntImm(K)],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[IntImm(N), IntImm(K)],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B = Tensor(
            shape=[IntImm(N)],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        OP = (
            ops.gemm_rcr_bias_fast_gelu() if use_fast_gelu else ops.gemm_rcr_bias_gelu()
        )
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(
            Y,
            detect_target(),
            "./tmp",
            f"gemm_rcr_bias_fast_gelu_{test_name}"
            if use_fast_gelu
            else f"gemm_rcr_bias_gelu_{test_name}",
        )

        for M in Ms:
            logging.info(f"Testing {M=}")

            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            Y_pt = torch.nn.GELU()(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt))
            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(
                {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt},
                [y],
            )
            torch.testing.assert_close(Y_pt, y, **_TOLERANCE_LIMITS[dtype])

    def test_gemm_rcr_bias_fast_gelu_fp16(self):
        self._test_gemm_rcr_bias_fast_gelu(
            Ms=[128],
            test_name="static_fp16_fast_gelu",
            use_fast_gelu=True,
            dtype="float16",
        )
        self._test_gemm_rcr_bias_fast_gelu(
            Ms=[1, 7, 64, 127],
            test_name="dynamic_m_fp16_fast_gelu",
            use_fast_gelu=True,
            dtype="float16",
        )
        self._test_gemm_rcr_bias_fast_gelu(
            Ms=[128],
            test_name="static_fp16_gelu",
            use_fast_gelu=False,
            dtype="float16",
        )
        self._test_gemm_rcr_bias_fast_gelu(
            Ms=[1, 7, 64, 127],
            test_name="dynamic_m_fp16_gelu",
            use_fast_gelu=False,
            dtype="float16",
        )

    def test_gemm_rcr_bias_fast_gelu_fp16_rocm(self):
        self._test_gemm_rcr_bias_fast_gelu(
            Ms=[128],
            test_name="static_fp16_rocm_fast_gelu",
            use_fast_gelu=True,
            dtype="float16",
        )

    def test_gemm_rcr_bias_fast_gelu_fp32_sm80(self):
        self._test_gemm_rcr_bias_fast_gelu(
            Ms=[1, 7, 64, 127],
            test_name="dynamic_m_fp32_fast_gelu",
            use_fast_gelu=True,
            dtype="float32",
        )
        self._test_gemm_rcr_bias_fast_gelu(
            Ms=[1, 7, 64, 127],
            test_name="dynamic_m_fp32_gelu",
            use_fast_gelu=False,
            dtype="float32",
        )

    def test_gemm_rcr_bias_fast_gelu_bf16(self):
        self._test_gemm_rcr_bias_fast_gelu(
            Ms=[1, 7, 64, 127],
            test_name="dynamic_m_bf16_fast_gelu",
            use_fast_gelu=True,
            dtype="bfloat16",
        )
        self._test_gemm_rcr_bias_fast_gelu(
            Ms=[1, 7, 64, 127],
            test_name="dynamic_m_bf16_gelu",
            use_fast_gelu=False,
            dtype="bfloat16",
        )

    def test_gemm_rcr_bias_fast_gelu_sm90(self):
        with env_variables(
            AIT_FORCE_CUTLASS_SM90_KERNELS="1",
            INSIDE_RE_WORKER="1",
        ):
            with self.assertRaisesRegex(
                expected_exception=RuntimeError,
                expected_regex="No GEMM op instances are left after filtering",
            ):
                # input alignment < 8 not supported by SM90 kernels
                # use alignment 4 to avoid auto-padding to 8
                self._test_gemm_rcr_bias_fast_gelu(
                    Ms=[1, 7, 64, 127],
                    K=1020,
                    test_name="wrong_input_alignment_sm90",
                    use_fast_gelu=True,
                    dtype="float16",
                )

            with self.assertRaisesRegex(
                expected_exception=RuntimeError,
                expected_regex="No GEMM op instances are left after filtering",
            ):
                # output alignment < 8 not supported by SM90 TMA epilogues
                self._test_gemm_rcr_bias_fast_gelu(
                    Ms=[1, 7, 64, 127],
                    N=63,
                    test_name="wrong_output_alignment_sm90",
                    use_fast_gelu=True,
                    dtype="float16",
                )

            self._test_gemm_rcr_bias_fast_gelu(
                Ms=[1, 7, 64, 127],
                test_name="dynamic_m_fp16_fast_gelu_force_sm90",
                use_fast_gelu=True,
                dtype="float16",
            )
            self._test_gemm_rcr_bias_fast_gelu(
                Ms=[1, 7, 64, 127],
                test_name="dynamic_m_fp16_gelu_force_sm90",
                use_fast_gelu=False,
                dtype="float16",
            )
            self._test_gemm_rcr_bias_fast_gelu(
                Ms=[1, 7, 64, 127],
                test_name="dynamic_m_bf16_fast_gelu_force_sm90",
                use_fast_gelu=True,
                dtype="bfloat16",
            )
            self._test_gemm_rcr_bias_fast_gelu(
                Ms=[1, 7, 64, 127],
                test_name="dynamic_m_bf16_gelu_force_sm90",
                use_fast_gelu=False,
                dtype="bfloat16",
            )


filter_test_cases_by_test_env(GEMMRcrBiasFastGeluTestCase)


if __name__ == "__main__":
    unittest.main()
