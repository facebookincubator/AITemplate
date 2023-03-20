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

import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
)
from aitemplate.utils import shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GEMMRrrSmallNKTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_count = 0

    def _test_rrr(
        self, M, N, K, use_fp16_acc=True, dtype="float16", atol=1e-1, rtol=1e-1
    ):
        target = detect_target(use_fp16_acc=use_fp16_acc)
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(M, name="batch_dim"), K],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[K, N], dtype=dtype, name="input_1", is_input=True)
        OP = ops.gemm_rrr_small_nk()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y, target, "./tmp", f"gemm_rrr_small_nk_{self.test_count}"
        )

        for m in M:
            X_pt = get_random_torch_tensor([m, K], dtype)
            W_pt = get_random_torch_tensor([K, N], dtype)
            Y_pt = torch.matmul(X_pt, W_pt)

            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = torch.empty_like(Y_pt)
            module.run_with_tensors(inputs, [y])
            if X_pt.nelement() == 0 or W_pt.nelement() == 0:
                pass
            else:
                torch.testing.assert_close(Y_pt, y, atol=atol, rtol=rtol)
        self.test_count += 1

        # from aitemplate.testing.benchmark_pt import benchmark_torch_function
        # t = benchmark_torch_function(100, torch.matmul, X_pt, W_pt)
        # print("pt time: ", t)
        # module.benchmark_with_tensors(inputs, [y])

    def test_rrr(self):
        self._test_rrr([0, 1], 6, 3)
        self._test_rrr([1000], 6, 0)
        self._test_rrr([1, 1000], 6, 3)
        self._test_rrr([10000], 6, 3, False)
        self._test_rrr([10000], 6, 10, False)
        self._test_rrr([10, 13], 6, 3)
        self._test_rrr([105], 7, 1)
        # self._test_rrr([1000000], 6, 3)
        # self._test_rrr([1000000], 6, 10)
        # self._test_rrr([1000000], 8, 16)
        # self._test_rrr([1000000], 6, 3, False)

    def test_gemm_rrr_small_nk_float_sm80(self):
        self._test_rrr([0, 1], 6, 3, False, dtype="float32", atol=1e-5, rtol=1.3e-6)
        self._test_rrr([100001], 7, 10, False, dtype="float32", atol=1e-5, rtol=1.3e-6)

    def test_gemm_rrr_small_nk_bfloat16_sm80(self):
        self._test_rrr([0, 1], 6, 3, False, dtype="bfloat16", atol=1e-1, rtol=1e-1)
        self._test_rrr([100001], 7, 10, False, dtype="bfloat16", atol=1e-1, rtol=1e-1)


filter_test_cases_by_test_env(GEMMRrrSmallNKTestCase)

if __name__ == "__main__":
    unittest.main()
