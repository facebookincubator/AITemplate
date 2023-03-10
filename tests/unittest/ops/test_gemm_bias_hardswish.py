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
    filter_test_cases_by_params,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from parameterized import parameterized


_TOLERANCE_LIMITS = {
    "float16": {"atol": 1e-1, "rtol": 1e-1},
    "float32": {"atol": 1e-1, "rtol": 1e-1},
    "bfloat16": {"atol": 3e-1, "rtol": 3e-1},
}


def hard_swish(x):
    return x * torch.clamp((x + 3), 0, 6) / 6


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GEMMBiasHardSwishTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GEMMBiasHardSwishTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_rcr(self, dtype="float16"):
        M = 128
        K = 1024
        N = 64
        target = detect_target()
        X = Tensor(shape=[M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype=dtype, name="input_1", is_input=True)
        B = Tensor(shape=[N], dtype=dtype, name="input_2", is_input=True)
        OP = ops.gemm_rcr_bias_hardswish()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        test_name = f"gemm_rcr_bias_hardswish_{dtype}_{self._test_id}"
        module = compile_model(Y, target, "./tmp", test_name)
        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([N, K], dtype)
        B_pt = get_random_torch_tensor([N], dtype)
        Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
        Y_pt = hard_swish(Y_pt)

        inputs = {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}
        y = get_torch_empty_tensor([M, N], dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, **_TOLERANCE_LIMITS[dtype]))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float32"), ("bfloat16")],
            }
        )
    )
    def test_rcr(self, dtype):
        self._test_rcr(dtype)


if __name__ == "__main__":
    unittest.main()
