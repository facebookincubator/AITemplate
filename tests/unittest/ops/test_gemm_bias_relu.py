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
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from parameterized import parameterized


def _tolerance_limits(dtype):
    if dtype == "float16":
        return {"atol": 1e-1, "rtol": 1e-1}
    elif dtype == "float32":
        return {"atol": 1e-1, "rtol": 1e-1}
    elif dtype == "bfloat16":
        return {"atol": 2e-1, "rtol": 2e-1}
    else:
        return {}


def _skip_target(target, ait_dtype):
    if ait_dtype == "float16":
        return None
    if target.name() != "cuda":
        return "Not supported for non-CUDA target"
    if int(target._arch) < 80:
        return "Not supported for CUDA SM<80."
    return None


class GEMMBiasReluTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GEMMBiasReluTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_gemm_rcr_bias_relu(self, dtype="float16", target=None):
        M = 128
        K = 1024
        N = 64
        tolerance_limits = _tolerance_limits(dtype)
        X = Tensor(shape=[M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype=dtype, name="input_1", is_input=True)
        B = Tensor(shape=[N], dtype=dtype, name="input_2", is_input=True)
        OP = ops.gemm_rcr_bias_relu()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        test_name = f"gemm_rcr_bias_relu_{dtype}_{self._test_id}"
        self._test_id += 1
        module = compile_model(Y, target, "./tmp", test_name)
        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([N, K], dtype)
        B_pt = get_random_torch_tensor([N], dtype)
        Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
        Y_pt = torch.relu(Y_pt)

        inputs = {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}
        y = get_torch_empty_tensor([M, N], dtype)
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(Y_pt, y, **tolerance_limits)

    @parameterized.expand(("float16", "float32", "bfloat16"))
    def test_gemm_rcr_bias_relu(self, ait_dtype):
        target = detect_target()
        skipped_reason = _skip_target(target, ait_dtype)
        if skipped_reason is not None:
            self.skipTest(skipped_reason)
        self._test_gemm_rcr_bias_relu(ait_dtype, target)

    def _test_gemm_rcr_bias_add_relu(self, dtype="float16", target=None):
        M = 128
        K = 1024
        N = 64
        tolerance_limits = _tolerance_limits(dtype)
        X = Tensor(shape=[M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype=dtype, name="input_1", is_input=True)
        B = Tensor(shape=[N], dtype=dtype, name="input_2", is_input=True)
        D = Tensor(shape=[M, N], dtype=dtype, name="input_3", is_input=True)
        OP = ops.gemm_rcr_bias_add_relu()
        Y = OP(X, W, B, D)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        test_name = f"gemm_rcr_bias_add_relu_{dtype}_{self._test_id}"
        self._test_id += 1
        module = compile_model(Y, target, "./tmp", test_name)
        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([N, K], dtype)
        B_pt = get_random_torch_tensor([N], dtype)
        D_pt = get_random_torch_tensor([M, N], dtype)
        Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) + D_pt
        Y_pt = torch.relu(Y_pt)

        inputs = [X_pt, W_pt, B_pt, D_pt]
        y = get_torch_empty_tensor([M, N], dtype)
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(Y_pt, y, **tolerance_limits)

    @parameterized.expand(("float16", "float32", "bfloat16"))
    def test_gemm_rcr_bias_add_relu(self, ait_dtype):
        target = detect_target()
        skipped_reason = _skip_target(target, ait_dtype)
        if skipped_reason is not None:
            self.skipTest(skipped_reason)
        self._test_gemm_rcr_bias_add_relu(ait_dtype, target)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
