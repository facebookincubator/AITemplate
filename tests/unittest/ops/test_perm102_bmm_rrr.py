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
"""
C[m, b, n](row) = bmm(A[m, b, k](row), B[b, k, n](row))
in torch it is
# _2905_2929 = _2904.view(B, 25, -1).permute(1, 0, 2)
# _2930_2954 = torch.baddbmm(
#      self._1085_1133, _2905_2929, self._1084_1132) # baddbmm(bias, X, W)
"""


import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from parameterized import parameterized


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class Perm102BMMTestCase(unittest.TestCase):
    @parameterized.expand([("float16"), ("float32"), ("bfloat16")])
    def test_perm102_bmm_rrr(self, dtype="float16"):
        if dtype != "float16" and int(detect_target()._arch) < 80:
            self.skipTest(f"{dtype} BMM not supported in {detect_target()._arch}")
        B = 25
        M = 128
        K = 256
        N = 100
        target = detect_target()
        X = Tensor(shape=[M, B, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[B, K, N], dtype=dtype, name="input_1", is_input=True)
        OP = ops.perm102_bmm_rrr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "perm102_bmm_rrr")

        X_pt = get_random_torch_tensor(shape=(M, B, K), dtype=dtype)
        W_pt = get_random_torch_tensor(shape=(B, K, N), dtype=dtype)

        XT = X_pt.permute(1, 0, 2).contiguous()
        Y_pt = torch.bmm(XT, W_pt)
        Y_pt = Y_pt.permute(1, 0, 2).contiguous()
        y = torch.empty_like(Y_pt)
        module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])

        torch.testing.assert_close(Y_pt, y, atol=1e-1, rtol=1e-1)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class Perm102BMMBiasTestCase(unittest.TestCase):
    @parameterized.expand([("float16"), ("float32"), ("bfloat16")])
    def test_perm102_bmm_rrr_bias(self, dtype="float16"):
        if dtype != "float16" and int(detect_target()._arch) < 80:
            self.skipTest(f"{dtype} BMM not supported in {detect_target()._arch}")
        B = 25
        M = 128
        K = 256
        N = 100
        target = detect_target()
        X = Tensor(shape=[M, B, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[B, K, N], dtype=dtype, name="input_1", is_input=True)
        BIAS = Tensor(shape=[B, N], dtype=dtype, name="input_2", is_input=True)
        OP = ops.perm102_bmm_rrr_bias()
        Y = OP(X, W, BIAS)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "perm102_bmm_rrr_bias")

        X_pt = get_random_torch_tensor(shape=(M, B, K), dtype=dtype)
        W_pt = get_random_torch_tensor(shape=(B, K, N), dtype=dtype)
        B_pt = get_random_torch_tensor(shape=(B, N), dtype=dtype)

        XT = X_pt.permute(1, 0, 2).contiguous()
        Bias = B_pt.unsqueeze(1)
        Y_pt = torch.baddbmm(Bias, XT, W_pt)
        Y_pt = Y_pt.permute(1, 0, 2).contiguous()

        y = torch.empty_like(Y_pt)
        module.run_with_tensors(
            {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}, [y]
        )

        torch.testing.assert_close(Y_pt, y, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
