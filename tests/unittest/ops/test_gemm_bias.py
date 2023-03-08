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
from aitemplate.compiler.base import IntImm
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
    streamk_enabled_and_not_supported,
)
from aitemplate.utils import shape_utils
from parameterized import parameterized


def _tolerance_limits(dtype):
    if dtype == "float16":
        return {"atol": 1e-1, "rtol": 1e-1}
    elif dtype == "float32":
        return {"atol": 1e-1, "rtol": 1e-1}
    elif dtype == "bfloat16":
        return {"atol": 3e-1, "rtol": 3e-1}
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


class GEMMBiasTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GEMMBiasTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_rcr(self, Ms, N, K, test_name, dtype="float16"):
        target = detect_target()
        tolerance_limits = _tolerance_limits(dtype)
        MDim = shape_utils.gen_int_var_min_max(Ms, name="m")
        X = Tensor(shape=[MDim, IntImm(K)], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(
            shape=[IntImm(N), IntImm(K)], dtype=dtype, name="input_1", is_input=True
        )
        B = Tensor(shape=[IntImm(N)], dtype=dtype, name="input_2", is_input=True)
        OP = ops.gemm_rcr_bias()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y, target, "./tmp", f"gemm_rcr_bias_{test_name}_{self._test_id}"
        )
        self._test_id += 1

        for M in Ms:
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)

            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(
                {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt},
                [y],
            )
            if X_pt.nelement() == 0 or W_pt.nelement() == 0:
                pass
            else:
                torch.testing.assert_close(Y_pt, y, **tolerance_limits)

    def test_rcr_zero_size(self):
        target = detect_target()
        if target.name() == "cuda":
            # This test triggered a c10 assertion failure internally
            # caffe2/c10/util/SmallVector.h:338:
            # Assertion `idx < size()' failed
            if type(target).__name__ != "FBCUDA":
                self._test_rcr([2], N=64, K=0, test_name="zero_k")
            self._test_rcr([2], N=0, K=4, test_name="zero_n")
            self._test_rcr([0], N=4, K=4, test_name="zero_m")

    @unittest.skipIf(streamk_enabled_and_not_supported(), "StreamK is not supported")
    def test_rcr_static(self):
        self._test_rcr([4096], N=4, K=4, test_name="static")
        self._test_rcr([1000], N=81, K=1024, test_name="static")
        self._test_rcr([67200], N=3, K=256, test_name="static")

    @parameterized.expand(("bfloat16",))
    def test_rcr_all_floats(self, dtype):
        skipped_reason = _skip_target(detect_target(), dtype)
        if skipped_reason is not None:
            self.skipTest(skipped_reason)
        self._test_rcr([4], N=2, K=11, test_name=f"static_{dtype}", dtype=dtype)
        self._test_rcr([128], N=64, K=1024, test_name=f"static_{dtype}", dtype=dtype)
        self._test_rcr(
            [1, 7, 64, 127],
            N=64,
            K=1024,
            test_name=f"dynamic_m_{dtype}",
            dtype=dtype,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
