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


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class BMMTestCase(unittest.TestCase):
    def _test_rrr(self, B, M, K, N, test_name):
        target = detect_target()
        X = Tensor(shape=[B, M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[B, K, N], dtype="float16", name="input_1", is_input=True)
        OP = ops.bmm_rrr_k1_tanh()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)
        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()

        Y_pt = torch.bmm(X_pt, W_pt)
        Y_pt = torch.tanh(Y_pt)

        y = torch.empty([B, M, N]).cuda().half()
        module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
        if X_pt.nelement() == 0 or W_pt.nelement() == 0:
            pass
        else:
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rrr(self):
        self._test_rrr(B=1024, M=32, K=1, N=32, test_name="bmm_rrr_k1")
        self._test_rrr(B=1024, M=0, K=1, N=32, test_name="bmm_rrr_k1_zero_m")
        self._test_rrr(B=1024, M=32, K=0, N=32, test_name="bmm_rrr_k1_zero_k")


if __name__ == "__main__":
    unittest.main()
