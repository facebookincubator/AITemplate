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
import os
import unittest

import numpy as np
import torch
from aitemplate.compiler import compile_model, Model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target


# @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
@unittest.skip("GEMM + Softmax is disabled for now")
class GEMMTestCase(unittest.TestCase):
    def _test_gemm_rcr_bias_softmax(
        self, M=16, K=64, N=24, rebuild=True, test_name="gemm_bias_softmax"
    ):
        target = detect_target()
        X = Tensor(shape=[M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype="float16", name="input_1", is_input=True)
        B = Tensor(shape=[N], dtype="float16", name="input_2", is_input=True)
        OP = ops.gemm_rcr_bias_softmax()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        X_pt = torch.randn(M, K).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()
        B_pt = torch.randn(N).cuda().half()
        Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
        Y_pt = torch.softmax(Y_pt, dim=1)
        Y_np = Y_pt.cpu().numpy()

        if rebuild:
            target = detect_target()
            module = compile_model(Y, target, "./tmp", test_name)
        else:
            module = Model(os.path.join("./tmp", test_name, "test.so"))
        inputs = {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}
        y = torch.empty([M, N]).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

        np.testing.assert_allclose(
            np.argmax(Y_np, axis=1),
            np.argmax(y.cpu().numpy(), axis=1),
            atol=1e-1,
            rtol=1e-1,
        )

    def test_gemm_bias_softmax(self):
        self._test_gemm_rcr_bias_softmax(N=81)


if __name__ == "__main__":
    unittest.main()
