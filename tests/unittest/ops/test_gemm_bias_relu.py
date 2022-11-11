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


class GEMMBiasReluTestCase(unittest.TestCase):
    def test_gemm_rcr_bias_relu(self):
        M = 128
        K = 1024
        N = 64
        target = detect_target()
        X = Tensor(shape=[M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype="float16", name="input_1", is_input=True)
        B = Tensor(shape=[N], dtype="float16", name="input_2", is_input=True)
        OP = ops.gemm_rcr_bias_relu()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "gemm_rcr_bias_relu")
        X_pt = torch.randn(M, K).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()
        B_pt = torch.randn(N).cuda().half()
        Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
        Y_pt = torch.relu(Y_pt)

        inputs = {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}
        y = torch.empty([M, N]).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_gemm_rcr_bias_add_relu(self):
        M = 128
        K = 1024
        N = 64
        target = detect_target()
        X = Tensor(shape=[M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype="float16", name="input_1", is_input=True)
        B = Tensor(shape=[N], dtype="float16", name="input_2", is_input=True)
        D = Tensor(shape=[M, N], dtype="float16", name="input_3", is_input=True)
        OP = ops.gemm_rcr_bias_add_relu()
        Y = OP(X, W, B, D)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "gemm_rcr_bias_add_relu")
        X_pt = torch.randn(M, K).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()
        B_pt = torch.randn(N).cuda().half()
        D_pt = torch.randn(M, N).cuda().half()
        Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) + D_pt
        Y_pt = torch.relu(Y_pt)

        inputs = [X_pt, W_pt, B_pt, D_pt]
        y = torch.empty([M, N]).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
