#  Copyright (c) Meta Platform, Inc. and its affiliates.
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
from aitemplate.utils import logger


@unittest.skipIf(detect_target()._arch == "75", "DualGemm not supported on sm75.")
class DUALGEMMTestCase(unittest.TestCase):
    def _test_dual_gemm(self, M=4096, N=4096, K=8192, benchmark=False):
        target = detect_target(use_fp16_acc=False)
        X = Tensor(shape=[M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype="float16", name="input_1", is_input=True)
        B = Tensor(shape=[N, K], dtype="float16", name="input_2", is_input=True)
        OP = ops.dual_gemm_rcr_silu()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "dual_gemm")
        X_pt = torch.randn(M, K).cuda().half() * 0.01
        W_pt = torch.randn(N, K).cuda().half()
        B_pt = torch.randn(N, K).cuda().half()

        def pt_func(X_pt, W_pt, B_pt):
            Y_pt1 = torch.nn.functional.linear(X_pt, W_pt)
            Y_pt2 = torch.nn.functional.linear(X_pt, B_pt)
            Y_pt = torch.nn.functional.silu(Y_pt1) * Y_pt2
            # Y_pt =  0.5 * Y_pt1 * ( 1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (Y_pt1 + 0.044715 * torch.pow(Y_pt1, 3.0))) ) * Y_pt2
            return Y_pt

        Y_pt = pt_func(X_pt, W_pt, B_pt)

        inputs = {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}
        y = torch.empty([M, N]).cuda().half()
        module.run_with_tensors(inputs, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

        if benchmark:
            # Warm up.
            for _ in range(5):
                module.run_with_tensors(inputs, [y])
            # Benchmark AIT
            time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
                inputs,
                [y],
                count=100,
            )
            logger.info(__file__, f"AIT GEMMxGEMM time: {time_per_iter_ms:.5f}ms")
            # Benchmark PT
            from aitemplate.testing.benchmark_pt import benchmark_torch_function

            func = pt_func
            args = (X_pt, W_pt, B_pt)
            duration = benchmark_torch_function(100, func, *args)
            logger.info(__file__, f"PT GEMMxGEMM Time: {duration:.5f}ms")

    def test_dual_gemm(self):
        self._test_dual_gemm(M=128, N=128, K=256)
        self._test_dual_gemm(M=1024, N=1024, K=2048)
        self._test_dual_gemm(M=4096, N=4096, K=8192)


if __name__ == "__main__":
    unittest.main()
