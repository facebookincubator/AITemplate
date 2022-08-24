# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GEMMTestCase(unittest.TestCase):
    def _test_rrr(self, M, N, K, use_fp16_acc=True):
        target = detect_target(use_fp16_acc=use_fp16_acc)
        X = Tensor(shape=[*M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[K, N], dtype="float16", name="input_1", is_input=True)
        OP = ops.gemm_rrr_small_nk()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(Y, target, "./tmp", "gemm_rrr_small_nk")
        X_pt = torch.randn(*M, K).cuda().half()
        W_pt = torch.randn(K, N).cuda().half()
        Y_pt = torch.matmul(X_pt, W_pt)

        inputs = {"input_0": X_pt, "input_1": W_pt}
        y = torch.empty([*M, N]).cuda().half()
        module.RunWithTensors(inputs, [y])
        if X_pt.nelement() == 0 or W_pt.nelement() == 0:
            pass
        else:
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

        # from aitemplate.testing.test_utils import benchmark_torch_function
        # t = benchmark_torch_function(100, torch.matmul, X_pt, W_pt)
        # print("pt time: ", t)
        # module.BenchmarkWithTensors(inputs, [y])

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


if __name__ == "__main__":
    unittest.main()
