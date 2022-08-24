# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import logger


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class BMMSoftmaxTestCase(unittest.TestCase):
    def _test_bmm_rcr_softmax(
        self, B=16, M=16, K=64, N=24, test_name="bmm_rcr_softmax"
    ):

        X = Tensor(shape=[B, M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[B, N, K], dtype="float16", name="input_1", is_input=True)
        OP = ops.bmm_rcr_softmax()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Skip this test on SM75")
            return
        if type(target).__name__ == "FBCUDA":
            logger.warning(__file__, "Skip this test for special profiling requirement")
            return
        module = gen_execution_module(Y, target, "./tmp", test_name)
        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()

        WT = torch.transpose(W_pt, 2, 1)
        Y_pt = torch.bmm(X_pt, WT)
        Y_pt = torch.softmax(Y_pt, dim=-1)

        y = torch.empty([B, M, N]).cuda().half()
        module.RunWithTensors({"input_0": X_pt, "input_1": W_pt}, [y])
        eps = 1e-1
        self.assertTrue(torch.allclose(Y_pt, y, atol=eps, rtol=eps))

    def test_bmm_softmax(self):
        self._test_bmm_rcr_softmax()


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
