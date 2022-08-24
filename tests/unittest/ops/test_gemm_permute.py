# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GEMMTestCase(unittest.TestCase):
    def _test_rcr(self, ms, k, n, shape, test_name):
        target = detect_target()
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[n, k], dtype="float16", name="input_1", is_input=True)
        OP = ops.gemm_rcr_permute(shape)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(
            Y, target, "./tmp", "gemm_rcr_{}".format(test_name)
        )

        for m in ms:
            X_pt = torch.randn(m, k).cuda().half()
            W_pt = torch.randn(n, k).cuda().half()
            Y_l = torch.nn.functional.linear(X_pt, W_pt)
            Y_r = Y_l.reshape(16, *shape, 16)
            Y_pt = torch.permute(Y_r, [2, 0, 3, 1, 4])

            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = torch.empty(Y_pt.shape).cuda().half()
            module.RunWithTensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rcr(self):
        self._test_rcr([80], 32, 96, (5, 3, 2), "permute1")
        self._test_rcr([128], 64, 256, (8, 4, 4), "permute2")

    def _test_rrr(self, ms, k, n, shape, test_name):
        target = detect_target()
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[k, n], dtype="float16", name="input_1", is_input=True)
        OP = ops.gemm_rrr_permute(shape)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(
            Y, target, "./tmp", "gemm_rrr_{}".format(test_name)
        )

        for m in ms:
            X_pt = torch.randn(m, k).cuda().half()
            W_pt = torch.randn(k, n).cuda().half()
            Y_l = torch.matmul(X_pt, W_pt)
            Y_r = Y_l.reshape(16, *shape, 16)
            Y_pt = torch.permute(Y_r, [2, 0, 3, 1, 4])
            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = torch.empty(Y_pt.shape).cuda().half()
            module.RunWithTensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rrr(self):
        self._test_rrr([80], 32, 96, (5, 3, 2), "permute1")
        self._test_rrr([128], 64, 256, (8, 4, 4), "permute2")


if __name__ == "__main__":
    unittest.main()
