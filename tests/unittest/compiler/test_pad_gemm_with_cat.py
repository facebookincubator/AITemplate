# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

import numpy as np
import torch

from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import logger


class PadGemmWithCatTestCase(unittest.TestCase):
    def test_pad_gemm_rcr_with_cat(self):
        M = 128
        N = 32
        K1 = 3
        K2 = 10
        K = K1 + K2

        X1 = Tensor(shape=[M, K1], dtype="float16", name="x1", is_input=True)
        W1 = Tensor(shape=[N, K], dtype="float16", name="w1", is_input=True)
        B1 = Tensor(shape=[N], dtype="float16", name="b1", is_input=True)

        X2 = Tensor(shape=[M, K2], dtype="float16", name="x2", is_input=True)
        W2 = Tensor(shape=[N, K], dtype="float16", name="w2", is_input=True)
        B2 = Tensor(shape=[N], dtype="float16", name="b2", is_input=True)

        X3 = ops.elementwise(FuncEnum.ADD)(X1, X1)
        X4 = ops.concatenate()([X2, X3], dim=1)
        X5 = ops.gemm_rcr_bias()(X4, W1, B1)
        X6 = ops.gemm_rcr_bias()(X4, W2, B2)
        Y = ops.concatenate()([X5, X6], dim=1)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Skip this test on SM75")
            return
        dll_name = "test_rcr.so"
        module = gen_execution_module(
            [Y], target, "./tmp", "pad_gemm_with_cat", dll_name=dll_name
        )

        X1_pt = torch.randn(M, K1).cuda().half()
        X2_pt = torch.randn(M, K2).cuda().half()
        W1_pt = torch.randn(N, K).cuda().half()
        W2_pt = torch.randn(N, K).cuda().half()
        B1_pt = torch.randn(N).cuda().half()
        B2_pt = torch.randn(N).cuda().half()
        X3_pt = torch.add(X1_pt, X1_pt)
        X4_pt = torch.cat([X2_pt, X3_pt], dim=1)
        X5_pt = torch.nn.functional.linear(X4_pt, W1_pt, bias=B1_pt)
        X6_pt = torch.nn.functional.linear(X4_pt, W2_pt, bias=B2_pt)
        Y_pt = torch.cat([X5_pt, X6_pt], dim=1)

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_pt.size())

        inputs = [0] * 6
        name_to_idx = module.GetInputNameToIndexMap()
        inputs[name_to_idx["x1"]] = X1_pt
        inputs[name_to_idx["x2"]] = X2_pt

        inputs[name_to_idx["w1"]] = W1_pt
        inputs[name_to_idx["w2"]] = W2_pt

        inputs[name_to_idx["b1"]] = B1_pt
        inputs[name_to_idx["b2"]] = B2_pt

        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
