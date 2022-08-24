# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

import numpy as np
import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import logger


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GEMMTestCase(unittest.TestCase):
    def test_rcr_cat(self):
        M = 256
        K1 = 128
        N1 = 60
        K2 = 192
        N2 = 64
        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Group Gemm need SM80 HW")
            return
        X1 = Tensor(shape=[M, K1], dtype="float16", name="x1", is_input=True)
        X2 = Tensor(shape=[M, K2], dtype="float16", name="x2", is_input=True)
        W1 = Tensor(shape=[N1, K1], dtype="float16", name="w1", is_input=True)
        W2 = Tensor(shape=[N2, K2], dtype="float16", name="w2", is_input=True)
        OP = ops.group_gemm_rcr()
        Y = OP(operand_groups=[[X1, W1], [X2, W2]], output_stride_dim=1)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        module = gen_execution_module([Y], target, "./tmp", "group_gemm_rcr_cat")

        X1_pt = torch.randn(M, K1).cuda().half()
        X2_pt = torch.randn(M, K2).cuda().half()
        W1_pt = torch.randn(N1, K1).cuda().half()
        W2_pt = torch.randn(N2, K2).cuda().half()
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt)
        Y_pt = torch.cat([Y1_pt, Y2_pt], dim=1)
        Y_np = Y_pt.cpu().numpy()

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_np.shape)

        inputs = {
            "x1": X1_pt,
            "w1": W1_pt,
            "x2": X2_pt,
            "w2": W2_pt,
        }
        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
