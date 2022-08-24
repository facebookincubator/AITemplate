# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import logger, shape_utils


class PadGemmWithCatTestCase(unittest.TestCase):
    def _test_pad_gemm_rrr_with_cat(self, test_name, ms, n, k1, k2):
        k = k1 + k2
        m_dim = shape_utils.gen_int_var_min_max(ms, name="m")
        X1 = Tensor(shape=[m_dim, k1], dtype="float16", name="x1", is_input=True)
        W1 = Tensor(shape=[k, n], dtype="float16", name="w1", is_input=True)
        X2 = Tensor(shape=[m_dim, k2], dtype="float16", name="x2", is_input=True)
        W2 = Tensor(shape=[k, n], dtype="float16", name="w2", is_input=True)
        X4 = ops.concatenate()([X1, X2], dim=1)
        Y1 = ops.gemm_rrr()(X4, W1)
        Y2 = ops.gemm_rrr()(X4, W2)
        Y = ops.concatenate()([Y1, Y2], dim=1)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Skip this test on SM75")
            return
        dll_name = f"test_rrr_padding_{test_name}.so"
        module = gen_execution_module(
            [Y], target, "./tmp", "pad_gemm_with_cat_rrr", dll_name=dll_name
        )

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate y_shape: {}".format(y_shape))

        for m in ms:
            X1_pt = torch.randn(m, k1).cuda().half()
            W1_pt = torch.randn(k, n).cuda().half()
            X2_pt = torch.randn(m, k2).cuda().half()
            W2_pt = torch.randn(k, n).cuda().half()
            X4_pt = torch.cat([X1_pt, X2_pt], dim=1)
            Y1_pt = torch.matmul(X4_pt, W1_pt)
            Y2_pt = torch.matmul(X4_pt, W2_pt)
            Y_pt = torch.cat([Y1_pt, Y2_pt], dim=1)

            inputs = [0] * 4
            name_to_idx = module.GetInputNameToIndexMap()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["x2"]] = X2_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["w2"]] = W2_pt
            y = torch.empty(Y_pt.size()).cuda().half()
            module.RunWithTensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_pad_gemm_rrr_with_cat(self):
        self._test_pad_gemm_rrr_with_cat("static_odd_k", ms=[128], n=32, k1=3, k2=10)
        self._test_pad_gemm_rrr_with_cat("static_odd_kn", ms=[128], n=31, k1=1, k2=8)
        self._test_pad_gemm_rrr_with_cat(
            "dynamic_odd_kn", ms=[2, 5, 7], n=15, k1=1, k2=2
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
