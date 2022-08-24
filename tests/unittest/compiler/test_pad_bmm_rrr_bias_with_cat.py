# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import itertools
import logging
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import logger, shape_utils


class PadBmmBiasWithCatTestCase(unittest.TestCase):
    def _test_pad_bmm_rrr_bias_with_cat(self, test_name, bs, ms, n, k1, k2):
        k = k1 + k2
        b_dim = shape_utils.gen_int_var_min_max(bs, name="b")
        m_dim = shape_utils.gen_int_var_min_max(ms, name="m")
        X1 = Tensor(shape=[b_dim, m_dim, k1], dtype="float16", name="x1", is_input=True)
        X2 = Tensor(shape=[b_dim, m_dim, k2], dtype="float16", name="x2", is_input=True)
        X4 = ops.concatenate()([X1, X2], dim=2)

        W1 = Tensor(shape=[b_dim, k, n], dtype="float16", name="w1", is_input=True)
        B1 = Tensor(shape=[b_dim, m_dim, n], dtype="float16", name="b1", is_input=True)
        W2 = Tensor(shape=[b_dim, k, n], dtype="float16", name="w2", is_input=True)
        B2 = Tensor(shape=[b_dim, m_dim, n], dtype="float16", name="b2", is_input=True)
        Y1 = ops.bmm_rrr_add()(X4, W1, B1)
        Y2 = ops.bmm_rrr_add()(X4, W2, B2)

        Y = ops.concatenate()([Y1, Y2], dim=2)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Skip this test on SM75")
            return
        module = gen_execution_module(
            [Y], target, "./tmp", f"test_bmm_rrr_padding_{test_name}"
        )

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate y_shape: {}".format(y_shape))

        for b, m in itertools.product(bs, ms):
            X1_pt = torch.randn(b, m, k1).cuda().half()
            X2_pt = torch.randn(b, m, k2).cuda().half()
            X4_pt = torch.cat([X1_pt, X2_pt], dim=2)

            W1_pt = torch.randn(b, k, n).cuda().half()
            B1_pt = torch.randn(b, m, n).cuda().half()
            W2_pt = torch.randn(b, k, n).cuda().half()
            B2_pt = torch.randn(b, m, n).cuda().half()

            Y1_pt = torch.baddbmm(B1_pt, X4_pt, W1_pt)
            Y2_pt = torch.baddbmm(B2_pt, X4_pt, W2_pt)
            Y_pt = torch.cat([Y1_pt, Y2_pt], dim=2)

            inputs = [0] * 6
            name_to_idx = module.GetInputNameToIndexMap()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["x2"]] = X2_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["w2"]] = W2_pt
            inputs[name_to_idx["b1"]] = B1_pt
            inputs[name_to_idx["b2"]] = B2_pt

            y = torch.empty(Y_pt.size()).cuda().half()
            module.RunWithTensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_pad_bmm_rrr_bias_with_cat(self):
        self._test_pad_bmm_rrr_bias_with_cat(
            "static_odd_k", bs=[2], ms=[64], n=32, k1=3, k2=10
        )
        self._test_pad_bmm_rrr_bias_with_cat(
            "static_odd_kn", bs=[2], ms=[128], n=31, k1=1, k2=8
        )
        self._test_pad_bmm_rrr_bias_with_cat(
            "dynamic_odd_kn", bs=[1, 2, 3], ms=[2, 5, 7], n=15, k1=1, k2=2
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
