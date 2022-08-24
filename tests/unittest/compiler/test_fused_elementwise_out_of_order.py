# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Unittests for elementwise fusion out-of-order cases.
"""
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from torch import nn


class FusedElementwiseOutOfOrderTestCase(unittest.TestCase):
    def test_fused_elementwise_out_of_order(self):
        r"""
           X0   X1
            \   /
             Add_1      Gemm_2(X2, X4)
              |    \      /
        Gemm_1(X3)   Sub_1
              |        |
              |      Gemm_3(X4)
               \      /
                 Sub_2

        Add_1 and Sub_1 will be fused together.
        However tensor order needs to be re-adjusted.

        Original order:
        [X0, X1, R0, X3, R1, X2, X4, R2, R3, R4, R5]

        New order:
        [X2, X4, R2, X0, X1, R0, X3, R1, R3, R4, R5]
        """

        M = 10
        K = 4
        N = 4
        X0 = Tensor(
            shape=[M, K],
            dtype="float16",
            name="X0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[],
            dtype="float16",
            name="X1",
            value=3.0,
        )
        X2 = Tensor(
            shape=[M, K],
            dtype="float16",
            name="X2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[K, N],
            dtype="float16",
            name="X3",
            is_input=True,
        )
        X4 = Tensor(
            shape=[K, N],
            dtype="float16",
            name="X4",
            is_input=True,
        )

        R0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        R1 = ops.gemm_rcr()(R0, X3)
        R2 = ops.gemm_rcr()(X2, X4)
        R3 = ops.elementwise(FuncEnum.SUB)(R0, R2)
        R4 = ops.gemm_rcr()(R3, X4)
        R5 = ops.elementwise(FuncEnum.SUB)(R1, R4)
        R5._attrs["name"] = "R5"
        R5._attrs["is_output"] = True

        target = detect_target()
        module = gen_execution_module(
            R5,
            target,
            "./tmp",
            "fused_elementwise_out_of_order",
        )

        x0_pt = torch.rand(M, K).cuda().half()
        x2_pt = torch.rand(M, K).cuda().half()
        x3_pt = torch.rand(K, N).cuda().half()
        x4_pt = torch.rand(K, N).cuda().half()

        r0_pt = x0_pt + 3
        r1_pt = nn.functional.linear(r0_pt, x3_pt)
        r2_pt = nn.functional.linear(x2_pt, x4_pt)
        r3_pt = r0_pt - r2_pt
        r4_pt = nn.functional.linear(r3_pt, x4_pt)
        r5_pt = r1_pt - r4_pt

        r5 = torch.empty([M, N]).cuda().half()

        input_name_to_idx_mapping = module.GetInputNameToIndexMap()
        inputs = [None] * len(input_name_to_idx_mapping)
        input_name_to_pt_mapping = {
            "X0": x0_pt,
            "X2": x2_pt,
            "X3": x3_pt,
            "X4": x4_pt,
        }
        for input_name, pt in input_name_to_pt_mapping.items():
            inputs[input_name_to_idx_mapping[input_name]] = pt
        module.RunWithTensors(inputs, [r5])
        self.assertTrue(torch.allclose(r5, r5_pt, atol=1e-2, rtol=1e-2))

    def test_fused_elementwise_out_of_order_with_size(self):
        pass


if __name__ == "__main__":
    unittest.main()
