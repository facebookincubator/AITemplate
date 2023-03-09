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
"""
Unittests for elementwise fusion out-of-order cases.
"""
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)

from parameterized import parameterized
from torch import nn


class FusedElementwiseOutOfOrderTestCase(unittest.TestCase):
    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_fused_elementwise_out_of_order(self, dtype):
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
        target = detect_target()
        if dtype == "float" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        M = 10
        K = 4
        N = 4
        X0 = Tensor(
            shape=[M, K],
            dtype=dtype,
            name="X0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[],
            dtype=dtype,
            name="X1",
            value=3.0,
        )
        X2 = Tensor(
            shape=[M, K],
            dtype=dtype,
            name="X2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[K, N],
            dtype=dtype,
            name="X3",
            is_input=True,
        )
        X4 = Tensor(
            shape=[K, N],
            dtype=dtype,
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

        module = compile_model(
            R5,
            target,
            "./tmp",
            f"fused_elementwise_out_of_order_{dtype}",
        )

        x0_pt = get_random_torch_tensor([M, K], dtype)
        x2_pt = get_random_torch_tensor([M, K], dtype)
        x3_pt = get_random_torch_tensor([K, N], dtype)
        x4_pt = get_random_torch_tensor([K, N], dtype)

        r0_pt = x0_pt + 3
        r1_pt = nn.functional.linear(r0_pt, x3_pt)
        r2_pt = nn.functional.linear(x2_pt, x4_pt)
        r3_pt = r0_pt - r2_pt
        r4_pt = nn.functional.linear(r3_pt, x4_pt)
        r5_pt = r1_pt - r4_pt

        r5 = get_torch_empty_tensor([M, N], dtype)

        input_name_to_idx_mapping = module.get_input_name_to_index_map()
        inputs = [None] * len(input_name_to_idx_mapping)
        input_name_to_pt_mapping = {
            "X0": x0_pt,
            "X2": x2_pt,
            "X3": x3_pt,
            "X4": x4_pt,
        }
        for input_name, pt in input_name_to_pt_mapping.items():
            inputs[input_name_to_idx_mapping[input_name]] = pt
        module.run_with_tensors(inputs, [r5])
        self.assertTrue(torch.allclose(r5, r5_pt, atol=1e-2, rtol=1e-2))

    def test_fused_elementwise_out_of_order_with_size(self):
        pass


if __name__ == "__main__":
    unittest.main()
