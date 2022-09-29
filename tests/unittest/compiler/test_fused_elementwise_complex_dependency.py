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
Unittests for elementwise fusion with complex dependencies.
"""
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import graph_utils
from torch import nn


class FusedElementwiseComplexDependencyTestCase(unittest.TestCase):
    def test_fused_elementwise_direct_input_dependency(self):
        r"""
            X0   X1
             \   /
              Add_1        X2
               |    \      /
               |      Add_2
                \      /
                  Sub_1

        Add_1, Add_2, and Sub_1 should be fused together.
        """

        M = 10
        N = 4
        X0 = Tensor(
            shape=[M, N],
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
            shape=[M, N],
            dtype="float16",
            name="X2",
            is_input=True,
        )

        R0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        R1 = ops.elementwise(FuncEnum.ADD)(R0, X2)
        R2 = ops.elementwise(FuncEnum.SUB)(R0, R1)
        R2._attrs["name"] = "R2"
        R2._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            R2,
            target,
            "./tmp",
            "fused_elementwise_direct_input_dependency",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)

        x0_pt = torch.rand(M, N).cuda().half()
        x2_pt = torch.rand(M, N).cuda().half()

        r0_pt = x0_pt + 3 + x2_pt
        r1_pt = r0_pt + x2_pt
        r2_pt = r0_pt - r1_pt

        r2 = torch.empty([M, N]).cuda().half()

        input_name_to_idx_mapping = module.get_input_name_to_index_map()
        inputs = [None] * len(input_name_to_idx_mapping)
        input_name_to_pt_mapping = {
            "X0": x0_pt,
            "X2": x2_pt,
        }
        for input_name, pt in input_name_to_pt_mapping.items():
            inputs[input_name_to_idx_mapping[input_name]] = pt
        module.run_with_tensors(inputs, [r2])
        self.assertTrue(torch.allclose(r2, r2_pt, atol=1e-2, rtol=1e-2))

    def test_fused_elementwise_indirect_input_dependency(self):
        r"""
            X0   X1
             \   /
              Add_1        X2
               |    \      /
               |     Gemm_1
               |        |
               |      Tanh_1
                \      /
                  Sub_1

        Tanh_1 and Sub_1 should be fused together.
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
            shape=[K, N],
            dtype="float16",
            name="X2",
            is_input=True,
        )

        R0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        R1 = ops.gemm_rcr()(R0, X2)
        R2 = ops.elementwise(FuncEnum.TANH)(R1)
        R3 = ops.elementwise(FuncEnum.SUB)(R0, R2)
        R3._attrs["name"] = "R3"
        R3._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            R3,
            target,
            "./tmp",
            "fused_elementwise_indirect_input_dependency",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 3)

        x0_pt = torch.rand(M, K).cuda().half()
        x2_pt = torch.rand(K, N).cuda().half()

        r0_pt = x0_pt + 3
        r1_pt = nn.functional.linear(r0_pt, x2_pt)
        r2_pt = torch.tanh(r1_pt)
        r3_pt = r0_pt - r2_pt

        r3 = torch.empty([M, N]).cuda().half()

        input_name_to_idx_mapping = module.get_input_name_to_index_map()
        inputs = [None] * len(input_name_to_idx_mapping)
        input_name_to_pt_mapping = {
            "X0": x0_pt,
            "X2": x2_pt,
        }
        for input_name, pt in input_name_to_pt_mapping.items():
            inputs[input_name_to_idx_mapping[input_name]] = pt
        module.run_with_tensors(inputs, [r3])
        self.assertTrue(torch.allclose(r3, r3_pt, atol=1e-2, rtol=1e-2))

    def test_fused_elementwise_multi_dependency(self):
        r"""
            X0   X1                X3
             \   /                 |
              Add_1        X2     Tanh_2      X4
               |    \      /       |    \     /
               |     Gemm_1        |    Gemm_2
               |        |           \    /
               |     Tanh_1         Sub_2
                \      /             |
                  Sub_1              |
                      \             /
                        \         /
                           Add_2

        Tanh_1, Sub_1, Sub_2 and Add_2 should be fused together.
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
            shape=[K, N],
            dtype="float16",
            name="X2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[M, K],
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
        R1 = ops.gemm_rcr()(R0, X2)
        R2 = ops.elementwise(FuncEnum.TANH)(R1)
        R3 = ops.elementwise(FuncEnum.SUB)(R0, R2)
        R4 = ops.elementwise(FuncEnum.TANH)(X3)
        R5 = ops.gemm_rcr()(R4, X4)
        R6 = ops.elementwise(FuncEnum.SUB)(R4, R5)
        R7 = ops.elementwise(FuncEnum.ADD)(R6, R3)
        R7._attrs["name"] = "R7"
        R7._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            R7,
            target,
            "./tmp",
            "fused_elementwise_multi_dependency",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 5)

        x0_pt = torch.rand(M, K).cuda().half()
        x2_pt = torch.rand(K, N).cuda().half()
        x3_pt = torch.rand(M, K).cuda().half()
        x4_pt = torch.rand(K, N).cuda().half()

        r0_pt = x0_pt + 3
        r1_pt = nn.functional.linear(r0_pt, x2_pt)
        r2_pt = torch.tanh(r1_pt)
        r3_pt = r0_pt - r2_pt
        r4_pt = torch.tanh(x3_pt)
        r5_pt = nn.functional.linear(r4_pt, x4_pt)
        r6_pt = r4_pt - r5_pt
        r7_pt = r6_pt + r3_pt

        r7 = torch.empty([M, N]).cuda().half()

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
        module.run_with_tensors(inputs, [r7])
        self.assertTrue(torch.allclose(r7, r7_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
