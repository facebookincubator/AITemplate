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
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from aitemplate.utils import graph_utils

from parameterized import parameterized
from torch import nn


class FusedElementwiseComplexDependencyTestCase(unittest.TestCase):
    @parameterized.expand([("float16"), ("float")])
    def test_fused_elementwise_direct_input_dependency(self, dtype):
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
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by rocm")

        M = 10
        N = 4
        X0 = Tensor(
            shape=[M, N],
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
            shape=[M, N],
            dtype=dtype,
            name="X2",
            is_input=True,
        )

        R0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        R1 = ops.elementwise(FuncEnum.ADD)(R0, X2)
        R2 = ops.elementwise(FuncEnum.SUB)(R0, R1)
        R2._attrs["name"] = "R2"
        R2._attrs["is_output"] = True

        module = compile_model(
            R2,
            target,
            "./tmp",
            f"fused_elementwise_direct_input_dependency_{dtype}",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)

        x0_pt = get_random_torch_tensor([M, N], dtype)
        x2_pt = get_random_torch_tensor([M, N], dtype)

        r0_pt = x0_pt + 3 + x2_pt
        r1_pt = r0_pt + x2_pt
        r2_pt = r0_pt - r1_pt

        r2 = get_torch_empty_tensor([M, N], dtype)

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

    @parameterized.expand([("float16"), ("float")])
    def test_fused_elementwise_direct_input_dependency_split_subgraph(self, dtype):
        r"""
        X3[K,N]   X0[N]   X1[]
           |         \   /
           |     Add_1[N]  X2[M,N]
            \      /  |  \    /
             Add[K,N] |  Add_2[M, N]
                       \     /
                       Sub_1 [M,N]

           Add_1, Add_2, and Sub_1 should be fused together.
        """
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by rocm")

        M = 10
        N = 4
        K = 15
        X0 = Tensor(
            shape=[N],
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
            shape=[M, N],
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

        R0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        R1 = ops.elementwise(FuncEnum.ADD)(R0, X2)
        R2 = ops.elementwise(FuncEnum.SUB)(R0, R1)
        R3 = ops.elementwise(FuncEnum.ADD)(R0, X3)
        R2._attrs["name"] = "R2"
        R2._attrs["is_output"] = True
        R3._attrs["name"] = "R3"
        R3._attrs["is_output"] = True

        module = compile_model(
            [R3, R2],
            target,
            "./tmp",
            f"fused_elementwise_direct_input_dependency_split_subgraph{dtype}",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        x0_pt = get_random_torch_tensor([N], dtype)  # N
        x2_pt = get_random_torch_tensor([M, N], dtype)
        x3_pt = get_random_torch_tensor([K, N], dtype)

        r0_pt = x0_pt + 3
        r3_pt = r0_pt + x3_pt
        r1_pt = r0_pt + x2_pt
        r2_pt = r0_pt - r1_pt

        r2 = get_torch_empty_tensor([M, N], dtype)
        r3 = get_torch_empty_tensor([K, N], dtype)  # N

        input_name_to_idx_mapping = module.get_input_name_to_index_map()
        inputs = [None] * len(input_name_to_idx_mapping)
        input_name_to_pt_mapping = {
            "X0": x0_pt,
            "X2": x2_pt,
            "X3": x3_pt,
        }
        for input_name, pt in input_name_to_pt_mapping.items():
            inputs[input_name_to_idx_mapping[input_name]] = pt
        module.run_with_tensors(inputs, [r3, r2])
        self.assertTrue(torch.allclose(r2, r2_pt, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(r3, r3_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand([("float16"), ("float")])
    def test_fused_elementwise_non_elementwise_ops(self, dtype):
        r"""
                X0   X1 (3)
                 \   /
                  Add_1 (R0)   X2
                   |    \      /
                   |      Add_2 (R1, is_output)
                  / \      /
        (R3) reshape   Sub_1 (R2)
               |
              Add_3 (R4)


            Add_1, Add_2, and Sub_1 should be fused together.
        """
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by rocm")

        M = 10
        N = 4
        X0 = Tensor(
            shape=[M, N],
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
            shape=[M, N],
            dtype=dtype,
            name="X2",
            is_input=True,
        )

        R0 = ops.elementwise(FuncEnum.ADD)(X0, X1)  # Add_1
        R1 = ops.elementwise(FuncEnum.ADD)(R0, X2)  # Add_2
        R2 = ops.elementwise(FuncEnum.SUB)(R0, R1)
        R3 = ops.reshape()(R0, [-1])
        R4 = ops.elementwise(FuncEnum.ADD)(R3, R3)  # Add3
        R1._attrs["name"] = "R1"
        R1._attrs["is_output"] = True
        R2._attrs["name"] = "R2"
        R2._attrs["is_output"] = True
        R4._attrs["name"] = "R4"
        R4._attrs["is_output"] = True

        module = compile_model(
            [R1, R2, R4],
            target,
            "./tmp",
            f"test_fused_elementwise_non_elementwise_ops_{dtype}",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 4)

        x0_pt = get_random_torch_tensor([M, N], dtype)
        x2_pt = get_random_torch_tensor([M, N], dtype)

        r0_pt = x0_pt + 3
        r1_pt = r0_pt + x2_pt
        r2_pt = r0_pt - r1_pt
        r3_pt = r0_pt.reshape([-1])
        r4_pt = r3_pt + r3_pt

        r1 = get_torch_empty_tensor(r1_pt.shape, dtype)
        r2 = get_torch_empty_tensor([M, N], dtype)
        r4 = get_torch_empty_tensor(r4_pt.shape, dtype)

        input_name_to_idx_mapping = module.get_input_name_to_index_map()
        inputs = [None] * len(input_name_to_idx_mapping)
        input_name_to_pt_mapping = {
            "X0": x0_pt,
            "X2": x2_pt,
        }
        for input_name, pt in input_name_to_pt_mapping.items():
            inputs[input_name_to_idx_mapping[input_name]] = pt
        module.run_with_tensors(inputs, {"R1": r1, "R2": r2, "R4": r4})
        self.assertTrue(torch.allclose(r1, r1_pt, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(r2, r2_pt, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(r4, r4_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_fused_elementwise_indirect_input_dependency(self, dtype):
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
            shape=[K, N],
            dtype=dtype,
            name="X2",
            is_input=True,
        )

        R0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        R1 = ops.gemm_rcr()(R0, X2)
        R2 = ops.elementwise(FuncEnum.TANH)(R1)
        R3 = ops.elementwise(FuncEnum.SUB)(R0, R2)
        R3._attrs["name"] = "R3"
        R3._attrs["is_output"] = True

        module = compile_model(
            R3,
            target,
            "./tmp",
            f"fused_elementwise_indirect_input_dependency_{dtype}",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 3)

        x0_pt = get_random_torch_tensor([M, K], dtype)
        x2_pt = get_random_torch_tensor([K, N], dtype)

        r0_pt = x0_pt + 3
        r1_pt = nn.functional.linear(r0_pt, x2_pt)
        r2_pt = torch.tanh(r1_pt)
        r3_pt = r0_pt - r2_pt

        r3 = get_torch_empty_tensor([M, N], dtype)

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

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_fused_elementwise_indirect_input_dependency_split_subgraph(self, dtype):
        r"""
                X0[M,K] X1[]
                 \      /
                  Add_1      X2[K,N]
                   |    \      /
                   |     Gemm_1
                   |        |
        X3[P,M,N]  |      Tanh_1 (output)
              \    |           |
                Sub_1          |
                   |          /
                Sub_2 (output)
            Tanh_1 and Sub_1 should be fused together.
        """
        target = detect_target()
        if dtype == "float" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        M = 10
        K = 4
        N = 4
        P = 15
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
            shape=[K, N],
            dtype=dtype,
            name="X2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[P, M, N],
            dtype=dtype,
            name="X3",
            is_input=True,
        )

        R0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        R1 = ops.gemm_rcr()(R0, X2)
        R2 = ops.elementwise(FuncEnum.TANH)(R1)
        R3 = ops.elementwise(FuncEnum.SUB)(X3, R0)
        R4 = ops.elementwise(FuncEnum.SUB)(R3, R2)
        R3._attrs["name"] = "R3"
        R3._attrs["is_output"] = True
        R4._attrs["name"] = "R4"
        R4._attrs["is_output"] = True

        module = compile_model(
            [R3, R4],
            target,
            "./tmp",
            f"fused_elementwise_indirect_input_dependency_split_subgraph{dtype}",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 4)

        x0_pt = get_random_torch_tensor([M, K], dtype)
        x2_pt = get_random_torch_tensor([K, N], dtype)
        x3_pt = get_random_torch_tensor([P, M, N], dtype)

        r0_pt = x0_pt + 3
        r1_pt = nn.functional.linear(r0_pt, x2_pt)
        r2_pt = torch.tanh(r1_pt)
        r3_pt = x3_pt - r0_pt
        r4_pt = r3_pt - r2_pt

        r3 = get_torch_empty_tensor([P, M, N], dtype)
        r4 = get_torch_empty_tensor([P, M, N], dtype)

        input_name_to_idx_mapping = module.get_input_name_to_index_map()
        inputs = [None] * len(input_name_to_idx_mapping)
        input_name_to_pt_mapping = {
            "X0": x0_pt,
            "X2": x2_pt,
            "X3": x3_pt,
        }
        for input_name, pt in input_name_to_pt_mapping.items():
            inputs[input_name_to_idx_mapping[input_name]] = pt
        module.run_with_tensors(inputs, [r3, r4])
        self.assertTrue(torch.allclose(r3, r3_pt, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(r4, r4_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_fused_elementwise_multi_dependency(self, dtype):
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
            shape=[K, N],
            dtype=dtype,
            name="X2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[M, K],
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

        x0_pt = get_random_torch_tensor([M, K], dtype)
        x2_pt = get_random_torch_tensor([K, N], dtype)
        x3_pt = get_random_torch_tensor([M, K], dtype)
        x4_pt = get_random_torch_tensor([K, N], dtype)

        r0_pt = x0_pt + 3
        r1_pt = nn.functional.linear(r0_pt, x2_pt)
        r2_pt = torch.tanh(r1_pt)
        r3_pt = r0_pt - r2_pt
        r4_pt = torch.tanh(x3_pt)
        r5_pt = nn.functional.linear(r4_pt, x4_pt)
        r6_pt = r4_pt - r5_pt
        r7_pt = r6_pt + r3_pt

        r7 = get_torch_empty_tensor([M, N], dtype)

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

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_fused_elementwise_find_fusable_graph(self, dtype):
        r"""
                     X0
                     |
                    Abs
                   /   \
            X1   Tanh  |
             \    /    |
              Gemm   Relu
                \      |
                 Exp   |
                   \  /
                   Sub

        Tanh, Abs, Relu should be fused together;  Sub, Exp should be fused together.
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
            shape=[K, N],
            dtype=dtype,
            name="X1",
            is_input=True,
        )

        R0 = ops.elementwise(FuncEnum.ABS)(X0)
        R1 = ops.elementwise(FuncEnum.TANH)(R0)
        R2 = ops.gemm_rcr()(R1, X1)
        R3 = ops.elementwise(FuncEnum.EXP)(R2)
        R4 = ops.elementwise(FuncEnum.RELU)(R0)
        R5 = ops.elementwise(FuncEnum.SUB)(R4, R3)
        R5._attrs["name"] = "R5"
        R5._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            R5,
            target,
            "./tmp",
            "fused_elementwise_find_fusable_graph",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 3)

        x0_pt = get_random_torch_tensor([M, K], dtype)
        x1_pt = get_random_torch_tensor([K, N], dtype)
        relu = torch.nn.ReLU()
        r0_pt = torch.abs(x0_pt)
        r1_pt = torch.tanh(r0_pt)
        r2_pt = nn.functional.linear(r1_pt, x1_pt)
        r3_pt = torch.exp(r2_pt)
        r4_pt = relu(r0_pt)
        r5_pt = r4_pt - r3_pt

        r5 = get_torch_empty_tensor([M, N], dtype)

        input_name_to_idx_mapping = module.get_input_name_to_index_map()
        inputs = [None] * len(input_name_to_idx_mapping)
        input_name_to_pt_mapping = {
            "X0": x0_pt,
            "X1": x1_pt,
        }
        for input_name, pt in input_name_to_pt_mapping.items():
            inputs[input_name_to_idx_mapping[input_name]] = pt
        module.run_with_tensors(inputs, [r5])
        self.assertTrue(torch.allclose(r5, r5_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
