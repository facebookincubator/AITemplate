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
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.testing import detect_target, test_utils
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from aitemplate.utils import graph_utils

from parameterized import parameterized


_TOLERANCE_LIMITS = {
    "float16": {"atol": 5e-2, "rtol": 5e-2},
    "float32": {"atol": 5e-2, "rtol": 5e-2},
    "bfloat16": {"atol": 3e-1, "rtol": 3e-1},
}


class SliceViewStridedOpTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [],
            }
        )
    )
    def test_slice_view_gemm_fusible(self, dtype):
        N = 4
        batch_dim = IntVar([1, 2, 3], "batch_size")

        input0 = test_utils.gen_input_tensor(
            [batch_dim, 2 * N, N], dtype=dtype, name="input0"
        )
        X0 = ops.dynamic_slice()(input0, [None, None, None], [None, N, None])
        X1 = ops.reshape()(X0, [-1, N * N])
        input1 = test_utils.gen_input_tensor([N, N * N], dtype=dtype, name="input1")
        Y = ops.gemm_rcr()(X1, input1)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(
            [Y], target, "./tmp", f"slice_reshape_gemm_fusible_{dtype}"
        )

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)

        # Prepare PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            # Run PyTorch baseline.
            input0_pt = get_random_torch_tensor([batch_size, 2 * N, N], dtype)
            x0_pt = input0_pt[:, :N, :]
            x1_pt = torch.reshape(x0_pt, [-1, N * N])
            input1_pt = get_random_torch_tensor([N, N * N], dtype)
            y_pt = torch.nn.functional.linear(x1_pt, input1_pt)
            y = get_torch_empty_tensor(y_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors(
                {
                    "input0": input0_pt,
                    "input1": input1_pt,
                },
                [y],
            )

            # Do comparisons.
            torch.testing.assert_close(y, y_pt, **_TOLERANCE_LIMITS[dtype])

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [],
            }
        )
    )
    def test_slice_view_gemm_non_fusible(self, dtype):
        N = 4
        batch_dim = IntVar([1, 2, 3], "batch_size")

        input0 = test_utils.gen_input_tensor(
            [batch_dim, N, 2 * N], dtype=dtype, name="input0"
        )
        X0 = ops.dynamic_slice()(input0, [None, None, None], [None, None, N])
        X1 = ops.reshape()(X0, [-1, N * N])
        input1 = test_utils.gen_input_tensor([N, N * N], dtype=dtype, name="input1")
        Y = ops.gemm_rcr()(X1, input1)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(
            [Y], target, "./tmp", f"slice_reshape_gemm_non_fusible_{dtype}"
        )

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepare PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            # Run PyTorch baseline.
            input0_pt = get_random_torch_tensor([batch_size, N, 2 * N], dtype)
            x0_pt = input0_pt[:, :, :N]
            x1_pt = torch.reshape(x0_pt, [-1, N * N])
            input1_pt = get_random_torch_tensor([N, N * N], dtype) * 0.5
            y_pt = torch.nn.functional.linear(x1_pt, input1_pt)
            y = get_torch_empty_tensor(y_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors(
                {
                    "input0": input0_pt,
                    "input1": input1_pt,
                },
                [y],
            )

            # Do comparisons.
            torch.testing.assert_close(y, y_pt, **_TOLERANCE_LIMITS[dtype])

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [],
            }
        )
    )
    def test_slice_flatten_concat_fusible_1(self, dtype):
        test_name = f"slice_flatten_concat_fusible_{dtype}"
        batch_dim = IntVar([3, 10], "batch_size")
        X0 = test_utils.gen_input_tensor([batch_dim, 12, 1], dtype=dtype, name="x0")
        X1 = test_utils.gen_input_tensor([batch_dim, 12, 1], dtype=dtype, name="x1")
        X2 = test_utils.gen_input_tensor([batch_dim, 10], dtype=dtype, name="x2")
        A = test_utils.gen_input_tensor([batch_dim, 8, 48], dtype=dtype, name="a")
        B = test_utils.gen_input_tensor([batch_dim, 48, 40], dtype=dtype, name="b")

        start_indices = [0, 0, 0]
        end_indices = [None, None, 39]
        squeeze_dim = 2
        cat_dim = 1
        flatten_start_dim = 1
        flatten_end_dim = -1

        Y0 = ops.bmm_rrr()(A, B)
        Y1 = ops.dynamic_slice()(Y0, start_indices, end_indices)
        Y2 = ops.flatten(start_dim=flatten_start_dim, end_dim=flatten_end_dim)(Y1)
        Y3 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        Y4 = ops.squeeze(squeeze_dim)(Y3)
        Y = ops.concatenate()([X2, Y2, Y4], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model([Y], target, "./tmp", test_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 7)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 3)

        # Prepare PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            # Run PyTorch baseline.
            x0_pt = get_random_torch_tensor([batch_size, 12, 1], dtype)
            x1_pt = get_random_torch_tensor([batch_size, 12, 1], dtype)
            x2_pt = get_random_torch_tensor([batch_size, 10], dtype)
            a_pt = get_random_torch_tensor([batch_size, 8, 48], dtype)
            b_pt = get_random_torch_tensor([batch_size, 48, 40], dtype)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]

            y0_pt = torch.bmm(a_pt, b_pt)
            y1_pt = y0_pt[slice_indices]
            y2_pt = torch.flatten(
                y1_pt, start_dim=flatten_start_dim, end_dim=flatten_end_dim
            )
            y3_pt = x0_pt + x1_pt
            y4_pt = torch.squeeze(y3_pt, dim=squeeze_dim)
            y_pt = torch.cat([x2_pt, y2_pt, y4_pt], dim=cat_dim)
            y = get_torch_empty_tensor(y_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors(
                {
                    "x0": x0_pt,
                    "x1": x1_pt,
                    "x2": x2_pt,
                    "a": a_pt,
                    "b": b_pt,
                },
                [y],
            )

            # Do comparisons.
            torch.testing.assert_close(y, y_pt, **_TOLERANCE_LIMITS[dtype])

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [],
            }
        )
    )
    def test_slice_flatten_concat_fusible_2(self, dtype):
        test_name = f"slice_flatten_concat_fusible_{dtype}_2"
        batch_dim = IntVar([1, 2], "batch_size")
        X0 = test_utils.gen_input_tensor([batch_dim, 2, 1], dtype=dtype, name="x0")
        X1 = test_utils.gen_input_tensor([batch_dim, 2, 1], dtype=dtype, name="x1")
        X2 = test_utils.gen_input_tensor([batch_dim, 1], dtype=dtype, name="x2")
        A = test_utils.gen_input_tensor([batch_dim, 2, 1], dtype=dtype, name="a")
        B = test_utils.gen_input_tensor([batch_dim, 1, 2], dtype=dtype, name="b")

        start_indices = [0, 0, 0]
        end_indices = [None, None, 3]
        reshape_to = [-1, 2]
        cat_dim = 1
        flatten_start_dim = 1
        flatten_end_dim = -1

        Y0 = ops.bmm_rrr()(A, B)
        Y1 = ops.dynamic_slice()(Y0, start_indices, end_indices)
        Y2 = ops.flatten(start_dim=flatten_start_dim, end_dim=flatten_end_dim)(Y1)
        Y3 = X0 + X1
        Y4 = ops.reshape()(Y3, reshape_to)
        Y = ops.concatenate()([Y4, Y2, X2, Y4], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model([Y], target, "./tmp", test_name)

        # Prepare PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            # Run PyTorch baseline.
            x0_pt = get_random_torch_tensor([batch_size, 2, 1], dtype)
            x1_pt = get_random_torch_tensor([batch_size, 2, 1], dtype)
            x2_pt = get_random_torch_tensor([batch_size, 1], dtype)
            a_pt = get_random_torch_tensor([batch_size, 2, 1], dtype)
            b_pt = get_random_torch_tensor([batch_size, 1, 2], dtype)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]

            y0_pt = torch.bmm(a_pt, b_pt)
            y1_pt = y0_pt[slice_indices]
            y2_pt = torch.flatten(
                y1_pt, start_dim=flatten_start_dim, end_dim=flatten_end_dim
            )
            y3_pt = x0_pt + x1_pt
            y4_pt = y3_pt.reshape(*reshape_to)
            y_pt = torch.cat([y4_pt, y2_pt, x2_pt, y4_pt], dim=cat_dim)
            y = get_torch_empty_tensor(y_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors(
                {
                    "x0": x0_pt,
                    "x1": x1_pt,
                    "x2": x2_pt,
                    "a": a_pt,
                    "b": b_pt,
                },
                [y],
            )

            # Do comparisons.
            torch.testing.assert_close(y, y_pt, **_TOLERANCE_LIMITS[dtype])

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [],
            }
        )
    )
    def test_slice_reshape_concat_fusible_1(self, dtype):
        test_name = f"slice_reshape_concat_fusible_{dtype}_1"
        batch_dim = IntVar([1, 2], "batch_size")
        M = 2
        N = 2
        K = 1

        X0 = test_utils.gen_input_tensor([batch_dim, 1], dtype=dtype, name="x0")
        X1 = test_utils.gen_input_tensor([batch_dim, 1], dtype=dtype, name="x1")
        A = test_utils.gen_input_tensor([batch_dim, K, M], dtype=dtype, name="a")
        B = test_utils.gen_input_tensor([batch_dim, K, N], dtype=dtype, name="b")
        D = test_utils.gen_input_tensor([N], dtype=dtype, name="d")

        start_indices = [0, 0, 0]
        end_indices = [None, None, 1]
        reshape_to = [-1, M * (N - 1)]
        cat_dim = 1

        Y0 = ops.bmm_crr_add()(A, B, D)
        Y1 = ops.dynamic_slice()(Y0, start_indices, end_indices)
        Y2 = ops.reshape()(Y1, reshape_to)
        Y3 = ops.concatenate()([Y2, X0], dim=cat_dim)
        Y = ops.concatenate()([Y3, X1], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model([Y], target, "./tmp", test_name)

        # Prepare PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            # Run PyTorch baseline.
            x0_pt = get_random_torch_tensor([batch_size, 1], dtype)
            x1_pt = get_random_torch_tensor([batch_size, 1], dtype)
            a_pt = get_random_torch_tensor([batch_size, K, M], dtype)
            b_pt = get_random_torch_tensor([batch_size, K, N], dtype)
            d_pt = get_random_torch_tensor([N], dtype)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]

            y0_pt = torch.bmm(a_pt.permute([0, 2, 1]), b_pt)
            y0_pt = y0_pt + d_pt
            y1_pt = y0_pt[slice_indices]
            y2_pt = y1_pt.reshape(*reshape_to)
            y3_pt = torch.cat([y2_pt, x0_pt], dim=cat_dim)
            y_pt = torch.cat([y3_pt, x1_pt], dim=cat_dim)
            y = get_torch_empty_tensor(y_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors(
                {
                    "x0": x0_pt,
                    "x1": x1_pt,
                    "a": a_pt,
                    "b": b_pt,
                    "d": d_pt,
                },
                [y],
            )

            # Do comparisons.
            torch.testing.assert_close(y, y_pt, **_TOLERANCE_LIMITS[dtype])

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [],
            }
        )
    )
    def test_slice_reshape_concat_fusible_2(self, dtype):
        test_name = f"slice_reshape_concat_fusible_{dtype}_2"
        batch_dim = IntVar([1, 8], "batch_size")
        M = 8
        N = 64
        K = 4

        K1_0 = 32 * 8
        K1_1 = 3

        # K1 = 259: need padding
        K1 = K1_0 + K1_1
        N1 = 256

        X0 = test_utils.gen_input_tensor([batch_dim, M, K], dtype=dtype, name="x0")
        W0 = test_utils.gen_input_tensor([N, K], dtype=dtype, name="w0")
        X1 = test_utils.gen_input_tensor([batch_dim, K1_1], dtype=dtype, name="x1")
        W1 = test_utils.gen_input_tensor([N1, K1], dtype=dtype, name="w1")

        start_indices = [0, 0, 32]
        end_indices = [None, None, 64]
        reshape_to = [-1, K1_0]
        cat_dim = 1

        Y0 = ops.gemm_rcr()(X0, W0)
        Y1 = ops.dynamic_slice()(Y0, start_indices, end_indices)
        Y2 = ops.reshape()(Y1, reshape_to)
        Y3 = ops.concatenate()([Y2, X1], dim=cat_dim)
        Y = ops.gemm_rcr()(Y3, W1)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model([Y], target, "./tmp", test_name)

        # Prepare PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            # Run PyTorch baseline.
            x0_pt = get_random_torch_tensor([batch_size, M, K], dtype)
            w0_pt = get_random_torch_tensor([N, K], dtype)
            x1_pt = get_random_torch_tensor([batch_size, K1_1], dtype)
            w1_pt = get_random_torch_tensor([N1, K1], dtype)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]

            y0_pt = torch.nn.functional.linear(x0_pt, w0_pt)
            y1_pt = y0_pt[slice_indices]
            y2_pt = y1_pt.reshape(*reshape_to)
            y3_pt = torch.cat([y2_pt, x1_pt], dim=cat_dim)
            y_pt = torch.nn.functional.linear(y3_pt, w1_pt)
            y = get_torch_empty_tensor(y_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors(
                {
                    "x0": x0_pt,
                    "w0": w0_pt,
                    "x1": x1_pt,
                    "w1": w1_pt,
                },
                [y],
            )

            # Do comparisons.
            torch.testing.assert_close(y, y_pt, **_TOLERANCE_LIMITS[dtype])


if __name__ == "__main__":
    unittest.main()
