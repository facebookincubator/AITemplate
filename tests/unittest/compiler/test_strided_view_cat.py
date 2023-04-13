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
from typing import List

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm, IntVar, Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.testing import detect_target, test_utils
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from aitemplate.utils import graph_utils
from parameterized import param, parameterized


_TOLERANCE_LIMITS = {
    "float16": {"atol": 1e-2, "rtol": 1e-2},
    "float32": {"atol": 1e-2, "rtol": 1e-2},
    "bfloat16": {"atol": 3e-1, "rtol": 3e-1},
}


def custom_name_func(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_{param.args[0]}"


class StridedViewCatOpTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    @parameterized.expand(
        [
            param(
                "gemm_reshape_cat_fusible_simple",
                n=2,
                new_shape=[-1, 2, 2],
                cat_dim=2,
                expected_num_tensors=10,
                expected_num_ops=9,
            ),
            param(
                "gemm_reshape_cat_fusible_expand_2",
                n=4,
                new_shape=[-1, 4, 4, 1],
                cat_dim=2,
                expected_num_tensors=10,
                expected_num_ops=9,
            ),
            param(
                "gemm_reshape_cat_fusible_expand_3",
                n=2,
                new_shape=[-1, 2, 2, 1],
                cat_dim=2,
                expected_num_tensors=10,
                expected_num_ops=9,
            ),
            param(
                "gemm_reshape_cat_fusible_expand_4",
                n=4,
                new_shape=[-1, 4, 2, 2],
                cat_dim=2,
                expected_num_tensors=10,
                expected_num_ops=9,
            ),
            param(
                "gemm_reshape_cat_non_fusible_dynamic_dim",
                n=2,
                new_shape=[-1, 2],
                cat_dim=1,
                expected_num_tensors=25,
                expected_num_ops=18,
            ),
            param(
                "gemm_reshape_cat_non_fusible_stride_dim",
                n=2,
                new_shape=[-1, 2 * 2],
                cat_dim=1,
                expected_num_tensors=14,
                expected_num_ops=9,
            ),
            param(
                # Concat along rightmost unsqueezed dim - not fusible.
                "gemm_reshape_cat_non_fusible_stride_dim_rightmost_unsqueezed",
                n=2,
                new_shape=[-1, 2, 2, 1],
                cat_dim=3,
                expected_num_tensors=16,
                expected_num_ops=9,
            ),
            param(
                # Concat along inner unsqueezed dim - fusible.
                "gemm_reshape_cat_fusible_stride_dim_inner_unsqueezed",
                n=2,
                new_shape=[-1, 2, 1, 2],
                cat_dim=2,
                expected_num_tensors=10,
                expected_num_ops=9,
            ),
        ],
        name_func=custom_name_func,
    )
    def test_strided_gemm_view_cat_fusible(
        self,
        test_name: str,
        n: int,
        new_shape: List[int],
        cat_dim: int,
        expected_num_tensors: int,
        expected_num_ops: int,
        dtype: str = "float16",
    ):
        self._test_strided_gemm_view_cat_fusible(
            test_name,
            n,
            new_shape,
            cat_dim,
            expected_num_tensors,
            expected_num_ops,
            dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_strided_gemm_view_cat_fusible_dtype(self, dtype):
        self._test_strided_gemm_view_cat_fusible(
            f"gemm_reshape_cat_non_fusible_expand_{dtype}",
            n=4,
            new_shape=[-1, 4, 2, 2],
            cat_dim=3,
            expected_num_tensors=16,
            expected_num_ops=9,
            dtype=dtype,
        )
        self._test_strided_gemm_view_cat_fusible(
            f"gemm_reshape_cat_fusible_expand_{dtype}",
            n=2,
            new_shape=[-1, 2, 1, 2],
            cat_dim=3,
            expected_num_tensors=10,
            expected_num_ops=9,
            dtype=dtype,
        )

    def _test_strided_gemm_view_cat_fusible(
        self,
        test_name: str,
        n: int,
        new_shape: List[int],
        cat_dim: int,
        expected_num_tensors: int,
        expected_num_ops: int,
        dtype: str = "float16",
    ):
        target = detect_target()

        batch_dim = IntVar([1, 2, 3], "batch_size")
        input0 = test_utils.gen_input_tensor(
            [batch_dim, n, n], name="input0", dtype=dtype
        )
        input1 = test_utils.gen_input_tensor([n, n], name="input1", dtype=dtype)
        input2 = test_utils.gen_input_tensor(
            [batch_dim, n, n], name="input2", dtype=dtype
        )
        input3 = test_utils.gen_input_tensor([n], name="input3", dtype=dtype)
        input4 = test_utils.gen_input_tensor(
            [batch_dim, n, n], name="input4", dtype=dtype
        )
        input5 = test_utils.gen_input_tensor(
            [batch_dim, n, n], name="input5", dtype=dtype
        )
        input6 = test_utils.gen_input_tensor([n, n, n], name="input6", dtype=dtype)

        X0 = ops.gemm_rcr()(input0, input1)
        X1 = ops.gemm_rcr_bias()(input0, input1, input3)
        X2 = ops.gemm_rcr_bias_add()(input0, input1, input3, input4)
        X3 = ops.gemm_rcr_bias_add_add()(input0, input1, input3, input4, input4)
        X4 = ops.bmm_rcr()(input0, input2)
        X5 = ops.bmm_rrr_add()(input0, input2, input3)

        # [m, b, k] x [b, n, k] -> [m, b, n] b = n, k = n
        X6 = ops.perm102_bmm_rcr()(input0, input6)
        X7 = ops.perm102_bmm_rrr()(input0, input6)

        Xs = [X2, X1, X0, X3, X4, X5, X6, X7]
        Ys = [ops.reshape()(X, new_shape) for X in Xs]
        Ys.insert(2, ops.reshape()(input5, new_shape))
        Z = ops.concatenate()(Ys, dim=cat_dim)
        Z._attrs["name"] = "output0"
        Z._attrs["is_output"] = True

        # Gen module.
        module = compile_model([Z], target, "./tmp", test_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), expected_num_tensors)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_num_ops)

        # Prepare PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input0_pt = get_random_torch_tensor([batch_size, n, n], dtype)
            input1_pt = get_random_torch_tensor([n, n], dtype)
            input2_pt = get_random_torch_tensor([batch_size, n, n], dtype)
            input3_pt = get_random_torch_tensor([n], dtype)
            input4_pt = get_random_torch_tensor([batch_size, n, n], dtype)
            input5_pt = get_random_torch_tensor([batch_size, n, n], dtype)
            input6_pt = get_random_torch_tensor([n, n, n], dtype)

            # Run PyTorch baseline.
            x0_pt = torch.nn.functional.linear(input0_pt, input1_pt)
            x1_pt = torch.nn.functional.linear(input0_pt, input1_pt, input3_pt)
            x2_pt = (
                torch.nn.functional.linear(input0_pt, input1_pt, input3_pt) + input4_pt
            )
            x3_pt = (
                torch.nn.functional.linear(input0_pt, input1_pt, input3_pt)
                + input4_pt
                + input4_pt
            )
            x4_pt = torch.bmm(input0_pt, input2_pt.transpose(1, 2))
            x5_pt = torch.bmm(input0_pt, input2_pt) + input3_pt
            x6_pt = torch.bmm(
                input0_pt.permute(1, 0, 2), input6_pt.permute(0, 2, 1)
            ).permute(1, 0, 2)
            x7_pt = torch.bmm(input0_pt.permute(1, 0, 2), input6_pt).permute(1, 0, 2)

            xs_pt = [x2_pt, x1_pt, x0_pt, x3_pt, x4_pt, x5_pt, x6_pt, x7_pt]
            ys_pt = [torch.reshape(x, new_shape) for x in xs_pt]
            ys_pt.insert(2, torch.reshape(input5_pt, new_shape))
            z_pt = torch.cat(ys_pt, dim=cat_dim)
            z = get_torch_empty_tensor(z_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors(
                {
                    "input0": input0_pt,
                    "input1": input1_pt,
                    "input2": input2_pt,
                    "input3": input3_pt,
                    "input4": input4_pt,
                    "input5": input5_pt,
                    "input6": input6_pt,
                },
                [z],
            )

            # Do comparisons.
            torch.testing.assert_close(z, z_pt, **_TOLERANCE_LIMITS[dtype])

    def _test_strided_layernorm_view_cat_fusible(self, dtype="float16"):
        def _create_layernorm_sigmoid_mul(
            input: Tensor,
            normalized_shape: List[int],
            gamma: Tensor = None,
            beta: Tensor = None,
        ) -> Tensor:
            X1 = ops.layernorm([IntImm(s) for s in normalized_shape])(
                input, gamma, beta
            )
            X2 = ops.elementwise(FuncEnum.SIGMOID)(X1)
            X3 = ops.elementwise(FuncEnum.MUL)(X2, input)
            return X3

        batch_dim = IntVar([1, 2, 3], "batch_size")
        m = 5
        n = 10
        new_shape = [-1, m, n * 2]
        cat_dim = 1
        # layernorm + reshape
        input0 = test_utils.gen_input_tensor(
            [batch_dim, m, 2, n], name="input0", dtype=dtype
        )
        # group layernorm + reshape
        gamma = test_utils.gen_input_tensor([m * n], name="g", dtype=dtype)
        beta = test_utils.gen_input_tensor([m * n], name="b", dtype=dtype)
        input1 = test_utils.gen_input_tensor(
            [batch_dim, 2, m * n], name="input1", dtype=dtype
        )
        input2 = test_utils.gen_input_tensor(
            [batch_dim, 2, m * n], name="input2", dtype=dtype
        )
        # layernorm + nop reshape
        input3 = test_utils.gen_input_tensor(
            [batch_dim, m, n * 2], name="input3", dtype=dtype
        )

        X0 = _create_layernorm_sigmoid_mul(input0, [n])
        X1 = _create_layernorm_sigmoid_mul(input1, [m * n], gamma, beta)
        X2 = _create_layernorm_sigmoid_mul(input2, [m * n], gamma, beta)
        X3 = _create_layernorm_sigmoid_mul(input3, [n * 2])
        Xs = [X0, X1, X2, X3]
        Ys = [ops.reshape()(X, new_shape) for X in Xs]
        Z = ops.concatenate()(Ys, dim=cat_dim)

        Z._attrs["name"] = "output0"
        Z._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(
            Z, target, "./tmp", f"strided_layernorm_view_cat_fusion_{dtype}"
        )

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 7)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 3)

        # Prepare PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input0_pt = get_random_torch_tensor([batch_size, m, 2, n], dtype)
            input1_pt = get_random_torch_tensor([batch_size, 2, m * n], dtype)
            input2_pt = get_random_torch_tensor([batch_size, 2, m * n], dtype)
            gamma_pt = get_random_torch_tensor([m * n], dtype)
            beta_pt = get_random_torch_tensor([m * n], dtype)
            input3_pt = get_random_torch_tensor([batch_size, m, n * 2], dtype)

            # Run PyTorch baseline.
            x0_pt = torch.nn.functional.layer_norm(input0_pt, [n])
            x0_pt = torch.mul(input0_pt, torch.sigmoid(x0_pt))
            x1_pt = torch.nn.functional.layer_norm(
                input1_pt, [m * n], weight=gamma_pt, bias=beta_pt
            )
            x1_pt = torch.mul(input1_pt, torch.sigmoid(x1_pt))
            x2_pt = torch.nn.functional.layer_norm(
                input2_pt, [m * n], weight=gamma_pt, bias=beta_pt
            )
            x2_pt = torch.mul(input2_pt, torch.sigmoid(x2_pt))
            x3_pt = torch.nn.functional.layer_norm(input3_pt, [n * 2])
            x3_pt = torch.mul(input3_pt, torch.sigmoid(x3_pt))

            xs_pt = [x0_pt, x1_pt, x2_pt, x3_pt]
            ys_pt = [torch.reshape(x, new_shape) for x in xs_pt]
            z_pt = torch.cat(ys_pt, dim=cat_dim)
            z = get_torch_empty_tensor(z_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors(
                {
                    "input0": input0_pt,
                    "input1": input1_pt,
                    "input2": input2_pt,
                    "input3": input3_pt,
                    "g": gamma_pt,
                    "b": beta_pt,
                },
                [z],
            )

            # Do comparisons.
            for x, x_pt in zip(z, z_pt):
                torch.testing.assert_close(x, x_pt, **_TOLERANCE_LIMITS[dtype])

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [],
            }
        )
    )
    def test_strided_layernorm_view_cat_fusible(self, dtype):
        self._test_strided_layernorm_view_cat_fusible(dtype)


if __name__ == "__main__":
    unittest.main()
