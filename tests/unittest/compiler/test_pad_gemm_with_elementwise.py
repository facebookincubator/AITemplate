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
import itertools
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
from aitemplate.utils import shape_utils
from parameterized import param, parameterized


class PadGemmWithElementwise(unittest.TestCase):
    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [
                    param("static_M_float16", [23], 7, 3, "float16"),
                    param("dynamic_M_float16", [1, 78, 99], 7, 3, "float16"),
                ],
                TestEnv.CUDA_SM80: [
                    param("dynamic_M_float32", [1, 78, 99], 7, 3, "float32"),
                ],
            }
        )
    )
    def test_pad_gemm_rcr_bias_broadcast_with_elementwise(
        self, test_name, ms, n, k, dtype
    ):
        target = detect_target()
        if dtype == "float32" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        m_dim = shape_utils.gen_int_var_min_max(ms, "M")

        X1 = Tensor(shape=[m_dim, k], dtype=dtype, name="x1", is_input=True)
        W1 = Tensor(shape=[n, k], dtype=dtype, name="w1", is_input=True)
        B1 = Tensor(shape=[n], dtype=dtype, name="b1", is_input=True)
        S1 = Tensor(shape=[m_dim, n], dtype=dtype, name="s1", is_input=True)
        S2 = Tensor(shape=[m_dim, n], dtype=dtype, name="s2", is_input=True)

        X2 = ops.gemm_rcr_bias_mul_add()(X1, W1, B1, S1, S2)
        Y = ops.elementwise(FuncEnum.ADD)(X2, X2)

        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        module = compile_model(
            [Y], target, "./tmp", f"pad_gemm_with_elementwise_{test_name}"
        )

        for m in ms:
            X1_pt = get_random_torch_tensor([m, k], dtype)
            W1_pt = get_random_torch_tensor([n, k], dtype)
            B1_pt = get_random_torch_tensor([n], dtype)
            S1_pt = get_random_torch_tensor([m, n], dtype)
            S2_pt = get_random_torch_tensor([m, n], dtype)

            X2_pt = torch.nn.functional.linear(X1_pt, W1_pt, B1_pt) * S1_pt + S2_pt
            Y_pt = X2_pt + X2_pt

            inputs = [0] * 5
            name_to_idx = module.get_input_name_to_index_map()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["b1"]] = B1_pt
            inputs[name_to_idx["s1"]] = S1_pt
            inputs[name_to_idx["s2"]] = S2_pt
            y = get_torch_empty_tensor(Y_pt.size(), dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [
                    ("static_shape_float16", [3], [1], 5, 3, "float16"),
                    ("dynamic_M_float16", [3], [1, 78, 99], 7, 3, "float16"),
                    ("dynamic_B_float16", [3, 5, 8], [3], 11, 15, "float16"),
                    (
                        "dynamic_BM_float16",
                        [3, 5, 8],
                        [3, 9, 10],
                        17,
                        21,
                        "float16",
                    ),
                ],
                TestEnv.CUDA_SM80: [
                    ("static_shape_float32", [3], [1], 5, 3, "float32"),
                    (
                        "dynamic_BM_float32",
                        [3, 5, 8],
                        [3, 9, 10],
                        17,
                        21,
                        "float32",
                    ),
                ],
            }
        )
    )
    def test_pad_bmm_rrr_add_with_elementwise(self, test_name, bs, ms, n, k, dtype):
        target = detect_target()
        if dtype == "float32" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        b_dim = shape_utils.gen_int_var_min_max(bs, "B")
        m_dim = shape_utils.gen_int_var_min_max(ms, "M")

        X1 = Tensor(shape=[b_dim, m_dim, k], dtype=dtype, name="x1", is_input=True)
        W1 = Tensor(shape=[b_dim, k, n], dtype=dtype, name="w1", is_input=True)
        B1 = Tensor(shape=[b_dim, m_dim, n], dtype=dtype, name="b1", is_input=True)

        X2 = ops.bmm_rrr_add()(X1, W1, B1)
        Y = ops.elementwise(FuncEnum.ADD)(X2, X2)

        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        module = compile_model(
            [Y], target, "./tmp", f"pad_bmm_with_elementwise_{test_name}"
        )

        for b, m in itertools.product(bs, ms):
            X1_pt = get_random_torch_tensor([b, m, k], dtype)
            W1_pt = get_random_torch_tensor([b, k, n], dtype)
            B1_pt = get_random_torch_tensor([b, m, n], dtype)

            X2_pt = torch.matmul(X1_pt, W1_pt) + B1_pt
            Y_pt = X2_pt + X2_pt

            inputs = [0, 0, 0]
            name_to_idx = module.get_input_name_to_index_map()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["b1"]] = B1_pt
            y = get_torch_empty_tensor(Y_pt.size(), dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [
                    ("static_shape_float16", [3], [1], 5, 3, "float16"),
                    ("dynamic_M_float16", [3], [1, 78, 99], 7, 3, "float16"),
                    ("dynamic_B_float16", [3, 5, 8], [3], 11, 15, "float16"),
                    (
                        "dynamic_BM_float16",
                        [3, 5, 8],
                        [3, 9, 10],
                        17,
                        21,
                        "float16",
                    ),
                ],
                TestEnv.CUDA_SM80: [
                    ("static_shape_float32", [3], [1], 5, 3, "float32"),
                    (
                        "dynamic_BM_float32",
                        [3, 5, 8],
                        [3, 9, 10],
                        17,
                        21,
                        "float32",
                    ),
                ],
            }
        )
    )
    def test_pad_perm102_bmm_rrr_with_elementwise(self, test_name, bs, ms, n, k, dtype):
        target = detect_target()
        if dtype == "float32" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        b_dim = shape_utils.gen_int_var_min_max(bs, "B")
        m_dim = shape_utils.gen_int_var_min_max(ms, "M")

        # (M, B, K) * (B, K, N) = (M, B, N)
        X1 = Tensor(shape=[m_dim, b_dim, k], dtype=dtype, name="x1", is_input=True)
        W1 = Tensor(shape=[b_dim, k, n], dtype=dtype, name="w1", is_input=True)
        B1 = Tensor(shape=[b_dim, n], dtype=dtype, name="b1", is_input=True)

        X2 = ops.perm102_bmm_rrr_bias()(X1, W1, B1)
        Y = ops.elementwise(FuncEnum.ADD)(X2, X2)

        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        module = compile_model(
            [Y], target, "./tmp", f"pad_perm102_with_elementwise_{test_name}"
        )

        for b, m in itertools.product(bs, ms):
            X1_pt = get_random_torch_tensor([m, b, k], dtype)
            W1_pt = get_random_torch_tensor([b, k, n], dtype)
            B1_pt = get_random_torch_tensor([b, n], dtype)
            Bias_pt = B1_pt.unsqueeze(1)

            X2_pt = torch.permute(
                torch.baddbmm(Bias_pt, torch.permute(X1_pt, (1, 0, 2)), W1_pt),
                (1, 0, 2),
            )
            Y_pt = X2_pt + X2_pt
            inputs = [0, 0, 0]
            name_to_idx = module.get_input_name_to_index_map()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["b1"]] = B1_pt
            y = get_torch_empty_tensor(Y_pt.size(), dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [
                    param("static_M_float16", [23], 7, 3, "float16"),
                    param("dynamic_M_float16", [1, 78, 99], 7, 3, "float16"),
                ],
                TestEnv.CUDA_SM80: [
                    param("dynamic_M_float32", [1, 78, 99], 7, 3, "float32"),
                ],
            }
        )
    )
    def test_pad_gemm_rcr_bias_broadcast_with_elementwise_2(
        self, test_name, ms, n, k, dtype
    ):
        target = detect_target()
        if dtype == "float32" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        # S1 is fed to gemm twice
        m_dim = shape_utils.gen_int_var_min_max(ms, "M")

        X1 = Tensor(shape=[m_dim, k], dtype=dtype, name="x1", is_input=True)
        W1 = Tensor(shape=[n, k], dtype=dtype, name="w1", is_input=True)
        B1 = Tensor(shape=[n], dtype=dtype, name="b1", is_input=True)
        S1 = Tensor(shape=[m_dim, n], dtype=dtype, name="s1", is_input=True)

        X2 = ops.gemm_rcr_bias_mul_add()(X1, W1, B1, S1, S1)
        Y = ops.elementwise(FuncEnum.ADD)(X2, X2)

        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        module = compile_model(
            [Y], target, "./tmp", f"pad_gemm_with_elementwise_2_{test_name}"
        )

        for m in ms:
            X1_pt = get_random_torch_tensor([m, k], dtype)
            W1_pt = get_random_torch_tensor([n, k], dtype)
            B1_pt = get_random_torch_tensor([n], dtype)
            S1_pt = get_random_torch_tensor([m, n], dtype)

            X2_pt = torch.nn.functional.linear(X1_pt, W1_pt, B1_pt) * S1_pt + S1_pt
            Y_pt = X2_pt + X2_pt

            inputs = [0] * 4
            name_to_idx = module.get_input_name_to_index_map()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["b1"]] = B1_pt
            inputs[name_to_idx["s1"]] = S1_pt
            y = get_torch_empty_tensor(Y_pt.size(), dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
