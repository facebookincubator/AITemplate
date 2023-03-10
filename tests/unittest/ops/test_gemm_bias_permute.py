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
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
)


class GEMMBiasPermuteTestCase(unittest.TestCase):
    def _test_gemm_rcr_bias_permute_m2n3(
        self,
        copy_op=False,
        test_name="gemm_rcr_bias_permute_m2n3",
        dtype="float16",
    ):
        M0 = 4
        M1 = 256
        N0 = 4
        N1 = 16
        N2 = 128
        M = M0 * M1
        N = N0 * N1 * N2
        K = 256
        shape = (M1, N0, N1)
        target = detect_target()
        X = Tensor(
            shape=[M, K],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B = Tensor(
            shape=[N],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        OP = ops.gemm_rcr_bias_permute(shape, layout="m2n3")
        if copy_op:
            OP = ops.gemm_rcr_bias_permute(**OP._get_op_attributes())
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)
        X_pt = get_random_torch_tensor([M, K], dtype=dtype)
        W_pt = get_random_torch_tensor([N, K], dtype=dtype)
        B_pt = get_random_torch_tensor([N], dtype=dtype)

        Y_l = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
        Y_r = Y_l.reshape(M0, M1, N0, N1, N2)
        Y_pt = torch.permute(Y_r, [2, 0, 3, 1, 4])

        inputs = [X_pt, W_pt, B_pt]
        y = torch.empty_like(Y_pt)
        module.run_with_tensors(inputs, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_gemm_rcr_bias_permute_m2n3_fp16_rocm(self):
        self._test_gemm_rcr_bias_permute_m2n3(
            test_name="gemm_rcr_bias_permute_m2n3_fp16",
            dtype="float16",
        )
        self._test_gemm_rcr_bias_permute_m2n3(
            copy_op=True,
            test_name="gemm_rcr_bias_permute_m2n3_fp16_copy_op",
            dtype="float16",
        )

    def _test_gemm_rcr_bias_permute_m3n2(
        self,
        copy_op=False,
        test_name="gemm_rcr_bias_permute_m3n2",
        dtype="float16",
    ):
        M0 = 4
        M1 = 16
        M2 = 32
        N0 = 8
        N1 = 128
        M = M0 * M1 * M2
        N = N0 * N1
        K = 256
        shape = (M1, M2, N0)
        target = detect_target()
        X = Tensor(
            shape=[M, K],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B = Tensor(
            shape=[N],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        OP = ops.gemm_rcr_bias_permute(shape, layout="m3n2")
        if copy_op:
            OP = ops.gemm_rcr_bias_permute(**OP._get_op_attributes())
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "gemm_rcr_bias_permute_m3n2")
        X_pt = get_random_torch_tensor([M, K], dtype=dtype)
        W_pt = get_random_torch_tensor([N, K], dtype=dtype)
        B_pt = get_random_torch_tensor([N], dtype=dtype)
        Y_l = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
        Y_r = Y_l.reshape(M0, M1, M2, N0, N1)
        Y_pt = torch.permute(Y_r, [2, 0, 3, 1, 4])

        inputs = [X_pt, W_pt, B_pt]
        y = torch.empty_like(Y_pt)
        module.run_with_tensors(inputs, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_gemm_rcr_bias_permute_m3n2_fp16_rocm(self):
        self._test_gemm_rcr_bias_permute_m3n2(
            test_name="gemm_rcr_bias_permute_m3n2_fp16",
            dtype="float16",
        )
        self._test_gemm_rcr_bias_permute_m3n2(
            copy_op=True,
            test_name="gemm_rcr_bias_permute_m3n2_fp16_copy_op",
            dtype="float16",
        )

    def _test_gemm_rcr_permute_m2n3(
        self,
        copy_op=False,
        test_name="gemm_rcr_permute_m2n3",
        dtype="float16",
    ):
        M0 = 4
        M1 = 256
        N0 = 4
        N1 = 16
        N2 = 128
        M = M0 * M1
        N = N0 * N1 * N2
        K = 256
        shape = (M1, N0, N1)
        target = detect_target()
        X = Tensor(
            shape=[M, K],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        OP = ops.gemm_rcr_permute(shape, layout="m2n3")
        if copy_op:
            OP = ops.gemm_rcr_permute(**OP._get_op_attributes())
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "gemm_rcr_permute_m2n3")
        X_pt = get_random_torch_tensor([M, K], dtype=dtype)
        W_pt = get_random_torch_tensor([N, K], dtype=dtype)

        Y_l = torch.nn.functional.linear(X_pt, W_pt)
        Y_r = Y_l.reshape(M0, M1, N0, N1, N2)
        Y_pt = torch.permute(Y_r, [2, 0, 3, 1, 4])

        inputs = [X_pt, W_pt]
        y = torch.empty_like(Y_pt)
        module.run_with_tensors(inputs, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_gemm_rcr_permute_m2n3_fp16_rocm(self):
        self._test_gemm_rcr_permute_m2n3(
            test_name="test_gemm_rcr_permute_m2n3_fp16",
            dtype="float16",
        )
        self._test_gemm_rcr_permute_m2n3(
            copy_op=True,
            test_name="test_gemm_rcr_permute_m2n3_fp16_copy_op",
            dtype="float16",
        )

    # ========== enable them after fix profiler =========
    # def test_gemm_rcr_bias_relu_rocm(self):
    #     M0 = 4
    #     M1 = 32
    #     M2 = 128
    #     N0 = 16
    #     N1 = 256
    #     M = M0 * M1 * M2
    #     N = N0 * N1
    #     K = 128
    #     shape = (M1, M2, N0)
    #     target = detect_target()
    #     X = Tensor(shape=[M, K], dtype="float16", name="input_0", is_input=True)
    #     W = Tensor(shape=[N, K], dtype="float16", name="input_1", is_input=True)
    #     B = Tensor(shape=[N], dtype="float16", name="input_2", is_input=True)
    #     OP = ops.gemm_rcr_bias_permute(shape)
    #     Y = OP(X, W, B)
    #     Y._attrs["name"] = "output_0"
    #     Y._attrs["is_output"] = True
    #     module = compile_model(Y, target, "./tmp", "gemm_rcr_bias_permute")
    #     X_pt = torch.randn(M, K).cuda().half()
    #     W_pt = torch.randn(N, K).cuda().half()
    #     B_pt = torch.randn(N).cuda().half()
    #     Y_l = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
    #     Y_r = Y_l.reshape(M0, M1, M2, N0, N1)
    #     Y_pt = torch.permute(Y_r, [2, 0, 3, 1, 4])

    #     inputs = {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}
    #     y = torch.empty(Y_pt.shape).cuda().half()
    #     module.run_with_tensors(inputs, [y])
    #     self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    # def test_gemm_rrr_bias_relu_rocm(self):
    #     M0 = 4
    #     M1 = 32
    #     M2 = 128
    #     N0 = 16
    #     N1 = 256
    #     M = M0 * M1 * M2
    #     N = N0 * N1
    #     K = 128
    #     shape = (M1, M2, N0)
    #     target = detect_target()
    #     X = Tensor(shape=[M, K], dtype="float16", name="input_0", is_input=True)
    #     W = Tensor(shape=[K, N], dtype="float16", name="input_1", is_input=True)
    #     B = Tensor(shape=[N], dtype="float16", name="input_2", is_input=True)
    #     OP = ops.gemm_rrr_bias_permute(shape)
    #     Y = OP(X, W, B)
    #     Y._attrs["name"] = "output_0"
    #     Y._attrs["is_output"] = True
    #     module = compile_model(Y, target, "./tmp", "gemm_rrr_bias_permute")
    #     X_pt = torch.randn(M, K).cuda().half()
    #     W_pt = torch.randn(K, N).cuda().half()
    #     B_pt = torch.randn(N).cuda().half()
    #     Y_l = torch.matmul(X_pt, W_pt) + B_pt
    #     Y_r = Y_l.reshape(M0, M1, M2, N0, N1)
    #     Y_pt = torch.permute(Y_r, [2, 0, 3, 1, 4])

    #     inputs = {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}
    #     y = torch.empty(Y_pt.shape).cuda().half()
    #     module.run_with_tensors(inputs, [y])
    #     self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


filter_test_cases_by_test_env(GEMMBiasPermuteTestCase)


if __name__ == "__main__":
    unittest.main()
