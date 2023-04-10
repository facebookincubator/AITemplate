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
    get_torch_empty_tensor,
)

from parameterized import parameterized


def custom_name_func_with_funcname(testcase_func, param_num, param):
    return "%s_%s_%s" % (
        testcase_func.__name__[:-5],
        str(param.args[0].__name__),
        testcase_func.__name__[-4:],
    )


class GEMMBiasBroadcastTestCase(unittest.TestCase):
    def _init_tensors(self, m, k, n, m0=None, m1=None, dtype="float16"):
        m_shape = [m] if m is not None else [m0, m1]
        self.X = Tensor(shape=m_shape + [k], dtype=dtype, name="input_0", is_input=True)
        self.W = Tensor(shape=[n, k], dtype=dtype, name="input_1", is_input=True)
        self.B = Tensor(shape=[n], dtype=dtype, name="input_2", is_input=True)
        self.D0 = Tensor(shape=m_shape + [n], dtype=dtype, name="d0", is_input=True)
        self.D1 = Tensor(shape=m_shape + [n], dtype=dtype, name="d1", is_input=True)
        self.X_pt = get_random_torch_tensor([*m_shape, k], dtype)
        self.W_pt = get_random_torch_tensor([n, k], dtype)
        self.B_pt = get_random_torch_tensor([n], dtype)
        self.D0_pt = get_random_torch_tensor([*m_shape, n], dtype)
        self.D1_pt = get_random_torch_tensor([*m_shape, n], dtype)

    def _test_and_verify(
        self,
        module,
        torch_output,
        dtype,
        has_d1=False,
        module_output_name="output_0",
    ):
        inputs = {
            "input_0": self.X_pt,
            "input_1": self.W_pt,
            "input_2": self.B_pt,
            "d0": self.D0_pt,
        }
        if has_d1:
            inputs["d1"] = self.D1_pt
        y = get_torch_empty_tensor(list(torch_output.shape), dtype)
        module.run_with_tensors(inputs, [y])
        if self.X_pt.nelement() == 0 or self.W_pt.nelement() == 0:
            pass
        else:
            torch.testing.assert_close(torch_output, y, atol=1e-1, rtol=1e-1)

    def _test_bias_rcr_mul_add(
        self,
        m,
        m0,
        m1,
        k,
        n,
        dtype="float16",
        test_name_suffix="",
    ):
        target = detect_target()
        self._init_tensors(m, k, n, m0, m1, dtype)
        OP = ops.gemm_rcr_bias_mul_add()
        Y = OP(self.X, self.W, self.B, self.D0, self.D1)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"gemm_rcr_bias_mul_add_k_{k}_n_{n}_{dtype}{test_name_suffix}",
        )
        Y_pt = (
            torch.nn.functional.linear(self.X_pt, self.W_pt, bias=self.B_pt)
            * self.D0_pt
            + self.D1_pt
        )
        self._test_and_verify(module, Y_pt, dtype, has_d1=True)

    def test_bias_rcr_mul_add(self):
        self._test_bias_rcr_mul_add(8, None, None, 8, 8)
        self._test_bias_rcr_mul_add(None, 2, 32, 256, 128)
        self._test_bias_rcr_mul_add(None, 21, 5, 1024, 512)

    def test_bias_rcr_mul_add_rocm(self):
        self._test_bias_rcr_mul_add(8, None, None, 8, 8, test_name_suffix="_rocm")

    def _test_bias_rcr_sigmoid_mul(
        self,
        m,
        m0,
        m1,
        k,
        n,
        dtype="float16",
        test_name_suffix="",
    ):
        target = detect_target()
        self._init_tensors(m, k, n, m0, m1, dtype)
        OP = ops.gemm_rcr_bias_sigmoid_mul()
        Y = OP(self.X, self.W, self.B, self.D0)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"gemm_rcr_bias_sigmoid_mul_k_{k}_n_{n}_{dtype}{test_name_suffix}",
        )

        Y_pt = (
            torch.sigmoid(
                torch.nn.functional.linear(self.X_pt, self.W_pt, bias=self.B_pt)
            )
            * self.D0_pt
        )
        self._test_and_verify(module, Y_pt, dtype)

    def test_bias_rcr_sigmoid_mul(self):
        self._test_bias_rcr_sigmoid_mul(8, None, None, 8, 8)
        self._test_bias_rcr_sigmoid_mul(None, 2, 32, 256, 128)
        self._test_bias_rcr_sigmoid_mul(None, 21, 5, 1024, 512)

    def test_bias_rcr_sigmoid_mul_rocm(self):
        self._test_bias_rcr_sigmoid_mul(8, None, None, 8, 8, test_name_suffix="_rocm")

    def _test_bias_rcr_sigmoid_mul_tanh(
        self,
        m,
        m0,
        m1,
        k,
        n,
        dtype="float16",
        test_name_suffix="",
    ):
        target = detect_target()
        self._init_tensors(m, k, n, m0, m1, dtype)
        OP = ops.gemm_rcr_bias_sigmoid_mul_tanh()
        Y = OP(self.X, self.W, self.B, self.D0)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"gemm_rcr_bias_sigmoid_mul_tanh_k_{k}_n_{n}_{dtype}{test_name_suffix}",
        )

        Y_pt = torch.tanh(
            torch.sigmoid(
                torch.nn.functional.linear(self.X_pt, self.W_pt, bias=self.B_pt)
            )
            * self.D0_pt
        )
        self._test_and_verify(module, Y_pt, dtype)

    def test_bias_rcr_sigmoid_mul_tanh(self):
        self._test_bias_rcr_sigmoid_mul_tanh(8, None, None, 8, 8)
        self._test_bias_rcr_sigmoid_mul_tanh(None, 2, 32, 256, 128)
        self._test_bias_rcr_sigmoid_mul_tanh(None, 21, 5, 1024, 512)
        self._test_bias_rcr_sigmoid_mul_tanh(None, 21, 5, 1024, 0)

    def test_bias_rcr_sigmoid_mul_tanh_rocm(self):
        self._test_bias_rcr_sigmoid_mul_tanh(
            8, None, None, 8, 8, test_name_suffix="_rocm"
        )

    def _test_bias_rcr_add(
        self,
        m,
        m0,
        m1,
        k,
        n,
        dtype="float16",
        test_name_suffix="",
    ):
        target = detect_target()
        self._init_tensors(m, k, n, m0, m1, dtype)
        OP = ops.gemm_rcr_bias_add()
        Y = OP(self.X, self.W, self.B, self.D0)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"gemm_rcr_bias_add_k_{k}_n_{n}_{dtype}{test_name_suffix}",
        )

        Y_pt = (
            torch.nn.functional.linear(self.X_pt, self.W_pt, bias=self.B_pt)
            + self.D0_pt
        )
        self._test_and_verify(module, Y_pt, dtype)

    def test_bias_rcr_add(self):
        self._test_bias_rcr_add(8, None, None, 8, 8)
        self._test_bias_rcr_add(None, 2, 32, 256, 128)
        self._test_bias_rcr_add(None, 21, 5, 1024, 512)

    def test_bias_rcr_add_rocm(self):
        self._test_bias_rcr_add(8, None, None, 8, 8, test_name_suffix="_rocm")

    def _test_bias_rcr_add_relu(
        self,
        m,
        m0,
        m1,
        k,
        n,
        dtype="float16",
        test_name_suffix="",
    ):
        target = detect_target()
        self._init_tensors(m, k, n, m0, m1, dtype)
        OP = ops.gemm_rcr_bias_add_relu()
        Y = OP(self.X, self.W, self.B, self.D0)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"gemm_rcr_bias_add_relu_k_{k}_n_{n}_{dtype}{test_name_suffix}",
        )

        Y_pt = torch.relu(
            torch.nn.functional.linear(self.X_pt, self.W_pt, bias=self.B_pt)
            + self.D0_pt
        )
        self._test_and_verify(module, Y_pt, dtype)

    def test_bias_rcr_add_relu(self):
        self._test_bias_rcr_add_relu(8, None, None, 8, 8)
        self._test_bias_rcr_add_relu(None, 2, 32, 256, 128)
        self._test_bias_rcr_add_relu(None, 21, 5, 1024, 512)

    def test_bias_rcr_add_relu_rocm(self):
        self._test_bias_rcr_add_relu(8, None, None, 8, 8, test_name_suffix="_rocm")

    def _test_bias_rcr_add_add_relu(
        self,
        m,
        m0,
        m1,
        k,
        n,
        dtype="float16",
        test_name_suffix="",
    ):
        target = detect_target()
        self._init_tensors(m, k, n, m0, m1, dtype)
        OP = ops.gemm_rcr_bias_add_add_relu()
        Y = OP(self.X, self.W, self.B, self.D0, self.D1)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"gemm_rcr_bias_add_add_relu_k_{k}_n_{n}_{dtype}{test_name_suffix}",
        )

        Y_pt = torch.relu(
            torch.nn.functional.linear(self.X_pt, self.W_pt, bias=self.B_pt)
            + self.D0_pt
            + self.D1_pt
        )
        self._test_and_verify(module, Y_pt, dtype, has_d1=True)

    def test_bias_rcr_add_add_relu(self):
        self._test_bias_rcr_add_add_relu(8, None, None, 8, 8)
        self._test_bias_rcr_add_add_relu(None, 2, 32, 256, 128)
        self._test_bias_rcr_add_add_relu(None, 21, 5, 1024, 512)
        self._test_bias_rcr_add_add_relu(None, 21, 5, 1024, 0)
        # This test triggered a c10 assertion failure internally
        # caffe2/c10/util/SmallVector.h:338:
        # Assertion `idx < size()' failed
        target = detect_target()
        if type(target).__name__ != "FBCUDA":
            self._test_bias_rcr_add_add_relu(21, None, None, 0, 512)

    def test_bias_rcr_add_add_relu_rocm(self):
        self._test_bias_rcr_add_add_relu(8, None, None, 8, 8, test_name_suffix="_rocm")

    def _test_bias_rcr_mul(
        self,
        m,
        m0,
        m1,
        k,
        n,
        use_fp16_acc=False,
        dtype="float16",
        test_name_suffix="",
    ):
        target = detect_target(use_fp16_acc=use_fp16_acc)
        self._init_tensors(m, k, n, m0, m1, dtype)
        OP = ops.gemm_rcr_bias_mul()
        Y = OP(self.X, self.W, self.B, self.D0)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"gemm_rcr_bias_mul_k_{k}_n_{n}_{dtype}{test_name_suffix}",
        )

        Y_pt = (
            torch.nn.functional.linear(self.X_pt, self.W_pt, bias=self.B_pt)
            * self.D0_pt
        )
        self._test_and_verify(module, Y_pt, dtype)

    def test_bias_rcr_mul(self):
        self._test_bias_rcr_mul(8, None, None, 8, 8)
        self._test_bias_rcr_mul(None, 2, 32, 256, 128)
        self._test_bias_rcr_mul(None, 21, 5, 1024, 512)

    def test_bias_rcr_mul_rocm(self):
        self._test_bias_rcr_mul(8, None, None, 8, 8, test_name_suffix="_rocm")

    def _test_bias_rcr_add_add(
        self,
        m,
        m0,
        m1,
        k,
        n,
        dtype="float16",
        test_name_suffix="",
    ):
        target = detect_target()
        self._init_tensors(m, k, n, m0, m1, dtype)
        OP = ops.gemm_rcr_bias_add_add()
        Y = OP(self.X, self.W, self.B, self.D0, self.D1)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"gemm_rcr_bias_add_add_k_{k}_n_{n}_{dtype}{test_name_suffix}",
        )

        Y_pt = (
            torch.nn.functional.linear(self.X_pt, self.W_pt, bias=self.B_pt)
            + self.D0_pt
            + self.D1_pt
        )
        self._test_and_verify(module, Y_pt, dtype, has_d1=True)

    def test_bias_rcr_add_add(self):
        self._test_bias_rcr_add_add(8, None, None, 8, 8)
        self._test_bias_rcr_add_add(None, 2, 32, 256, 128)
        self._test_bias_rcr_add_add(None, 21, 5, 1024, 512)
        self._test_bias_rcr_add_add(None, 0, 5, 1024, 512)

    def test_bias_rcr_add_add_rocm(self):
        self._test_bias_rcr_add_add(8, None, None, 8, 8, test_name_suffix="_rocm")

    def _test_bias_rcr_mul_tanh(
        self,
        m,
        m0,
        m1,
        k,
        n,
        dtype="float16",
        test_name_suffix="",
    ):
        target = detect_target()
        self._init_tensors(m, k, n, m0, m1, dtype)
        OP = ops.gemm_rcr_bias_mul_tanh()
        Y = OP(self.X, self.W, self.B, self.D0)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"gemm_rcr_bias_mul_tanh_k_{k}_n_{n}_{dtype}{test_name_suffix}",
        )

        Y_pt = torch.tanh(
            torch.nn.functional.linear(self.X_pt, self.W_pt, bias=self.B_pt)
            * self.D0_pt
        )
        self._test_and_verify(module, Y_pt, dtype)

    def test_bias_rcr_mul_tanh(self):
        self._test_bias_rcr_mul_tanh(8, None, None, 8, 8)
        self._test_bias_rcr_mul_tanh(None, 2, 32, 256, 128)
        self._test_bias_rcr_mul_tanh(None, 21, 5, 1024, 512)

    def test_bias_rcr_mul_tanh_rocm(self):
        self._test_bias_rcr_mul_tanh(8, None, None, 8, 8, test_name_suffix="_rocm")

    @parameterized.expand(
        [
            (_test_bias_rcr_mul_add, None, 2, 32, 256, 128, "float32"),
            (_test_bias_rcr_sigmoid_mul, None, 2, 32, 256, 128, "float32"),
            (_test_bias_rcr_sigmoid_mul_tanh, None, 2, 32, 256, 128, "float32"),
            (_test_bias_rcr_add, None, 2, 32, 256, 128, "float32"),
            (_test_bias_rcr_add_relu, None, 2, 32, 256, 128, "float32"),
            (_test_bias_rcr_add_add_relu, None, 2, 32, 256, 128, "float32"),
            (_test_bias_rcr_mul, None, 2, 32, 256, 128, "float32"),
            (_test_bias_rcr_add_add, None, 2, 32, 256, 128, "float32"),
            (_test_bias_rcr_mul_tanh, None, 2, 32, 256, 128, "float32"),
        ],
        name_func=custom_name_func_with_funcname,
    )
    def test_gemm_bias_broadcast_float32_sm80(self, func, m, m0, m1, k, n, dtype):
        func(self, m, m0, m1, k, n, dtype)

    @parameterized.expand(
        [
            (_test_bias_rcr_mul_add, None, 2, 32, 256, 128, "bfloat16"),
            (_test_bias_rcr_sigmoid_mul, None, 2, 32, 256, 128, "bfloat16"),
            (_test_bias_rcr_sigmoid_mul_tanh, None, 2, 32, 256, 128, "bfloat16"),
            (_test_bias_rcr_add, None, 2, 32, 256, 128, "bfloat16"),
            (_test_bias_rcr_add_relu, None, 2, 32, 256, 128, "bfloat16"),
            (_test_bias_rcr_add_add_relu, None, 2, 32, 256, 128, "bfloat16"),
            (_test_bias_rcr_mul, None, 2, 32, 256, 128, "bfloat16"),
            (_test_bias_rcr_add_add, None, 2, 32, 256, 128, "bfloat16"),
            (_test_bias_rcr_mul_tanh, None, 2, 32, 256, 128, "bfloat16"),
        ],
        name_func=custom_name_func_with_funcname,
    )
    def test_gemm_bias_broadcast_bfloat16_bf16(self, func, m, m0, m1, k, n, dtype):
        func(self, m, m0, m1, k, n, dtype)

    def test_gemm_bias_broadcast_use_fp16_acc_sm80(self):
        self._test_bias_rcr_mul(
            None, 2, 32, 256, 128, use_fp16_acc=True, dtype="float32"
        )
        self._test_bias_rcr_mul(
            None, 2, 32, 256, 128, use_fp16_acc=True, dtype="bfloat16"
        )


filter_test_cases_by_test_env(GEMMBiasBroadcastTestCase)


if __name__ == "__main__":
    unittest.main()
