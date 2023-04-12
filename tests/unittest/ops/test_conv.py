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
from aitemplate.frontend import IntImm, nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)

from parameterized import parameterized


class ConvTestCase(unittest.TestCase):
    def _test_conv(
        self,
        batch=4,
        copy_op=False,
        test_name="conv2d",
        dtype="float16",
    ):
        target = detect_target()
        X = Tensor(
            shape=[IntImm(batch), 28, 28, 128],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[256, 3, 3, 128],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        OP = ops.conv2d(stride=1, pad=1, dilate=1)
        if copy_op:
            OP = ops.conv2d(**OP._get_op_attributes())
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        X_pt = get_random_torch_tensor([batch, 128, 28, 28], dtype=dtype)
        W_pt = get_random_torch_tensor([256, 128, 3, 3], dtype=dtype)
        Y_pt = torch.nn.functional.conv2d(X_pt.float(), W_pt.float(), padding=1).to(
            dtype=X_pt.dtype
        )
        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
        module.run_with_tensors({"input_0": x, "input_1": w}, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        if target.name() == "cuda":
            if dtype == "float32":
                torch.testing.assert_close(Y_pt, y_transpose, atol=1e-1, rtol=1e-1)
            else:
                torch.testing.assert_close(Y_pt, y_transpose, atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1)

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float32"), ("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_conv2d(self, dtype):
        self._test_conv(
            test_name=f"conv2d_{dtype}",
            dtype=dtype,
        )
        self._test_conv(
            copy_op=True,
            test_name=f"conv2d_{dtype}_copy_op",
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float32"), ("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_conv1d(self, dtype):
        self._test_conv1d(dtype=dtype, bias=False)

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float32"), ("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_conv1d_bias(self, dtype):
        self._test_conv1d(dtype=dtype, bias=True)

    def _test_conv1d(self, dtype, bias):
        target = detect_target()
        batch = 4
        C_in = 80
        C_out = 512
        K = 3
        L = 28
        stride = 1
        padding = 1
        dilation = 1
        test_name = "test_conv1d"

        X_pt = get_random_torch_tensor([batch, C_in, L], dtype=dtype)
        W_pt = get_random_torch_tensor([C_out, C_in, K], dtype=dtype)
        bias_pt = get_random_torch_tensor([C_out], dtype=dtype) if bias else None

        X = Tensor(
            shape=[IntImm(batch), L, C_in],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        mod = nn.Conv1d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=K,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dtype=dtype,
            bias=bias,
        )

        Y = mod(X)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)
        module.set_constant_with_tensor(
            "conv1d_weight", W_pt.permute((0, 2, 1)).contiguous()
        )
        if bias:
            module.set_constant_with_tensor("conv1d_bias", bias_pt)
        Y_pt = torch.nn.functional.conv1d(
            X_pt.float(),
            W_pt.float(),
            bias=bias_pt.float() if bias else None,
            padding=padding,
            stride=stride,
            dilation=dilation,
        ).to(dtype=X_pt.dtype)

        x = X_pt.permute((0, 2, 1)).contiguous()

        y = torch.empty_like(Y_pt).permute((0, 2, 1)).contiguous()
        module.run_with_tensors({"input_0": x}, [y])
        y_transpose = y.permute((0, 2, 1))
        if target.name() == "cuda":
            if dtype == "float32":
                torch.testing.assert_close(Y_pt, y_transpose, atol=1.5e-1, rtol=1e-1)
            else:
                torch.testing.assert_close(Y_pt, y_transpose, atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
