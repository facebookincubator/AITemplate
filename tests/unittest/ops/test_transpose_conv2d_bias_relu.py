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
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class Conv2dTransposeBiasReluTestCase(unittest.TestCase):
    def _test_transpose_conv2d_bias_relu(
        self,
        batch=32,
        copy_op=False,
        test_name="transpose_conv2d_bias_relu",
        dtype="float16",
    ):
        target = detect_target()
        X = Tensor(
            shape=[IntImm(batch), 14, 14, 256],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[256, 2, 2, 256],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B = Tensor(
            shape=[256],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        OP = ops.transposed_conv2d_bias_relu(stride=2, pad=0, dilate=1)
        if copy_op:
            OP = ops.transposed_conv2d_bias_relu(**OP._get_op_attributes())
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        X_pt = get_random_torch_tensor([batch, 256, 14, 14], dtype=dtype)
        W_pt = get_random_torch_tensor([256, 256, 2, 2], dtype=dtype)
        B_pt = get_random_torch_tensor([1, 256, 1, 1], dtype=dtype)
        Y_pt = torch.nn.functional.conv_transpose2d(X_pt, W_pt, padding=0, stride=2)
        Y_pt = Y_pt + B_pt
        Y_pt = torch.relu(Y_pt)

        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
        module.run_with_tensors(
            {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}, [y]
        )
        y_transpose = y.permute((0, 3, 1, 2))
        if target.name() == "cuda":
            if dtype == "float32":
                self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=5e-2, rtol=1e-2))
            else:
                self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1))

    def test_fp16(self):
        self._test_transpose_conv2d_bias_relu(
            test_name="transpose_conv2d_bias_relu_fp16",
            dtype="float16",
        )
        self._test_transpose_conv2d_bias_relu(
            copy_op=True,
            test_name="transpose_conv2d_bias_relu_fp16_copy_op",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_fp32_sm80(self):
        self._test_transpose_conv2d_bias_relu(
            test_name="transpose_conv2d_bias_relu_fp32",
            dtype="float32",
        )
        self._test_transpose_conv2d_bias_relu(
            copy_op=True,
            test_name="transpose_conv2d_bias_relu_fp32_copy_op",
            dtype="float32",
        )


filter_test_cases_by_test_env(Conv2dTransposeBiasReluTestCase)

if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
