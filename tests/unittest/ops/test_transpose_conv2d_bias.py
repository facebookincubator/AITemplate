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


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class conv2dTransposeTestCase(unittest.TestCase):
    def test_fp16(self, batch=4):
        target = detect_target()
        if int(target._arch) < 80:
            return
        X = Tensor(
            shape=[IntImm(batch), 14, 14, 256],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[256, 2, 2, 256], dtype="float16", name="input_1", is_input=True
        )
        B = Tensor(shape=[256], dtype="float16", name="input_2", is_input=True)
        OP = ops.transposed_conv2d_bias(stride=2, pad=0, dilate=1)
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "transpose_conv2d_bias")

        X_pt = torch.randn(batch, 256, 14, 14).cuda().half()
        W_pt = torch.randn(256, 256, 2, 2).cuda().half()
        B_pt = torch.randn(1, 256, 1, 1).cuda().half()
        Y_pt = torch.nn.functional.conv_transpose2d(X_pt, W_pt, padding=0, stride=2)
        Y_pt = Y_pt + B_pt

        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        y = torch.empty([batch, 28, 28, 256]).cuda().half()
        module.run_with_tensors(
            {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}, [y]
        )
        y_transpose = y.permute((0, 3, 1, 2))
        if target.name() == "cuda":
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
