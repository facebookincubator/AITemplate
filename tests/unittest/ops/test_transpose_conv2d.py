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


class conv2dTransposeTestCase(unittest.TestCase):
    def test_fp16(self, batch=32):
        target = detect_target()
        if target.name() == "cuda" and int(target._arch) < 80:
            return
        X = Tensor(
            shape=[IntImm(batch), 28, 28, 256],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[256, 2, 2, 256], dtype="float16", name="input_1", is_input=True
        )
        OP = ops.transposed_conv2d(stride=2, pad=0, dilate=1)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "transpose_conv2d")

        X_pt = torch.randn(batch, 256, 28, 28).cuda().half()
        W_pt = torch.randn(256, 256, 2, 2).cuda().half()
        Y_pt = torch.nn.functional.conv_transpose2d(X_pt, W_pt, padding=0, stride=2)

        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        y = torch.empty([batch, 56, 56, 256]).cuda().half()
        module.run_with_tensors({"input_0": x, "input_1": w}, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
