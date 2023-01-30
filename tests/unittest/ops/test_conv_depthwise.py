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


class ConvDepthwiseTestCase(unittest.TestCase):
    def test_fp16(self, batch=4):
        groups = 32
        size = (12, 12)
        target = detect_target()
        X = Tensor(
            shape=[IntImm(batch), *size, 32],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[32, 3, 3, 1], dtype="float16", name="input_1", is_input=True)
        OP = ops.conv2d_depthwise(stride=1, pad=1, dilate=1, group=groups)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "conv2d_dw")

        X_pt = torch.randn(batch, 32, *size).cuda().half()
        W_pt = torch.randn(32, 1, 3, 3).cuda().half()
        Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=1, groups=groups)
        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        y = torch.empty([batch, *size, 32]).cuda().half()
        module.run_with_tensors({"input_0": x, "input_1": w}, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        self.assertFalse(y_transpose.isnan().any())
        self.assertFalse(y_transpose.isinf().any())
        if target.name() == "cuda":
            torch.testing.assert_close(Y_pt, y_transpose, atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
