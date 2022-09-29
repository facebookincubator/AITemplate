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
from aitemplate.compiler import compile_model

from aitemplate.frontend import IntVar, nn, Tensor
from aitemplate.testing import detect_target


class MaxPool2dTestCase(unittest.TestCase):
    def test_max_pool_2d_fp16(self):
        batch_size = [1, 3]
        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), 112, 112, 64],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        OP = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "max_pool2d")
        for batch in batch_size:
            X_pt = torch.randn(batch, 64, 112, 112).cuda().half()
            OP_pt = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            Y_pt = OP_pt(X_pt)
            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            y = torch.empty([batch, 56, 56, 64]).cuda().half()
            module.run_with_tensors([x], [y])
            y_transpose = y.permute((0, 3, 1, 2))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
