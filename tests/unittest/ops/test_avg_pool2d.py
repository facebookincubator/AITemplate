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


class AvgPoolTestCase(unittest.TestCase):
    def test_fp16(self):
        target = detect_target()
        batch_size = [1, 3]
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), 7, 7, 2048],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        OP = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "avg_pool2d")
        for b in batch_size:
            X_pt = torch.randn(b, 2048, 7, 7).cuda().half()
            OP_pt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            Y_pt = OP_pt(X_pt)
            y = torch.empty([b, 1, 1, 2048]).cuda().half()
            x = torch.permute(X_pt, (0, 2, 3, 1)).contiguous()
            module.run_with_tensors([x], [y])
            y_transpose = torch.permute(y, (0, 3, 1, 2))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
