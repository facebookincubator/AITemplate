# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
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
from aitemplate.compiler.base import DynamicProfileStrategy
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ConvTestCase(unittest.TestCase):
    def test_fp16(self):
        target = detect_target()
        batch_size = [2, 32]
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), 24, 24, 4],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[36, 3, 3, 4], dtype="float16", name="input_1", is_input=True)
        OP = ops.conv2d(stride=2, pad=1, dilate=1)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            "dynamic_conv",
            dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
        )
        for batch in batch_size:
            print("Test batch: %d" % batch)
            X_pt = torch.randn(batch, 4, 24, 24).cuda().half()
            W_pt = torch.randn(36, 4, 3, 3).cuda().half()
            Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, stride=2, padding=1)
            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w = W_pt.permute((0, 2, 3, 1)).contiguous()
            y = torch.empty([batch, 12, 12, 36]).cuda().half()
            module.run_with_tensors({"input_0": x, "input_1": w}, [y])
            y_transpose = y.permute((0, 3, 1, 2))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
