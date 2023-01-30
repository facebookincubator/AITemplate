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
from aitemplate.compiler.base import DynamicProfileStrategy
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ConvDynamicTestCase(unittest.TestCase):
    def _test_conv_dynamic(
        self,
        test_name="conv_dynamic",
        dtype="float16",
    ):
        target = detect_target()
        batch_size = [2, 32]
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), 24, 24, 4],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[36, 3, 3, 4],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        OP = ops.conv2d(stride=2, pad=1, dilate=1)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
        )
        for batch in batch_size:
            print("Test batch: %d" % batch)
            X_pt = get_random_torch_tensor([batch, 4, 24, 24], dtype=dtype)
            W_pt = get_random_torch_tensor([36, 4, 3, 3], dtype=dtype)
            Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, stride=2, padding=1)
            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w = W_pt.permute((0, 2, 3, 1)).contiguous()
            y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
            module.run_with_tensors({"input_0": x, "input_1": w}, [y])
            y_transpose = y.permute((0, 3, 1, 2))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))

    def test_fp16(self):
        self._test_conv_dynamic(
            test_name="conv_dynamic_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_fp32(self):
        self._test_conv_dynamic(
            test_name="conv_dynamic_fp32",
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
