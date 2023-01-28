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
from aitemplate.testing.test_utils import get_random_torch_tensor


_DEFAULT_BATCH_SIZE = [1, 3]


class UpsamplingTestCase(unittest.TestCase):
    def _test_single_op(
        self,
        scale_factor=2.0,
        mode="bilinear",
        batch_size=_DEFAULT_BATCH_SIZE,
        test_name="bilinear_upsampling2d_fp16",
        dtype="float16",
    ):
        channels = 1024
        HH, WW = 8, 8
        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), HH, WW, channels],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        OP = nn.Upsampling2d(scale_factor=scale_factor, mode=mode)
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_pt = get_random_torch_tensor([b, channels, HH, WW], dtype=dtype)
            Y_pt = torch.nn.functional.interpolate(
                X_pt, scale_factor=scale_factor, mode=mode
            )
            x = torch.permute(X_pt, (0, 2, 3, 1)).contiguous()
            y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
            module.run_with_tensors([x], [y])
            y_transpose = torch.permute(y, (0, 3, 1, 2))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))

    def test_bilinear_upsample_fp16(self):
        self._test_single_op(
            scale_factor=3.5,
            mode="bilinear",
            test_name="bilinear_upsampling2d_fp16",
            dtype="float16",
        )

    def test_nearest_upsample_fp16(self):
        self._test_single_op(
            scale_factor=2.0,
            mode="nearest",
            test_name="nearest_upsampling2d_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_bilinear_upsample_fp32(self):
        self._test_single_op(
            scale_factor=3.5,
            mode="bilinear",
            test_name="bilinear_upsampling2d_fp32",
            dtype="float32",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_nearest_upsample_fp32(self):
        self._test_single_op(
            scale_factor=2.0,
            mode="nearest",
            test_name="nearest_upsampling2d_fp32",
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
