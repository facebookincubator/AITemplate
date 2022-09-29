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

_DEFAULT_BATCH_SIZE = [1, 3]


class UpsamplingTestCase(unittest.TestCase):
    def _test_fp16_single_op(
        self,
        scale_factor=2.0,
        mode="bilinear",
        batch_size=_DEFAULT_BATCH_SIZE,
        test_name="bilinear_upsampling2d",
    ):
        channels = 1024
        HH, WW = 8, 8
        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), HH, WW, channels],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        OP = nn.Upsampling2d(scale_factor=scale_factor, mode=mode)
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_pt = torch.randn(b, channels, HH, WW).cuda().half()
            Y_pt = torch.nn.functional.interpolate(
                X_pt, scale_factor=scale_factor, mode=mode
            )
            x = torch.permute(X_pt, (0, 2, 3, 1)).contiguous()
            y = (
                torch.empty(
                    [b, int(HH * scale_factor), int(WW * scale_factor), channels]
                )
                .cuda()
                .half()
            )
            module.run_with_tensors([x], [y])
            y_transpose = torch.permute(y, (0, 3, 1, 2))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))

    def test_bilinear_upsample(self):
        self._test_fp16_single_op(
            scale_factor=3.5, mode="bilinear", test_name="bilinear_upsampling2d"
        )

    def test_nearest_upsample(self):
        self._test_fp16_single_op(
            scale_factor=2.0, mode="nearest", test_name="nearest_upsampling2d"
        )


if __name__ == "__main__":
    unittest.main()
