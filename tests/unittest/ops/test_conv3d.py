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

import numpy as np

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target


@unittest.skipIf(detect_target()._arch == "75", "Conv3d not supported on sm75.")
class ConvTestCase(unittest.TestCase):
    def _test_fp16(
        self,
        tt,
        hh,
        ww,
        ci,
        co,
        kt,
        kh,
        kw,
        stride=(1, 1, 1),
        pad=(1, 1, 1),
        batch=4,
        test_case="",
    ):
        target = detect_target()

        X = Tensor(
            shape=[IntImm(batch), tt, hh, ww, ci],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[co, kt, kh, kw, ci], dtype="float16", name="input_1", is_input=True
        )
        OP = ops.conv3d(stride=stride, pad=pad, dilate=1)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"conv3d_{test_case}")

        X_pt = torch.randn(batch, ci, tt, hh, ww).cuda().half()
        W_pt = torch.randn(co, ci, kt, kh, kw).cuda().half()
        Y_pt = torch.nn.functional.conv3d(X_pt, W_pt, stride=stride, padding=pad)
        x = X_pt.permute((0, 2, 3, 4, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 4, 1)).contiguous()

        Y_pt_transpose = Y_pt.permute(0, 2, 3, 4, 1)
        y_shape = list(Y_pt_transpose.shape)
        y = torch.empty(y_shape).cuda().half()
        module.run_with_tensors({"input_0": x, "input_1": w}, [y])

        np.testing.assert_allclose(
            Y_pt_transpose.cpu().numpy(), y.cpu().numpy(), atol=1e-2, rtol=1e-2
        )

    def test_fp16(self):
        self._test_fp16(
            4, 224, 224, 8, 96, 3, 5, 5, stride=(2, 4, 4), pad=(1, 2, 2), test_case=1
        )
        self._test_fp16(56, 56, 56, 64, 256, 1, 1, 1, test_case=2)
        self._test_fp16(56, 56, 56, 64, 64, 1, 1, 1, test_case=3)
        self._test_fp16(56, 56, 56, 64, 64, 3, 3, 3, test_case=4)
        self._test_fp16(56, 56, 56, 256, 64, 1, 1, 1, test_case=5)
        self._test_fp16(56, 56, 56, 256, 512, 1, 1, 1, stride=(2, 2, 2), test_case=6)
        self._test_fp16(56, 56, 56, 128, 128, 3, 3, 3, stride=(2, 2, 2), test_case=7)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
