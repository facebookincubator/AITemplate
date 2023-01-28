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
from aitemplate.testing.test_utils import get_random_torch_tensor


@unittest.skipIf(detect_target()._arch == "75", "Conv3d not supported on sm75.")
class Conv3dTestCase(unittest.TestCase):
    def _test_conv3d(
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
        test_name="conv3d",
        dtype="float16",
    ):
        target = detect_target()

        X = Tensor(
            shape=[IntImm(batch), tt, hh, ww, ci],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[co, kt, kh, kw, ci],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        OP = ops.conv3d(stride=stride, pad=pad, dilate=1)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        X_pt = get_random_torch_tensor([batch, ci, tt, hh, ww], dtype=dtype)
        W_pt = get_random_torch_tensor([co, ci, kt, kh, kw], dtype=dtype)
        Y_pt = torch.nn.functional.conv3d(X_pt, W_pt, stride=stride, padding=pad)
        x = X_pt.permute((0, 2, 3, 4, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 4, 1)).contiguous()
        y = torch.empty_like(Y_pt).permute((0, 2, 3, 4, 1)).contiguous()
        module.run_with_tensors({"input_0": x, "input_1": w}, [y])
        y_transpose = y.permute((0, 4, 1, 2, 3))

        if dtype == "float32":
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=5e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))

    def test_fp16(self):
        self._test_conv3d(
            4,
            224,
            224,
            8,
            96,
            3,
            5,
            5,
            stride=(2, 4, 4),
            pad=(1, 2, 2),
            test_name="conv3d_fp16_1",
            dtype="float16",
        )
        self._test_conv3d(
            56,
            56,
            56,
            64,
            256,
            1,
            1,
            1,
            test_name="conv3d_fp16_2",
            dtype="float16",
        )
        self._test_conv3d(
            56,
            56,
            56,
            64,
            64,
            1,
            1,
            1,
            test_name="conv3d_fp16_3",
            dtype="float16",
        )
        self._test_conv3d(
            56,
            56,
            56,
            64,
            64,
            3,
            3,
            3,
            test_name="conv3d_fp16_4",
            dtype="float16",
        )
        self._test_conv3d(
            56,
            56,
            56,
            256,
            64,
            1,
            1,
            1,
            test_name="conv3d_fp16_5",
            dtype="float16",
        )
        self._test_conv3d(
            56,
            56,
            56,
            256,
            512,
            1,
            1,
            1,
            stride=(2, 2, 2),
            test_name="conv3d_fp16_6",
            dtype="float16",
        )
        self._test_conv3d(
            56,
            56,
            56,
            128,
            128,
            3,
            3,
            3,
            stride=(2, 2, 2),
            test_name="conv3d_fp16_7",
            dtype="float16",
        )

    @unittest.skip("no fp32 kernels are available for conv3d")
    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_fp32(self):
        self._test_conv3d(
            4,
            224,
            224,
            8,
            96,
            3,
            5,
            5,
            stride=(2, 4, 4),
            pad=(1, 2, 2),
            test_name="conv3d_fp32_1",
            dtype="float32",
        )
        self._test_conv3d(
            56,
            56,
            56,
            64,
            256,
            1,
            1,
            1,
            test_name="conv3d_fp32_2",
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
