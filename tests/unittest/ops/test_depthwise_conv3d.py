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


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class DepthwiseConv3dTestCase(unittest.TestCase):
    def _test_fp16(self, batch=4, copy_op=False):
        target = detect_target()
        tt, hh, ww, ci, co, groups = 28, 28, 28, 128, 128, 128
        X = Tensor(
            shape=[IntImm(batch), tt, hh, ww, ci],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[co, 3, 3, 3, 1], dtype="float16", name="input_1", is_input=True
        )
        OP = ops.depthwise_conv3d(stride=1, pad=1, dilate=1, group=groups)
        if copy_op:
            OP = ops.depthwise_conv3d(**OP._get_op_attributes())
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"depthwise_conv3d_{copy_op}")

        X_pt = torch.randn(batch, ci, tt, hh, ww).cuda().half()
        W_pt = torch.randn(co, 1, 3, 3, 3).cuda().half()
        Y_pt = torch.nn.functional.conv3d(X_pt, W_pt, padding=1, groups=groups)
        x = X_pt.permute((0, 2, 3, 4, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 4, 1)).contiguous()
        y = torch.empty([batch, tt, hh, ww, co]).cuda().half()
        module.run_with_tensors({"input_0": x, "input_1": w}, [y])

        Y_pt_transpose = Y_pt.permute(0, 2, 3, 4, 1)
        self.assertTrue(torch.allclose(Y_pt_transpose, y, atol=1e-2, rtol=1e-2))

    def test_fp16(self):
        self._test_fp16()
        self._test_fp16(copy_op=True)

    def _test_mvit_shape(
        self,
        batch,
        tt,
        hh,
        ww,
        ci,
        co,
        groups,
        kernel_size,
        strides,
        test_case,
    ):
        assert ci == co and ci == groups

        target = detect_target()
        X = Tensor(
            shape=[IntImm(batch), tt, hh, ww, ci],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W_shape = [co] + list(kernel_size) + [1]
        W = Tensor(shape=W_shape, dtype="float16", name="input_1", is_input=True)

        OP = ops.depthwise_conv3d(stride=strides, pad=1, dilate=1, group=groups)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"depthwise_conv3d_mvit_{test_case}")

        X_pt = torch.randn(batch, ci, tt, hh, ww).cuda().half()
        W_pt = (
            torch.randn(co, 1, kernel_size[0], kernel_size[1], kernel_size[2])
            .cuda()
            .half()
        )
        Y_pt = torch.nn.functional.conv3d(
            X_pt, W_pt, stride=strides, padding=1, groups=groups
        )
        Y_pt_transpose = Y_pt.permute(0, 2, 3, 4, 1)

        x = X_pt.permute((0, 2, 3, 4, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 4, 1)).contiguous()
        y = torch.empty(Y_pt_transpose.shape).cuda().half()
        module.run_with_tensors({"input_0": x, "input_1": w}, [y])

        self.assertTrue(torch.allclose(Y_pt_transpose, y, atol=1e-2, rtol=1e-2))

    def test_mvit(self):
        self._test_mvit_shape(1, 2, 56, 56, 96, 96, 96, (3, 3, 3), (1, 1, 1), "0")
        self._test_mvit_shape(2, 2, 28, 28, 96, 96, 96, (3, 3, 3), (1, 1, 1), "1")
        self._test_mvit_shape(4, 2, 14, 14, 96, 96, 96, (3, 3, 3), (1, 1, 1), "2")
        self._test_mvit_shape(8, 2, 7, 7, 96, 96, 96, (3, 3, 3), (1, 1, 1), "3")
        self._test_mvit_shape(128, 2, 56, 56, 96, 96, 96, (3, 3, 3), (1, 2, 2), "4")
        self._test_mvit_shape(128, 2, 56, 56, 96, 96, 96, (3, 3, 3), (1, 4, 4), "5")
        self._test_mvit_shape(128, 2, 56, 56, 96, 96, 96, (3, 3, 3), (2, 8, 8), "6")
        self._test_mvit_shape(128, 2, 56, 56, 96, 96, 96, (1, 3, 3), (2, 8, 8), "7")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
