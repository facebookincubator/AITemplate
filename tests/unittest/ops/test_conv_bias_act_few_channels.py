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


def hard_swish(x):
    # return x * F.relu6(x + 3) / 6
    return x * torch.clamp((x + 3), 0, 6) / 6


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ConvBiasReluTestCase(unittest.TestCase):
    def test_relu(self, HH=224, WW=224, CI=4, CO=64, batch=1):
        KK = 7
        stride = 2
        pad = 3
        target = detect_target()
        X = Tensor(
            shape=[IntImm(batch), HH, WW, CI],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[CO, KK, KK, CI], dtype="float16", name="input_1", is_input=True
        )
        B = Tensor(shape=[CO], dtype="float16", name="input_2", is_input=True)
        OP = ops.conv2d_bias_relu_few_channels(stride=stride, pad=pad, dilate=1)
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "conv_bias_relu_few_channels")

        X_pt = torch.randn(batch, CI, HH, WW).cuda().half()
        W_pt = torch.randn(CO, CI, KK, KK).cuda().half()
        B_pt = torch.randn(1, CO, 1, 1).cuda().half()
        Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=pad, stride=stride)
        Y_pt = Y_pt + B_pt
        Y_pt = torch.nn.functional.relu(Y_pt)
        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        inputs = {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}
        y = torch.empty([batch, HH // stride, WW // stride, CO]).cuda().half()
        module.run_with_tensors(inputs, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))

    def test_hardswish(self, HH=224, WW=224, CI=4, CO=64, batch=1):
        KK = 7
        stride = 2
        pad = 3
        target = detect_target()
        X = Tensor(
            shape=[IntImm(batch), HH, WW, CI],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[CO, KK, KK, CI], dtype="float16", name="input_1", is_input=True
        )
        B = Tensor(shape=[CO], dtype="float16", name="input_2", is_input=True)
        OP = ops.conv2d_bias_hardswish_few_channels(stride=stride, pad=pad, dilate=1)
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "conv_bias_hardswish_few_channels")

        X_pt = torch.randn(batch, CI, HH, WW).cuda().half()
        W_pt = torch.randn(CO, CI, KK, KK).cuda().half()
        B_pt = torch.randn(1, CO, 1, 1).cuda().half()
        Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=pad, stride=stride)
        Y_pt = Y_pt + B_pt
        Y_pt = hard_swish(Y_pt)
        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        inputs = {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}
        y = torch.empty([batch, HH // stride, WW // stride, CO]).cuda().half()
        module.run_with_tensors(inputs, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
