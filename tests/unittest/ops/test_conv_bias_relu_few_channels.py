# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ConvBiasReluTestCase(unittest.TestCase):
    def test_fp16(self, HH=224, WW=224, CI=4, CO=64, batch=4):
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
        module = gen_execution_module(Y, target, "./tmp", "conv_bias_relu_few_channels")

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
        module.RunWithTensors(inputs, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
