# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from aitemplate.compiler import ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target, gen_execution_module


class ConvBiasReluAddTestCase(unittest.TestCase):
    def test_fp16(self, batch=4):
        target = detect_target()
        CO, HH, WW, CI = 256, 28, 28, 128
        X = Tensor(
            shape=[IntImm(batch), HH, WW, CI],
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        W = Tensor(shape=[CO, 3, 3, CI], dtype="float16", name="input_1", is_input=True)
        B = Tensor(shape=[CO], dtype="float16", name="input_2", is_input=True)
        R = Tensor(
            shape=[IntImm(batch), HH, WW, CO],
            dtype="float16",
            name="input_3",
            is_input=True,
        )
        OP = ops.conv2d_bias_add_relu(stride=1, pad=1, dilate=1)
        Y = OP(X, W, B, R)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(Y, target, "./tmp", "conv2d_bias_add_relu")

        X_pt = torch.randn(batch, CI, HH, WW).cuda().half()
        W_pt = torch.randn(CO, CI, 3, 3).cuda().half()
        B_pt = torch.randn(1, CO, 1, 1).cuda().half()
        R_pt = torch.randn(batch, CO, HH, WW).cuda().half()
        Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=1)
        Y_pt = Y_pt + B_pt + R_pt
        Y_pt = torch.nn.functional.relu(Y_pt)

        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        r = R_pt.permute((0, 2, 3, 1)).contiguous()
        inputs = {"input_0": x, "input_1": w, "input_2": B_pt.squeeze(), "input_3": r}
        y = torch.empty([batch, HH, WW, CO]).cuda().half()
        module.RunWithTensors(inputs, [y])
        y_transpose = y.permute(0, 3, 1, 2)
        if target.name() == "cuda":
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
