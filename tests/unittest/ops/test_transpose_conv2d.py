# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class conv2dTransposeTestCase(unittest.TestCase):
    def test_fp16(self, batch=4):
        target = detect_target()
        if int(target._arch) < 80:
            return
        X = Tensor(
            shape=[IntImm(batch), 28, 28, 256],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[256, 2, 2, 256], dtype="float16", name="input_1", is_input=True
        )
        OP = ops.transposed_conv2d(stride=2, pad=0, dilate=1)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(Y, target, "./tmp", "transpose_conv2d")

        X_pt = torch.randn(batch, 256, 28, 28).cuda().half()
        W_pt = torch.randn(256, 256, 2, 2).cuda().half()
        Y_pt = torch.nn.functional.conv_transpose2d(X_pt, W_pt, padding=0, stride=2)

        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        y = torch.empty([batch, 56, 56, 256]).cuda().half()
        module.RunWithTensors({"input_0": x, "input_1": w}, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        if target.name() == "cuda":
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
