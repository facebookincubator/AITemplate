# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.frontend import IntVar, nn, Tensor
from aitemplate.testing import detect_target, gen_execution_module


class AvgPoolTestCase(unittest.TestCase):
    def test_fp16(self):
        target = detect_target()
        batch_size = [1, 3]
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), 7, 7, 2048],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        OP = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(Y, target, "./tmp", "avg_pool2d")
        for b in batch_size:
            X_pt = torch.randn(b, 2048, 7, 7).cuda().half()
            OP_pt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            Y_pt = OP_pt(X_pt)
            y = torch.empty([b, 1, 1, 2048]).cuda().half()
            x = torch.permute(X_pt, (0, 2, 3, 1)).contiguous()
            module.RunWithTensors([x], [y])
            y_transpose = torch.permute(y, (0, 3, 1, 2))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
