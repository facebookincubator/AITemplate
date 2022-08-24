# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import numpy as np
import torch

from aitemplate.frontend import IntVar, nn, Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class Nhcw3To8TestCase(unittest.TestCase):
    def test_nhcw3to8_fp16(self):
        target = detect_target()
        batch_size = [1, 3]
        if target.name() == "rocm":
            return True
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), 224, 224, 3],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        OP = nn.Nhwc3to8()
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(Y, target, "./tmp", "nhwc3to8")
        for batch in batch_size:
            X_np = np.random.uniform(-1, 1, (batch, 224, 224, 3)).astype("float16")
            Y_np = np.zeros((batch, 224, 224, 8)).astype("float16")
            Y_np[:, :, :, 0] = X_np[:, :, :, 0]
            Y_np[:, :, :, 1] = X_np[:, :, :, 1]
            Y_np[:, :, :, 2] = X_np[:, :, :, 2]
            Y_pt = torch.from_numpy(Y_np).cuda()
            X_pt = torch.from_numpy(X_np).cuda()
            y = torch.empty([batch, 224, 224, 8]).cuda().half()
            module.RunWithTensors([X_pt], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
