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

from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class Nhcw3To4TestCase(unittest.TestCase):
    def _test_nhcw3to4(self, dtype="float16"):
        target = detect_target()
        batch_size = [1, 3]
        if target.name() == "rocm":
            return True
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), 224, 224, 3],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        OP = ops.nhwc3to4()
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "nhwc3to4")
        for batch in batch_size:
            X_np = np.random.uniform(-1, 1, (batch, 224, 224, 3)).astype(dtype)
            Y_np = np.zeros((batch, 224, 224, 4)).astype(dtype)
            Y_np[:, :, :, 0] = X_np[:, :, :, 0]
            Y_np[:, :, :, 1] = X_np[:, :, :, 1]
            Y_np[:, :, :, 2] = X_np[:, :, :, 2]
            Y_pt = torch.from_numpy(Y_np).cuda()
            X_pt = torch.from_numpy(X_np).cuda()
            y = torch.empty_like(Y_pt)
            module.run_with_tensors([X_pt], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_nhcw3to4_f16(self):
        self._test_nhcw3to4()

    def test_nhcw3to4_f32(self):
        self._test_nhcw3to4(dtype="float32")


if __name__ == "__main__":
    unittest.main()
