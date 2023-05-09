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
from aitemplate.compiler import compile_model

from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.pool3d import MaxPool3d
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


class MaxPool3dTestCase(unittest.TestCase):
    def _test_max_pool_3d(
        self,
        kernel_size,
        stride,
        padding,
        pt_input_shape,
        ait_input_shape,
        dtype="float16",
    ):
        X_pt = get_random_torch_tensor(pt_input_shape, dtype=dtype)
        OP_pt = (
            torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
            .cuda()
            .half()
        )
        Y_pt = OP_pt(X_pt)
        X_ait = Tensor(
            shape=ait_input_shape,
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        OP_ait = MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
        Y_ait = OP_ait(X_ait)

        Y_ait._attrs["name"] = "output_0"
        Y_ait._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y_ait, target, "./tmp", "max_pool3d")

        x = X_pt.permute((0, 2, 3, 4, 1)).contiguous()
        y = torch.empty_like(Y_pt).permute(0, 2, 3, 4, 1).contiguous()
        module.run_with_tensors([x], [y])
        y_transpose = y.permute((0, 4, 1, 2, 3))

        self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))

    def test_max_pool_3d_fp16(self):
        for batch in [1, 3]:
            self._test_max_pool_3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                pt_input_shape=[batch, 4, 8, 256, 256],
                ait_input_shape=[batch, 8, 256, 256, 4],
                dtype="float16",
            )
            self._test_max_pool_3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                pt_input_shape=[batch, 4, 8, 256, 256],
                ait_input_shape=[batch, 8, 256, 256, 4],
                dtype="float16",
            )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_max_pool_3d_fp32(self):
        for batch in [1, 3]:
            self._test_max_pool_3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                pt_input_shape=[batch, 4, 8, 256, 256],
                ait_input_shape=[batch, 8, 256, 256, 4],
                dtype="float32",
            )
            self._test_max_pool_3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                pt_input_shape=[batch, 4, 8, 256, 256],
                ait_input_shape=[batch, 8, 256, 256, 4],
                dtype="float32",
            )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
