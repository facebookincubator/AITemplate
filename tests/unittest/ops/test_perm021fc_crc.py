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
"""
# _3306 = _3305.permute(0, 2, 1)  # Transpose
# _3307 = _3306  # torch.reshape(_3306, (-1, 745))  # Reshape
# _3308 = torch.nn.functional.linear(_3307, self._1184, bias=self._1185)  # FC
"""

import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class Perm021FCCRCTestCase(unittest.TestCase):
    def _test_perm021fc_crc(
        self,
        test_name="perm021fc_crc",
        dtype="float16",
    ):
        B = 1024
        M = 128
        K = 742
        # K = 752
        N = 64
        target = detect_target()
        X = Tensor(
            shape=[1, K, N],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[B, K, M],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        OP = ops.perm021fc_crc()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        X_pt = get_random_torch_tensor([B, K, M], dtype=dtype)
        W_pt = get_random_torch_tensor([N, K], dtype=dtype)

        XT = X_pt.permute(0, 2, 1)
        XT = torch.reshape(XT, (-1, K))
        Y_pt = torch.nn.functional.linear(XT, W_pt)
        Y_pt = torch.reshape(Y_pt, (B, M, N))

        WT = W_pt.transpose(0, 1).contiguous()
        y = torch.empty_like(Y_pt)
        module.run_with_tensors({"input_0": WT.unsqueeze(0), "input_1": X_pt}, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_perm021fc_crc_fp16(self):
        self._test_perm021fc_crc(
            test_name="perm021fc_crc_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @unittest.skipIf(
        int(detect_target()._arch) < 80,
        f"fp32 BMM not supported in {detect_target()._arch}",
    )
    def test_perm021fc_crc_fp32(self):
        self._test_perm021fc_crc(
            test_name="perm021fc_crc_fp32",
            dtype="float32",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @unittest.skipIf(
        int(detect_target()._arch) < 80,
        f"bf16 BMM not supported in {detect_target()._arch}",
    )
    def test_perm021fc_crc_bf16(self):
        self._test_perm021fc_crc(
            test_name="perm021fc_crc_bf16",
            dtype="bfloat16",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
