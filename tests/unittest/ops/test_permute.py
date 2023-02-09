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
from typing import Sequence

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils.torch_utils import torch_dtype_to_string
from parameterized import param, parameterized


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GenericPermuteTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GenericPermuteTest, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_generic_permute(
        self,
        input_shapes: Sequence[int],
        dims: Sequence[int],
        torch_dtype: torch.dtype,
        testname: str,
    ) -> None:
        ait_dtype = torch_dtype_to_string(torch_dtype)
        X = Tensor(shape=input_shapes, name="X", dtype=ait_dtype, is_input=True)
        op = ops.permute()
        Y = op(X, dims)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = compile_model(Y, target, "./tmp", f"{testname}_{self._test_id}")
        self._test_id += 1

        X_pt = torch.randn(input_shapes, dtype=torch_dtype).cuda()
        Y_pt = torch.permute(X_pt, dims)

        y = torch.empty(Y_pt.size(), dtype=torch_dtype).cuda()
        module.run_with_tensors([X_pt], [y])

        # mean, _, _ = module.benchmark_with_tensors([X_pt], [y], count=1000)
        # mem = 1
        # for dim in input_shapes:
        #     mem *= dim
        # bw = 2 * 2 * mem / (mean * 1e-3 * 1e9)  # GB/s
        # print(f"bw: {bw} GB/s")

        self.assertTrue(torch.equal(y, Y_pt))

    @parameterized.expand(
        [
            param((80, 300, 2), (0, 2, 1)),
            param((80, 300, 2), (1, 0, 2)),
            param((80, 300, 2), (2, 1, 0)),
            param((5, 113, 15, 31), (0, 2, 1, 3)),
            param((3, 1, 113, 15, 64), (2, 0, 3, 1, 4)),
            param((8, 29, 100000, 3), (0, 2, 1, 3)),
            param((32, 12, 4096, 64), (0, 2, 1, 3)),
            param((1, 12, 128, 64), (0, 2, 1, 3)),
            param((2, 3, 4, 5), (3, 2, 1, 0)),
            param((3, 5, 128, 514), (2, 3, 0, 1)),
            param((128, 512), (1, 0)),
            param((5, 113, 15, 31), (0, 1, 3, 2)),
            param((3, 1, 113, 15, 64), (0, 1, 2, 4, 3)),
        ]
    )
    def test_generic_permute_fp16(self, input_shapes, dims):
        self._test_generic_permute(
            input_shapes=input_shapes,
            dims=dims,
            torch_dtype=torch.float16,
            testname="test_generic_permute_fp16",
        )

    @parameterized.expand(
        [
            param((80, 300, 2), (0, 2, 1)),
            param((80, 300, 2), (1, 0, 2)),
            param((80, 300, 2), (2, 1, 0)),
            param((5, 113, 15, 31), (0, 2, 1, 3)),
            param((3, 1, 113, 15, 64), (2, 0, 3, 1, 4)),
            param((8, 29, 100000, 3), (0, 2, 1, 3)),
            param((32, 12, 4096, 64), (0, 2, 1, 3)),
            param((1, 12, 128, 64), (0, 2, 1, 3)),
            param((2, 3, 4, 5), (3, 2, 1, 0)),
            param((3, 5, 128, 514), (2, 3, 0, 1)),
            param((128, 512), (1, 0)),
            param((5, 113, 15, 31), (0, 1, 3, 2)),
            param((3, 1, 113, 15, 64), (0, 1, 2, 4, 3)),
        ]
    )
    @unittest.skipIf(detect_target().name() == "rocm", "FP32 is not supported by ROCm.")
    def test_generic_permute_fp32(self, input_shapes, dims):
        self._test_generic_permute(
            input_shapes=input_shapes,
            dims=dims,
            torch_dtype=torch.float32,
            testname="test_generic_permute_fp32",
        )

    @parameterized.expand(
        [
            param((80, 300, 2), (0, 2, 1)),
            param((80, 300, 2), (1, 0, 2)),
            param((80, 300, 2), (2, 1, 0)),
            param((5, 113, 15, 31), (0, 2, 1, 3)),
            param((3, 1, 113, 15, 64), (2, 0, 3, 1, 4)),
            param((8, 29, 100000, 3), (0, 2, 1, 3)),
            param((32, 12, 4096, 64), (0, 2, 1, 3)),
            param((1, 12, 128, 64), (0, 2, 1, 3)),
            param((2, 3, 4, 5), (3, 2, 1, 0)),
            param((3, 5, 128, 514), (2, 3, 0, 1)),
            param((128, 512), (1, 0)),
            param((5, 113, 15, 31), (0, 1, 3, 2)),
            param((3, 1, 113, 15, 64), (0, 1, 2, 4, 3)),
        ]
    )
    @unittest.skipIf(detect_target().name() == "rocm", "bf16 is not supported by ROCm.")
    def test_generic_permute_bf16(self, input_shapes, dims):
        self._test_generic_permute(
            input_shapes=input_shapes,
            dims=dims,
            torch_dtype=torch.bfloat16,
            testname="test_generic_permute_bf16",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
