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
from aitemplate.testing.test_utils import get_random_torch_tensor
from parameterized import param, parameterized


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class TransposeTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TransposeTest, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_transpose_static_shape(
        self,
        input_shape: Sequence[int],
        dim0: int,
        dim1: int,
        dtype: str = "float16",
        test_name: str = "transpose_static_shape",
    ) -> None:
        X = Tensor(
            shape=input_shape,
            name="X",
            dtype=dtype,
            is_input=True,
        )
        op = ops.transpose()
        Y = op(X, dim0, dim1)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1

        X_pt = get_random_torch_tensor(input_shape, dtype=dtype)
        Y_pt = torch.transpose(X_pt, dim0, dim1).contiguous()

        y = torch.empty_like(Y_pt)
        module.run_with_tensors([X_pt], [y])

        torch.testing.assert_close(y, Y_pt, atol=1e-2, rtol=1e-2)

    @parameterized.expand(
        [
            param((80, 300, 2), 1, 2),
            param((80, 300, 2), 2, -2),
            param((32, 12, 4096, 64), 2, 1),
            param((128, 512), -1, -2),
            param((128, 512), 0, 0),
        ]
    )
    def test_transpose_static_shape_fp16(self, input_shape, dim0, dim1):
        self._test_transpose_static_shape(
            input_shape=input_shape,
            dim0=dim0,
            dim1=dim1,
            test_name="test_transpose_static_shape_fp16",
            dtype="float16",
        )

    @parameterized.expand(
        [
            param((80, 300, 2), 1, 2),
        ]
    )
    def test_transpose_static_shape_fp32(self, input_shape, dim0, dim1):
        self._test_transpose_static_shape(
            input_shape=input_shape,
            dim0=dim0,
            dim1=dim1,
            test_name="test_transpose_static_shape_fp32",
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
