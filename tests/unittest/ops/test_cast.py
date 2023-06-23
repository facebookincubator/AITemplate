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

import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils.torch_utils import string_to_torch_dtype
from parameterized import param, parameterized


class TestCast(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_id = 0

    def _test_cast(
        self,
        shape,
        dtype="float32",
        cast_dtype="bfloat16",
        test_name="cast",
    ) -> None:
        if not isinstance(shape, list):
            shape = [shape]

        X = Tensor(
            shape=shape,
            name="X",
            dtype=dtype,
            is_input=True,
        )

        Y = ops.cast()(X, cast_dtype)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1

        x = get_random_torch_tensor(shape, dtype=dtype)
        y = get_torch_empty_tensor(shape, dtype=cast_dtype)
        inputs = {"X": x}
        outputs = {"Y": y}
        module.run_with_tensors(inputs, outputs)

        y_pt = x.to(string_to_torch_dtype(cast_dtype))
        torch.testing.assert_close(y, y_pt, atol=1e-2, rtol=1e-2)

    @parameterized.expand(
        [
            param(1, "float16", "bfloat16", [1], "float16_to_bfloat16"),
            param(2, "float16", "float32", [10, 20], "float16_to_float32"),
            param(3, "bfloat16", "float16", [10, 20, 30], "bfloat16_to_float16"),
            param(4, "bfloat16", "float32", 123, "bfloat16_to_float32"),
            param(5, "float32", "float16", [20, 30], "float32_to_float16"),
            param(6, "float32", "bfloat16", [1, 128], "float32_to_bfloat16"),
        ]
    )
    def test_cast(
        self,
        i,
        dtype,
        cast_dtype,
        shape,
        test_name,
    ):
        self._test_cast(
            shape=shape,
            dtype=dtype,
            cast_dtype=cast_dtype,
            test_name=test_name,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
