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
from aitemplate.compiler.public import FuncEnum
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from parameterized import param, parameterized


class TestFull(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_id = 0

    def _test_full(
        self,
        shape,
        fill_value,
        dtype="float16",
        test_name="full",
    ) -> None:
        Y = ops.full()(shape, fill_value, dtype)
        Y._attrs["name"] = "Y"

        if not isinstance(shape, list):
            shape = [shape]

        X = Tensor(
            shape=shape,
            name="X",
            dtype=dtype,
            is_input=True,
        )

        Z = ops.elementwise(FuncEnum.ADD)(X, Y)
        Z._attrs["name"] = "Z"
        Z._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Z, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1

        if isinstance(shape[0], IntVar):
            shapes = [[val] + shape[1:] for val in shape[0]._attrs["values"]]
        else:
            shapes = [shape]

        for shape in shapes:
            x_pt = get_random_torch_tensor(shape, dtype=dtype)
            z_pt = x_pt + fill_value

            z = torch.empty_like(z_pt)

            module.run_with_tensors([x_pt], [z])
            torch.testing.assert_close(z, z_pt, atol=1e-2, rtol=1e-2)

    @parameterized.expand(
        [
            param(1, [1], 1, "float16"),
            param(2, [10, 20, 30], 3.14, "float16"),
            param(3, [IntVar([10, 20]), 30], 0, "float16"),
            param(4, 123, -5, "float16"),
            param(5, [20, 30], 2.71, "float32"),
            param(6, [IntVar([1, 128]), 10], -1.23, "float32"),
            param(7, IntVar([1, 128]), 1234, "float32"),
        ]
    )
    def test_full(self, i, shape, fill_value, dtype):
        self._test_full(
            shape=shape,
            fill_value=fill_value,
            dtype=dtype,
            test_name=f"test_full_{i}",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
