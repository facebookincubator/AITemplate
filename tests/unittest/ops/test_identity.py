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
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from parameterized import param, parameterized


class TestIdentity(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_id = 0

    def _test_identity(
        self,
        shape,
        elementwise,
        dtype="float16",
        test_name="identity",
    ) -> None:
        X = Tensor(
            shape=shape,
            name="X",
            dtype=dtype,
            is_input=True,
        )
        Y = ops.identity()(X)
        if elementwise:
            Y = ops.elementwise(FuncEnum.ADD)(X, Y)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self.assertEqual(len(module.debug_sorted_graph), 2)
        self._test_id += 1

        x_pt = get_random_torch_tensor(shape, dtype=dtype)
        if elementwise:
            y_pt = 2 * x_pt
        else:
            y_pt = x_pt

        y = torch.empty_like(y_pt)

        module.run_with_tensors([x_pt], [y])
        torch.testing.assert_close(y, y_pt, atol=1e-2, rtol=1e-2)

    @parameterized.expand(
        [
            param(1, [3, 4], True, "float16"),
            param(2, [3, 4], True, "float32"),
            param(3, [3, 4], False, "float16"),
            param(4, [3, 4], False, "float32"),
        ]
    )
    def test_identity(self, i, shape, elementwise, dtype):
        self._test_identity(
            shape=shape,
            elementwise=elementwise,
            dtype=dtype,
            test_name=f"test_identity_{i}",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
