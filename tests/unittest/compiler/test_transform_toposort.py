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
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.testing import detect_target


class TestTopoSort(unittest.TestCase):
    def test_very_deep_toposort(self):
        x = Tensor(
            [2, 10],
            is_input=True,
            name="x",
        )

        for _ in range(1000):
            x = ops.elementwise(FuncEnum.RELU)(x)

        x._attrs["is_output"] = True
        x._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(x, target, "./tmp", "test_very_deep_toposort")

        x_pt = torch.randn((2, 10)).half().cuda()
        out_pt = torch.relu(x_pt)

        out_ait = torch.empty_like(out_pt)
        module.run_with_tensors({"x": x_pt}, {"output": out_ait})

        self.assertTrue(torch.equal(out_ait, out_pt))


if __name__ == "__main__":
    unittest.main()
