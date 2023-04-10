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
from aitemplate.compiler.transform.toposort import (
    _dfsSort,
    _priSort,
    SizePriTensorHelper,
)
from aitemplate.testing import detect_target


class TestTopoSort(unittest.TestCase):
    def _get_diff_size_graph(self):
        X1 = Tensor(shape=[10, 50], dtype="float16", name="in_10_50")
        X2 = Tensor(shape=[50, 1000], dtype="float16", name="in_50_1000")
        X3 = Tensor(shape=[1000, 5], dtype="float16", name="in_1000_5")
        X4 = Tensor(shape=[5, 5], dtype="float16", name="in_5_5")
        X5 = ops.gemm_rrr()(X1, X2)
        X5._attrs["name"] = "MUL_10_1000"
        X6 = ops.gemm_rrr()(X3, X4)
        X6._attrs["name"] = "MUL_1000_5"
        X7 = ops.gemm_rrr()(X5, X6)
        X7._attrs["name"] = "MUL_10_5"
        X7._attrs["is_output"] = True
        return X7

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

    def test_size_pri_toposort(self):
        tensor = self._get_diff_size_graph()
        expected_order = [
            "in_10_50",
            "in_50_1000",
            "in_1000_5",
            "in_5_5",
            "MUL_10_1000",
            "MUL_1000_5",
            "MUL_10_5",
        ]
        self.assertEqual(
            [node._attrs["name"] for node in _priSort(tensor, SizePriTensorHelper())],
            expected_order,
        )

        # dfs don't follow size pri order
        self.assertNotEqual(
            [node._attrs["name"] for node in _dfsSort(tensor)], expected_order
        )


if __name__ == "__main__":
    unittest.main()
