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
from aitemplate.compiler.base import IntVar, Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import graph_has_op
from parameterized import param, parameterized


class TestFuseExpand(unittest.TestCase):
    @parameterized.expand(
        [
            param(True, "test_fuse_expand_elementwise_exact"),
            param(False, "test_fuse_expand_elementwise_non_exact"),
        ]
    )
    def test_fuse_expand_elementwise(self, exact_match: bool, name: str):
        N, M = (2, 10) if exact_match else (1, 1)
        x = Tensor(
            [IntVar([1, 10], name="batch"), 2, 10],
            is_input=True,
            name="x",
        )
        B = ops.size()(x, 0)

        y = Tensor([1, N, M], is_input=True, name="y")
        y_expanded = ops.expand()(y, [B, -1, -1])

        z = ops.elementwise(FuncEnum.ADD)(x, y_expanded)
        z._attrs["is_output"] = True
        z._attrs["name"] = "z"

        with compile_model(z, detect_target(), "./tmp", name) as mod:
            self.assertFalse(graph_has_op(mod.debug_sorted_graph, "expand"))
            for batch_size in (1, 5, 10):
                x_pt = torch.randn((batch_size, 2, 10)).half().cuda()
                y_pt = torch.randn((1, N, M)).half().cuda()
                z_pt = x_pt + y_pt.expand(batch_size, -1, -1)

                z_ait = torch.empty_like(z_pt)
                mod.run_with_tensors({"x": x_pt, "y": y_pt}, {"z": z_ait})
                self.assertTrue(torch.equal(z_ait, z_pt))


if __name__ == "__main__":
    unittest.main()
