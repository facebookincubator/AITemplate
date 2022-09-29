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


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ExpandTestCase(unittest.TestCase):
    def test_expand_fails_mismatched_ndim(self):
        x = Tensor(shape=[5, IntVar([1, 10]), 5])
        expand_shape = [5, -1]
        self.assertRaises(ValueError, ops.expand().__call__, x, expand_shape)

    def test_expand_fails_non_singleton_dim(self):
        x = Tensor(shape=[5, 1, 2])
        expand_shape = [6, 1, 2]
        self.assertRaises(ValueError, ops.expand().__call__, x, expand_shape)

        x = Tensor(shape=[IntVar([1, 10])])
        expand_shape = [20]
        self.assertRaises(ValueError, ops.expand().__call__, x, expand_shape)

    def test_no_op_expands_removed_static_shapes(self):
        x = Tensor([1, 2, 3], name="input_0", is_input=True)
        y = ops.expand()(x, [1, -1, -1])
        z = ops.elementwise(FuncEnum.MUL)(y, y)
        z._attrs["is_output"] = True
        z._attrs["name"] = "output_0"

        x_pt = torch.randn((1, 2, 3)).half().cuda()
        z_pt = x_pt * x_pt
        z_ait = torch.empty_like(z_pt)
        with compile_model(
            z, detect_target(), "./tmp", "test_no_op_expands_removed_static_shapes"
        ) as module:
            module.run_with_tensors({"input_0": x_pt}, {"output_0": z_ait})
            self.assertFalse(graph_has_op(module.debug_sorted_graph, "expand"))
            self.assertTrue(torch.equal(z_ait, z_pt))

    def test_no_op_expands_removed_dynamic_shapes(self):
        dynamic_dim = IntVar([1, 5], name="dynamic_dim")
        x = Tensor([1, dynamic_dim, 3], name="input_0", is_input=True)
        y = ops.expand()(x, [IntVar([1, 1]), -1, -1])
        z = ops.elementwise(FuncEnum.MUL)(y, y)
        z._attrs["is_output"] = True
        z._attrs["name"] = "output_0"

        x_pt = torch.randn((1, 2, 3)).half().cuda()
        z_pt = x_pt * x_pt
        z_ait = torch.empty_like(z_pt)
        with compile_model(
            z, detect_target(), "./tmp", "test_no_op_expands_removed_dynamic_shapes"
        ) as module:
            module.run_with_tensors({"input_0": x_pt}, {"output_0": z_ait})
            self.assertFalse(graph_has_op(module.debug_sorted_graph, "expand"))
            self.assertTrue(torch.equal(z_ait, z_pt))

    def test_no_op_expands_removed_size_op(self):
        x = Tensor([1, 2, 3], name="input_0", is_input=True)
        y = Tensor([IntVar([1, 1]), 2, 3], name="input_1", is_input=True)
        x_size = ops.size()(x, 0)
        y_size = ops.size()(y, 0)
        x_expand = ops.expand()(x, [x_size, -1, -1])
        y_expand = ops.expand()(y, [y_size, -1, -1])
        z = ops.elementwise(FuncEnum.MUL)(x_expand, y_expand)
        z._attrs["is_output"] = True
        z._attrs["name"] = "output_0"

        x_pt = torch.randn((1, 2, 3)).half().cuda()
        y_pt = torch.randn((1, 2, 3)).half().cuda()
        z_pt = x_pt * y_pt
        z_ait = torch.empty_like(z_pt)
        with compile_model(
            z, detect_target(), "./tmp", "test_no_op_expands_removed_dynamic_shapes"
        ) as module:
            module.run_with_tensors(
                {"input_0": x_pt, "input_1": y_pt}, {"output_0": z_ait}
            )
            self.assertFalse(graph_has_op(module.debug_sorted_graph, "expand"))
            self.assertTrue(torch.equal(z_ait, z_pt))


if __name__ == "__main__":
    unittest.main()
