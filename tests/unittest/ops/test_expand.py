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
from aitemplate.testing.test_utils import get_random_torch_tensor, graph_has_op


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

    def _test_no_op_expands_removed_static_shapes(
        self,
        test_name="no_op_expands_removed_static_shapes",
        dtype="float16",
    ):
        x = Tensor(
            [1, 2, 3],
            name="input_0",
            is_input=True,
            dtype=dtype,
        )
        y = ops.expand()(x, [1, -1, -1])
        z = ops.elementwise(FuncEnum.MUL)(y, y)
        z._attrs["is_output"] = True
        z._attrs["name"] = "output_0"

        x_pt = get_random_torch_tensor([1, 2, 3], dtype=dtype)
        z_pt = x_pt * x_pt
        z_ait = torch.empty_like(z_pt)
        with compile_model(z, detect_target(), "./tmp", test_name) as module:
            module.run_with_tensors({"input_0": x_pt}, {"output_0": z_ait})
            self.assertFalse(graph_has_op(module.debug_sorted_graph, "expand"))
            self.assertTrue(torch.equal(z_ait, z_pt))

    def test_no_op_expands_removed_static_shapes_fp16(self):
        self._test_no_op_expands_removed_static_shapes(
            test_name="no_op_expands_removed_static_shapes_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_no_op_expands_removed_static_shapes_fp32(self):
        self._test_no_op_expands_removed_static_shapes(
            test_name="no_op_expands_removed_static_shapes_fp32",
            dtype="float32",
        )

    def _test_no_op_expands_removed_dynamic_shapes(
        self,
        test_name="no_op_expands_removed_dynamic_shapes",
        dtype="float16",
    ):
        dynamic_dim = IntVar([1, 5], name="dynamic_dim")
        x = Tensor(
            [1, dynamic_dim, 3],
            name="input_0",
            is_input=True,
            dtype=dtype,
        )
        y = ops.expand()(x, [IntVar([1, 1]), -1, -1])
        z = ops.elementwise(FuncEnum.MUL)(y, y)
        z._attrs["is_output"] = True
        z._attrs["name"] = "output_0"

        x_pt = get_random_torch_tensor([1, 2, 3], dtype=dtype)
        z_pt = x_pt * x_pt
        z_ait = torch.empty_like(z_pt)
        with compile_model(z, detect_target(), "./tmp", test_name) as module:
            module.run_with_tensors({"input_0": x_pt}, {"output_0": z_ait})
            self.assertFalse(graph_has_op(module.debug_sorted_graph, "expand"))
            self.assertTrue(torch.equal(z_ait, z_pt))

    def test_no_op_expands_removed_dynamic_shapes_fp16(self):
        self._test_no_op_expands_removed_dynamic_shapes(
            test_name="no_op_expands_removed_dynamic_shapes_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_no_op_expands_removed_dynamic_shapes_fp32(self):
        self._test_no_op_expands_removed_dynamic_shapes(
            test_name="no_op_expands_removed_dynamic_shapes_fp32",
            dtype="float32",
        )

    def _test_no_op_expands_removed_size_op(
        self,
        test_name="no_op_expands_removed_size_op",
        dtype="float16",
    ):
        x = Tensor(
            [1, 2, 3],
            name="input_0",
            is_input=True,
            dtype=dtype,
        )
        y = Tensor(
            [IntVar([1, 1]), 2, 3],
            name="input_1",
            is_input=True,
            dtype=dtype,
        )
        x_size = ops.size()(x, 0)
        y_size = ops.size()(y, 0)
        x_expand = ops.expand()(x, [x_size, -1, -1])
        y_expand = ops.expand()(y, [y_size, -1, -1])
        z = ops.elementwise(FuncEnum.MUL)(x_expand, y_expand)
        z._attrs["is_output"] = True
        z._attrs["name"] = "output_0"

        x_pt = get_random_torch_tensor([1, 2, 3], dtype=dtype)
        y_pt = get_random_torch_tensor([1, 2, 3], dtype=dtype)
        z_pt = x_pt * y_pt
        z_ait = torch.empty_like(z_pt)
        with compile_model(z, detect_target(), "./tmp", test_name) as module:
            module.run_with_tensors(
                {"input_0": x_pt, "input_1": y_pt}, {"output_0": z_ait}
            )
            self.assertFalse(graph_has_op(module.debug_sorted_graph, "expand"))
            self.assertTrue(torch.equal(z_ait, z_pt))

    def test_no_op_expands_removed_size_op_fp16(self):
        self._test_no_op_expands_removed_size_op(
            test_name="no_op_expands_removed_size_op_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_no_op_expands_removed_size_op_fp32(self):
        self._test_no_op_expands_removed_size_op(
            test_name="no_op_expands_removed_size_op_fp32",
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
