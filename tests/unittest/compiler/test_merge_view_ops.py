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
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    count_ops,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    graph_has_op,
)
from aitemplate.utils import graph_utils


class MergeViewOpsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def test_basic(self):
        """
        Check that we convert a sequence of reshape(unsqueeze(...)) into a
        single reshape() call.
        """
        dtype = "float"
        x0_shape = [2, 4, 8]
        y_shape = [8, 8]

        x0 = Tensor(
            shape=x0_shape,
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        x1 = ops.reshape()(x0, [8, 8])
        x2 = ops.unsqueeze(dim=1)(x1)
        y = ops.reduce_sum(dim=1)(x2)
        y._attrs["name"] = "y"
        y._attrs["is_output"] = True

        x0_pt = get_random_torch_tensor(x0_shape, dtype)
        y_pt = get_torch_empty_tensor(y_shape, dtype)

        target = detect_target()
        module = compile_model(
            y,
            target,
            "./tmp",
            "test_basic",
        )
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x0": x0_pt}, {"y": y_pt})

        self.assertEqual(len(result_graph), 3)
        self.assertFalse(graph_has_op(result_graph, "unsqueeze"))

        expected = torch.reshape(x0_pt, y_shape)
        torch.testing.assert_close(expected, y_pt, atol=5e-2, rtol=5e-2)

    def test_multiple_sequential_views(self):
        """
        Check that we convert a sequence of reshape(unsqueeze(reshape(...)))
        into a single reshape() call.
        """
        dtype = "float"
        x0_shape = [2, 4, 8]
        y_shape = [8, 2, 4]

        x0 = Tensor(
            shape=x0_shape,
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        x1 = ops.reshape()(x0, [8, 8])
        x2 = ops.unsqueeze(dim=1)(x1)
        x3 = ops.reshape()(x2, [8, 1, 2, 4])
        y = ops.reduce_sum(dim=1)(x3)
        y._attrs["name"] = "y"
        y._attrs["is_output"] = True

        x0_pt = get_random_torch_tensor(x0_shape, dtype)
        y_pt = get_torch_empty_tensor(y_shape, dtype)

        target = detect_target()
        module = compile_model(
            y,
            target,
            "./tmp",
            "test_multiple_sequential_views",
        )
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x0": x0_pt}, {"y": y_pt})

        self.assertEqual(len(result_graph), 3)
        self.assertFalse(graph_has_op(result_graph, "unsqueeze"))

        expected = torch.reshape(x0_pt, y_shape)
        torch.testing.assert_close(expected, y_pt, atol=5e-2, rtol=5e-2)

    def test_multiple_dst_view_ops(self):
        """
        Given

          x0 -> reshape -> x1 -> unsqueeze -> ...
                            |--> unsqueeze -> ...

        We want to merge both unsqueeze calls into the preceding reshape call.
        """
        dtype = "float"
        x0_shape = [2, 4, 8]
        y_shape = [8, 8]

        x0 = Tensor(
            shape=x0_shape,
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        x1 = ops.reshape()(x0, [8, 8])
        x2 = ops.unsqueeze(dim=1)(x1)
        x3 = ops.unsqueeze(dim=2)(x1)

        y0 = ops.reduce_sum(dim=1)(x2)
        y0._attrs["name"] = "y0"
        y0._attrs["is_output"] = True

        y1 = ops.reduce_sum(dim=2)(x3)
        y1._attrs["name"] = "y1"
        y1._attrs["is_output"] = True

        x0_pt = get_random_torch_tensor(x0_shape, dtype)
        y0_pt = get_torch_empty_tensor(y_shape, dtype)
        y1_pt = get_torch_empty_tensor(y_shape, dtype)

        target = detect_target()
        module = compile_model(
            [y0, y1],
            target,
            "./tmp",
            "test_multiple_dst_view_ops",
        )
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x0": x0_pt}, {"y0": y0_pt, "y1": y1_pt})

        self.assertEqual(len(result_graph), 5)
        self.assertFalse(graph_has_op(result_graph, "unsqueeze"))
        sorted_ops = graph_utils.get_sorted_ops(result_graph)
        self.assertEqual(count_ops(sorted_ops, "reshape"), 2)

        y_expected = torch.reshape(x0_pt, [8, 8])
        torch.testing.assert_close(y_expected, y0_pt, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(y_expected, y1_pt, atol=5e-2, rtol=5e-2)

    def test_multiple_dst_ops(self):
        """
        Given

          x0 -> reshape -> x1 -> unsqueeze -> ...
                            |--> ...

        We cannot eliminate x1 since it has a non-view-op destination, but we
        can still merge the reshape and unsqueeze operators.
        """
        dtype = "float"
        x0_shape = [2, 4, 8]
        y0_shape = [8]
        y1_shape = [8, 8]

        x0 = Tensor(
            shape=x0_shape,
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        x1 = ops.reshape()(x0, [8, 8])
        x2 = ops.unsqueeze(dim=1)(x1)

        y0 = ops.reduce_sum(dim=1)(x1)
        y0._attrs["name"] = "y0"
        y0._attrs["is_output"] = True

        y1 = ops.reduce_sum(dim=1)(x2)
        y1._attrs["name"] = "y1"
        y1._attrs["is_output"] = True

        x0_pt = get_random_torch_tensor(x0_shape, dtype)
        y0_pt = get_torch_empty_tensor(y0_shape, dtype)
        y1_pt = get_torch_empty_tensor(y1_shape, dtype)

        target = detect_target()
        module = compile_model(
            [y0, y1],
            target,
            "./tmp",
            "test_multiple_dst_ops",
        )
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x0": x0_pt}, {"y0": y0_pt, "y1": y1_pt})

        self.assertEqual(len(result_graph), 5)
        self.assertFalse(graph_has_op(result_graph, "unsqueeze"))
        sorted_ops = graph_utils.get_sorted_ops(result_graph)
        self.assertEqual(count_ops(sorted_ops, "reshape"), 2)

        y0_expected = torch.sum(torch.reshape(x0_pt, [8, 8]), 1)
        y1_expected = torch.reshape(x0_pt, y1_shape)
        torch.testing.assert_close(y0_expected, y0_pt, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(y1_expected, y1_pt, atol=5e-2, rtol=5e-2)

    def test_identity_reshape(self):
        """
        Given reshape(reshape(x, shape0), shape1), where shape1 is identical to
        x's original shape, we can eliminate both reshape ops.
        """
        dtype = "float"
        x0_shape = [2, 4, 8]
        y_shape = [2, 8]

        x0 = Tensor(
            shape=x0_shape,
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        x1 = ops.reshape()(x0, [8, 8])
        x2 = ops.reshape()(x1, x0_shape)

        y = ops.reduce_sum(dim=1)(x2)
        y._attrs["name"] = "y"
        y._attrs["is_output"] = True

        x0_pt = get_random_torch_tensor(x0_shape, dtype)
        y_pt = get_torch_empty_tensor(y_shape, dtype)

        target = detect_target()
        module = compile_model(
            y,
            target,
            "./tmp",
            "test_identity_reshape",
        )
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x0": x0_pt}, {"y": y_pt})

        self.assertEqual(len(result_graph), 2)
        self.assertFalse(graph_has_op(result_graph, "reshape"))
        expected = torch.sum(x0_pt, 1)
        torch.testing.assert_close(expected, y_pt, atol=5e-2, rtol=5e-2)

    def test_identity_reshape_multiple_dst_ops(self):
        """
        Given

          x0 -> reshape -> x1 -> reshape -> x2 -> op1
                              -> op2

        If x2 == x0, we can transform that into

          x0 -> op1
             -> reshape -> x1 -> op2
        """
        dtype = "float"
        x0_shape = [2, 4, 8]
        y0_shape = [2, 8]
        y1_shape = [8]

        x0 = Tensor(
            shape=x0_shape,
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        x1 = ops.reshape()(x0, [8, 8])
        x2 = ops.reshape()(x1, x0_shape)

        y0 = ops.reduce_sum(dim=1)(x2)
        y0._attrs["name"] = "y0"
        y0._attrs["is_output"] = True

        y1 = ops.reduce_sum(dim=1)(x1)
        y1._attrs["name"] = "y1"
        y1._attrs["is_output"] = True

        x0_pt = get_random_torch_tensor(x0_shape, dtype)
        y0_pt = get_torch_empty_tensor(y0_shape, dtype)
        y1_pt = get_torch_empty_tensor(y1_shape, dtype)

        target = detect_target()
        module = compile_model(
            [y0, y1],
            target,
            "./tmp",
            "test_identity_reshape_multiple_dst_ops",
        )
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x0": x0_pt}, {"y0": y0_pt, "y1": y1_pt})

        self.assertEqual(len(result_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(result_graph)
        self.assertEqual(count_ops(sorted_ops, "reshape"), 1)

        y0_expected = torch.sum(x0_pt, 1)
        y1_expected = torch.sum(torch.reshape(x0_pt, [8, 8]), 1)
        torch.testing.assert_close(y0_expected, y0_pt, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(y1_expected, y1_pt, atol=5e-2, rtol=5e-2)

    def test_identity_reshape_in_out_conflict(self):
        """
        If x is an input and y is an output tensor, then we can only eliminate
        one view op in the following example:

          y = reshape(reshape(x, y_shape), x_original_shape)
        """
        dtype = "float"
        x0_shape = [2, 4, 8]

        x0 = Tensor(
            shape=x0_shape,
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        x1 = ops.reshape()(x0, [8, 8])

        y = ops.reshape()(x1, x0_shape)
        y._attrs["name"] = "y"
        y._attrs["is_output"] = True

        x0_pt = get_random_torch_tensor(x0_shape, dtype)
        y_pt = get_torch_empty_tensor(x0_shape, dtype)

        target = detect_target()
        module = compile_model(
            y,
            target,
            "./tmp",
            "test_identity_reshape_in_out_conflict",
        )
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x0": x0_pt}, {"y": y_pt})

        self.assertEqual(len(result_graph), 2)
        self.assertTrue(graph_has_op(result_graph, "reshape"))
        torch.testing.assert_close(x0_pt, y_pt, atol=5e-2, rtol=5e-2)

    def test_identity_reshape_out_out_conflict(self):
        """
        If y0 and y1 are both output tensors, then we can only eliminate one
        view op in the following example:

          y1 = reshape(reshape(y0, some_shape), y0_original_shape)
        """
        dtype = "float"
        x0_shape = [2, 4, 8]
        y_shape = [2, 8]

        x0 = Tensor(
            shape=x0_shape,
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        y0 = ops.reduce_sum(dim=1)(x0)
        y0._attrs["name"] = "y0"
        y0._attrs["is_output"] = True

        x1 = ops.reshape()(y0, [4, 4])

        y1 = ops.reshape()(x1, y_shape)
        y1._attrs["name"] = "y1"
        y1._attrs["is_output"] = True

        x0_pt = get_random_torch_tensor(x0_shape, dtype)
        y0_pt = get_torch_empty_tensor(y_shape, dtype)
        y1_pt = get_torch_empty_tensor(y_shape, dtype)

        target = detect_target()
        module = compile_model(
            [y0, y1],
            target,
            "./tmp",
            "test_identity_reshape_out_out_conflict",
        )
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x0": x0_pt}, {"y0": y0_pt, "y1": y1_pt})

        y_expected = torch.sum(x0_pt, 1)

        self.assertEqual(len(result_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(result_graph)
        self.assertEqual(count_ops(sorted_ops, "reshape"), 1)
        torch.testing.assert_close(y_expected, y0_pt, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(y_expected, y1_pt, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
