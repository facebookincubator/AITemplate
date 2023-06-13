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
from typing import Callable

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.transform.remove_elementwise_no_ops import (
    remove_elementwise_no_ops,
)
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    gen_input_tensor,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils, shape_utils


class RemoveElementwiseNoOpsTestCase(unittest.TestCase):
    def _test_remove_elementwise_op_impl(
        self, elementwise_op_getter: Callable[[Tensor], Tensor], should_remove: bool
    ) -> None:
        batch_sizes = [1, 1024]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        M = 10
        X1 = gen_input_tensor([batch_dim, IntImm(M)], name="x1", dtype="float16")
        X2 = gen_input_tensor([batch_dim, IntImm(M)], name="x2", dtype="float16")
        add_0 = elementwise_op_getter(X1)
        Y = ops.elementwise(FuncEnum.ADD)(add_0, X2)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        sorted_graph = toposort([Y])
        modified_graph = remove_elementwise_no_ops(sorted_graph)
        if should_remove:
            self.assertEqual(len(modified_graph), len(sorted_graph) - 1)
            self.assertTrue(add_0 in sorted_graph)
            self.assertFalse(add_0 in modified_graph)
        else:
            self.assertEqual(sorted_graph, modified_graph)

    def test_remove_elementwise_op(self) -> None:
        test_cases = [
            (lambda x: ops.elementwise(FuncEnum.ADD)(x, 0), True),
            (lambda x: ops.elementwise(FuncEnum.ADD)(0, x), True),
            (lambda x: ops.elementwise(FuncEnum.ADD)(x, 1), False),
            (lambda x: ops.elementwise(FuncEnum.ADD)(1, x), False),
            (lambda x: ops.elementwise(FuncEnum.SUB)(x, 0), True),
            (lambda x: ops.elementwise(FuncEnum.SUB)(0, x), False),
            (lambda x: ops.elementwise(FuncEnum.SUB)(x, 1), False),
            (lambda x: ops.elementwise(FuncEnum.MUL)(x, 1), True),
            (lambda x: ops.elementwise(FuncEnum.MUL)(1, x), True),
            (lambda x: ops.elementwise(FuncEnum.MUL)(x, 2), False),
            (lambda x: ops.elementwise(FuncEnum.MUL)(2, x), False),
            (lambda x: ops.elementwise(FuncEnum.DIV)(x, 1), True),
            (lambda x: ops.elementwise(FuncEnum.DIV)(x, 2), False),
            (lambda x: ops.elementwise(FuncEnum.DIV)(1, x), False),
        ]
        for test_no, test in enumerate(test_cases):
            with self.subTest(test_no=test_no):
                self._test_remove_elementwise_op_impl(
                    elementwise_op_getter=test[0], should_remove=test[1]
                )

    def test_not_remove_connecting_input_output(
        self,
    ):
        batch_sizes = [1, 1024]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        M = 10
        X1 = gen_input_tensor([batch_dim, IntImm(M)], name="x1", dtype="float16")
        Y = ops.elementwise(FuncEnum.ADD)(X1, 0)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        sorted_graph = toposort([Y])
        modified_graph = remove_elementwise_no_ops(sorted_graph)
        self.assertEqual(sorted_graph, modified_graph)


class RemoveElementwiseNoOpsIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(RemoveElementwiseNoOpsIntegrationTest, self).__init__(*args, **kwargs)
        torch.manual_seed(0)
        self.BATCH_SIZES = [1, 218]
        self.M = 10

    def test_remove_elementwise_op(self) -> None:
        test_cases = [
            (lambda x: ops.elementwise(FuncEnum.ADD)(x, 0), lambda x: x + 0),
            (lambda x: ops.elementwise(FuncEnum.SUB)(x, 0), lambda x: x - 0),
            (lambda x: ops.elementwise(FuncEnum.MUL)(x, 1), lambda x: x * 1),
            (lambda x: ops.elementwise(FuncEnum.DIV)(x, 1), lambda x: x * 1),
        ]
        for test_no, test in enumerate(test_cases):
            with self.subTest(test_no=test_no):
                self._test_remove_elementwise_no_ops_impl(
                    elementwise_op_getter=test[0], expected_op=test[1]
                )

    def _test_remove_elementwise_no_ops_impl(
        self,
        elementwise_op_getter: Callable[[Tensor], Tensor],
        expected_op: Callable[[Tensor], Tensor],
    ):
        dtype = "float16"
        batch_dim = shape_utils.gen_int_var_min_max(self.BATCH_SIZES, "batch_0")
        reduce_dim = 0
        X0 = gen_input_tensor([batch_dim, IntImm(self.M)], name="x0", dtype=dtype)
        elementwise_op_0 = elementwise_op_getter(X0)
        Y = ops.reduce_mean(reduce_dim)(elementwise_op_0)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(
            Y,
            detect_target(),
            "./tmp",
            "test_remove_elementwise_no_ops",
        )

        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)

        for batch in self.BATCH_SIZES:
            x0_pt = get_random_torch_tensor([batch, self.M], dtype)
            add_0_pt = expected_op(x0_pt)
            y_pt = torch.mean(add_0_pt, dim=reduce_dim)
            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
