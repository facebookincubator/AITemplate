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
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
    graph_has_op,
)


class EliminatePermutationTestCase(unittest.TestCase):
    def test_eliminate_permutation(self):
        dtype = "float"
        shape = [32, 64, 112, 112]
        new_shape = [32, 64 * 112 * 112]
        target = detect_target()

        x = Tensor(shape, name="x", dtype=dtype, is_input=True)
        p1 = ops.permute()(x, dims=[0, 2, 3, 1])
        p2 = ops.permute()(p1, dims=[0, 3, 1, 2])
        z = ops.reshape()(p2, new_shape)
        z._attrs["is_output"] = True
        z._attrs["name"] = "z"

        x_pt = get_random_torch_tensor(shape, dtype)
        y_pt = get_torch_empty_tensor(new_shape, dtype)

        module = compile_model(z, target, "./tmp", "test_eliminate_permutation")
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x": x_pt}, {"z": y_pt})

        self.assertEqual(len(result_graph), 2)
        self.assertFalse(graph_has_op(result_graph, "permute"))
        self.assertTrue(torch.equal(torch.reshape(x_pt, new_shape), y_pt))

    def test_eliminate_last_permutation(self):
        dtype = "float"
        shape = [32, 64, 112, 112]
        target = detect_target()

        x = Tensor(shape, name="x", dtype=dtype, is_input=True)
        p1 = ops.permute()(x, dims=[0, 2, 3, 1])
        p2 = ops.permute()(p1, dims=[0, 2, 3, 1])
        z = ops.permute()(p2, dims=[0, 3, 1, 2])
        z._attrs["is_output"] = True
        z._attrs["name"] = "z"

        x_pt = get_random_torch_tensor(shape, dtype)
        y_pt = get_torch_empty_tensor(shape, dtype)

        module = compile_model(z, target, "./tmp", "test_eliminate_permutation")
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x": x_pt}, {"z": y_pt})

        self.assertEqual(len(result_graph), 2)
        self.assertTrue(graph_has_op(result_graph, "permute"))

    def test_eliminate_permutation_names(self):
        dtype = "float"
        shape = [32, 64, 112]
        new_shape = [32, 64 * 112]
        target = detect_target()

        x = Tensor(shape, name="x", dtype=dtype, is_input=True)
        p1 = ops.permute021()(x)
        p2 = ops.permute021()(p1)
        z = ops.reshape()(p2, new_shape)
        z._attrs["is_output"] = True
        z._attrs["name"] = "z"

        x_pt = get_random_torch_tensor(shape, dtype)
        y_pt = get_torch_empty_tensor(new_shape, dtype)

        module = compile_model(z, target, "./tmp", "test_eliminate_permutation_names")
        result_graph = module.debug_sorted_graph
        module.run_with_tensors({"x": x_pt}, {"z": y_pt})
        self.assertEqual(len(result_graph), 2)
        self.assertFalse(graph_has_op(result_graph, "permute"))
        self.assertTrue(torch.equal(torch.reshape(x_pt, new_shape), y_pt))

    def test_eliminate_permutation_multiple_operations(self):
        dtype = "float"
        shape = [2, 4]
        target = detect_target()

        x0 = Tensor(shape, name="x", dtype=dtype, is_input=True)
        p1 = ops.permute()(x0, dims=[1, 0])
        p2 = ops.permute()(p1, dims=[1, 0])
        r = ops.reshape()(p1, shape)
        z = ops.elementwise(FuncEnum.ADD)(r, p2)
        z._attrs["is_output"] = True
        z._attrs["name"] = "z"

        x_pt = get_random_torch_tensor(shape, dtype)
        y_pt = get_torch_empty_tensor(shape, dtype)

        module = compile_model(z, target, "./tmp", "test_eliminate_permutation")
        module.run_with_tensors({"x": x_pt}, {"z": y_pt})
        self.assertTrue(
            torch.equal(
                torch.reshape(torch.permute(x_pt, (1, 0)), shape) + x_pt,
                y_pt,
            )
        )

    def test_eliminate_permutation_multiple_operations_2(self):
        dtype = "float"
        shape = [2, 4]
        target = detect_target()

        x0 = Tensor(shape, name="x", dtype=dtype, is_input=True)
        p1 = ops.permute()(x0, dims=[1, 0])
        p2 = ops.permute()(p1, dims=[1, 0])
        r = ops.reshape()(p1, shape)
        a1 = ops.elementwise(FuncEnum.ADD)(r, p2)
        a2 = ops.elementwise(FuncEnum.ADD)(x0, p2)
        z = ops.elementwise(FuncEnum.MUL)(a1, a2)
        z._attrs["is_output"] = True
        z._attrs["name"] = "z"

        x_pt = get_random_torch_tensor(shape, dtype)
        y_pt = get_torch_empty_tensor(shape, dtype)

        module = compile_model(z, target, "./tmp", "test_eliminate_permutation")
        module.run_with_tensors({"x": x_pt}, {"z": y_pt})
        self.assertTrue(
            torch.equal(
                (torch.reshape(torch.permute(x_pt, (1, 0)), shape) + x_pt) * 2 * x_pt,
                y_pt,
            )
        )

    def test_eliminate_permutation_different_shapes(self):
        dtype = "float"
        shape = [32, 64, 112, 112]
        new_shape = [32, 64 * 112 * 112]
        target = detect_target()

        x = Tensor(shape, dtype=dtype, is_input=True)
        p1 = ops.permute()(x, dims=[0, 2, 3, 1])
        p2 = ops.permute()(p1, dims=[0, 2, 3, 1])
        z = ops.reshape()(p2, new_shape)
        z._attrs["is_output"] = True
        z._attrs["name"] = "z"
        module = compile_model(
            z, target, "./tmp", "test_eliminate_permutation_different_shapes"
        )
        result_graph = module.debug_sorted_graph
        self.assertEqual(len(result_graph), 4)
        self.assertTrue(graph_has_op(result_graph, "permute"))

    def test_eliminate_permutation_all_permutations(self):
        dtype = "float"
        target = detect_target()
        shape = [32, 64, 112, 112]

        x = Tensor(shape, dtype=dtype, is_input=True)
        p1 = ops.permute()(x, dims=[0, 2, 3, 1])
        p2 = ops.permute()(p1, dims=[0, 3, 1, 2])
        p2._attrs["is_output"] = True

        module = compile_model(
            p2,
            target,
            "./tmp",
            "test_eliminate_permutation_all_permutations",
        )
        result_graph = module.debug_sorted_graph
        self.assertEqual(len(result_graph), 3)
        self.assertTrue(graph_has_op(result_graph, "permute"))
