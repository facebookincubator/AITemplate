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
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    gen_input_tensor,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils, shape_utils


class RemoveIdOpsTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(RemoveIdOpsTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0
        self.BATCH_SIZE = 1024

    def test_remove_id_simple(
        self,
        test_name="remove_id_simple",
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor
        # x1 = tensor
        # add_0 = add(x0, x0)
        # id_1 = id(add_0)
        # y = add(x1, id_1)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        M = 10
        X0 = gen_input_tensor([batch_dim, IntImm(M)], name="x0", dtype=dtype)
        X1 = gen_input_tensor([batch_dim, IntImm(M)], name="x1", dtype=dtype)
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X0)
        id_1 = ops.identity()(add_0)
        Y = ops.elementwise(FuncEnum.ADD)(X1, id_1)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self._test_id}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self._test_id += 1

        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M], dtype)
            x1_pt = get_random_torch_tensor([batch, M], dtype)
            add_0_pt = x0_pt + x0_pt
            id_1_pt = add_0_pt
            y_pt = x1_pt + id_1_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)

    def test_remove_id_simple_2(
        self,
        test_name="remove_id_simple_2",
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor
        # x1 = tensor
        # add_0 = add(x0, x0)
        # id_1 = id(add_0)
        # id_2 = id(x1)
        # y = add(id_1, id_2)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        M = 10
        X0 = gen_input_tensor([batch_dim, IntImm(M)], name="x0", dtype=dtype)
        X1 = gen_input_tensor([batch_dim, IntImm(M)], name="x1", dtype=dtype)
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X0)
        id_1 = ops.identity()(add_0)
        id_2 = ops.identity()(X1)
        Y = ops.elementwise(FuncEnum.ADD)(id_1, id_2)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self._test_id}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self._test_id += 1

        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M], dtype)
            x1_pt = get_random_torch_tensor([batch, M], dtype)
            add_0_pt = x0_pt + x0_pt
            id_1_pt = add_0_pt
            id_2_pt = x1_pt
            y_pt = id_1_pt + id_2_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)

    def test_remove_consecutive_ids_1(
        self,
        test_name="remove_consecutive_ids_1",
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor
        # x1 = tensor
        # add_0 = add(x0, x0)
        # id_1 = id(add_0)
        # id_2 = id(id_1)
        # id_3 = id(id_2)
        # id_4 = id(id_1)
        # add_1 = add(id_3, id_4)
        # add_2 = add(id_1, id_4)
        # y = add(add_1, add_2)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        M = 10
        X0 = gen_input_tensor([batch_dim, IntImm(M)], name="x0", dtype=dtype)
        X1 = gen_input_tensor([batch_dim, IntImm(M)], name="x1", dtype=dtype)
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X0)
        id_1 = ops.identity()(add_0)
        id_2 = ops.identity()(id_1)
        id_3 = ops.identity()(id_2)
        id_4 = ops.identity()(id_1)
        add_1 = ops.elementwise(FuncEnum.ADD)(id_3, id_4)
        id_5 = ops.identity()(X1)
        add_2 = ops.elementwise(FuncEnum.ADD)(id_5, id_2)
        Y = ops.elementwise(FuncEnum.ADD)(add_1, add_2)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self._test_id}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self._test_id += 1

        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M], dtype)
            x1_pt = get_random_torch_tensor([batch, M], dtype)
            add_0_pt = x0_pt + x0_pt
            id_1_pt = add_0_pt
            id_2_pt = id_1_pt
            id_3_pt = id_2_pt
            id_4_pt = id_1_pt
            add_1_pt = id_3_pt + id_4_pt
            id_5_pt = x1_pt
            add_2_pt = id_5_pt + id_2_pt
            y_pt = add_1_pt + add_2_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)

    def test_remove_consecutive_ids_2(
        self,
        test_name="remove_consecutive_ids_2",
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor
        # x1 = tensor
        # add_0 = add(x0, x1)
        # id_1 = id(add_0)
        # id_2 = id(id_1)
        # id_3 = id(x1)
        # id_4 = id(id_3)
        # add_1 = add(id_2, id_4)
        # id_5 = id(add_1)
        # y = id(id_5)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        M = 10
        X0 = gen_input_tensor([batch_dim, IntImm(M)], name="x0", dtype=dtype)
        X1 = gen_input_tensor([batch_dim, IntImm(M)], name="x1", dtype=dtype)
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        id_1 = ops.identity()(add_0)
        id_2 = ops.identity()(id_1)
        id_3 = ops.identity()(X1)
        id_4 = ops.identity()(id_3)
        add_1 = ops.elementwise(FuncEnum.ADD)(id_2, id_4)
        id_5 = ops.identity()(add_1)
        Y = ops.identity()(id_5)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self._test_id}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self._test_id += 1

        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M], dtype)
            x1_pt = get_random_torch_tensor([batch, M], dtype)
            add_0_pt = x0_pt + x1_pt
            id_1_pt = add_0_pt
            id_2_pt = id_1_pt
            id_3_pt = x1_pt
            id_4_pt = id_3_pt
            add_1_pt = id_2_pt + id_4_pt
            id_5_pt = add_1_pt
            y_pt = id_5_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)

    def test_non_removable_id(
        self,
        test_name="non_removable_id",
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor
        # y = id(x0)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        M = 10
        X0 = gen_input_tensor([batch_dim, IntImm(M)], name="x0", dtype=dtype)
        Y = ops.identity()(X0)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self._test_id}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self._test_id += 1

        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "identity")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M], dtype)
            y_pt = x0_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)

    def test_non_removable_id_2(
        self,
        test_name="non_removable_id_2",
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor
        # x1 = tensor
        # add_0 = add(x0, x1)
        # id_1 = id(add_0)
        # y0 = id(id_1)
        # y1 = id(x0)
        # y2 = add(y0, y1)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        M = 10
        X0 = gen_input_tensor([batch_dim, IntImm(M)], name="x0", dtype=dtype)
        X1 = gen_input_tensor([batch_dim, IntImm(M)], name="x1", dtype=dtype)
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        id_1 = ops.identity()(add_0)
        Y0 = ops.identity()(id_1)
        Y0._attrs["name"] = "output_0"
        Y0._attrs["is_output"] = True
        Y1 = ops.identity()(X0)
        Y1._attrs["name"] = "output_1"
        Y1._attrs["is_output"] = True
        Y2 = ops.elementwise(FuncEnum.ADD)(Y0, Y1)
        Y2._attrs["name"] = "output_2"
        Y2._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self._test_id}.so"
        module = compile_model(
            [Y0, Y1, Y2], target, "./tmp", test_name, dll_name=dll_name
        )
        self._test_id += 1

        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 3)
        self.assertEqual(sorted_ops[0]._attrs["op"], "identity")
        id_cnt = 0
        add_cnt = 0
        for sorted_op in sorted_ops:
            if sorted_op._attrs["op"] == "identity":
                id_cnt += 1
            elif sorted_op._attrs["op"] == "fused_elementwise":
                add_cnt += 1
        self.assertEqual(id_cnt, 1)
        self.assertEqual(add_cnt, 2)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M], dtype)
            x1_pt = get_random_torch_tensor([batch, M], dtype)
            add_0_pt = x0_pt + x1_pt
            id_1_pt = add_0_pt
            y0_pt = id_1_pt
            y1_pt = x0_pt
            y2_pt = y0_pt + y1_pt

            y0 = get_torch_empty_tensor(y0_pt.size(), dtype)
            y1 = get_torch_empty_tensor(y1_pt.size(), dtype)
            y2 = get_torch_empty_tensor(y2_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt}
            module.run_with_tensors(inputs, [y0, y1, y2])
            torch.testing.assert_close(y0_pt, y0, atol=0.01, rtol=0.01)
            torch.testing.assert_close(y1_pt, y1, atol=0.01, rtol=0.01)
            torch.testing.assert_close(y2_pt, y2, atol=0.01, rtol=0.01)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
