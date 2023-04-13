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
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils, shape_utils


class MoveViewOpsTestCase(unittest.TestCase):
    BATCH_SIZE = 1024

    def __init__(self, *args, **kwargs):
        super(MoveViewOpsTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _test_non_movable_reshape_cat(self, M0, M1, N, test_name, dtype="float16"):
        # make a graph like below:
        # concat_0 = concatenate(x0, x1, dim=1)
        # reshape_1 = reshape(concat_0)
        # y = concatenate(reshape_1, x2, dim=2)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        M2 = M0 + M1
        X2 = Tensor(
            shape=[batch_dim, M2, IntImm(N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim_1 = 1
        concat_0 = ops.concatenate()([X0, X1], dim=cat_dim_1)
        reshape_to_shape_1 = [-1, M2, N]
        reshape_1 = ops.reshape()(concat_0, reshape_to_shape_1)
        cat_dim_2 = 2
        Y = ops.concatenate()([reshape_1, X2], dim=cat_dim_2)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 5)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)
        self.assertEqual(sorted_ops[0]._attrs["op"], "concatenate")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N], dtype)
            concat_0_pt = torch.cat([x0_pt, x1_pt], dim=cat_dim_1)
            reshape_1_pt = torch.reshape(concat_0_pt, reshape_to_shape_1)
            y_pt = torch.cat([reshape_1_pt, x2_pt], dim=cat_dim_2)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)

    def test_non_movable_reshape_cat(self):
        self._test_non_movable_reshape_cat(
            M0=4,
            M1=2,
            N=4,
            test_name="test_non_movable_reshape_cat",
            dtype="float16",
        )

    def _test_move_reshape_cat_basic(
        self, M0, M1, M2, N, test_name, dtype="float16", non_movable=False
    ):
        # make a graph like below:
        # concat_0 = concatenate(x0, x1)
        # reshape_1 = reshape(concat_0)
        # y = concatenate(reshape_1, x2)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        if non_movable is True:
            assert (M0 + M1) % 3 == 0, "(M0 + M1) * N must be divisible by 3"
            X2_M = (M0 + M1) * N // 3
            X2_N = 3
            reshape_to_shape_1 = [-1, X2_M, X2_N]
        else:
            reshape_to_shape_1 = [-1, M0 + M1, N]
            X2_M = M2
            X2_N = N
        X2 = Tensor(
            shape=[batch_dim, IntImm(X2_M), IntImm(X2_N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim = 1
        concat_0 = ops.concatenate()([X0, X1], dim=cat_dim)
        reshape_1 = ops.reshape()(concat_0, reshape_to_shape_1)
        Y = ops.concatenate()([reshape_1, X2], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        if non_movable is True:
            expected_num_tensors = 5
            # reshape can be fused into the second cat
            expected_num_ops = 2
        else:
            expected_num_tensors = 4
            expected_num_ops = 1
        self.assertEqual(len(sorted_graph), expected_num_tensors)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_num_ops)
        self.assertEqual(sorted_ops[0]._attrs["op"], "concatenate")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, X2_M, X2_N], dtype)
            concat_0_pt = torch.cat([x0_pt, x1_pt], dim=cat_dim)
            reshape_1_pt = torch.reshape(concat_0_pt, reshape_to_shape_1)
            y_pt = torch.cat([reshape_1_pt, x2_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)

    def test_move_reshape_cat_basic(self):
        self._test_move_reshape_cat_basic(
            M0=4,
            M1=2,
            M2=6,
            N=4,
            test_name="test_move_reshape_cat_basic_non_movable",
            dtype="float16",
            non_movable=True,
        )
        self._test_move_reshape_cat_basic(
            M0=1,
            M1=5,
            M2=7,
            N=3,
            test_name="test_move_reshape_cat_basic",
            dtype="float16",
        )
        self._test_move_reshape_cat_basic(
            M0=2,
            M1=2,
            M2=6,
            N=8,
            test_name="test_move_reshape_cat_basic",
            dtype="float16",
        )

    def _test_move_reshape_cat_basic_2(self, M0, M1, M2, N, test_name, dtype="float16"):
        # make a graph like below:
        # reshape_0 = reshape(x0)
        # reshape_1 = reshape(x1)
        # concat_2 = concatenate(reshape_0, x3, reshape_1)
        # reshape_3 = reshape(concat_2)
        # y = concatenate(x2, reshape_3, x2)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        assert M0 % 2 == 0, f"{M0=} must be divisible by 2"
        assert N % 2 == 0, f"{N=} must be divisible by 2"
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 // 2), IntImm(N * 2)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1), IntImm(N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2), IntImm(N // 2)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N)],
            dtype=dtype,
            name="x3",
            is_input=True,
        )
        cat_dim = 1
        reshape_0 = ops.reshape()(X0, [-1, M0, N])
        reshape_1 = ops.reshape()(X1, [-1, M1, N])
        concat_2 = ops.concatenate()([reshape_0, X3, reshape_1], dim=cat_dim)
        reshape_to_shape_3 = [-1, (M0 + M0 + M1) * 2, N // 2]
        reshape_3 = ops.reshape()(concat_2, reshape_to_shape_3)
        Y = ops.concatenate()([X2, reshape_3, X2], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 5)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "concatenate")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 // 2, N * 2], dtype)
            x1_pt = get_random_torch_tensor([batch, M1, N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N // 2], dtype)
            x3_pt = get_random_torch_tensor([batch, M0, N], dtype)
            reshape_0_pt = torch.reshape(x0_pt, [-1, M0, N])
            reshape_1_pt = torch.reshape(x1_pt, [-1, M1, N])
            concat_2_pt = torch.cat([reshape_0_pt, x3_pt, reshape_1_pt], dim=cat_dim)
            reshape_3_pt = torch.reshape(concat_2_pt, reshape_to_shape_3)
            y_pt = torch.cat([x2_pt, reshape_3_pt, x2_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)

    def test_move_reshape_cat_basic_2(self):
        self._test_move_reshape_cat_basic_2(
            M0=2,
            M1=2,
            M2=6,
            N=8,
            test_name="test_move_reshape_cat_basic_2",
            dtype="float16",
        )

    def _test_move_reshape_cat_basic_3(self, M0, M2, N, test_name, dtype="float16"):
        # make a graph like below:
        # concat_0 = concatenate(x0, x0)
        # reshape_1 = reshape(concat_0)
        # y = concatenate(reshape_1, x2)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2), IntImm(N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim = 1
        concat_0 = ops.concatenate()([X0, X0], dim=cat_dim)
        reshape_to_shape_1 = [-1, M0 + M0, N]
        reshape_1 = ops.reshape()(concat_0, reshape_to_shape_1)
        Y = ops.concatenate()([reshape_1, X2], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "concatenate")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N], dtype)
            concat_0_pt = torch.cat([x0_pt, x0_pt], dim=cat_dim)
            reshape_1_pt = torch.reshape(concat_0_pt, reshape_to_shape_1)
            y_pt = torch.cat([reshape_1_pt, x2_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x2": x2_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)

    def test_move_reshape_cat_basic_3(self):
        self._test_move_reshape_cat_basic_3(
            M0=1,
            M2=7,
            N=3,
            test_name="test_move_reshape_cat_basic_3",
            dtype="float16",
        )

    def _test_move_reshape_cat_1(self, M0, M1, M2, N, test_name, dtype="float16"):
        # make a graph like below:
        # concat_0 = concatenate(x0, x1)
        # reshape_2 = reshape(concat_0)
        # concat_4 = concatenate(x2, reshape_2)
        # flatten_5 = flatten(concat_4)
        # concat_6 = concatenate(x0, flatten_5)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2), IntImm(N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim = 1
        concat_0 = ops.concatenate()([X0, X1], dim=cat_dim)
        reshape_2 = ops.reshape()(concat_0, [-1, M0 + M1, N])
        concat_4 = ops.concatenate()([X2, reshape_2], dim=cat_dim)
        flatten_5 = ops.flatten(start_dim=1, end_dim=-1)(concat_4)
        Y = ops.concatenate()([X0, flatten_5], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "concatenate")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N], dtype)
            concat_0_pt = torch.cat([x0_pt, x1_pt], dim=cat_dim)
            reshape_2_pt = torch.reshape(concat_0_pt, [-1, M0 + M1, N])
            concat_4_pt = torch.cat([x2_pt, reshape_2_pt], dim=cat_dim)
            flatten_5_pt = torch.flatten(concat_4_pt, 1, -1)
            y_pt = torch.cat([x0_pt, flatten_5_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_move_reshape_cat_1(self):
        self._test_move_reshape_cat_1(
            M0=2,
            M1=2,
            M2=6,
            N=8,
            test_name="test_move_reshape_cat_1",
            dtype="float16",
        )

    def _test_move_reshape_cat_2(self, M0, M1, M2, M3, N, test_name, dtype="float16"):
        # make a graph like below:
        # concat_0 = concatenate(x0, x1)
        # concat_1 = concatenate(x0, x1)
        # reshape_2 = reshape(concat_0)
        # reshape_3 = reshape(concat_1)
        # concat_4 = concatenate(x2, reshape_2, reshape_3, x3, reshape_2)
        # flatten_5 = flatten(concat_4)
        # concat_6 = concatenate(x0, flatten_5, x1, flatten_5)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2), IntImm(N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[batch_dim, IntImm(M3), IntImm(N)],
            dtype=dtype,
            name="x3",
            is_input=True,
        )
        cat_dim = 1
        concat_0 = ops.concatenate()([X0, X1], dim=cat_dim)
        concat_1 = ops.concatenate()([X0, X1], dim=cat_dim)
        reshape_2 = ops.reshape()(concat_0, [-1, M0 + M1, N])
        reshape_3 = ops.reshape()(concat_1, [-1, M0 + M1, N])
        concat_4 = ops.concatenate()(
            [X2, reshape_2, reshape_3, X3, reshape_2], dim=cat_dim
        )
        flatten_5 = ops.flatten(start_dim=1, end_dim=-1)(concat_4)
        Y = ops.concatenate()([X0, flatten_5, X1, flatten_5], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "concatenate")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N], dtype)
            x3_pt = get_random_torch_tensor([batch, M3, N], dtype)
            concat_0_pt = torch.cat([x0_pt, x1_pt], dim=cat_dim)
            concat_1_pt = torch.cat([x0_pt, x1_pt], dim=cat_dim)
            reshape_2_pt = torch.reshape(concat_0_pt, [-1, M0 + M1, N])
            reshape_3_pt = torch.reshape(concat_1_pt, [-1, M0 + M1, N])
            concat_4_pt = torch.cat(
                [x2_pt, reshape_2_pt, reshape_3_pt, x3_pt, reshape_2_pt], dim=cat_dim
            )
            flatten_5_pt = torch.flatten(concat_4_pt, 1, -1)
            y_pt = torch.cat([x0_pt, flatten_5_pt, x1_pt, flatten_5_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_move_reshape_cat_2(self):
        self._test_move_reshape_cat_2(
            M0=2,
            M1=2,
            M2=6,
            M3=4,
            N=8,
            test_name="test_move_reshape_cat_2",
            dtype="float16",
        )

    def _test_move_reshape_cat_3(self, M0, M1, M2, N, test_name, dtype="float16"):
        # make a graph like below:
        # concat_0 = concatenate(x0, x1)
        # reshape_1 = reshape(concat_0)
        # reshape_2 = reshape(x2)
        # y = concatenate(reshape_2, reshape_1, reshape_2)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1), IntImm(N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2), IntImm(N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim = 1
        concat_0 = ops.concatenate()([X0, X1], dim=cat_dim)
        reshape_1 = ops.reshape()(concat_0, [-1, (M0 + M1) * N])
        reshape_2 = ops.reshape()(X2, [-1, M2 * N])
        Y = ops.concatenate()([reshape_2, reshape_1, reshape_2], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            if sorted_op._attrs["op"] == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1, N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N], dtype)

            concat_0_pt = torch.cat([x0_pt, x1_pt], dim=cat_dim)
            reshape_1_pt = torch.reshape(concat_0_pt, [-1, (M0 + M1) * N])
            reshape_2_pt = torch.reshape(x2_pt, [-1, M2 * N])
            y_pt = torch.cat([reshape_2_pt, reshape_1_pt, reshape_2_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_move_reshape_cat_3(self):
        self._test_move_reshape_cat_3(
            M0=4,
            M1=6,
            M2=3,
            N=4,
            test_name="test_move_reshape_cat_3",
            dtype="float16",
        )

    def _test_move_strided_reshape_cat(
        self, M0, M1, M2, M3, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # add_0 = add(x0, x1)
        # concat_1 = concatenate(add_0, x2)
        # reshape_2 = reshape(concat_1)
        # y = concatenate(reshape_2, x3)
        assert M0 == M1, f"expected {M0=} to be equal to {M1=}"
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[batch_dim, IntImm(M3), IntImm(N)],
            dtype=dtype,
            name="x3",
            is_input=True,
        )
        cat_dim = 1
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        concat_1 = ops.concatenate()([add_0, X2], dim=cat_dim)
        reshape_2 = ops.reshape()(concat_1, [-1, M0 + M2, N])
        Y = ops.concatenate()([reshape_2, X3], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            if sorted_op._attrs["op"] == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x3_pt = get_random_torch_tensor([batch, M3, N], dtype)
            add_0_pt = x0_pt + x1_pt
            concat_1_pt = torch.cat([add_0_pt, x2_pt], dim=cat_dim)
            reshape_2_pt = torch.reshape(concat_1_pt, [-1, M0 + M2, N])
            y_pt = torch.cat([reshape_2_pt, x3_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_move_strided_reshape_cat(self):
        self._test_move_strided_reshape_cat(
            M0=4,
            M1=4,
            M2=6,
            M3=3,
            N=8,
            test_name="test_move_strided_reshape_cat",
            dtype="float16",
        )
        self._test_move_strided_reshape_cat(
            M0=4,
            M1=4,
            M2=5,
            M3=10,
            N=7,
            test_name="test_move_strided_reshape_cat",
            dtype="float16",
        )

    def _test_move_strided_reshape_cat_2(
        self, M0, M1, M2, M3, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # add_0 = add(x0, x0)
        # reshape_1 = reshape(add_0)
        # add_2 = add(x1, x1)
        # concat_3 = concatenate(x2, reshape_1, x2, add_2)
        # reshape_4 = reshape(concat_3)
        # y = concatenate(x3, reshape_4, x3)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[batch_dim, IntImm(M3), IntImm(N)],
            dtype=dtype,
            name="x3",
            is_input=True,
        )

        cat_dim = 1
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X0)
        reshape_1 = ops.reshape()(add_0, [-1, M0 * N])
        add_2 = ops.elementwise(FuncEnum.ADD)(X1, X1)
        concat_3 = ops.concatenate()([X2, reshape_1, X2, add_2], dim=cat_dim)
        reshape_to_shape_4 = (
            sum([t.shape()[cat_dim].value() for t in [X2, reshape_1, X2, add_2]]) // N
        )
        reshape_4 = ops.reshape()(concat_3, [-1, reshape_to_shape_4, N])
        Y = ops.concatenate()([X3, reshape_4, X3], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(Y, target, "./tmp", test_name)
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 3)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            if sorted_op._attrs["op"] == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)
        output_tensors = {op._attrs["outputs"][0] for op in sorted_ops}
        self.assertEqual(len(output_tensors), 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x3_pt = get_random_torch_tensor([batch, M3, N], dtype)
            add_0_pt = x0_pt + x0_pt
            reshape_1_pt = torch.reshape(add_0_pt, [batch, M0 * N])
            add_2_pt = x1_pt + x1_pt
            concat_3_pt = torch.cat([x2_pt, reshape_1_pt, x2_pt, add_2_pt], dim=cat_dim)
            reshape_4_pt = torch.reshape(concat_3_pt, [-1, reshape_to_shape_4, N])
            y_pt = torch.cat([x3_pt, reshape_4_pt, x3_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_move_strided_reshape_cat_2(self):
        self._test_move_strided_reshape_cat_2(
            M0=4,
            M1=6,
            M2=9,
            M3=16,
            N=8,
            test_name="test_move_strided_reshape_cat_2",
            dtype="float16",
        )

    def _test_move_strided_reshape_cat_3(
        self, M0, M1, M2, M3, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # slice_0 = slice(x4)
        # slice_1 = slice(x4)
        # slice_2 = slice(x4)
        # add_0 = add(x0, x0)
        # reshape_1 = reshape(add_0)
        # add_2 = add(x1, x1)
        # flatten_3 = flatten(add_2)
        # concat_4 = concatenate(x2, slice_0, slice_1, reshape_1, slice_2, flatten_3) # 2d
        # add_5 = add(x3, x3)
        # reshape_6 = reshape(add_5)
        # reshape_7 = reshape(concat_4)
        # concat_8 = concatenate(x0, reshape_7, reshape_6) # 3d
        # add_9 = add(x0, x0)
        # flatten_10 = flatten(concat_8) # 2d
        # reshape_11 = reshape(add_9) # 2d
        # y = concatenate(x1, reshape_11, flatten_10, x2) # 2d
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[batch_dim, IntImm(M3), IntImm(N)],
            dtype=dtype,
            name="x3",
            is_input=True,
        )
        M4 = 10 * M0
        X4 = Tensor(
            shape=[batch_dim, IntImm(M4 * N)],
            dtype=dtype,
            name="x4",
            is_input=True,
        )

        slice_start_indices_0 = [None, 0]
        slice_end_indices_0 = [None, N]
        slice_start_indices_1 = [None, 3 * N]
        slice_end_indices_1 = [None, 4 * N]
        slice_start_indices_2 = [None, 4 * N]
        slice_end_indices_2 = [None, 8 * N]
        slice_0 = ops.dynamic_slice()(X4, slice_start_indices_0, slice_end_indices_0)
        slice_1 = ops.dynamic_slice()(X4, slice_start_indices_1, slice_end_indices_1)
        slice_2 = ops.dynamic_slice()(X4, slice_start_indices_2, slice_end_indices_2)
        cat_dim = 1
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X0)
        reshape_1 = ops.reshape()(add_0, [-1, M0 * N])
        add_2 = ops.elementwise(FuncEnum.ADD)(X1, X1)
        flatten_3 = ops.flatten(start_dim=1, end_dim=-1)(add_2)
        concat_4 = ops.concatenate()(
            [X2, slice_0, slice_1, reshape_1, slice_2, flatten_3], dim=cat_dim
        )
        add_5 = ops.elementwise(FuncEnum.ADD)(X3, X3)
        reshape_6 = ops.reshape()(add_5, [-1, M3, N])
        reshape_to_shape_7 = (
            sum(
                [
                    t.shape()[cat_dim].value()
                    for t in [X2, slice_0, slice_1, reshape_1, slice_2, flatten_3]
                ]
            )
            // N
        )
        reshape_7 = ops.reshape()(concat_4, [-1, reshape_to_shape_7, N])
        concat_8 = ops.concatenate()([X0, reshape_7, reshape_6], dim=cat_dim)
        add_9 = ops.elementwise(FuncEnum.ADD)(X0, X0)
        flatten_10 = ops.flatten(start_dim=1, end_dim=-1)(concat_8)
        reshape_11 = ops.reshape()(add_9, [-1, M0 * N])
        Y = ops.concatenate()([X1, reshape_11, flatten_10, X2], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(Y, target, "./tmp", test_name)
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 5)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            if sorted_op._attrs["op"] == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)
        output_tensors = {op._attrs["outputs"][0] for op in sorted_ops}
        self.assertEqual(len(output_tensors), 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x3_pt = get_random_torch_tensor([batch, M3, N], dtype)
            x4_pt = get_random_torch_tensor([batch, M4 * N], dtype)

            slice_indices_0 = [
                slice(i, j) for i, j in zip(slice_start_indices_0, slice_end_indices_0)
            ]
            slice_indices_1 = [
                slice(i, j) for i, j in zip(slice_start_indices_1, slice_end_indices_1)
            ]
            slice_indices_2 = [
                slice(i, j) for i, j in zip(slice_start_indices_2, slice_end_indices_2)
            ]
            slice_0_pt = x4_pt[slice_indices_0]
            slice_1_pt = x4_pt[slice_indices_1]
            slice_2_pt = x4_pt[slice_indices_2]

            add_0_pt = x0_pt + x0_pt
            reshape_1_pt = torch.reshape(add_0_pt, [batch, M0 * N])
            add_2_pt = x1_pt + x1_pt
            flatten_3_pt = torch.flatten(add_2_pt, 1, -1)
            concat_4_pt = torch.cat(
                [x2_pt, slice_0_pt, slice_1_pt, reshape_1_pt, slice_2_pt, flatten_3_pt],
                dim=cat_dim,
            )
            add_5_pt = x3_pt + x3_pt
            reshape_6_pt = torch.reshape(add_5_pt, [-1, M3, N])
            reshape_7_pt = torch.reshape(concat_4_pt, [-1, reshape_to_shape_7, N])
            concat_8_pt = torch.cat([x0_pt, reshape_7_pt, reshape_6_pt], dim=cat_dim)
            add_9_pt = x0_pt + x0_pt
            flatten_10_pt = torch.flatten(concat_8_pt, 1, -1)
            reshape_11_pt = torch.reshape(add_9_pt, [-1, M0 * N])
            y_pt = torch.cat([x1_pt, reshape_11_pt, flatten_10_pt, x2_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt, "x4": x4_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_move_strided_reshape_cat_3(self):
        self._test_move_strided_reshape_cat_3(
            M0=4,
            M1=6,
            M2=9,
            M3=16,
            N=8,
            test_name="test_move_strided_reshape_cat_3",
            dtype="float16",
        )

    def _test_move_strided_reshape_cat_4(self, M0, M2, N, test_name, dtype="float16"):
        # make a graph like below:
        # slice_0 = slice(x4)
        # concat_4 = concatenate(x2, slice_0) # 2d
        # reshape_7 = reshape(concat_4)
        # y = concatenate(x0, reshape_7) # 3d
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        M4 = 10 * M0
        X4 = Tensor(
            shape=[batch_dim, IntImm(M4 * N)],
            dtype=dtype,
            name="x4",
            is_input=True,
        )

        slice_start_indices_0 = [None, 0]
        slice_end_indices_0 = [None, N]
        slice_0 = ops.dynamic_slice()(X4, slice_start_indices_0, slice_end_indices_0)
        cat_dim = 1
        concat_4 = ops.concatenate()([X2, slice_0], dim=cat_dim)
        reshape_to_shape_7 = (
            sum([t.shape()[cat_dim].value() for t in [X2, slice_0]]) // N
        )
        reshape_7 = ops.reshape()(concat_4, [-1, reshape_to_shape_7, N])
        Y = ops.concatenate()([X0, reshape_7], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(Y, target, "./tmp", test_name)
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            if sorted_op._attrs["op"] == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)
        output_tensors = {op._attrs["outputs"][0] for op in sorted_ops}
        self.assertEqual(len(output_tensors), 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x4_pt = get_random_torch_tensor([batch, M4 * N], dtype)

            slice_indices_0 = [
                slice(i, j) for i, j in zip(slice_start_indices_0, slice_end_indices_0)
            ]
            slice_0_pt = x4_pt[slice_indices_0]

            concat_4_pt = torch.cat([x2_pt, slice_0_pt], dim=cat_dim)
            reshape_7_pt = torch.reshape(concat_4_pt, [-1, reshape_to_shape_7, N])
            y_pt = torch.cat([x0_pt, reshape_7_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x2": x2_pt, "x4": x4_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_move_strided_reshape_cat_4(self):
        self._test_move_strided_reshape_cat_4(
            M0=4,
            M2=9,
            N=8,
            test_name="test_move_strided_reshape_cat_4",
            dtype="float16",
        )

    def _test_move_strided_reshape_cat_5(self, M0, M2, N, test_name, dtype="float16"):
        # make a graph like below:
        # slice_0 = slice(x4)
        # concat_4 = concatenate(x2, slice_0) # 2d
        # reshape_7 = reshape(concat_4)
        # concat_8 = concatenate(x0, reshape_7) # 3d
        # flatten_10 = reshape(concat_8) # 2d
        # y = concatenate(flatten_10, x2) # 2d
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        M4 = 10 * M0
        X4 = Tensor(
            shape=[batch_dim, IntImm(M4 * N)],
            dtype=dtype,
            name="x4",
            is_input=True,
        )

        slice_start_indices_0 = [None, 0]
        slice_end_indices_0 = [None, N]
        slice_0 = ops.dynamic_slice()(X4, slice_start_indices_0, slice_end_indices_0)
        cat_dim = 1
        concat_4 = ops.concatenate()([X2, slice_0], dim=cat_dim)
        reshape_to_shape_7 = (
            sum([t.shape()[cat_dim].value() for t in [X2, slice_0]]) // N
        )
        reshape_7 = ops.reshape()(concat_4, [-1, reshape_to_shape_7, N])
        concat_8 = ops.concatenate()([X0, reshape_7], dim=cat_dim)
        flatten_10 = ops.flatten(start_dim=1, end_dim=-1)(concat_8)
        Y = ops.concatenate()([flatten_10, X2], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(Y, target, "./tmp", test_name)
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            if sorted_op._attrs["op"] == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)
        output_tensors = {op._attrs["outputs"][0] for op in sorted_ops}
        self.assertEqual(len(output_tensors), 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x4_pt = get_random_torch_tensor([batch, M4 * N], dtype)

            slice_indices_0 = [
                slice(i, j) for i, j in zip(slice_start_indices_0, slice_end_indices_0)
            ]
            slice_0_pt = x4_pt[slice_indices_0]

            concat_4_pt = torch.cat([x2_pt, slice_0_pt], dim=cat_dim)
            reshape_7_pt = torch.reshape(concat_4_pt, [-1, reshape_to_shape_7, N])
            concat_8_pt = torch.cat([x0_pt, reshape_7_pt], dim=cat_dim)
            flatten_10_pt = torch.flatten(concat_8_pt, 1, -1)
            y_pt = torch.cat([flatten_10_pt, x2_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x2": x2_pt, "x4": x4_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_move_strided_reshape_cat_5(self):
        self._test_move_strided_reshape_cat_5(
            M0=4,
            M2=9,
            N=8,
            test_name="test_move_strided_reshape_cat_5",
            dtype="float16",
        )

    def _test_move_strided_reshape_cat_6(self, M0, M2, N, test_name, dtype="float16"):
        # make a graph like below:
        # add_0 = add(x4, x4)
        # concat_4 = concatenate(x2, add_0) # 2d
        # reshape_7 = reshape(concat_4)
        # concat_8 = concatenate(x0, reshape_7) # 3d
        # flatten_10 = reshape(concat_8) # 2d
        # y = concatenate(flatten_10, x2) # 2d
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        M4 = 10 * M0
        X4 = Tensor(
            shape=[batch_dim, IntImm(M4 * N)],
            dtype=dtype,
            name="x4",
            is_input=True,
        )

        add_0 = ops.elementwise(FuncEnum.ADD)(X4, X4)
        cat_dim = 1
        concat_4 = ops.concatenate()([X2, add_0], dim=cat_dim)
        reshape_to_shape_7 = sum([t.shape()[cat_dim].value() for t in [X2, add_0]]) // N
        reshape_7 = ops.reshape()(concat_4, [-1, reshape_to_shape_7, N])
        concat_8 = ops.concatenate()([X0, reshape_7], dim=cat_dim)
        flatten_10 = ops.flatten(start_dim=1, end_dim=-1)(concat_8)
        Y = ops.concatenate()([flatten_10, X2], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(Y, target, "./tmp", test_name)
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            if sorted_op._attrs["op"] == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)
        output_tensors = {op._attrs["outputs"][0] for op in sorted_ops}
        self.assertEqual(len(output_tensors), 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x4_pt = get_random_torch_tensor([batch, M4 * N], dtype)

            add_0_pt = x4_pt + x4_pt
            concat_4_pt = torch.cat([x2_pt, add_0_pt], dim=cat_dim)
            reshape_7_pt = torch.reshape(concat_4_pt, [-1, reshape_to_shape_7, N])
            concat_8_pt = torch.cat([x0_pt, reshape_7_pt], dim=cat_dim)
            flatten_10_pt = torch.flatten(concat_8_pt, 1, -1)
            y_pt = torch.cat([flatten_10_pt, x2_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x2": x2_pt, "x4": x4_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_move_strided_reshape_cat_6(self):
        self._test_move_strided_reshape_cat_6(
            M0=4,
            M2=9,
            N=8,
            test_name="test_move_strided_reshape_cat_6",
            dtype="float16",
        )

    def _test_move_strided_reshape_cat_7(
        self, M0, M1, M2, M3, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # add_0 = add(x0, x1)
        # concat_1 = concatenate(add_0, x2)
        # reshape_2 = reshape(concat_1)
        # add_3 = add(x4, reshape_2)
        # concat_4 = concatenate(x3, reshape_2, x3)
        # reduce_5 = reduce_sum(add_3)
        # reduce_6 = reduce_sum(concat_5)
        # y = add(reduce_5, reduce_6)
        assert M0 == M1, f"expected {M0=} to be equal to {M1=}"
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[batch_dim, IntImm(M3), IntImm(N)],
            dtype=dtype,
            name="x3",
            is_input=True,
        )
        M4 = M0 + M2
        X4 = Tensor(
            shape=[batch_dim, IntImm(M0 + M2), IntImm(N)],
            dtype=dtype,
            name="x4",
            is_input=True,
        )
        cat_dim = 1
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        concat_1 = ops.concatenate()([add_0, X2], dim=cat_dim)
        reshape_2 = ops.reshape()(concat_1, [-1, M0 + M2, N])
        add_3 = ops.elementwise(FuncEnum.ADD)(X4, reshape_2)
        concat_4 = ops.concatenate()([X3, reshape_2, X3], dim=cat_dim)
        reduce_dim = cat_dim
        reduce_5 = ops.reduce_sum(reduce_dim)(add_3)
        reduce_6 = ops.reduce_sum(reduce_dim)(concat_4)
        Y = ops.elementwise(FuncEnum.ADD)(reduce_5, reduce_6)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 6)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            if sorted_op._attrs["op"] == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x3_pt = get_random_torch_tensor([batch, M3, N], dtype)
            x4_pt = get_random_torch_tensor([batch, M4, N], dtype)
            add_0_pt = x0_pt + x1_pt
            concat_1_pt = torch.cat([add_0_pt, x2_pt], dim=cat_dim)
            reshape_2_pt = torch.reshape(concat_1_pt, [-1, M0 + M2, N])
            add_3_pt = x4_pt + reshape_2_pt
            concat_4_pt = torch.cat([x3_pt, reshape_2_pt, x3_pt], dim=cat_dim)
            reduce_5_pt = torch.sum(add_3_pt, reduce_dim)
            reduce_6_pt = torch.sum(concat_4_pt, reduce_dim)
            y_pt = reduce_5_pt + reduce_6_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt, "x4": x4_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_move_strided_reshape_cat_7(self):
        self._test_move_strided_reshape_cat_7(
            M0=4,
            M1=4,
            M2=6,
            M3=3,
            N=8,
            test_name="test_move_strided_reshape_cat_7",
            dtype="float16",
        )
        self._test_move_strided_reshape_cat_7(
            M0=4,
            M1=4,
            M2=5,
            M3=3,
            N=7,
            test_name="test_move_strided_reshape_cat_7",
            dtype="float16",
        )

    def _test_move_strided_reshape_cat_8(
        self, M0, M1, M2, M3, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # add_0 = add(x0, x1)  # 2d
        # concat_1 = concatenate(add_0, x2) # 2d
        # reshape_2 = reshape(concat_1) # 3d
        # bmm_crr_add_3 = bmm_crr_add(reshape_2, x4, x5) # 3d
        # concat_4 = concatenate(x3, reshape_2, x3) # 3d
        # reshape_5 = reshape(concat_4) # 2d
        # add_6 = add(reshape_5, x6) # 2d
        # concat_7 = concatenate(x0, reshape_5, x0)
        # reshape_8 = reshape(bmm_crr_add_3) # 2d
        # reduce_9 = reduce_sum(reshape_8)
        # reduce_10 = reduce_sum(add_6)
        # reduce_11 = reduce_sum(concat_7)
        # add_12 = add(reduce_9, reduce_10)
        # y = add(add_12, reduce_11)
        assert M0 == M1, f"expected {M0=} to be equal to {M1=}"
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[batch_dim, IntImm(M3), IntImm(N)],
            dtype=dtype,
            name="x3",
            is_input=True,
        )
        M4 = M0 + M2
        X4 = Tensor(
            shape=[IntImm(M4), IntImm(N)],
            dtype=dtype,
            name="x4",
            is_input=True,
        )
        X5 = Tensor(
            shape=[IntImm(N)],
            dtype=dtype,
            name="x5",
            is_input=True,
        )
        cat_dim = 1
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        concat_1 = ops.concatenate()([add_0, X2], dim=cat_dim)
        bmm_K = M0 + M2
        reshape_2 = ops.reshape()(concat_1, [-1, bmm_K, N])
        # bmm_crr_add_3[batch, N, N] = bmm_crr_add(
        #     reshape_2[batch, bmm_K, N], X4[bmm_K, N], X5[N]
        # )
        bmm_crr_add_3 = ops.bmm_crr_add()(reshape_2, X4, X5)
        concat_4 = ops.concatenate()([X3, reshape_2, X3], dim=cat_dim)  # 3d
        reshape_to_shape_5 = (
            sum([t.shape()[cat_dim].value() for t in [X3, reshape_2, X3]]) * N
        )
        reshape_5 = ops.reshape()(concat_4, [-1, reshape_to_shape_5])  # 2d
        X6 = Tensor(
            shape=[batch_dim, IntImm(reshape_to_shape_5)],
            dtype=dtype,
            name="x6",
            is_input=True,
        )
        add_6 = ops.elementwise(FuncEnum.ADD)(reshape_5, X6)
        concat_7 = ops.concatenate()([X0, reshape_5, X0], dim=cat_dim)  # 2d
        reshape_8 = ops.reshape()(bmm_crr_add_3, [-1, N * N])  # 2d
        reduce_dim = cat_dim
        reduce_9 = ops.reduce_sum(reduce_dim)(reshape_8)
        reduce_10 = ops.reduce_sum(reduce_dim)(add_6)
        reduce_11 = ops.reduce_sum(reduce_dim)(concat_7)
        add_12 = ops.elementwise(FuncEnum.ADD)(reduce_9, reduce_10)
        Y = ops.elementwise(FuncEnum.ADD)(add_12, reduce_11)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        # dynamic_slice + bmm cannot be fused because we can't generate
        # any valid strided access
        self.assertEqual(len(sorted_ops), 9)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            op_type = sorted_op._attrs["op"]
            if op_type == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x3_pt = get_random_torch_tensor([batch, M3, N], dtype)
            x4_pt = get_random_torch_tensor([M4, N], dtype)
            x5_pt = get_random_torch_tensor([N], dtype)
            x6_pt = get_random_torch_tensor([batch, reshape_to_shape_5], dtype)
            add_0_pt = x0_pt + x1_pt
            concat_1_pt = torch.cat([add_0_pt, x2_pt], dim=cat_dim)
            reshape_2_pt = torch.reshape(concat_1_pt, [-1, bmm_K, N])
            reshape_2_trans_pt = torch.transpose(reshape_2_pt, -2, -1)
            bmm_crr_add_3_pt = torch.matmul(reshape_2_trans_pt, x4_pt) + x5_pt
            concat_4_pt = torch.cat([x3_pt, reshape_2_pt, x3_pt], dim=cat_dim)
            reshape_5_pt = torch.reshape(concat_4_pt, [-1, reshape_to_shape_5])
            add_6_pt = reshape_5_pt + x6_pt
            concat_7_pt = torch.cat([x0_pt, reshape_5_pt, x0_pt], dim=cat_dim)
            reshape_8_pt = torch.reshape(bmm_crr_add_3_pt, [-1, N * N])
            reduce_9_pt = torch.sum(reshape_8_pt, reduce_dim)
            reduce_10_pt = torch.sum(add_6_pt, reduce_dim)
            reduce_11_pt = torch.sum(concat_7_pt, reduce_dim)
            add_12_pt = reduce_9_pt + reduce_10_pt
            y_pt = add_12_pt + reduce_11_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
                "x2": x2_pt,
                "x3": x3_pt,
                "x4": x4_pt,
                "x5": x5_pt,
                "x6": x6_pt,
            }
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_move_strided_reshape_cat_8(self):
        self._test_move_strided_reshape_cat_8(
            M0=4,
            M1=4,
            M2=6,
            M3=4,
            N=4,
            test_name="test_move_strided_reshape_cat_8",
            dtype="float16",
        )
        return
        self._test_move_strided_reshape_cat_8(
            M0=4,
            M1=4,
            M2=6,
            M3=3,
            N=4,
            test_name="test_move_strided_reshape_cat_8",
            dtype="float16",
        )

    def _test_move_strided_reshape_cat_9(
        self, M0, M1, M2, M3, M7, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # add_0 = add(x0, x1)  # 2d
        # concat_1 = concatenate(add_0, x2) # 2d
        # reshape_2 = reshape(concat_1) # 3d
        # bmm_crr_add_3 = bmm_crr_add(reshape_2, x4, x5) # 3d
        # concat_4 = concatenate(x3, concat_1, x3) # 2d
        # reshape_5 = reshape(concat_4) # 3d
        # add_6 = add(reshape_5, x6) # 3d
        # concat_7 = concatenate(x7, reshape_5, x7) # 3d
        # reduce_8 = reduce_sum(bmm_crr_add_3)
        # reduce_9 = reduce_sum(add_6)
        # reduce_10 = reduce_sum(concat_7)
        # add_11 = add(reduce_8, reduce_9)
        # y = add(add_11, reduce_10)
        assert M0 == M1, f"expected {M0=} to be equal to {M1=}"
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[batch_dim, IntImm(M3 * N)],
            dtype=dtype,
            name="x3",
            is_input=True,
        )
        M4 = M0 + M2
        X4 = Tensor(
            shape=[IntImm(M4), IntImm(N)],
            dtype=dtype,
            name="x4",
            is_input=True,
        )
        X5 = Tensor(
            shape=[IntImm(N)],
            dtype=dtype,
            name="x5",
            is_input=True,
        )
        cat_dim = 1
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)  # 2d
        concat_1 = ops.concatenate()([add_0, X2], dim=cat_dim)  # 2d
        bmm_K = M0 + M2
        reshape_2 = ops.reshape()(concat_1, [-1, bmm_K, N])
        # bmm_crr_add_3[batch, N, N] = bmm_crr_add(
        #     reshape_2[batch, bmm_K, N], X4[bmm_K, N], X5[N]
        # )
        bmm_crr_add_3 = ops.bmm_crr_add()(reshape_2, X4, X5)
        concat_4 = ops.concatenate()([X3, concat_1, X3], dim=cat_dim)  # 2d
        M6 = sum([t.shape()[cat_dim].value() for t in [X3, concat_1, X3]])
        assert M6 % N == 0, f"expected {M6=} is divisible by {N=}"
        M6 = M6 // N
        reshape_5 = ops.reshape()(concat_4, [-1, M6, N])  # 3d
        X6 = Tensor(
            shape=[batch_dim, IntImm(M6), IntImm(N)],
            dtype=dtype,
            name="x6",
            is_input=True,
        )
        add_6 = ops.elementwise(FuncEnum.ADD)(reshape_5, X6)
        X7 = Tensor(
            shape=[batch_dim, IntImm(M7), IntImm(N)],
            dtype=dtype,
            name="x7",
            is_input=True,
        )
        concat_7 = ops.concatenate()([X7, reshape_5, X7], dim=cat_dim)  # 3d
        reduce_dim = cat_dim
        reduce_8 = ops.reduce_sum(reduce_dim)(bmm_crr_add_3)
        reduce_9 = ops.reduce_sum(reduce_dim)(add_6)
        reduce_10 = ops.reduce_sum(reduce_dim)(concat_7)
        add_11 = ops.elementwise(FuncEnum.ADD)(reduce_8, reduce_9)
        Y = ops.elementwise(FuncEnum.ADD)(add_11, reduce_10)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 8)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            op_type = sorted_op._attrs["op"]
            # dynamic_slice is fused into add
            self.assertTrue(op_type != "dynamic_slice")
            if op_type == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x3_pt = get_random_torch_tensor([batch, M3 * N], dtype)
            x4_pt = get_random_torch_tensor([M4, N], dtype)
            x5_pt = get_random_torch_tensor([N], dtype)
            x6_pt = get_random_torch_tensor([batch, M6, N], dtype)
            x7_pt = get_random_torch_tensor([batch, M7, N], dtype)

            add_0_pt = x0_pt + x1_pt
            concat_1_pt = torch.cat([add_0_pt, x2_pt], dim=cat_dim)
            reshape_2_pt = torch.reshape(concat_1_pt, [-1, bmm_K, N])
            reshape_2_trans_pt = torch.transpose(reshape_2_pt, -2, -1)
            bmm_crr_add_3_pt = torch.matmul(reshape_2_trans_pt, x4_pt) + x5_pt
            concat_4_pt = torch.cat([x3_pt, concat_1_pt, x3_pt], dim=cat_dim)
            reshape_5_pt = torch.reshape(concat_4_pt, [-1, M6, N])
            add_6_pt = reshape_5_pt + x6_pt
            concat_7_pt = torch.cat([x7_pt, reshape_5_pt, x7_pt], dim=cat_dim)
            reduce_8_pt = torch.sum(bmm_crr_add_3_pt, reduce_dim)
            reduce_9_pt = torch.sum(add_6_pt, reduce_dim)
            reduce_10_pt = torch.sum(concat_7_pt, reduce_dim)
            add_11_pt = reduce_8_pt + reduce_9_pt
            y_pt = add_11_pt + reduce_10_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
                "x2": x2_pt,
                "x3": x3_pt,
                "x4": x4_pt,
                "x5": x5_pt,
                "x6": x6_pt,
                "x7": x7_pt,
            }
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_move_strided_reshape_cat_9(self):
        self._test_move_strided_reshape_cat_9(
            M0=4,
            M1=4,
            M2=6,
            M3=4,
            M7=8,
            N=4,
            test_name="test_move_strided_reshape_cat_9",
            dtype="float16",
        )

    def _test_move_strided_reshape_cat_multi_dsts(
        self, M0, M1, M2, M3, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # add_0 = add(x0, x1)  # 2d
        # concat_1 = concatenate(add_0, x2) # 2d
        # reshape_2 = reshape(concat_1) # 3d
        # bmm_crr_add_3 = bmm_crr_add(reshape_2, x4, x5) # 3d
        # reshape_4 = reshape(concat_1) # 3d
        # concat_5 = concatenate(x3, reshape_4, x3) # 3d
        # reduce_8 = reduce_sum(bmm_crr_add_3)
        # reduce_9 = reduce_sum(concat_5)
        # y = add(reduce_8, reduce_9)
        assert M0 == M1, f"expected {M0=} to be equal to {M1=}"
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1 * N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[batch_dim, IntImm(M3), IntImm(N)],
            dtype=dtype,
            name="x3",
            is_input=True,
        )
        M4 = M0 + M2
        X4 = Tensor(
            shape=[IntImm(M4), IntImm(N)],
            dtype=dtype,
            name="x4",
            is_input=True,
        )
        X5 = Tensor(
            shape=[IntImm(N)],
            dtype=dtype,
            name="x5",
            is_input=True,
        )
        cat_dim = 1
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)  # 2d
        concat_1 = ops.concatenate()([add_0, X2], dim=cat_dim)  # 2d
        bmm_K = M0 + M2
        reshape_2 = ops.reshape()(concat_1, [-1, bmm_K, N])
        # bmm_crr_add_3[batch, N, N] = bmm_crr_add(
        #     reshape_2[batch, bmm_K, N], X4[bmm_K, N], X5[N]
        # )
        bmm_crr_add_3 = ops.bmm_crr_add()(reshape_2, X4, X5)
        reshape_to_shape_4 = M0 + M2
        reshape_4 = ops.reshape()(concat_1, [-1, reshape_to_shape_4, N])  # 3d
        concat_5 = ops.concatenate()([X3, reshape_4, X3], dim=cat_dim)  # 2d
        reduce_dim = cat_dim
        reduce_8 = ops.reduce_sum(reduce_dim)(bmm_crr_add_3)
        reduce_9 = ops.reduce_sum(reduce_dim)(concat_5)
        Y = ops.elementwise(FuncEnum.ADD)(reduce_8, reduce_9)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 6)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            if sorted_op._attrs["op"] == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x3_pt = get_random_torch_tensor([batch, M3, N], dtype)
            x4_pt = get_random_torch_tensor([M4, N], dtype)
            x5_pt = get_random_torch_tensor([N], dtype)

            add_0_pt = x0_pt + x1_pt
            concat_1_pt = torch.cat([add_0_pt, x2_pt], dim=cat_dim)
            reshape_2_pt = torch.reshape(concat_1_pt, [-1, bmm_K, N])
            reshape_2_trans_pt = torch.transpose(reshape_2_pt, -2, -1)
            bmm_crr_add_3_pt = torch.matmul(reshape_2_trans_pt, x4_pt) + x5_pt
            reshape_4_pt = torch.reshape(concat_1_pt, [-1, reshape_to_shape_4, N])
            concat_5_pt = torch.cat([x3_pt, reshape_4_pt, x3_pt], dim=cat_dim)
            reduce_8_pt = torch.sum(bmm_crr_add_3_pt, reduce_dim)
            reduce_9_pt = torch.sum(concat_5_pt, reduce_dim)
            y_pt = reduce_8_pt + reduce_9_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
                "x2": x2_pt,
                "x3": x3_pt,
                "x4": x4_pt,
                "x5": x5_pt,
            }
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_move_strided_reshape_cat_multi_dsts(self):
        self._test_move_strided_reshape_cat_multi_dsts(
            M0=4,
            M1=4,
            M2=6,
            M3=4,
            N=4,
            test_name="test_move_strided_reshape_cat_multi_dsts",
            dtype="float16",
        )

    def _test_non_movable_cat_reshape_cat_2(
        self, M0, M1, M2, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # concat_0 = concatenate(x0, x0) # 2d
        # reshape_1 = reshape(concat_0) # 3d
        # concat_2 = concat(reshape_1, x1) # 3d
        # concat_3 = concatenate(concat_0, x2) # 2d
        # reduce_4 = reduce_sum(concat_2)
        # reduce_5 = reduce_sum(concat_3)
        # y = add(reduce_4, reduce_5)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0 * N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M1), IntImm(N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2 * N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim = 1
        concat_0 = ops.concatenate()([X0, X0], dim=cat_dim)  # 2d
        reshape_1_to_shape = [-1, M0 + M0, N]
        reshape_1 = ops.reshape()(concat_0, reshape_1_to_shape)
        concat_2 = ops.concatenate()([reshape_1, X1], dim=cat_dim)  # 3d
        concat_3 = ops.concatenate()([concat_0, X2], dim=cat_dim)  # 2d
        reduce_dim = cat_dim
        reduce_4 = ops.reduce_sum(reduce_dim)(concat_2)
        reduce_4_2 = ops.reduce_sum(reduce_dim)(reduce_4)
        reduce_5 = ops.reduce_sum(reduce_dim)(concat_3)
        Y = ops.elementwise(FuncEnum.ADD)(reduce_4_2, reduce_5)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 7)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            op_type = sorted_op._attrs["op"]
            if op_type == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 3)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1, N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)

            concat_0_pt = torch.cat([x0_pt, x0_pt], dim=cat_dim)
            reshape_1_pt = torch.reshape(concat_0_pt, reshape_1_to_shape)
            concat_2_pt = torch.cat([reshape_1_pt, x1_pt], dim=cat_dim)
            concat_3_pt = torch.cat([concat_0_pt, x2_pt], dim=cat_dim)
            reduce_4_pt = torch.sum(concat_2_pt, reduce_dim)
            reduce_4_2_pt = torch.sum(reduce_4_pt, reduce_dim)
            reduce_5_pt = torch.sum(concat_3_pt, reduce_dim)
            y_pt = reduce_4_2_pt + reduce_5_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
                "x2": x2_pt,
            }
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_non_movable_cat_reshape_cat_2(self):
        self._test_non_movable_cat_reshape_cat_2(
            M0=3,
            M1=4,
            M2=6,
            N=4,
            test_name="test_non_movable_cat_reshape_cat_2",
            dtype="float16",
        )


if __name__ == "__main__":
    unittest.main()
