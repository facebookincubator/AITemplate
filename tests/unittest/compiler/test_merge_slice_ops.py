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


class MergeSliceOpsTestCase(unittest.TestCase):
    BATCH_SIZE = 1024

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def __init__(self, *args, **kwargs):
        super(MergeSliceOpsTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _test_slice_slice_basic(
        self,
        M0,
        N0,
        first_slice_start_indices,
        first_slice_end_indices,
        second_slice_start_indices,
        second_slice_end_indices,
        expected_ops_cnt,
        expected_slice_cnt,
        test_name,
        dtype="float16",
    ):
        # make a graph like below
        # add_0 = add(x0, x1)
        # slice_1 = slice(add_0)
        # slice_2 = slice(slice_1)
        # y = concat(x2, slice_2)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N0)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N0)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )

        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        slice_1 = ops.dynamic_slice()(
            add_0,
            start_indices=first_slice_start_indices,
            end_indices=first_slice_end_indices,
        )
        slice_2 = ops.dynamic_slice()(
            slice_1,
            start_indices=second_slice_start_indices,
            end_indices=second_slice_end_indices,
        )
        M2 = 3
        N2 = slice_2.shape()[-1].value()
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2), IntImm(N2)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim = 1
        Y = ops.concatenate()([X2, slice_2], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_ops_cnt)
        slice_ops = [op for op in sorted_ops if op._attrs["op"] == "dynamic_slice"]
        self.assertEqual(len(slice_ops), expected_slice_cnt)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N0], dtype)
            x1_pt = get_random_torch_tensor([batch, M0, N0], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N2], dtype)

            first_slice_indices = [
                slice(i, j)
                for i, j in zip(first_slice_start_indices, first_slice_end_indices)
            ]
            second_slice_indices = [
                slice(i, j)
                for i, j in zip(second_slice_start_indices, second_slice_end_indices)
            ]
            add_0_pt = x0_pt + x1_pt
            slice_1_pt = add_0_pt[first_slice_indices]
            slice_2_pt = slice_1_pt[second_slice_indices]
            y_pt = torch.cat([x2_pt, slice_2_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
                "x2": x2_pt,
            }
            outputs = [y]
            module.run_with_tensors(inputs, outputs)
            torch.testing.assert_close(y_pt, y, atol=1e-2, rtol=1e-2)

    def test_slice_slice_basic(self):
        self._test_slice_slice_basic(
            M0=10,
            N0=18,
            first_slice_start_indices=[0, 2, 1],
            first_slice_end_indices=[None, None, 15],
            second_slice_start_indices=[0, 1, 3],
            second_slice_end_indices=[None, None, 5],
            expected_ops_cnt=3,
            expected_slice_cnt=1,
            test_name="slice_slice_basic_0",
            dtype="float16",
        )
        self._test_slice_slice_basic(
            M0=10,
            N0=18,
            first_slice_start_indices=[0, 2, 0],
            first_slice_end_indices=[None, 10, None],
            second_slice_start_indices=[0, 2, 0],
            second_slice_end_indices=[None, 4, None],
            expected_ops_cnt=2,
            expected_slice_cnt=0,
            test_name="slice_slice_basic_1",
            dtype="float16",
        )
        self._test_slice_slice_basic(
            M0=10,
            N0=18,
            first_slice_start_indices=[0, 2, 3],
            first_slice_end_indices=[None, 10, 12],
            second_slice_start_indices=[0, 2, 1],
            second_slice_end_indices=[None, None, 6],
            expected_ops_cnt=3,
            expected_slice_cnt=1,
            test_name="slice_slice_basic_2",
            dtype="float16",
        )

    def _test_slice_slice_2(
        self,
        M0,
        N0,
        first_slice_start_indices,
        first_slice_end_indices,
        second_slice_start_indices,
        second_slice_end_indices,
        third_slice_start_indices,
        third_slice_end_indices,
        expected_ops_cnt,
        expected_slice_cnt,
        test_name,
        dtype="float16",
    ):
        # make a graph like below
        # add_0 = add(x0, x1)
        # slice_1 = slice(add_0)
        # slice_2 = slice(slice_1)
        # slice_3 = slice(slice_2)
        # y = add(slice_3, x2)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N0)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N0)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )

        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        slice_1 = ops.dynamic_slice()(
            add_0,
            start_indices=first_slice_start_indices,
            end_indices=first_slice_end_indices,
        )
        slice_2 = ops.dynamic_slice()(
            slice_1,
            start_indices=second_slice_start_indices,
            end_indices=second_slice_end_indices,
        )
        slice_3 = ops.dynamic_slice()(
            slice_2,
            start_indices=third_slice_start_indices,
            end_indices=third_slice_end_indices,
        )
        M2 = slice_3.shape()[-2].value()
        N2 = slice_3.shape()[-1].value()
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2), IntImm(N2)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        Y = ops.elementwise(FuncEnum.ADD)(slice_3, X2)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_ops_cnt)
        slice_ops = [op for op in sorted_ops if op._attrs["op"] == "dynamic_slice"]
        self.assertEqual(len(slice_ops), expected_slice_cnt)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N0], dtype)
            x1_pt = get_random_torch_tensor([batch, M0, N0], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N2], dtype)

            first_slice_indices = [
                slice(i, j)
                for i, j in zip(first_slice_start_indices, first_slice_end_indices)
            ]
            second_slice_indices = [
                slice(i, j)
                for i, j in zip(second_slice_start_indices, second_slice_end_indices)
            ]
            third_slice_indices = [
                slice(i, j)
                for i, j in zip(third_slice_start_indices, third_slice_end_indices)
            ]
            add_0_pt = x0_pt + x1_pt
            slice_1_pt = add_0_pt[first_slice_indices]
            slice_2_pt = slice_1_pt[second_slice_indices]
            slice_3_pt = slice_2_pt[third_slice_indices]
            y_pt = slice_3_pt + x2_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
                "x2": x2_pt,
            }
            outputs = [y]
            module.run_with_tensors(inputs, outputs)
            torch.testing.assert_close(y_pt, y, atol=1e-2, rtol=1e-2)

    def test_slice_slice_2(self):
        self._test_slice_slice_2(
            M0=20,
            N0=30,
            first_slice_start_indices=[0, 1, 2],
            first_slice_end_indices=[None, 15, 28],
            second_slice_start_indices=[0, 2, 2],
            second_slice_end_indices=[None, 10, 9],
            third_slice_start_indices=[0, 2, 1],
            third_slice_end_indices=[None, 5, 3],
            expected_ops_cnt=3,
            expected_slice_cnt=1,
            test_name="slice_slice_2",
            dtype="float16",
        )
        self._test_slice_slice_2(
            M0=20,
            N0=30,
            first_slice_start_indices=[0, 1, 2],
            first_slice_end_indices=[None, 15, 28],
            second_slice_start_indices=[0, 2, 2],
            second_slice_end_indices=[None, None, 9],
            third_slice_start_indices=[0, 2, 1],
            third_slice_end_indices=[None, 5, None],
            expected_ops_cnt=3,
            expected_slice_cnt=1,
            test_name="slice_slice_2",
            dtype="float16",
        )

    def _test_slice_slice_3(
        self,
        input_shape,
        first_slice_start_indices,
        first_slice_end_indices,
        second_slice_start_indices,
        second_slice_end_indices,
        expected_ops_cnt,
        expected_slice_cnt,
        test_name,
        dtype="float16",
    ):
        # make a graph like below
        # add_0 = add(x0, x0)
        # slice_1 = slice(add_0)
        # Y = slice(slice_1)
        X0 = Tensor(
            shape=input_shape,
            dtype=dtype,
            name="x0",
            is_input=True,
        )

        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X0)
        slice_1 = ops.dynamic_slice()(
            add_0,
            start_indices=first_slice_start_indices,
            end_indices=first_slice_end_indices,
        )
        Y = ops.dynamic_slice()(
            slice_1,
            start_indices=second_slice_start_indices,
            end_indices=second_slice_end_indices,
        )
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_ops_cnt)
        slice_ops = [op for op in sorted_ops if op._attrs["op"] == "dynamic_slice"]
        self.assertEqual(len(slice_ops), expected_slice_cnt)

        x0_pt = get_random_torch_tensor(input_shape, dtype)

        first_slice_indices = [
            slice(i, j)
            for i, j in zip(first_slice_start_indices, first_slice_end_indices)
        ]
        second_slice_indices = [
            slice(i, j)
            for i, j in zip(second_slice_start_indices, second_slice_end_indices)
        ]
        add_0_pt = x0_pt + x0_pt
        slice_1_pt = add_0_pt[first_slice_indices]
        y_pt = slice_1_pt[second_slice_indices]

        y = get_torch_empty_tensor(y_pt.size(), dtype)
        inputs = {"x0": x0_pt}
        outputs = [y]
        module.run_with_tensors(inputs, outputs)
        torch.testing.assert_close(y_pt, y, atol=1e-2, rtol=1e-2)

    def test_slice_slice_3(self):
        self._test_slice_slice_3(
            input_shape=[2, 3, 2],
            first_slice_start_indices=[0, 1, 0],
            first_slice_end_indices=[None, 2, None],
            second_slice_start_indices=[0, 0, 1],
            second_slice_end_indices=[None, None, 2],
            expected_ops_cnt=2,
            expected_slice_cnt=1,
            test_name="slice_slice_3",
            dtype="float16",
        )
        self._test_slice_slice_3(
            input_shape=[2, 1, 10, 10, 10],
            first_slice_start_indices=[0, 0, 1, 0, 0],
            first_slice_end_indices=[None, None, -1, None, None],
            second_slice_start_indices=[0, 0, 0, 1, 0],
            second_slice_end_indices=[None, None, None, 2, None],
            expected_ops_cnt=2,
            expected_slice_cnt=1,
            test_name="slice_slice_3",
            dtype="float16",
        )

    def _test_non_fusible_slice_slice(
        self,
        M0,
        N0,
        first_slice_start_indices,
        first_slice_end_indices,
        second_slice_start_indices,
        second_slice_end_indices,
        expected_ops_cnt,
        expected_slice_cnt,
        test_name,
        dtype="float16",
    ):
        # make a graph like below
        # add_0 = add(x0, x1)
        # slice_1 = slice(add_0)
        # slice_2 = slice(slice_1)
        # y = concat(x2, slice_1, slice_2)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N0)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N0)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )

        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        slice_1 = ops.dynamic_slice()(
            add_0,
            start_indices=first_slice_start_indices,
            end_indices=first_slice_end_indices,
        )
        slice_1_N = slice_1.shape()[-1].value()
        slice_2 = ops.dynamic_slice()(
            slice_1,
            start_indices=second_slice_start_indices,
            end_indices=second_slice_end_indices,
        )
        M2 = 3
        N2 = slice_2.shape()[-1].value()
        assert N0 == slice_1_N == N2, f"expected {N0=} == {slice_1_N=} == {N2=}"
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2), IntImm(N2)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim = 1
        Y = ops.concatenate()([X2, slice_1, slice_2], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_ops_cnt)
        slice_ops = [op for op in sorted_ops if op._attrs["op"] == "dynamic_slice"]
        self.assertEqual(len(slice_ops), expected_slice_cnt)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N0], dtype)
            x1_pt = get_random_torch_tensor([batch, M0, N0], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N2], dtype)

            first_slice_indices = [
                slice(i, j)
                for i, j in zip(first_slice_start_indices, first_slice_end_indices)
            ]
            second_slice_indices = [
                slice(i, j)
                for i, j in zip(second_slice_start_indices, second_slice_end_indices)
            ]
            add_0_pt = x0_pt + x1_pt
            slice_1_pt = add_0_pt[first_slice_indices]
            slice_2_pt = slice_1_pt[second_slice_indices]
            y_pt = torch.cat([x2_pt, slice_1_pt, slice_2_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
                "x2": x2_pt,
            }
            outputs = [y]
            module.run_with_tensors(inputs, outputs)
            torch.testing.assert_close(y_pt, y, atol=1e-2, rtol=1e-2)

    def test_non_fusible_slice_slice(self):
        self._test_non_fusible_slice_slice(
            M0=10,
            N0=18,
            first_slice_start_indices=[0, 2, 0],
            first_slice_end_indices=[None, 10, None],
            second_slice_start_indices=[0, 2, 0],
            second_slice_end_indices=[None, 4, None],
            expected_ops_cnt=3,
            expected_slice_cnt=1,
            test_name="slice_slice_non_fusible",
            dtype="float16",
        )


if __name__ == "__main__":
    unittest.main()
