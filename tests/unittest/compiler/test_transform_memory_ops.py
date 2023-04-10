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

from aitemplate.compiler import compile_model, ops, transform
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils, shape_utils

from parameterized import parameterized


class MemoryOpTransformationTestCase(unittest.TestCase):
    BATCH_SIZE = 1024
    M = 10
    N = 128
    USE_DYNAMIC_BATCH = False

    def __init__(self, *args, **kwargs):
        super(MemoryOpTransformationTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _prepare_cat_elimination_graph(self, dtype="float16"):
        X0 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch0")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=self.M),
                IntImm(value=self.N),
            ],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        X1 = ops.concatenate()([X0], dim=1)
        X2 = ops.concatenate()([X1], dim=2)
        X3 = ops.concatenate()([X2, X1], dim=1)
        X3._attrs["name"] = "output0"
        X3._attrs["is_output"] = True
        return X3

    def test_cat_elimination_graph_transformation(self):
        OUTPUT = self._prepare_cat_elimination_graph()
        graph = transform.toposort(OUTPUT)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        self.assertEqual(len(graph), 4)
        graph = transform.transform_memory_ops(graph)
        self.assertEqual(len(graph), 2)

    @parameterized.expand([("float16"), ("float")])
    def test_cat_elimination_e2e(self, dtype):
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by ROCM")
        OUTPUT = self._prepare_cat_elimination_graph(dtype)
        module = compile_model(OUTPUT, target, "./tmp", f"cat_elimination_{dtype}")

        x0_pt = get_random_torch_tensor([self.BATCH_SIZE, self.M, self.N], dtype)
        out_pt = torch.cat([x0_pt, x0_pt], dim=1)

        out = get_torch_empty_tensor(out_pt.size(), dtype)
        module.run_with_tensors([x0_pt], [out])
        self.assertTrue(torch.allclose(out_pt, out, atol=1e-1, rtol=1e-2))

    def _prepare_split_cat_elimination_graph(self, dtype="float16"):
        X0 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch0")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=self.M),
                IntImm(value=self.N),
            ],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        [X1, X2] = ops.split()(X0, int(self.M / 2), dim=1)
        X3 = ops.concatenate()([X1, X2], dim=1)
        [X4, X5] = ops.split()(X3, int(self.N / 2), dim=2)
        X6 = ops.concatenate()([X4, X5], dim=1)
        X6._attrs["name"] = "output0"
        X6._attrs["is_output"] = True

        Y0 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch1")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=self.M),
                IntImm(value=self.N),
            ],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        Y1 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch2")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=self.M),
                IntImm(value=self.N),
            ],
            dtype=dtype,
            name="input2",
            is_input=True,
        )
        [Y2, Y3] = ops.split()(Y0, int(self.M / 2), dim=1)
        Y4 = ops.concatenate()([Y1, Y2, Y3, Y0], dim=1)
        Y4._attrs["name"] = "output1"
        Y4._attrs["is_output"] = True

        return [X6, Y4]

    def test_split_cat_elimination_graph_transformation(self):
        OUTPUT = self._prepare_split_cat_elimination_graph()
        graph = transform.toposort(OUTPUT)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        self.assertEqual(len(graph), 12)
        graph = transform.transform_memory_ops(graph)
        self.assertEqual(len(graph), 7)

    @parameterized.expand([("float16"), ("float")])
    def test_split_cat_elimination_e2e(self, dtype):
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by ROCM")
        OUTPUT = self._prepare_split_cat_elimination_graph(dtype)
        module = compile_model(
            OUTPUT, target, "./tmp", f"split_cat_elimination_{dtype}"
        )

        x0_pt = get_random_torch_tensor([self.BATCH_SIZE, self.M, self.N], dtype)
        x4_pt, x5_pt = torch.split(x0_pt, int(self.N / 2), dim=2)
        out_pt0 = torch.cat([x4_pt, x5_pt], dim=1)
        y0_pt = get_random_torch_tensor([self.BATCH_SIZE, self.M, self.N], dtype)
        y1_pt = get_random_torch_tensor([self.BATCH_SIZE, self.M, self.N], dtype)
        out_pt1 = torch.cat([y1_pt, y0_pt, y0_pt], dim=1)

        out0 = get_torch_empty_tensor(out_pt0.size(), dtype)
        out1 = get_torch_empty_tensor(out_pt1.size(), dtype)
        module.run_with_tensors(
            {"input0": x0_pt, "input1": y0_pt, "input2": y1_pt},
            {"output0": out0, "output1": out1},
        )
        self.assertTrue(torch.allclose(out_pt0, out0, atol=1e-1, rtol=1e-2))
        self.assertTrue(torch.allclose(out_pt1, out1, atol=1e-1, rtol=1e-2))

    def _prepare_cat_cat_elimination_graph(self, dtype="float16"):
        X0 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch0")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=int(self.M / 2)),
                IntImm(value=self.N),
            ],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch1")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=int(self.M / 2)),
                IntImm(value=self.N),
            ],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch2")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=self.M),
                IntImm(value=self.N + 4),
            ],
            dtype=dtype,
            name="input2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch3")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=self.M),
                IntImm(value=self.N * 2),
            ],
            dtype=dtype,
            name="input3",
            is_input=True,
        )

        X5 = ops.concatenate()([X0, X1], dim=1)
        X6 = ops.concatenate()([X5, X2], dim=2)
        X7 = ops.concatenate()([X3, X6], dim=2)
        X8 = ops.concatenate()([X7, X2], dim=2)
        X8._attrs["name"] = "output0"
        X8._attrs["is_output"] = True

        return [X8]

    def test_cat_cat_elimination_graph_transformation(self):
        OUTPUT = self._prepare_cat_cat_elimination_graph()
        graph = transform.toposort(OUTPUT)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        self.assertEqual(len(graph), 8)
        self.assertEqual(len(graph_utils.get_sorted_ops(graph)), 4)
        graph = transform.transform_memory_ops(graph)
        self.assertEqual(len(graph), 6)
        self.assertEqual(len(graph_utils.get_sorted_ops(graph)), 2)

    @parameterized.expand([("float16"), ("float")])
    def test_cat_cat_elimination_e2e(self, dtype):
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by ROCM")
        OUTPUT = self._prepare_cat_cat_elimination_graph(dtype)
        module = compile_model(OUTPUT, target, "./tmp", f"cat_cat_elimination_{dtype}")

        x0_pt = get_random_torch_tensor(
            [self.BATCH_SIZE, int(self.M / 2), self.N], dtype
        )
        x1_pt = get_random_torch_tensor(
            [self.BATCH_SIZE, int(self.M / 2), self.N], dtype
        )
        x2_pt = get_random_torch_tensor([self.BATCH_SIZE, self.M, self.N + 4], dtype)
        x3_pt = get_random_torch_tensor([self.BATCH_SIZE, self.M, self.N * 2], dtype)
        x5_pt = torch.cat([x0_pt, x1_pt], dim=1)
        out_pt0 = torch.cat([x3_pt, x5_pt, x2_pt, x2_pt], dim=2)

        out0 = get_torch_empty_tensor(out_pt0.size(), dtype)
        module.run_with_tensors(
            {"input0": x0_pt, "input1": x1_pt, "input2": x2_pt, "input3": x3_pt},
            [out0],
        )
        self.assertTrue(torch.allclose(out_pt0, out0, atol=1e-1, rtol=1e-2))

    def _prepare_skip_cat_elimination_graph(self, dtype="float16"):
        X0 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch0")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=self.M),
                IntImm(value=self.N),
            ],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        X1 = ops.concatenate()([X0], dim=1)
        X2 = ops.concatenate()([X1], dim=2)
        X3 = ops.concatenate()([X2, X1], dim=1)
        X1._attrs["name"] = "output0"
        X1._attrs["is_output"] = True
        X3._attrs["name"] = "output1"
        X3._attrs["is_output"] = True
        return X1, X3

    def test_skip_cat_elimination_graph_transformation(self):
        OUTPUT = self._prepare_skip_cat_elimination_graph()
        graph = transform.toposort(OUTPUT)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        self.assertEqual(len(graph), 4)
        graph = transform.transform_memory_ops(graph)
        print(graph)
        self.assertEqual(len(graph), 3)

    @parameterized.expand([("float16"), ("float")])
    def test_skip_cat_elimination_e2e(self, dtype):
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by ROCM")
        OUTPUT = self._prepare_skip_cat_elimination_graph(dtype)
        module = compile_model(OUTPUT, target, "./tmp", f"skip_cat_elimination_{dtype}")

        x0_pt = get_random_torch_tensor([self.BATCH_SIZE, self.M, self.N], dtype)
        out0_pt = torch.cat([x0_pt], dim=1)
        out1_pt = torch.cat([x0_pt, x0_pt], dim=1)

        out0 = get_torch_empty_tensor(out0_pt.size(), dtype)
        out1 = get_torch_empty_tensor(out1_pt.size(), dtype)
        module.run_with_tensors([x0_pt], [out0, out1])
        self.assertTrue(torch.allclose(out0_pt, out0, atol=1e-1, rtol=1e-2))
        self.assertTrue(torch.allclose(out1_pt, out1, atol=1e-1, rtol=1e-2))

    def _prepare_skip_split_cat_elimination_graph(self, dtype="float16"):
        X0 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch0")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=self.M),
                IntImm(value=self.N),
            ],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        [X1, X2] = ops.split()(X0, int(self.M / 2), dim=1)
        X3 = ops.concatenate()([X1, X2], dim=1)
        [X4, X5] = ops.split()(X3, int(self.N / 2), dim=2)
        X6 = ops.concatenate()([X4, X5], dim=1)
        X3._attrs["name"] = "output0"
        X3._attrs["is_output"] = True
        X6._attrs["name"] = "output1"
        X6._attrs["is_output"] = True

        return [X3, X6]

    def test_skip_split_cat_elimination_graph_transformation(self):
        OUTPUT = self._prepare_skip_split_cat_elimination_graph()
        graph = transform.toposort(OUTPUT)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        self.assertEqual(len(graph), 7)
        graph = transform.transform_memory_ops(graph)
        self.assertEqual(len(graph), 5)

    @parameterized.expand([("float16")])  # , ("float")])
    def test_skip_split_cat_elimination_e2e(self, dtype):
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by ROCM")
        OUTPUT = self._prepare_skip_split_cat_elimination_graph(dtype)
        module = compile_model(
            OUTPUT, target, "./tmp", f"skip_split_cat_elimination_{dtype}"
        )

        x0_pt = get_random_torch_tensor([self.BATCH_SIZE, self.M, self.N], dtype)
        out_pt0 = x0_pt
        x4_pt, x5_pt = torch.split(x0_pt, int(self.N / 2), dim=2)
        out_pt1 = torch.cat([x4_pt, x5_pt], dim=1)

        out0 = get_torch_empty_tensor(out_pt0.size(), dtype)
        out1 = get_torch_empty_tensor(out_pt1.size(), dtype)
        module.run_with_tensors(
            {"input0": x0_pt},
            {"output0": out0, "output1": out1},
        )
        self.assertTrue(torch.allclose(out_pt0, out0, atol=1e-1, rtol=1e-2))
        self.assertTrue(torch.allclose(out_pt1, out1, atol=1e-1, rtol=1e-2))

    def _prepare_skip_cat_cat_elimination_graph(self, dtype="float16"):
        X0 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch0")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=int(self.M / 2)),
                IntImm(value=self.N),
            ],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch1")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=int(self.M / 2)),
                IntImm(value=self.N),
            ],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch2")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=self.M),
                IntImm(value=self.N + 4),
            ],
            dtype=dtype,
            name="input2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[
                IntVar(values=[1, self.BATCH_SIZE], name="input_batch3")
                if self.USE_DYNAMIC_BATCH
                else IntImm(value=self.BATCH_SIZE),
                IntImm(value=self.M),
                IntImm(value=self.N * 2),
            ],
            dtype=dtype,
            name="input3",
            is_input=True,
        )

        X5 = ops.concatenate()([X0, X1], dim=1)
        X6 = ops.concatenate()([X5, X2], dim=2)
        X7 = ops.concatenate()([X3, X6], dim=2)
        X8 = ops.concatenate()([X7, X2], dim=2)
        X6._attrs["name"] = "output0"
        X6._attrs["is_output"] = True
        X8._attrs["name"] = "output1"
        X8._attrs["is_output"] = True

        return [X6, X8]

    def test_skip_cat_cat_elimination_graph_transformation(self):
        OUTPUT = self._prepare_skip_cat_cat_elimination_graph()
        graph = transform.toposort(OUTPUT)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        self.assertEqual(len(graph), 8)
        self.assertEqual(len(graph_utils.get_sorted_ops(graph)), 4)
        graph = transform.transform_memory_ops(graph)
        self.assertEqual(len(graph), 7)
        self.assertEqual(len(graph_utils.get_sorted_ops(graph)), 3)

    @parameterized.expand([("float16"), ("float")])
    def test_skip_cat_cat_elimination_e2e(self, dtype):
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by ROCM")
        OUTPUT = self._prepare_skip_cat_cat_elimination_graph(dtype)
        module = compile_model(
            OUTPUT, target, "./tmp", f"skip_cat_cat_elimination_{dtype}"
        )

        x0_pt = get_random_torch_tensor(
            [self.BATCH_SIZE, int(self.M / 2), self.N], dtype
        )
        x1_pt = get_random_torch_tensor(
            [self.BATCH_SIZE, int(self.M / 2), self.N], dtype
        )
        x2_pt = get_random_torch_tensor([self.BATCH_SIZE, self.M, self.N + 4], dtype)
        x3_pt = get_random_torch_tensor([self.BATCH_SIZE, self.M, self.N * 2], dtype)
        x5_pt = torch.cat([x0_pt, x1_pt], dim=1)
        out_pt0 = torch.cat([x5_pt, x2_pt], dim=2)
        out_pt1 = torch.cat([x3_pt, x5_pt, x2_pt, x2_pt], dim=2)

        out0 = get_torch_empty_tensor(out_pt0.size(), dtype)
        out1 = get_torch_empty_tensor(out_pt1.size(), dtype)
        module.run_with_tensors(
            {"input0": x0_pt, "input1": x1_pt, "input2": x2_pt, "input3": x3_pt},
            [out0, out1],
        )
        self.assertTrue(torch.allclose(out_pt0, out0, atol=1e-1, rtol=1e-2))
        self.assertTrue(torch.allclose(out_pt1, out1, atol=1e-1, rtol=1e-2))

    def _test_fuse_strided_cat_cat(self, M0, M1, N, test_name, dtype="float16"):
        # make a graph like below:
        # concat_0 = concatenate(x0, x1)
        # add_1 = add(concat_0, x2)
        # concat_2 = concatenate(x0, concat_0)
        # reduce_3 = reduce_sum(add_1)
        # reduce_4 = reduce_sum(concat_2)
        # y = add(reduce_3, reduce_4)
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
        M2 = M0 + M1
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2), IntImm(N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim = 1
        concat_0 = ops.concatenate()([X0, X1], dim=cat_dim)
        add_1 = ops.elementwise(FuncEnum.ADD)(concat_0, X2)
        concat_2 = ops.concatenate()([X0, concat_0], dim=cat_dim)
        reduce_dim = cat_dim
        reduce_3 = ops.reduce_sum(reduce_dim)(add_1)
        reduce_4 = ops.reduce_sum(reduce_dim)(concat_2)
        Y = ops.elementwise(FuncEnum.ADD)(reduce_3, reduce_4)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 5)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            op_type = sorted_op._attrs["op"]
            # dynamic_slice is fused into add
            self.assertTrue(op_type != "dynamic_slice")
            if op_type == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1, N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N], dtype)

            concat_0_pt = torch.cat([x0_pt, x1_pt], dim=cat_dim)
            add_1_pt = concat_0_pt + x2_pt
            concat_2_pt = torch.cat([x0_pt, concat_0_pt], dim=cat_dim)
            reduce_3_pt = torch.sum(add_1_pt, reduce_dim)
            reduce_4_pt = torch.sum(concat_2_pt, reduce_dim)
            y_pt = reduce_3_pt + reduce_4_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
                "x2": x2_pt,
            }
            outputs = [y]
            module.run_with_tensors(inputs, outputs)
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_strided_cat_cat(self):
        self._test_fuse_strided_cat_cat(
            M0=3,
            M1=4,
            N=9,
            test_name="test_fuse_strided_cat_cat",
        )
        self._test_fuse_strided_cat_cat(
            M0=2,
            M1=4,
            N=8,
            test_name="test_fuse_strided_cat_cat",
        )

    def _test_fuse_strided_cat_reshape_cat(
        self, M0, M1, M3, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # concat_0 = concatenate(x0, x1)
        # reshape_1 = reshape(concat_0)
        # add_2 = add(reshape_1, x2)
        # concat_3 = concatenate(x0, reshape_1)
        # reduce_4 = reduce_sum(add_2)
        # reduce_5 = reduce_sum(concat_3)
        # y = add(reduce_4, reduce_5)
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
        M2 = M0 + M1
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
        cat_dim = 1
        concat_0 = ops.concatenate()([X0, X1], dim=cat_dim)
        reshape_to_shape_1 = [-1, M2 * N]
        reshape_1 = ops.reshape()(concat_0, reshape_to_shape_1)
        add_2 = ops.elementwise(FuncEnum.ADD)(reshape_1, X2)
        concat_3 = ops.concatenate()([X3, reshape_1], dim=cat_dim)
        reduce_dim = cat_dim
        reduce_4 = ops.reduce_sum(reduce_dim)(add_2)
        reduce_5 = ops.reduce_sum(reduce_dim)(concat_3)
        Y = ops.elementwise(FuncEnum.ADD)(reduce_4, reduce_5)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 5)
        concat_cnt = 0
        for sorted_op in sorted_ops:
            op_type = sorted_op._attrs["op"]
            # dynamic_slice is fused into add
            self.assertTrue(op_type != "dynamic_slice")
            if op_type == "concatenate":
                concat_cnt += 1
        self.assertEqual(concat_cnt, 1)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1, N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2 * N], dtype)
            x3_pt = get_random_torch_tensor([batch, M3 * N], dtype)

            concat_0_pt = torch.cat([x0_pt, x1_pt], dim=cat_dim)
            reshape_1_pt = torch.reshape(concat_0_pt, reshape_to_shape_1)
            add_2_pt = reshape_1_pt + x2_pt
            concat_3_pt = torch.cat([x3_pt, reshape_1_pt], dim=cat_dim)
            reduce_4_pt = torch.sum(add_2_pt, reduce_dim)
            reduce_5_pt = torch.sum(concat_3_pt, reduce_dim)
            y_pt = reduce_4_pt + reduce_5_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
                "x2": x2_pt,
                "x3": x3_pt,
            }
            outputs = [y]
            module.run_with_tensors(inputs, outputs)
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_strided_cat_reshape_cat(self):
        self._test_fuse_strided_cat_reshape_cat(
            M0=2,
            M1=4,
            M3=3,
            N=8,
            test_name="test_fuse_strided_cat_reshape_cat",
        )

    def _test_fuse_strided_cat_reshape_cat_2(
        self, M0, M1, M2, M3, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # add_0 = add(x0, x1)  # 2d
        # concat_1 = concatenate(add_0, x2) # 2d
        # reshape_2 = reshape(concat_1) # 3d
        # add_3 = add(reshape_2, x4) # 3d
        # concat_4 = concatenate(x3, reshape_2, x3) # 3d
        # reshape_5 = reshape(concat_4) # 2d
        # add_6 = add(reshape_5, x6) # 2d
        # concat_7 = concatenate(x0, reshape_5, x0)
        # reshape_8 = reshape(add_3) # 2d
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
            shape=[batch_dim, IntImm(M4), IntImm(N)],
            dtype=dtype,
            name="x4",
            is_input=True,
        )
        cat_dim = 1
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        concat_1 = ops.concatenate()([add_0, X2], dim=cat_dim)
        reshape_2 = ops.reshape()(concat_1, [-1, M0 + M2, N])
        add_3 = ops.elementwise(FuncEnum.ADD)(reshape_2, X4)
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
        reshape_8 = ops.reshape()(add_3, [-1, (M0 + M2) * N])  # 2d
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
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)
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
            x3_pt = get_random_torch_tensor([batch, M3, N], dtype)
            x4_pt = get_random_torch_tensor([batch, M4, N], dtype)
            x6_pt = get_random_torch_tensor([batch, reshape_to_shape_5], dtype)
            add_0_pt = x0_pt + x1_pt
            concat_1_pt = torch.cat([add_0_pt, x2_pt], dim=cat_dim)
            reshape_2_pt = torch.reshape(concat_1_pt, [-1, M0 + M2, N])
            add_3_pt = reshape_2_pt + x4_pt
            concat_4_pt = torch.cat([x3_pt, reshape_2_pt, x3_pt], dim=cat_dim)
            reshape_5_pt = torch.reshape(concat_4_pt, [-1, reshape_to_shape_5])
            add_6_pt = reshape_5_pt + x6_pt
            concat_7_pt = torch.cat([x0_pt, reshape_5_pt, x0_pt], dim=cat_dim)
            reshape_8_pt = torch.reshape(add_3_pt, [-1, (M0 + M2) * N])
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
                "x6": x6_pt,
            }
            outputs = [y]
            module.run_with_tensors(inputs, outputs)
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_strided_cat_reshape_cat_2(self):
        self._test_fuse_strided_cat_reshape_cat_2(
            M0=2,
            M1=2,
            M2=2,
            M3=1,
            N=2,
            test_name="test_fuse_strided_cat_reshape_cat_2",
        )

    def _test_fuse_strided_cat_reshape_cat_3(
        self, M0, M1, M2, M3, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # add_0 = add(x0, x1)  # 2d
        # concat_1 = concatenate(add_0, x2) # 2d
        # reshape_2 = reshape(concat_1) # 3d
        # add_3 = add(reshape_2, x4) # 3d
        # concat_4 = concatenate(x3, concat_1, x3) # 2d
        # reshape_5 = reshape(concat_4) # 2d
        # add_6 = add(reshape_5, x6) # 2d
        # concat_7 = concatenate(x0, reshape_5, x0)
        # reshape_8 = reshape(add_3) # 2d
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
            shape=[batch_dim, IntImm(M4), IntImm(N)],
            dtype=dtype,
            name="x4",
            is_input=True,
        )
        cat_dim = 1
        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        concat_1 = ops.concatenate()([add_0, X2], dim=cat_dim)
        reshape_2 = ops.reshape()(concat_1, [-1, M0 + M2, N])
        add_3 = ops.elementwise(FuncEnum.ADD)(reshape_2, X4)
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
        reshape_8 = ops.reshape()(add_3, [-1, (M0 + M2) * N])  # 2d
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
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)
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
            x3_pt = get_random_torch_tensor([batch, M3, N], dtype)
            x4_pt = get_random_torch_tensor([batch, M4, N], dtype)
            x6_pt = get_random_torch_tensor([batch, reshape_to_shape_5], dtype)
            add_0_pt = x0_pt + x1_pt
            concat_1_pt = torch.cat([add_0_pt, x2_pt], dim=cat_dim)
            reshape_2_pt = torch.reshape(concat_1_pt, [-1, M0 + M2, N])
            add_3_pt = reshape_2_pt + x4_pt
            concat_4_pt = torch.cat([x3_pt, reshape_2_pt, x3_pt], dim=cat_dim)
            reshape_5_pt = torch.reshape(concat_4_pt, [-1, reshape_to_shape_5])
            add_6_pt = reshape_5_pt + x6_pt
            concat_7_pt = torch.cat([x0_pt, reshape_5_pt, x0_pt], dim=cat_dim)
            reshape_8_pt = torch.reshape(add_3_pt, [-1, (M0 + M2) * N])
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
                "x6": x6_pt,
            }
            outputs = [y]
            module.run_with_tensors(inputs, outputs)
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_strided_cat_reshape_cat_3(self):
        self._test_fuse_strided_cat_reshape_cat_3(
            M0=2,
            M1=2,
            M2=2,
            M3=1,
            N=2,
            test_name="test_fuse_strided_cat_reshape_cat_3",
        )

    def _test_non_fusible_strided_cat_cat(self, M0, N, test_name, dtype="float16"):
        # make a graph like below:
        # concat_0 = concatenate(x0, x1)
        # add_1 = add(concat_0, x2)
        # y = concatenate(concat_0, add_1)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(M0), IntImm(N)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(M0 + M0), IntImm(N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim = 1
        concat_0 = ops.concatenate()([X0, X1], dim=cat_dim)
        add_1 = ops.elementwise(FuncEnum.ADD)(concat_0, X2)
        Y = ops.concatenate()([concat_0, add_1], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 3)
        concat_cnt = 0
        output_cat = None
        for sorted_op in sorted_ops:
            op_type = sorted_op._attrs["op"]
            if op_type == "concatenate":
                concat_cnt += 1
                if sorted_op._attrs["outputs"][0] == Y:
                    output_cat = sorted_op
        self.assertEqual(concat_cnt, 2)
        self.assertEqual(output_cat._attrs["input_masks"], [True, False])

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0, N], dtype)
            x1_pt = get_random_torch_tensor([batch, M0, N], dtype)
            x2_pt = get_random_torch_tensor([batch, M0 + M0, N], dtype)

            concat_0_pt = torch.cat([x0_pt, x1_pt], dim=cat_dim)
            add_1_pt = concat_0_pt + x2_pt
            y_pt = torch.cat([concat_0_pt, add_1_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
                "x2": x2_pt,
            }
            outputs = [y]
            module.run_with_tensors(inputs, outputs)
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_non_fusible_strided_cat_cat(self):
        self._test_non_fusible_strided_cat_cat(
            M0=2,
            N=8,
            test_name="test_non_fusible_strided_cat_cat",
        )

    def _test_non_fusible_split_reshape_cat(self, M, test_name, dtype="float16"):
        # make the following graph
        # split_0, split_1 = split(x0)
        # unsqueeze_2 = unsqueeze(dim=1)(split_0)
        # unsqueeze_3 = unsqueeze(dim=1)(split_1)
        # add_4 = add(x1, x1)
        # y = concat([unsqueeze_2, unsqueeze_3, add_4], dim=1)
        batch_sizes = [1, self.BATCH_SIZE]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        assert M % 2 == 0, f"expected {M=} % 2 == 0"
        X0 = Tensor(
            shape=[batch_dim, IntImm(M)],
            dtype=dtype,
            name="x0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[batch_dim, IntImm(2), IntImm(M // 2)],
            dtype=dtype,
            name="x1",
            is_input=True,
        )
        dim = 1
        split_0, split_1 = ops.split()(X0, [M // 2, M // 2], dim=dim)
        unsqueeze_2 = ops.unsqueeze(dim=dim)(split_0)
        unsqueeze_3 = ops.unsqueeze(dim=dim)(split_1)
        add_4 = ops.elementwise(FuncEnum.ADD)(X1, X1)
        Y = ops.concatenate()([unsqueeze_2, unsqueeze_3, add_4], dim=dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y, target, "./tmp", test_name)
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 3)

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M], dtype)
            x1_pt = get_random_torch_tensor([batch, 2, M // 2], dtype)

            split_0_pt, split_1_pt = torch.split(x0_pt, [M // 2, M // 2], dim=dim)
            unsqueeze_2_pt = torch.unsqueeze(split_0_pt, dim)
            unsqueeze_3_pt = torch.unsqueeze(split_1_pt, dim)
            add_4_pt = x1_pt + x1_pt
            y_pt = torch.cat([unsqueeze_2_pt, unsqueeze_3_pt, add_4_pt], dim=dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {
                "x0": x0_pt,
                "x1": x1_pt,
            }
            outputs = [y]
            module.run_with_tensors(inputs, outputs)
            torch.testing.assert_close(y_pt, y, atol=0.01, rtol=0.01)

    def test_non_fusible_split_reshape_cat(self):
        self._test_non_fusible_split_reshape_cat(
            M=32,
            test_name="test_non_fusible_split_reshape_cat",
        )


if __name__ == "__main__":
    unittest.main()
