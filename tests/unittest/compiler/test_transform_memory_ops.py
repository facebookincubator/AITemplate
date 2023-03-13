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

    def _test_fuse_cat_reshape_cat(self, M0, M1, M2, N, test_name, dtype="float16"):
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
        X2 = Tensor(
            shape=[batch_dim, IntImm(M2), IntImm(N)],
            dtype=dtype,
            name="x2",
            is_input=True,
        )
        cat_dim = 1
        concat_0 = ops.concatenate()([X0, X1], dim=cat_dim)
        reshape_1 = ops.reshape()(concat_0, [-1, M0 + M1, N])
        Y = ops.concatenate()([reshape_1, X2], dim=cat_dim)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "concatenate")

        for batch in [1, self.BATCH_SIZE]:
            x0_pt = get_random_torch_tensor([batch, M0 * N], dtype)
            x1_pt = get_random_torch_tensor([batch, M1 * N], dtype)
            x2_pt = get_random_torch_tensor([batch, M2, N], dtype)
            concat_0_pt = torch.cat([x0_pt, x1_pt], dim=cat_dim)
            reshape_1_pt = torch.reshape(concat_0_pt, [-1, M0 + M1, N])
            y_pt = torch.cat([reshape_1_pt, x2_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_fuse_cat_reshape_cat(self):
        self._test_fuse_cat_reshape_cat(
            M0=2,
            M1=2,
            M2=6,
            N=8,
            test_name="test_fuse_cat_reshape_cat",
            dtype="float16",
        )
        self._test_fuse_cat_reshape_cat(
            M0=1,
            M1=5,
            M2=7,
            N=3,
            test_name="test_fuse_cat_reshape_cat",
            dtype="float16",
        )
        self._test_fuse_cat_reshape_cat(
            M0=6,
            M1=3,
            M2=4,
            N=8,
            test_name="test_fuse_cat_reshape_cat",
            dtype="float16",
        )

    def _test_fuse_cat_reshape_cat_2(
        self, M0, M1, M2, M3, N, test_name, dtype="float16"
    ):
        # make a graph like below:
        # concat_0 = concatenate(x0, x1)
        # concat_1 = concatenate(x0, x1)
        # reshape_2 = reshape(concat_0)
        # reshape_3 = reshape(concat_1)
        # concat_4 = concatenate(x2, reshape_2, reshape_3, x3, reshape_2)
        # reshape_5 = reshape(concat_4)
        # flatten_6 = flatten(concat_4)
        # concat_7 = concatenate(x0, reshape_5, x1, flatten_6)
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
        reshape_to_shape_5 = (
            sum(
                [
                    t.shape()[cat_dim].value()
                    for t in [X2, reshape_2, reshape_3, X3, reshape_2]
                ]
            )
            * N
        )
        reshape_5 = ops.reshape()(concat_4, [-1, reshape_to_shape_5])
        flatten_6 = ops.flatten(start_dim=1, end_dim=-1)(concat_4)
        Y = ops.concatenate()([X0, reshape_5, X1, flatten_6], dim=cat_dim)
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
            reshape_5_pt = torch.reshape(concat_4_pt, [-1, reshape_to_shape_5])
            flatten_6_pt = torch.flatten(concat_4_pt, 1, -1)
            y_pt = torch.cat([x0_pt, reshape_5_pt, x1_pt, flatten_6_pt], dim=cat_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.05, rtol=0.05)

    def test_fuse_cat_reshape_cat_2(self):
        self._test_fuse_cat_reshape_cat_2(
            M0=2,
            M1=2,
            M2=6,
            M3=4,
            N=8,
            test_name="test_fuse_cat_reshape_cat_2",
            dtype="float16",
        )

    def _test_fuse_strided_cat_reshape_cat(
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

    def test_fuse_strided_cat_reshape_cat(self):
        self._test_fuse_strided_cat_reshape_cat(
            M0=4,
            M1=4,
            M2=6,
            M3=3,
            N=8,
            test_name="test_fuse_strided_cat_reshape_cat",
            dtype="float16",
        )
        self._test_fuse_strided_cat_reshape_cat(
            M0=4,
            M1=4,
            M2=5,
            M3=10,
            N=7,
            test_name="test_fuse_strided_cat_reshape_cat",
            dtype="float16",
        )

    def _test_fuse_strided_cat_reshape_cat_2(
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

    def test_fuse_strided_cat_reshape_cat_2(self):
        self._test_fuse_strided_cat_reshape_cat_2(
            M0=4,
            M1=6,
            M2=9,
            M3=16,
            N=8,
            test_name="test_fuse_strided_cat_reshape_cat_2",
            dtype="float16",
        )

    def _test_fuse_strided_cat_reshape_cat_3(
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

    def test_fuse_strided_cat_reshape_cat_3(self):
        self._test_fuse_strided_cat_reshape_cat_3(
            M0=4,
            M1=6,
            M2=9,
            M3=16,
            N=8,
            test_name="test_fuse_strided_cat_reshape_cat_3",
            dtype="float16",
        )

    def _test_fuse_strided_cat_reshape_cat_4(
        self, M0, M2, N, test_name, dtype="float16"
    ):
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

    def test_fuse_strided_cat_reshape_cat_4(self):
        self._test_fuse_strided_cat_reshape_cat_4(
            M0=4,
            M2=9,
            N=8,
            test_name="test_fuse_strided_cat_reshape_cat_4",
            dtype="float16",
        )

    def _test_fuse_strided_cat_reshape_cat_5(
        self, M0, M2, N, test_name, dtype="float16"
    ):
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

    def test_fuse_strided_cat_reshape_cat_5(self):
        self._test_fuse_strided_cat_reshape_cat_5(
            M0=4,
            M2=9,
            N=8,
            test_name="test_fuse_strided_cat_reshape_cat_5",
            dtype="float16",
        )

    def _test_fuse_strided_cat_reshape_cat_6(
        self, M0, M2, N, test_name, dtype="float16"
    ):
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

    def test_fuse_strided_cat_reshape_cat_6(self):
        self._test_fuse_strided_cat_reshape_cat_6(
            M0=4,
            M2=9,
            N=8,
            test_name="test_fuse_strided_cat_reshape_cat_6",
            dtype="float16",
        )


if __name__ == "__main__":
    unittest.main()
