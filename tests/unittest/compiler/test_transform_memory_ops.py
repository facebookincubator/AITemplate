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
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils

from parameterized import parameterized


class MemoryOpTransformationTestCase(unittest.TestCase):
    BATCH_SIZE = 1024
    M = 10
    N = 128
    USE_DYNAMIC_BATCH = False

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


if __name__ == "__main__":
    unittest.main()
