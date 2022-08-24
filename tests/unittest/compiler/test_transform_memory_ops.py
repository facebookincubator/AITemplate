# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops, transform
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import graph_utils


class MemoryOpTransformationTestCase(unittest.TestCase):
    BATCH_SIZE = 1024
    M = 10
    N = 128
    USE_DYNAMIC_BATCH = False

    def _prepare_cat_elimination_graph(self):
        dtype = "float16"
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

    def test_cat_elimination_e2e(self):
        OUTPUT = self._prepare_cat_elimination_graph()
        target = detect_target()
        module = gen_execution_module(OUTPUT, target, "./tmp", "cat_elimination")

        x0_pt = torch.randn([self.BATCH_SIZE, self.M, self.N]).cuda().half()
        out_pt = torch.cat([x0_pt, x0_pt], dim=1)

        out = torch.empty(out_pt.size()).cuda().half()
        module.RunWithTensors([x0_pt], [out])
        self.assertTrue(torch.allclose(out_pt, out, atol=1e-1, rtol=1e-2))

    def _prepare_split_cat_elimination_graph(self):
        dtype = "float16"
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

    def test_split_cat_elimination_e2e(self):
        OUTPUT = self._prepare_split_cat_elimination_graph()
        target = detect_target()
        module = gen_execution_module(OUTPUT, target, "./tmp", "split_cat_elimination")

        x0_pt = torch.randn([self.BATCH_SIZE, self.M, self.N]).cuda().half()
        x4_pt, x5_pt = torch.split(x0_pt, int(self.N / 2), dim=2)
        out_pt0 = torch.cat([x4_pt, x5_pt], dim=1)
        y0_pt = torch.randn([self.BATCH_SIZE, self.M, self.N]).cuda().half()
        y1_pt = torch.randn([self.BATCH_SIZE, self.M, self.N]).cuda().half()
        out_pt1 = torch.cat([y1_pt, y0_pt, y0_pt], dim=1)

        out0 = torch.empty(out_pt0.size()).cuda().half()
        out1 = torch.empty(out_pt1.size()).cuda().half()
        module.RunWithTensors(
            {"input0": x0_pt, "input1": y0_pt, "input2": y1_pt},
            {"output0": out0, "output1": out1},
        )
        self.assertTrue(torch.allclose(out_pt0, out0, atol=1e-1, rtol=1e-2))
        self.assertTrue(torch.allclose(out_pt1, out1, atol=1e-1, rtol=1e-2))

    def _prepare_cat_cat_elimination_graph(self):
        dtype = "float16"
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

    def test_cat_cat_elimination_e2e(self):
        OUTPUT = self._prepare_cat_cat_elimination_graph()
        target = detect_target()
        module = gen_execution_module(OUTPUT, target, "./tmp", "cat_cat_elimination")

        x0_pt = torch.randn([self.BATCH_SIZE, int(self.M / 2), self.N]).cuda().half()
        x1_pt = torch.randn([self.BATCH_SIZE, int(self.M / 2), self.N]).cuda().half()
        x2_pt = torch.randn([self.BATCH_SIZE, self.M, self.N + 4]).cuda().half()
        x3_pt = torch.randn([self.BATCH_SIZE, self.M, self.N * 2]).cuda().half()
        x5_pt = torch.cat([x0_pt, x1_pt], dim=1)
        out_pt0 = torch.cat([x3_pt, x5_pt, x2_pt, x2_pt], dim=2)

        out0 = torch.empty(out_pt0.size()).cuda().half()
        module.RunWithTensors(
            {"input0": x0_pt, "input1": x1_pt, "input2": x2_pt, "input3": x3_pt},
            [out0],
        )
        self.assertTrue(torch.allclose(out_pt0, out0, atol=1e-1, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
