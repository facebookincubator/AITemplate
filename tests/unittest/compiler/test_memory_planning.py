# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import random
import unittest

import torch
from aitemplate import compiler

from aitemplate.compiler import ops
from aitemplate.compiler.base import Operator
from aitemplate.frontend import IntImm, nn, Tensor
from aitemplate.testing import detect_target, gen_execution_module


class MemoryPlanningTestCase(unittest.TestCase):
    def test_memory_planning_with_tensor_views(self):
        target = detect_target()
        dtype = "float16"
        # batch_size = [4, 16] # reduce_sum doesn't work with dynamic shape
        batch_size = [4]
        in_shape = [32, 16, 8]
        X = Tensor(
            shape=[IntImm(value=batch_size[0], name="input_batch"), *in_shape],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )

        sum_0 = ops.reduce_sum(3, keepdim=True, dtype=dtype)
        sum_1 = ops.reduce_sum(2, keepdim=True, dtype=dtype)
        sum_2 = ops.reduce_sum(2, keepdim=False, dtype=dtype)
        reshape_0 = nn.Reshape()
        reshape_1 = nn.Reshape()
        add_0 = ops.elementwise(ops.common.FuncEnum.ADD)
        flatten_0 = nn.Flatten()

        T0 = sum_0(X)  # [b, 32, 16 1]
        T1 = sum_1(T0)  # [b, 32, 1, 1]

        # This reshape is fused into sum_1.
        T2 = reshape_0(T1, [-1, 32])  # [b, 32]

        # This reshape cannot be fused with sum_0 because T0 is used by multiple dst_ops.
        # This reshape cannot be fused with sum_2 because input_accessors haven't been added into reduction ops yet.
        T3 = reshape_1(T0, [-1, 32, 16])  # [b, 32, 16]

        T4 = sum_2(T3)  # [b, 32]
        T5 = add_0(T2, T4)  # [b, 32]

        # This flatten is fused with add_0.
        OUT = flatten_0(T5)  # [b * 32]

        OUT._attrs["name"] = "output_0"
        OUT._attrs["is_output"] = True

        module = gen_execution_module(OUT, target, "./tmp", "memory_planning")
        self.assertEqual(len(module.debug_sorted_graph), 6)

        assert T0._attrs["offset"] == T3._attrs["offset"]

        for b in batch_size:
            X_shape = [b] + in_shape
            x_pt = torch.randn(X_shape).cuda().half()
            t0_pt = torch.sum(x_pt, dim=3, keepdim=True)
            t1_pt = torch.sum(t0_pt, dim=2, keepdim=True)
            t2_pt = torch.reshape(t1_pt, [-1, 32])
            t3_pt = torch.reshape(t0_pt, [-1, 32, 16])
            t4_pt = torch.sum(t3_pt, dim=2, keepdim=False)
            out_pt = torch.add(t2_pt, t4_pt).flatten()

            out = torch.empty(out_pt.size()).cuda().half()
            module.RunWithTensors([x_pt], [out])
            self.assertTrue(torch.allclose(out_pt, out, atol=1e-1, rtol=1e-2))

    def test_memory_planning_workspace_offsets(self):
        class DummyOp(Operator):
            def __init__(self):
                super().__init__()
                self._attrs["op"] = "dummy_op"

            def __call__(
                self, inp: Tensor, workspace_size: int, unique_workspace_size: int
            ):
                self._attrs["inputs"] = [inp]
                self._set_depth()
                self._attrs["workspace"] = workspace_size
                self._attrs["unique_workspace"] = unique_workspace_size
                output = Tensor(shape=[1], src_ops={self})
                self._attrs["outputs"] = [output]
                return output

        X = Tensor(shape=[1], is_input=True)
        unique_workspace_expected_size = 0
        shared_workspace_expected_size = 0
        for _ in range(10):
            shared = random.randint(0, 10)
            unique = random.randint(0, 10)

            shared_workspace_expected_size = max(shared_workspace_expected_size, shared)
            unique_workspace_expected_size += unique
            X = DummyOp()(X, shared, unique)

        X._attrs["is_output"] = True
        graph = compiler.transform.toposort(X)
        compiler.transform.name_graph(graph)
        _, _, workspace = compiler.transform.memory_planning(graph)
        self.assertEqual(workspace.shared_size, shared_workspace_expected_size)
        self.assertEqual(workspace.unique_size, unique_workspace_expected_size)


if __name__ == "__main__":
    unittest.main()
