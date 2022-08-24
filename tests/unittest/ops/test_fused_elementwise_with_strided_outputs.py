# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Unittests for fused_elementwise Operator with strided outputs.
"""
import unittest

from typing import List

import torch

from aitemplate.compiler import ops, transform
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import shape_utils


class FusedElementwiseWithStridedOutputsTestCase(unittest.TestCase):
    def _fused_elementwise_e2e_helper(
        self, batch0_sizes: List[int], batch1_sizes: List[int], m1: int, m2: int, k: int
    ):
        # Construct one graph with 2 fused_elementwises + 1 cat.
        batch0_dim = shape_utils.gen_int_var_min_max(batch0_sizes, "batch0")
        batch1_dim = shape_utils.gen_int_var_min_max(batch1_sizes, "batch1")

        X1 = Tensor(
            shape=[
                batch0_dim,
                batch1_dim,
                IntImm(m1),
                IntImm(k),
            ],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype="float16",
            name="X2",
            value=3.0,
        )
        X3 = Tensor(
            shape=[
                batch0_dim,
                batch1_dim,
                IntImm(m2),
                IntImm(k),
            ],
            dtype="float16",
            name="input1",
            is_input=True,
        )

        X4 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        X5 = ops.elementwise(FuncEnum.TANH)(X4)
        X6 = ops.elementwise(FuncEnum.TANH)(X3)
        X7 = ops.concatenate()([X5, X6], dim=2)
        X7._attrs["name"] = "output0"
        X7._attrs["is_output"] = True

        # Trigger fuse_op() pass.
        target = detect_target()
        graph = transform.toposort(X7)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        transform.refine_graph(graph)
        graph = transform.fuse_ops(graph)

        # Manually set fused_elementwise output_accessors field.
        cat_op = list(X7._attrs["src_ops"])[0]
        cat_inputs = cat_op._attrs["inputs"]
        fe1 = list(cat_inputs[0]._attrs["src_ops"])[0]
        fe2 = list(cat_inputs[1]._attrs["src_ops"])[0]
        cat_inputs[0]._attrs["dst_ops"] = set()
        cat_inputs[1]._attrs["dst_ops"] = set()
        fe1._attrs["outputs"] = [X7]
        fe2._attrs["outputs"] = [X7]
        fe1._attrs["output_accessors"][0].update_base_tensor(
            X7, stride_dim=2, stride_dim_offset=0
        )
        fe2._attrs["output_accessors"][0].update_base_tensor(
            X7, stride_dim=2, stride_dim_offset=m1
        )
        X7._attrs["src_ops"] = {fe1, fe2}

        # Gen module.
        module = gen_execution_module(
            [X7],
            target,
            "./tmp",
            "fused_elementwise_with_strided_outputs_m1_{}_m2_{}_k_{}".format(m1, m2, k),
        )

        for batch0_size in batch0_sizes:
            for batch1_size in batch1_sizes:
                # Run PyTorch baseline.
                x1_pt = torch.randn(batch0_size, batch1_size, m1, k).cuda().half()
                x3_pt = torch.randn(batch0_size, batch1_size, m2, k).cuda().half()
                x5_pt = torch.tanh(x1_pt + 3.0)
                x6_pt = torch.tanh(x3_pt)
                x7_pt = torch.cat([x5_pt, x6_pt], dim=2)

                # Run AITemplate module.
                inputs = [0, 0]
                name_to_index_map = module.GetInputNameToIndexMap()
                inputs[name_to_index_map["input0"]] = x1_pt
                inputs[name_to_index_map["input1"]] = x3_pt

                x7 = torch.empty([batch0_size, batch1_size, m1 + m2, k]).cuda().half()
                module.RunWithTensors(inputs, [x7])
                # Do comparisons.
                self.assertTrue(torch.allclose(x7, x7_pt, atol=1e-2, rtol=1e-2))

    def test_all_aligned(self):
        self._fused_elementwise_e2e_helper(
            batch0_sizes=[1], batch1_sizes=[2, 4, 5], m1=8, m2=8, k=1
        )
        self._fused_elementwise_e2e_helper(
            batch0_sizes=[1, 99, 1024], batch1_sizes=[8], m1=8, m2=16, k=1
        )
        self._fused_elementwise_e2e_helper(
            batch0_sizes=[3, 5, 1024], batch1_sizes=[2, 5], m1=4, m2=4, k=2
        )
        self._fused_elementwise_e2e_helper(
            batch0_sizes=[1024], batch1_sizes=[2], m1=4, m2=2, k=4
        )
        self._fused_elementwise_e2e_helper(
            batch0_sizes=[1024], batch1_sizes=[2], m1=16, m2=64, k=32
        )

    def test_not_aligned(self):
        self._fused_elementwise_e2e_helper(
            batch0_sizes=[8], batch1_sizes=[23, 88, 100], m1=1, m2=1, k=1
        )
        self._fused_elementwise_e2e_helper(
            batch0_sizes=[88, 100, 234], batch1_sizes=[40], m1=4, m2=2, k=1
        )
        self._fused_elementwise_e2e_helper(
            batch0_sizes=[23, 56, 93], batch1_sizes=[12, 34, 55], m1=1, m2=2, k=2
        )
        self._fused_elementwise_e2e_helper(
            batch0_sizes=[2], batch1_sizes=[1024], m1=8, m2=2, k=1
        )


if __name__ == "__main__":
    unittest.main()
