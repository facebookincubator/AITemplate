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
from aitemplate.compiler.base import IntVar
from aitemplate.testing import detect_target, test_utils
from aitemplate.utils import graph_utils


class SliceViewStridedOpTestCase(unittest.TestCase):
    def test_slice_view_gemm_fusible(self):
        N = 4
        batch_dim = IntVar([1, 2, 3], "batch_size")

        input0 = test_utils.gen_input_tensor([batch_dim, 2 * N, N], name="input0")
        X0 = ops.dynamic_slice()(input0, [None, None, None], [None, N, None])
        X1 = ops.reshape()(X0, [-1, N * N])
        input1 = test_utils.gen_input_tensor([N, N * N], name="input1")
        Y = ops.gemm_rcr()(X1, input1)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model([Y], target, "./tmp", "slice_reshape_gemm_fusible")

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)

        # Prepare PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            # Run PyTorch baseline.
            input0_pt = torch.randn([batch_size, 2 * N, N]).cuda().half()
            x0_pt = input0_pt[:, :N, :]
            x1_pt = torch.reshape(x0_pt, [-1, N * N])
            input1_pt = torch.rand([N, N * N]).cuda().half()
            y_pt = torch.nn.functional.linear(x1_pt, input1_pt)
            y = torch.empty(y_pt.shape).cuda().half()

            # Run AITemplate module.
            module.run_with_tensors(
                {
                    "input0": input0_pt,
                    "input1": input1_pt,
                },
                [y],
            )

            # Do comparisons.
            self.assertTrue(
                torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2),
                f"batch_size: {batch_size}, y: {y}, y_pt: {y_pt}",
            )

    def test_slice_view_gemm_non_fusible(self):
        N = 4
        batch_dim = IntVar([1, 2, 3], "batch_size")

        input0 = test_utils.gen_input_tensor([batch_dim, N, 2 * N], name="input0")
        X0 = ops.dynamic_slice()(input0, [None, None, None], [None, None, N])
        X1 = ops.reshape()(X0, [-1, N * N])
        input1 = test_utils.gen_input_tensor([N, N * N], name="input1")
        Y = ops.gemm_rcr()(X1, input1)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model([Y], target, "./tmp", "slice_reshape_gemm_non_fusible")

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepare PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            # Run PyTorch baseline.
            input0_pt = torch.randn([batch_size, N, 2 * N]).cuda().half()
            x0_pt = input0_pt[:, :, :N]
            x1_pt = torch.reshape(x0_pt, [-1, N * N])
            input1_pt = torch.rand([N, N * N]).cuda().half()
            y_pt = torch.nn.functional.linear(x1_pt, input1_pt)
            y = torch.empty(y_pt.shape).cuda().half()

            # Run AITemplate module.
            module.run_with_tensors(
                {
                    "input0": input0_pt,
                    "input1": input1_pt,
                },
                [y],
            )

            # Do comparisons.
            self.assertTrue(
                torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2),
                f"batch_size: {batch_size}, y: {y}, y_pt: {y_pt}",
            )


if __name__ == "__main__":
    unittest.main()
