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
from aitemplate.frontend import Tensor

from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class SlicePermute021FusionTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SlicePermute021FusionTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_slice_permute021_fusion(
        self,
        N,
        K,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        dims,
        test_name,
        dtype="float16",
    ):
        X = Tensor(
            shape=slice_input_shape,
            dtype=dtype,
            name="input_x",
            is_input=True,
        )
        slice_op = ops.dynamic_slice()
        tensor_A = slice_op(
            X, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        tensor_A._attrs["name"] = "slice_output"

        permute_op = ops.permute021()
        Y = permute_op(tensor_A)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        with compile_model(
            Y,
            target,
            "./tmp",
            f"{test_name}_{self._test_id}",
            dll_name=f"test_{self._test_id}.so",
        ) as module:
            self._test_id += 1

            # Verify the generated graph.
            sorted_graph = module.debug_sorted_graph
            self.assertEqual(len(sorted_graph), 2)
            sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
            self.assertEqual(len(sorted_ops), 1)

            # Run PyTorch
            input_pt = get_random_torch_tensor(slice_input_shape, dtype)

            slice_indices = [
                slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
            ]
            a_pt = input_pt[slice_indices]
            y_pt = torch.permute(a_pt, dims)

            # Run AITemplate module.
            y = get_torch_empty_tensor(y_pt.size(), dtype)
            module.run_with_tensors([input_pt], [y])
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    def test_slice_permute021_fusion(self):
        self._test_slice_permute021_fusion(
            N=2,
            K=2,
            slice_input_shape=(2, 2, 8),
            slice_start_indices=(0, 0, 4),
            slice_end_indices=(2, 2, 8),
            dims=(0, 2, 1),
            test_name="slice_permute021",
            dtype="float16",
        )
        self._test_slice_permute021_fusion(
            N=2,
            K=2,
            slice_input_shape=(2, 2, 8),
            slice_start_indices=(0, 1, 0),
            slice_end_indices=(2, 3, 8),
            dims=(0, 2, 1),
            test_name="slice_permute021",
            dtype="float16",
        )
        self._test_slice_permute021_fusion(
            N=2,
            K=2,
            slice_input_shape=[2, 9, 4],
            slice_start_indices=[0, 0, 1],
            slice_end_indices=[None, None, 3],
            dims=(0, 2, 1),
            test_name="slice_permute021",
            dtype="float16",
        )
        self._test_slice_permute021_fusion(
            N=2,
            K=2,
            slice_input_shape=[120, 1211, 1200],
            slice_start_indices=[0, 0, 3],
            slice_end_indices=[None, None, 1100],
            dims=(0, 2, 1),
            test_name="slice_permute021",
            dtype="float16",
        )
        self._test_slice_permute021_fusion(
            N=2,
            K=2,
            slice_input_shape=[123, 1211, 1200],
            slice_start_indices=[0, 5, 0],
            slice_end_indices=[None, 1200, None],
            dims=(0, 2, 1),
            test_name="slice_permute021",
            dtype="float16",
        )
        self._test_slice_permute021_fusion(
            N=2,
            K=2,
            slice_input_shape=(2, 3, 8, 62),
            slice_start_indices=(0, 0, 0, 2),
            slice_end_indices=(2, 3, 8, 50),
            dims=(0, 1, 3, 2),
            test_name="slice_permute021",
            dtype="float16",
        )
        self._test_slice_permute021_fusion(
            N=2,
            K=2,
            slice_input_shape=(2, 3, 4, 4, 8),
            slice_start_indices=(0, 0, 0, 0, 0),
            slice_end_indices=(2, 3, 4, 4, 2),
            dims=(0, 1, 2, 4, 3),
            test_name="slice_permute021",
            dtype="float16",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
