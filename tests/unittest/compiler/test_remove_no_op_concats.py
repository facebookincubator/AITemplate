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
from typing import Sequence

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor, graph_has_op


class TestRemoveNoOpConcats(unittest.TestCase):
    """
    Tests the compiler's behavior of removing no-op concats.

    NOTE: Whenever we include an empty input tensor, the non-empty input tensor
    must be rank 1. That's because AIT's concat expects all its inputs to have
    the same rank and have matching dimension sizes except along the
    concatenating dimension.

    We run the following tests:
    # These are no-ops
    1. inputs=[non-empty]
    2. inputs=[rank-1 empty, rank-1 non-empty, rank-1 empty]
    3. inputs=[empty]
    4. inputs=[empty, empty]

    # These are meaningful
    5. inputs=[non-empty, non-empty]
    6. inputs=[non-empty, empty, non-empty]

    # These should have exceptions
    7. inputs=[rank-2 non-empty, rank-1 empty]
    8. inputs=[rank-2 non-empty, rank-2 empty]
    """

    def test_remove_no_op_concats_no_ops(self):
        self._test_remove_no_op_concats_impl(
            input_shapes=[[2, 4, 6]],
            should_keep_concat=False,
            test_name="test_remove_no_op_concats_single_non_empty",
        )

        self._test_remove_no_op_concats_impl(
            input_shapes=[[0], [3], [0]],
            should_keep_concat=False,
            test_name="test_remove_no_op_concats_single_non_empty_and_double_empty",
        )

    def test_remove_no_op_concats_no_ops_all_empty(self):
        """Below we test when all the input tensors are empty. fx2ait will fail
        in these cases. However, it's possible to create it directly in AIT.
        Therefore, we test this case and treat it as a no-op.
        """
        self._test_remove_no_op_concats_impl(
            input_shapes=[[0, 0, 0]],
            should_keep_concat=False,
            test_name="test_remove_no_op_concats_single_empty",
        )

        self._test_remove_no_op_concats_impl(
            input_shapes=[[0, 0, 0], [0, 0, 0]],
            should_keep_concat=False,
            test_name="test_remove_no_op_concats_double_empty",
        )

    def test_remove_no_op_concats_meaningful(self):
        self._test_remove_no_op_concats_impl(
            input_shapes=[[3, 5], [3, 5]],
            should_keep_concat=True,
            test_name="test_remove_no_op_concats_double_non_empty",
        )

        self._test_remove_no_op_concats_impl(
            input_shapes=[[3], [0], [5]],
            should_keep_concat=True,
            test_name="test_remove_no_op_concats_two_non_empty_and_empty",
        )

    def test_remove_no_op_concats_exceptions(self):
        """We expect this to raise an exception in these test cases."""

        # AIT expects all concat inputs to have the same rank.
        with self.assertRaises(RuntimeError):
            self._test_remove_no_op_concats_impl(
                input_shapes=[[2, 4], [0]],
                should_keep_concat=False,
                test_name="test_remove_no_op_concats_same_rank",
            )

        # AIT expects all concat inputs to have the same dimension sizes except for the concat_dim.
        with self.assertRaises(RuntimeError):
            self._test_remove_no_op_concats_impl(
                input_shapes=[[2, 4], [0, 0]],
                should_keep_concat=False,
                test_name="test_remove_no_ops_concat_same_dim_sizes",
            )

    def _test_remove_no_op_concats_impl(
        self,
        input_shapes: Sequence[Sequence[int]],
        should_keep_concat: bool,
        test_name: str,
    ):
        inputs = [
            Tensor(shape=shape, name=f"input_{i}", is_input=True)
            for i, shape in enumerate(input_shapes)
        ]
        concatenated = ops.concatenate()(inputs)
        c = Tensor(shape=[1], name="input_const", is_input=True)
        model_output = (concatenated * c) + (concatenated / c)
        model_output._attrs["name"] = "output_0"
        model_output._attrs["is_output"] = True

        inputs_pt = {
            f"input_{i}": get_random_torch_tensor(shape=shape)
            for i, shape in enumerate(input_shapes)
        }
        concatenated_pt = torch.concat(list(inputs_pt.values()))
        c_pt = get_random_torch_tensor(shape=[1])
        Y_pt = (concatenated_pt * c_pt) + (concatenated_pt / c_pt)
        Y_ait = torch.empty_like(Y_pt)

        with compile_model(model_output, detect_target(), "./tmp", test_name) as module:
            module.run_with_tensors(
                {**inputs_pt, "input_const": c_pt}, {"output_0": Y_ait}
            )

            self.assertEquals(
                graph_has_op(module.debug_sorted_graph, "concatenate"),
                should_keep_concat,
            )
            self.assertTrue(torch.allclose(Y_pt, Y_ait, atol=1e-2, rtol=1e-2))
