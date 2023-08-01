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
from typing import List

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.tensor.dynamic_slice import MAX_INT32
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    gen_input_tensor,
    get_random_torch_tensor,
    graph_has_op,
)


class TestRemoveNoOpDynamicSlices(unittest.TestCase):
    """
    Tests the compiler's behavior when removing no-op dynamic slices.
    """

    def test_remove_no_op_dynamic_slices(self):
        TEST_CASES = (
            # These are no-ops.
            {
                # X[:]
                "input_shape": [100],
                "start_indices": [None],
                "end_indices": [None],
                "should_keep_dynamic_slice": False,
            },
            {
                # X[0:]
                "input_shape": [100],
                "start_indices": [0],
                "end_indices": [None],
                "should_keep_dynamic_slice": False,
            },
            {
                # X[:2_147_483_647, ]
                "input_shape": [100, 100],
                "start_indices": [None, 0],
                "end_indices": [MAX_INT32, None],
                "should_keep_dynamic_slice": False,
            },
            # These are meaningful.
            {
                # X[-7:-7]
                "input_shape": [10],
                "start_indices": [-7],
                "end_indices": [-7],
                "should_keep_dynamic_slice": True,
            },
            {
                # X[7:, -7:, 0:]
                "input_shape": [10, 10, 10],
                "start_indices": [7, -7, 0],
                "end_indices": [None, None, None],
                "should_keep_dynamic_slice": True,
            },
            {
                # X[:7, :-7, :0]
                "input_shape": [10, 10, 10],
                "start_indices": [None, None, None],
                "end_indices": [7, -7, 0],
                "should_keep_dynamic_slice": True,
            },
            {
                # X[0:7, 0:-7]
                "input_shape": [10, 10],
                "start_indices": [0, 0],
                "end_indices": [7, -7],
                "should_keep_dynamic_slice": True,
            },
            {
                # X[-7:7, 7:-7]
                "input_shape": [10, 10],
                "start_indices": [-7, 7],
                "end_indices": [7, -7],
                "should_keep_dynamic_slice": True,
            },
            {
                # X[-7:7, 7:-7, :]
                "input_shape": [10, 10, 10],
                "start_indices": [-7, 7, None],
                "end_indices": [7, -7, None],
                "should_keep_dynamic_slice": True,
            },
        )

        for i, test_kwargs in enumerate(TEST_CASES):
            start_indices = ",".join(map(str, test_kwargs["start_indices"]))
            end_indices = ",".join(map(str, test_kwargs["end_indices"]))

            with self.subTest(
                start=start_indices,
                end=end_indices,
                keep=test_kwargs["should_keep_dynamic_slice"],
            ):
                self._test_remove_no_op_dynamic_slices_impl(
                    **test_kwargs,
                    test_name=f"test_remove_no_op_dynamic_slice_{i}",
                )

    def _test_remove_no_op_dynamic_slices_impl(
        self,
        input_shape: List[int],
        start_indices: List[int],
        end_indices: List[int],
        should_keep_dynamic_slice: bool,
        test_name: str,
    ):
        X = gen_input_tensor(shape=input_shape, name="input_0")
        X_sliced = ops.dynamic_slice()(X, start_indices, end_indices)
        c = gen_input_tensor(shape=[1], name="input_const")
        model_output = (X_sliced * c) + (X_sliced / c)
        model_output._attrs["name"] = "output_0"
        model_output._attrs["is_output"] = True

        X_pt = get_random_torch_tensor(shape=input_shape)
        slices = [slice(s, e) for s, e in zip(start_indices, end_indices)]
        X_sliced_pt = X_pt[slices]
        c_pt = get_random_torch_tensor(shape=[1])
        Y_pt = (X_sliced_pt * c_pt) + (X_sliced_pt / c_pt)
        Y_ait = torch.empty_like(Y_pt)

        # NOTE: We don't run every optimization pass to avoid fusion between
        # dynamic_slice and elementwise.
        with compile_model(
            model_output, detect_target(), "/tmp", test_name, do_optimize_graph=False
        ) as module:
            module.run_with_tensors(
                {"input_0": X_pt, "input_const": c_pt}, {"output_0": Y_ait}
            )

            self.assertEqual(
                graph_has_op(module.debug_sorted_graph, "dynamic_slice"),
                should_keep_dynamic_slice,
            )
            self.assertTrue(torch.allclose(Y_pt, Y_ait, atol=1e-2, rtol=1e-3))
