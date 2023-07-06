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
from typing import List, Sequence, Union

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    gen_input_tensor,
    get_random_torch_tensor,
    graph_has_op,
)


class TestRemoveNoOpSplits(unittest.TestCase):
    """
    Tests _remove_no_op_splits() in remove_no_ops.py
    """

    def test_remove_no_op_split(self):
        """
        Test cases:
        0. No-op split with split_size_or_sections as integer
        1. No-op split with split_size_or_sections as a singleton list
        2. No-op split with split_size > length along split_dim
        3. No-op split with split_dim = -1
        4. Meaningful split
        5. Meaningful split with split_dim = -1
        6. No-op split is a model output
        7. Meaningful split is a model output
        """

        test_cases = (
            # Split is a no-op.
            {
                "split_input_shape": (5,),
                "split_size_or_sections": 5,
                "split_dim": 0,
                "split_is_output": False,
                "should_remove_no_op_split": True,
                "test_name": "test_remove_no_op_split_no_op_0",
            },
            {
                "split_input_shape": (5,),
                "split_size_or_sections": [5],
                "split_dim": -1,
                "split_is_output": False,
                "should_remove_no_op_split": True,
                "test_name": "test_remove_no_op_split_no_op_1",
            },
            {
                "split_input_shape": (2, 3, 4),
                "split_size_or_sections": 10,  # split_size > length along dim=1
                "split_dim": 1,
                "split_is_output": False,
                "should_remove_no_op_split": True,
                "test_name": "test_remove_no_op_split_no_op_2",
            },
            {
                "split_input_shape": (2, 3, 4, 5),
                "split_size_or_sections": [5],
                "split_dim": -1,
                "split_is_output": False,
                "should_remove_no_op_split": True,
                "test_name": "test_remove_no_op_split_no_op_3",
            },
            # Split is meaningful.
            {
                "split_input_shape": (7,),
                "split_size_or_sections": 2,
                "split_dim": 0,
                "split_is_output": False,
                "should_remove_no_op_split": False,
                "test_name": "test_remove_no_op_split_meaningful_4",
            },
            {
                "split_input_shape": (2, 3, 4, 5),
                "split_size_or_sections": [2, 1, 2],
                "split_dim": -1,
                "split_is_output": False,
                "should_remove_no_op_split": False,
                "test_name": "test_remove_no_op_split_meaningful_5",
            },
            # Split is a model output.
            {
                "split_input_shape": (9,),
                "split_size_or_sections": [9],
                "split_dim": 0,
                "split_is_output": True,
                "should_remove_no_op_split": False,
                "test_name": "test_remove_no_op_split_output_6",
            },
            {
                "split_input_shape": (1, 9),
                "split_size_or_sections": [4, 5],
                "split_dim": -1,
                "split_is_output": True,
                "should_remove_no_op_split": False,
                "test_name": "test_remove_no_op_split_output_7",
            },
        )

        for i, test_kwargs in enumerate(test_cases):
            with self.subTest(test_no=i):
                self._test_remove_no_op_split_impl(**test_kwargs)

    def _test_remove_no_op_split_impl(
        self,
        split_input_shape: Sequence[int],
        split_size_or_sections: Union[int, List[int]],
        split_dim: int,
        split_is_output: bool,
        should_remove_no_op_split: bool,
        test_name: str,
    ):
        # Define model graph.
        X = gen_input_tensor(shape=split_input_shape, name="input_0")
        c = gen_input_tensor(shape=(1,), name="input_1")
        Zs = ops.split()(X, split_size_or_sections, split_dim)

        model_outputs = []
        for i, Z in enumerate(Zs):
            out = Z if split_is_output else Z + c
            out._attrs["name"] = f"output_{i}"
            out._attrs["is_output"] = True
            model_outputs.append(out)

        # Run PyTorch.
        X_pt = get_random_torch_tensor(shape=split_input_shape)
        c_pt = get_random_torch_tensor(shape=(1,))
        Zs_pt = torch.split(X_pt, split_size_or_sections, split_dim)
        outputs_pt = Zs_pt if split_is_output else [Z_pt + c_pt for Z_pt in Zs_pt]

        # Run AIT.
        with compile_model(
            model_outputs, detect_target(), "./tmp", test_name
        ) as module:
            inputs_pt = (
                {"input_0": X_pt}
                if split_is_output
                else {"input_0": X_pt, "input_1": c_pt}
            )
            outputs_ait = {
                f"output_{i}": torch.empty_like(out_pt)
                for (i, out_pt) in enumerate(outputs_pt)
            }
            module.run_with_tensors(inputs_pt, outputs_ait)

            self.assertNotEqual(
                graph_has_op(module.debug_sorted_graph, "split"),
                should_remove_no_op_split,
            )
            for out_pt, out_ait in zip(outputs_pt, outputs_ait.values()):
                self.assertTrue(torch.allclose(out_pt, out_ait, atol=1e-2, rtol=1e-3))
