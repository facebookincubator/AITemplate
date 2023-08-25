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
from aitemplate.compiler.base import IntVar, Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.transform.fuse_utils import is_elementwise_type
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import gen_input_tensor, get_random_torch_tensor
from aitemplate.utils.graph_utils import get_sorted_ops


class TestFuseDuplicateFusedElementwise(unittest.TestCase):
    """
    This tests the compiler's behavior when fusing duplicate fused-elementwise ops.
    See fuse_duplicate_fused_elementwise.

    We test the following test cases:
    1. Test duplicates
    2. Test duplicates with memory ops
    3. Test non-duplicates
    4. Test all interactions
    5. Test same input accessors
    6. Test different input accessors
    """

    SHAPE = [32, 64, 100]

    @staticmethod
    def _count_fused_elementwise_ops(
        graph: List[Tensor], target_elementwise_ops: List[FuncEnum]
    ) -> int:
        fused_elementwise_ops = filter(
            lambda op: op._attrs["op"] == "fused_elementwise", get_sorted_ops(graph)
        )

        count = 0
        for op in fused_elementwise_ops:
            elementwise_ops = op._attrs["elementwise_ops"]
            if len(target_elementwise_ops) != len(elementwise_ops):
                continue
            if all(
                is_elementwise_type(op, target)
                for op, target in zip(elementwise_ops, target_elementwise_ops)
            ):
                count += 1
        return count

    def test_fuse_duplicates(self):
        """When the input and elementwise ops are the same."""
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        sigmoid1 = ops.elementwise(FuncEnum.SIGMOID)(x)
        sigmoid2 = ops.elementwise(FuncEnum.SIGMOID)(x)
        softmax1 = ops.softmax()(sigmoid1, dim=0)
        softmax2 = ops.softmax()(sigmoid2, dim=0)
        model_output = softmax1 + softmax2
        model_output._attrs["is_output"] = True
        model_output._attrs["name"] = "output"

        x_pt = get_random_torch_tensor(self.SHAPE)
        sigmoid1_pt = torch.sigmoid(x_pt)
        sigmoid2_pt = torch.sigmoid(x_pt)
        softmax1_pt = torch.nn.functional.softmax(sigmoid1_pt, dim=0)
        softmax2_pt = torch.nn.functional.softmax(sigmoid2_pt, dim=0)
        y_pt = softmax1_pt + softmax2_pt
        y_ait = torch.empty_like(y_pt)

        with compile_model(
            model_output,
            detect_target(),
            "/tmp",
            "fuse_duplicate_fused_elementwise_dups",
        ) as module:
            module.run_with_tensors({"input_x": x_pt}, {"output": y_ait})
            nsigmoid = self._count_fused_elementwise_ops(
                module.debug_sorted_graph, [FuncEnum.SIGMOID]
            )
            self.assertEqual(nsigmoid, 1)
            self.assertTrue(torch.allclose(y_pt, y_ait, atol=1e-2, rtol=1e-2))

    def test_fuse_duplicates_with_concat_output_accessor(self):
        """Fused_elementwise ops' that have the same input and elementwise ops
        and they have output accessors that write to the same concat output."""
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        sigmoid1 = ops.elementwise(FuncEnum.SIGMOID)(x)
        sigmoid2 = ops.elementwise(FuncEnum.SIGMOID)(x)
        model_output = ops.concatenate()([sigmoid1, sigmoid2])
        model_output._attrs["is_output"] = True
        model_output._attrs["name"] = "output"

        x_pt = get_random_torch_tensor(self.SHAPE)
        sigmoid1_pt = torch.sigmoid(x_pt)
        sigmoid2_pt = torch.sigmoid(x_pt)
        y_pt = torch.concat([sigmoid1_pt, sigmoid2_pt])
        y_ait = torch.empty_like(y_pt)

        with compile_model(
            model_output,
            detect_target(),
            "/tmp",
            "fuse_duplicate_fused_elementwise_dups_with_accessors",
        ) as module:
            module.run_with_tensors({"input_x": x_pt}, {"output": y_ait})
            nsigmoid = self._count_fused_elementwise_ops(
                module.debug_sorted_graph, [FuncEnum.SIGMOID]
            )
            self.assertEqual(nsigmoid, 1)
            self.assertTrue(torch.allclose(y_pt, y_ait, atol=1e-2, rtol=1e-2))

    def test_dont_fuse_non_duplicates(self):
        """Fused-elementwise ops that have different inputs or different
        elementwise-ops aren't fused together.
        """
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        z = gen_input_tensor(shape=self.SHAPE, name="input_z")
        relu_x = ops.elementwise(FuncEnum.RELU)(x)
        gelu_x = ops.elementwise(FuncEnum.GELU)(x)
        gelu_z = ops.elementwise(FuncEnum.GELU)(z)
        softmax1 = ops.softmax()(relu_x, dim=0)
        softmax2 = ops.softmax()(gelu_x, dim=0)
        softmax3 = ops.softmax()(gelu_z, dim=0)
        model_output = softmax1 + softmax2 + softmax3
        model_output._attrs["is_output"] = True
        model_output._attrs["name"] = "output"

        x_pt = get_random_torch_tensor(self.SHAPE)
        z_pt = get_random_torch_tensor(self.SHAPE)
        relu_x_pt = torch.nn.functional.relu(x_pt)
        gelu_x_pt = torch.nn.functional.gelu(x_pt)
        gelu_z_pt = torch.nn.functional.gelu(z_pt)
        softmax1_pt = torch.nn.functional.softmax(relu_x_pt, dim=0)
        softmax2_pt = torch.nn.functional.softmax(gelu_x_pt, dim=0)
        softmax3_pt = torch.nn.functional.softmax(gelu_z_pt, dim=0)

        y_pt = softmax1_pt + softmax2_pt + softmax3_pt
        y_ait = torch.empty_like(y_pt)

        with compile_model(
            model_output,
            detect_target(),
            "/tmp",
            "fuse_duplicate_fused_elementwise_non_dups",
        ) as module:
            module.run_with_tensors(
                {"input_x": x_pt, "input_z": z_pt}, {"output": y_ait}
            )
            graph = module.debug_sorted_graph
            nrelu = self._count_fused_elementwise_ops(graph, [FuncEnum.RELU])
            ngelu = self._count_fused_elementwise_ops(graph, [FuncEnum.GELU])
            self.assertEqual(nrelu, 1)
            self.assertEqual(ngelu, 2)
            self.assertTrue(torch.allclose(y_pt, y_ait, atol=1e-2, rtol=1e-2))

    def test_all_interactions(self):
        """Test all interactions:
        1. Fusing duplicates
        2. Fusing duplicates with accessors that write to a concat's output tensor
        3. Avoid fusing non-duplicates
        """
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        z = gen_input_tensor(shape=self.SHAPE, name="input_z")
        p = gen_input_tensor(shape=self.SHAPE, name="input_p")

        # First ReLU op with x as the input.
        relu1 = ops.elementwise(FuncEnum.RELU)(x)
        tanh = ops.elementwise(FuncEnum.TANH)(relu1)
        concat1 = ops.concatenate()([relu1, tanh])

        # Fuse relu2 with relu1. This ReLU uses a tensor accessor to write
        # directly to concat2's output.
        relu2 = ops.elementwise(FuncEnum.RELU)(x)
        concat2 = ops.concatenate()([relu2, p])

        # Fuse relu3 with relu1.
        relu3 = ops.elementwise(FuncEnum.RELU)(x)
        softmax = ops.softmax()(relu3, dim=0)
        concat3 = ops.concatenate()([softmax, softmax])

        # Don't fuse operators with different input or elementwise-ops.
        gelu = ops.elementwise(FuncEnum.GELU)(x)
        relu4 = ops.elementwise(FuncEnum.RELU)(z)
        concat4 = ops.concatenate()([relu4, gelu])

        model_output = concat1 + concat2 + concat3 + concat4
        model_output._attrs["is_output"] = True
        model_output._attrs["name"] = "output"

        # Setup PyTorch
        x_pt = get_random_torch_tensor(self.SHAPE)
        z_pt = get_random_torch_tensor(self.SHAPE)
        p_pt = get_random_torch_tensor(self.SHAPE)

        relu1_pt = torch.nn.functional.relu(x_pt)
        tanh_pt = torch.nn.functional.tanh(relu1_pt)
        concat1_pt = torch.concat([relu1_pt, tanh_pt])

        relu2_pt = torch.nn.functional.relu(x_pt)
        concat2_pt = torch.concat([relu2_pt, p_pt])

        relu3_pt = torch.nn.functional.relu(x_pt)
        softmax_pt = torch.nn.functional.softmax(relu3_pt, dim=0)
        concat3_pt = torch.concat([softmax_pt, softmax_pt])

        relu4_pt = torch.nn.functional.relu(z_pt)
        gelu_pt = torch.nn.functional.gelu(x_pt)
        concat4_pt = torch.concat([relu4_pt, gelu_pt])

        y_pt = concat1_pt + concat2_pt + concat3_pt + concat4_pt
        y_ait = torch.empty_like(y_pt)

        with compile_model(
            model_output,
            detect_target(),
            "/tmp",
            "fuse_duplicate_fused_elementwise_all_interactions",
        ) as module:
            module.run_with_tensors(
                inputs={
                    "input_x": x_pt,
                    "input_z": z_pt,
                    "input_p": p_pt,
                },
                outputs={"output": y_ait},
            )
            graph = module.debug_sorted_graph
            nrelu = self._count_fused_elementwise_ops(graph, [FuncEnum.RELU])
            ngelu = self._count_fused_elementwise_ops(graph, [FuncEnum.GELU])
            self.assertEqual(nrelu, 2)
            self.assertEqual(ngelu, 1)
            self.assertTrue(torch.allclose(y_pt, y_ait, atol=1e-2, rtol=1e-2))

    def test_same_and_different_input_accessors(self):
        """
        Before _fuse_slice_and_strided_op the fused_elementwise ops have different
        input tensors. After _fuse_slice_and_strided_op, the fused_elementwise
        ops have the same input tensor and depending on the slice indices, the
        same or different input accessor.
        """

        # Input accessors are the same -- fuse them!
        self._test_input_accessors_impl(
            slice1_start=[0, 0, 0],
            slice1_end=[32, 64, 50],
            slice2_start=[0, 0, 0],
            slice2_end=[32, 64, 50],
            should_fuse=True,
        )
        # Input accessors are different -- don't fuse.
        self._test_input_accessors_impl(
            slice1_start=[0, 0, 0],
            slice1_end=[32, 64, 50],
            slice2_start=[0, 0, 50],
            slice2_end=[32, 64, 100],
            should_fuse=False,
        )

    def _test_input_accessors_impl(
        self,
        slice1_start: List[IntVar],
        slice1_end: List[IntVar],
        slice2_start: List[IntVar],
        slice2_end: List[IntVar],
        should_fuse: bool,
    ):
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        x_sliced_1 = ops.dynamic_slice()(x, slice1_start, slice1_end)
        x_sliced_2 = ops.dynamic_slice()(x, slice2_start, slice2_end)
        sigmoid1 = ops.elementwise(FuncEnum.SIGMOID)(x_sliced_1)
        sigmoid2 = ops.elementwise(FuncEnum.SIGMOID)(x_sliced_2)
        softmax1 = ops.softmax()(sigmoid1, dim=0)
        softmax2 = ops.softmax()(sigmoid2, dim=0)
        model_output = softmax1 + softmax2
        model_output._attrs["is_output"] = True
        model_output._attrs["name"] = "output"

        x_pt = get_random_torch_tensor(self.SHAPE)
        x_sliced_1_pt = x_pt[[slice(s, e) for s, e in zip(slice1_start, slice1_end)]]
        x_sliced_2_pt = x_pt[[slice(s, e) for s, e in zip(slice2_start, slice2_end)]]
        sigmoid1_pt = torch.sigmoid(x_sliced_1_pt)
        sigmoid2_pt = torch.sigmoid(x_sliced_2_pt)
        softmax1_pt = torch.nn.functional.softmax(sigmoid1_pt, dim=0)
        softmax2_pt = torch.nn.functional.softmax(sigmoid2_pt, dim=0)
        y_pt = softmax1_pt + softmax2_pt
        y_ait = torch.empty_like(y_pt)

        with compile_model(
            model_output,
            detect_target(),
            "/tmp",
            "fuse_duplicate_fused_elementwise_same_input_different_input_accessors",
        ) as module:
            module.run_with_tensors({"input_x": x_pt}, {"output": y_ait})
            nsigmoid = self._count_fused_elementwise_ops(
                module.debug_sorted_graph, [FuncEnum.SIGMOID]
            )
            self.assertEqual(nsigmoid, 1 if should_fuse else 2)
            self.assertTrue(torch.allclose(y_pt, y_ait, atol=1e-2, rtol=1e-2))
