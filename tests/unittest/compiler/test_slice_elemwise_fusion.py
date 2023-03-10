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
import itertools
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils, shape_utils


class SliceElemwiseFusionTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceElemwiseFusionTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    # "read_types" attribute contains a list of tuples like
    # [("input0", "uint4"), ("input1", "half")]. This helper function returns
    # the list of the second elements, i.e. read_t types for all inputs.
    def _get_read_types(self, op):
        return list({t for _, t in op._attrs["read_types"]})

    def _test_slice_elemwise_fusion(
        self,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        test_name,
        expected_max_read_t,
        expected_op_t,
        expected_data_t,
        input_x2_shape=None,
        dtype="float16",
    ):
        X1 = Tensor(
            shape=slice_input_shape,
            dtype=dtype,
            name="input_x1",
            is_input=True,
        )
        slice_op = ops.dynamic_slice()
        slice_output = slice_op(
            X1, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        slice_output._attrs["name"] = "slice_output"
        if input_x2_shape is None:
            input_x2_shape = [d.value() for d in slice_output._attrs["shape"]]

        # second input of the elemwise op
        X2 = Tensor(
            shape=input_x2_shape,
            dtype=dtype,
            name="input_x2",
            is_input=True,
        )

        Y = ops.elementwise(FuncEnum.ADD)(slice_output, X2)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        # 2 inputs + 1 output
        self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        # import pdb; pdb.set_trace()
        self.assertEqual(sorted_ops[0]._attrs["max_read_t"], expected_max_read_t)
        self.assertEqual(self._get_read_types(sorted_ops[0]), [expected_max_read_t])
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        # Run PyTorch
        x1_pt = get_random_torch_tensor(slice_input_shape, dtype)
        x2_pt = get_random_torch_tensor(input_x2_shape, dtype)

        slice_indices = [
            slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
        ]
        slice_output_pt = x1_pt[slice_indices]
        y_pt = slice_output_pt + x2_pt

        # Run AITemplate module.
        inputs = {
            "input_x1": x1_pt,
            "input_x2": x2_pt,
        }
        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_slice_elemwise_fusion(self):
        self._test_slice_elemwise_fusion(
            slice_input_shape=(10,),
            slice_start_indices=(2,),
            slice_end_indices=(None,),
            test_name="slice_elemwise_fusion",
            expected_max_read_t="uint",
            expected_op_t="half2",
            expected_data_t="half",
        )
        self._test_slice_elemwise_fusion(
            slice_input_shape=(10, 20),
            slice_start_indices=(0, 3),
            slice_end_indices=(None, 8),
            test_name="slice_elemwise_fusion",
            expected_max_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )
        self._test_slice_elemwise_fusion(
            slice_input_shape=(10, 20, 30),
            slice_start_indices=(0, 3, 0),
            slice_end_indices=(None, 5, None),
            test_name="slice_elemwise_fusion",
            expected_max_read_t="uint",
            expected_op_t="half2",
            expected_data_t="half",
        )

    def test_slice_elemwise_fusion_broadcast(self):
        # slice_output broadcasts to input_x2_shape with the same dimensionality
        self._test_slice_elemwise_fusion(
            slice_input_shape=(10, 16),
            slice_start_indices=(2, 0),
            slice_end_indices=(3, None),
            test_name="slice_elemwise_fusion_broadcast",
            expected_max_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
            input_x2_shape=(4, 16),
        )

        self._test_slice_elemwise_fusion(
            slice_input_shape=(10, 10),
            slice_start_indices=(0, 3),
            slice_end_indices=(None, 4),
            test_name="slice_elemwise_fusion_broadcast",
            expected_max_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            input_x2_shape=(10, 3),
        )

        # mixed
        self._test_slice_elemwise_fusion(
            slice_input_shape=(1, 1, 10),
            slice_start_indices=(0, 0, 2),
            slice_end_indices=(None, None, 7),
            test_name="slice_elemwise_fusion_broadcast",
            expected_max_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            input_x2_shape=(10, 3, 1),
        )

        # slice_output broadcasts to input_x2_shape with different dimensionalities
        self._test_slice_elemwise_fusion(
            slice_input_shape=(10, 10),
            slice_start_indices=(0, 3),
            slice_end_indices=(None, 4),
            test_name="slice_elemwise_fusion_broadcast",
            expected_max_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            input_x2_shape=(4, 10, 3),
        )
        self._test_slice_elemwise_fusion(
            slice_input_shape=(10, 20, 10),
            slice_start_indices=(0, 0, 3),
            slice_end_indices=(None, None, 4),
            test_name="slice_elemwise_fusion_broadcast",
            expected_max_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            input_x2_shape=(20, 3),
        )

    def _test_slice_elemwise_fusion_dynamic(
        self,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        test_name,
        expected_max_read_t,
        expected_op_t,
        expected_data_t,
        input_x2_shape=None,
        dtype="float16",
    ):
        x_shape = [
            shape_utils.gen_int_var_min_max(d) if isinstance(d, list) else IntImm(d)
            for d in slice_input_shape
        ]
        X1 = Tensor(
            shape=x_shape,
            dtype=dtype,
            name="input_x1",
            is_input=True,
        )
        slice_op = ops.dynamic_slice()
        slice_output = slice_op(
            X1, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        slice_output._attrs["name"] = "slice_output"
        slice_output_shape = slice_output._attrs["shape"]
        if input_x2_shape is None:
            x2_shape = slice_output_shape
        else:
            # iterate from right to handle cases with different dimensionalities
            x2_shape = []
            for x2_d, s_d in itertools.zip_longest(
                reversed(input_x2_shape), reversed(slice_output_shape)
            ):
                if x2_d is None:
                    break
                if s_d is None:
                    x2_shape.append(
                        shape_utils.gen_int_var_min_max(x2_d)
                        if isinstance(x2_d, list)
                        else IntImm(x2_d)
                    )
                elif isinstance(x2_d, list):
                    # must re-use the dynamic dim names from slice_output_shape
                    x2_shape.append(s_d)
                else:
                    x2_shape.append(IntImm(x2_d))
            x2_shape.reverse()

        X2 = Tensor(
            shape=x2_shape,
            dtype=dtype,
            name="input_x2",
            is_input=True,
        )

        Y1 = ops.elementwise(FuncEnum.RELU)(X2)
        Y2 = ops.elementwise(FuncEnum.SUB)(Y1, X2)
        Y = ops.elementwise(FuncEnum.ADD)(slice_output, Y2)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        # 2 inputs + 1 output
        self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["max_read_t"], expected_max_read_t)
        self.assertEqual(self._get_read_types(sorted_ops[0]), [expected_max_read_t])
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        for d in slice_input_shape:
            if isinstance(d, list):
                Ms = d
                break
        assert Ms is not None, "expected to have at least one dynamic dim"
        for idx in range(len(Ms)):
            # Run PyTorch
            x_shape_pt = [
                d[idx] if isinstance(d, list) else d for d in slice_input_shape
            ]
            if input_x2_shape is None:
                input_x2_shape_pt = [
                    d0[idx] if isinstance(d0, list) else d1.value()
                    for (d0, d1) in zip(slice_input_shape, slice_output_shape)
                ]
            else:
                input_x2_shape_pt = [
                    d[idx] if isinstance(d, list) else d for d in input_x2_shape
                ]
            x1_pt = get_random_torch_tensor(x_shape_pt, dtype)
            x2_pt = get_random_torch_tensor(input_x2_shape_pt, dtype)

            slice_indices = [
                slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
            ]
            slice_output_pt = x1_pt[slice_indices]
            y1_pt = torch.relu(x2_pt)
            y2_pt = y1_pt - x2_pt
            y_pt = slice_output_pt + y2_pt

            # Run AITemplate module.
            inputs = {
                "input_x1": x1_pt,
                "input_x2": x2_pt,
            }
            y = get_torch_empty_tensor(y_pt.size(), dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
            self.test_count += 1

    def test_slice_elemwise_fusion_dynamic(self):
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], 10),
            slice_start_indices=(0, 3),
            slice_end_indices=(None, 7),
            test_name="slice_elemwise_fusion_dynamic",
            expected_max_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], [4, 10], 16),
            slice_start_indices=(0, 0, 4),
            slice_end_indices=(None, None, 16),
            test_name="slice_elemwise_fusion_dynamic",
            expected_max_read_t="uint2",
            expected_op_t="half2",
            expected_data_t="half",
        )
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], [4, 10], 20, 16),
            slice_start_indices=(0, 0, 7, 0),
            slice_end_indices=(None, None, 10, None),
            test_name="slice_elemwise_fusion_dynamic",
            expected_max_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
        )

    def test_slice_elemwise_fusion_dynamic_broadcast(self):
        # slice_output broadcasts to input_x2
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], 8, 16),
            slice_start_indices=(0, 4, 0),
            slice_end_indices=(None, 5, None),
            test_name="slice_elemwise_fusion_dynamic_broadcast",
            expected_max_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
            input_x2_shape=([5, 16], 4, 16),
        )
        # mixed
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], 10, 10),
            slice_start_indices=(0, 0, 4),
            slice_end_indices=(None, None, 5),
            test_name="slice_elemwise_fusion_dynamic_broadcast",
            expected_max_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            input_x2_shape=(1, 10, 15),
        )
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([1, 1], [4, 20], 10),
            slice_start_indices=(0, 0, 4),
            slice_end_indices=(None, None, 5),
            test_name="slice_elemwise_fusion_dynamic_broadcast",
            expected_max_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            input_x2_shape=(10, 1, 15),
        )
        # with different dimensionalities
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], 10, 10),
            slice_start_indices=(0, 0, 0),
            slice_end_indices=(None, None, 8),
            test_name="slice_elemwise_fusion_dynamic_broadcast",
            expected_max_read_t="uint",
            expected_op_t="half2",
            expected_data_t="half",
            input_x2_shape=(10, 8),
        )
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], 10, 10),
            slice_start_indices=(0, 0, 4),
            slice_end_indices=(None, None, 5),
            test_name="slice_elemwise_fusion_dynamic_broadcast",
            expected_max_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            input_x2_shape=(3, [5, 16], 10, 15),
        )
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], 10, 20),
            slice_start_indices=(0, 0, 4),
            slice_end_indices=(None, None, 12),
            test_name="slice_elemwise_fusion_dynamic_broadcast",
            expected_max_read_t="uint2",
            expected_op_t="half2",
            expected_data_t="half",
            input_x2_shape=([3, 7], [5, 16], 10, 8),
        )

    def _test_two_slice_elemwise_fusion_dynamic(
        self,
        slice_input_shape,
        slice_start_indices1,
        slice_end_indices1,
        slice_start_indices2,
        slice_end_indices2,
        expected_max_read_t,
        expected_op_t,
        expected_data_t,
        test_name,
        dtype="float16",
    ):
        x_shape = [
            shape_utils.gen_int_var_min_max(d) if isinstance(d, list) else IntImm(d)
            for d in slice_input_shape
        ]
        X1 = Tensor(
            shape=x_shape,
            dtype=dtype,
            name="input_x1",
            is_input=True,
        )
        slice_op1 = ops.dynamic_slice()
        slice_output1 = slice_op1(
            X1, start_indices=slice_start_indices1, end_indices=slice_end_indices1
        )
        slice_output1._attrs["name"] = "slice_output1"

        slice_op2 = ops.dynamic_slice()
        slice_output2 = slice_op2(
            X1, start_indices=slice_start_indices2, end_indices=slice_end_indices2
        )
        slice_output2._attrs["name"] = "slice_output2"

        Y = ops.elementwise(FuncEnum.ADD)(slice_output1, slice_output2)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        # 1 inputs + 1 output
        self.assertEqual(len(sorted_graph), 2)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["max_read_t"], expected_max_read_t)
        self.assertEqual(self._get_read_types(sorted_ops[0]), [expected_max_read_t])
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        for d in slice_input_shape:
            if isinstance(d, list):
                Ms = d
                break
        assert Ms is not None, "expected to have at least one dynamic dim"
        for idx in range(len(Ms)):
            # Run PyTorch
            x_shape_pt = [
                d[idx] if isinstance(d, list) else d for d in slice_input_shape
            ]
            x1_pt = get_random_torch_tensor(x_shape_pt, dtype)

            slice_indices1 = [
                slice(i, j) for i, j in zip(slice_start_indices1, slice_end_indices1)
            ]
            slice_indices2 = [
                slice(i, j) for i, j in zip(slice_start_indices2, slice_end_indices2)
            ]
            slice_output1_pt = x1_pt[slice_indices1]
            slice_output2_pt = x1_pt[slice_indices2]
            y_pt = slice_output1_pt + slice_output2_pt

            # Run AITemplate module.
            inputs = {
                "input_x1": x1_pt,
            }
            y = get_torch_empty_tensor(y_pt.size(), dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
            self.test_count += 1

    def test_two_slice_elemwise_fusion_dynamic(self):
        self._test_two_slice_elemwise_fusion_dynamic(
            slice_input_shape=([3, 50], 100),
            slice_start_indices1=(0, 4),
            slice_end_indices1=(None, 8),
            slice_start_indices2=(0, 16),
            slice_end_indices2=(None, 20),
            expected_max_read_t="uint2",
            expected_op_t="half2",
            expected_data_t="half",
            test_name="two_slice_elemwise_fusion_dynamic",
        )
        self._test_two_slice_elemwise_fusion_dynamic(
            slice_input_shape=([3, 50], 100),
            slice_start_indices1=(0, 3),
            slice_end_indices1=(None, 7),
            slice_start_indices2=(0, 4),
            slice_end_indices2=(None, 8),
            expected_max_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            test_name="two_slice_elemwise_fusion_dynamic",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_slice_elemwise_fusion_float(self):
        self._test_slice_elemwise_fusion(
            slice_input_shape=(10, 20, 30),
            slice_start_indices=(0, 3, 0),
            slice_end_indices=(None, 5, None),
            test_name="slice_elemwise_fusion_float",
            expected_max_read_t="uint2",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float",
        )
        self._test_slice_elemwise_fusion(
            slice_input_shape=(10, 16),
            slice_start_indices=(2, 0),
            slice_end_indices=(3, None),
            test_name="slice_elemwise_fusion_broadcast_float",
            expected_max_read_t="uint4",
            expected_op_t="float",
            expected_data_t="float",
            input_x2_shape=(4, 16),
            dtype="float",
        )
        self._test_slice_elemwise_fusion(
            slice_input_shape=(1, 1, 10),
            slice_start_indices=(0, 0, 2),
            slice_end_indices=(None, None, 7),
            test_name="slice_elemwise_fusion_broadcast_float_2",
            expected_max_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            input_x2_shape=(10, 3, 1),
            dtype="float",
        )
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], [4, 10], 16),
            slice_start_indices=(0, 0, 4),
            slice_end_indices=(None, None, 16),
            test_name="slice_elemwise_fusion_dynamic_float",
            expected_max_read_t="uint4",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float",
        )
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], 10, 10),
            slice_start_indices=(0, 0, 4),
            slice_end_indices=(None, None, 5),
            test_name="slice_elemwise_fusion_dynamic_broadcast_float",
            expected_max_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            input_x2_shape=(1, 10, 15),
            dtype="float",
        )
        self._test_slice_elemwise_fusion_dynamic(
            slice_input_shape=([5, 16], 10, 10),
            slice_start_indices=(0, 0, 0),
            slice_end_indices=(None, None, 8),
            test_name="slice_elemwise_fusion_dynamic_broadcast_float",
            expected_max_read_t="uint2",
            expected_op_t="float",
            expected_data_t="float",
            input_x2_shape=(10, 8),
            dtype="float",
        )
        self._test_two_slice_elemwise_fusion_dynamic(
            slice_input_shape=([3, 50], 100),
            slice_start_indices1=(0, 4),
            slice_end_indices1=(None, 8),
            slice_start_indices2=(0, 16),
            slice_end_indices2=(None, 20),
            expected_max_read_t="uint4",
            expected_op_t="float",
            expected_data_t="float",
            test_name="two_slice_elemwise_fusion_dynamic_float",
            dtype="float",
        )


if __name__ == "__main__":
    unittest.main()
