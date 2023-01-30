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
import logging
import random
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils


_LOGGER = logging.getLogger(__name__)


class SplitLargeConcatTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SplitLargeConcatTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _make_tensors(self, num_inputs, input_shape, dtype, input_names=None):
        if input_names is not None:
            assert num_inputs == len(input_names)
        input_tensors = []
        for i in range(num_inputs):
            name = input_names[i] if input_names is not None else f"input_{i}"
            t = Tensor(
                shape=input_shape,
                dtype=dtype,
                name=name,
                is_input=True,
            )
            input_tensors.append(t)
        return input_tensors

    def _test_split_large_concat_simple(
        self, cat_dim, num_inputs, input_shape, split_count, test_name, dtype="float16"
    ):
        # a simple test: a concat takes num_inputs and the output of the concat
        # is a model output
        _LOGGER.info(f"test_split_large_concat with {num_inputs=}, {input_shape=}")
        input_tensors = self._make_tensors(num_inputs, input_shape, dtype)
        concat_op = ops.concatenate()
        Y = concat_op(input_tensors, cat_dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), num_inputs + 1)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), split_count)

        inputs_pt = [
            get_random_torch_tensor(input_shape, dtype) for _ in range(num_inputs)
        ]
        y_pt = torch.cat(inputs_pt, cat_dim)

        # run ait
        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(inputs_pt))]
        input_names = [x._attrs["name"] for x in input_tensors]
        for x_name, x_pt in zip(input_names, inputs_pt):
            inputs[input_name_to_index[x_name]] = x_pt

        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_split_large_concat_simple(self):
        self._test_split_large_concat_simple(
            cat_dim=1,
            num_inputs=136,
            input_shape=(2, 3),
            split_count=4,
            test_name="split_large_concat_simple",
        )
        self._test_split_large_concat_simple(
            cat_dim=1,
            num_inputs=34,
            input_shape=(2, 3),
            split_count=1,
            test_name="split_large_concat_simple",
        )
        self._test_split_large_concat_simple(
            cat_dim=1,
            num_inputs=35,
            input_shape=(2, 3),
            split_count=2,
            test_name="split_large_concat_simple",
        )

    def _test_split_large_concat_with_add(
        self, cat_dim, num_inputs, input_shape, test_name, dtype="float16"
    ):
        # make a model like below:
        # y1 = concat(x1,x2...)
        # y = add(y1, x_n) where x_n is not used by concat
        _LOGGER.info(f"test_split_large_concat with {num_inputs=}, {input_shape=}")
        input_tensors = self._make_tensors(num_inputs, input_shape, dtype)
        concat_op = ops.concatenate()
        Y1 = concat_op(input_tensors, cat_dim)
        x_n_shape = [1]
        X_ns = self._make_tensors(1, x_n_shape, dtype, ["input_x_n"])
        X_n = X_ns[0]
        Y = ops.elementwise(FuncEnum.ADD)(Y1, X_n)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        inputs_pt = [
            get_random_torch_tensor(input_shape, dtype) for _ in range(num_inputs)
        ]
        x_n_pt = get_random_torch_tensor(x_n_shape, dtype)
        y1_pt = torch.cat(inputs_pt, cat_dim)
        inputs_pt.append(x_n_pt)
        y_pt = y1_pt + x_n_pt

        # run ait
        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(inputs_pt))]
        input_names = [x._attrs["name"] for x in input_tensors + [X_n]]
        for x_name, x_pt in zip(input_names, inputs_pt):
            inputs[input_name_to_index[x_name]] = x_pt

        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_split_large_concat_with_add(self):
        self._test_split_large_concat_with_add(
            cat_dim=1,
            num_inputs=136,
            input_shape=(2, 3, 4),
            test_name="split_large_concat_with_add",
        )

    def _test_split_large_concat_with_strided_add(
        self, cat_dim, num_inputs, input_shape, test_name, dtype="float16"
    ):
        # make a model like below:
        # y1 = add(x1, x2)
        # y2 = concat(y1, x3, ...)
        # y = add(y1, x_n) where x_n is not used by concat
        _LOGGER.info(f"test_split_large_concat with {num_inputs=}, {input_shape=}")
        add_input_tensors = self._make_tensors(
            2, input_shape, dtype, ["add_input_0", "add_input_1"]
        )
        Y1 = ops.elementwise(FuncEnum.ADD)(add_input_tensors[0], add_input_tensors[1])
        concat_input_tensors = self._make_tensors(num_inputs, input_shape, dtype)
        concat_op = ops.concatenate()
        Y2 = concat_op([Y1] + concat_input_tensors, cat_dim)
        x_n_shape = [1]
        X_ns = self._make_tensors(1, x_n_shape, dtype, ["input_x_n"])
        X_n = X_ns[0]
        Y = ops.elementwise(FuncEnum.ADD)(Y2, X_n)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        add_inputs_pt = [get_random_torch_tensor(input_shape, dtype) for _ in range(2)]
        y1_pt = add_inputs_pt[0] + add_inputs_pt[1]
        concat_inputs_pt = [
            get_random_torch_tensor(input_shape, dtype) for _ in range(num_inputs)
        ]
        x_n_pt = get_random_torch_tensor(x_n_shape, dtype)
        y2_pt = torch.cat([y1_pt] + concat_inputs_pt, cat_dim)
        y_pt = y2_pt + x_n_pt

        input_tensors = add_input_tensors + concat_input_tensors + [X_n]
        inputs_pt = add_inputs_pt + concat_inputs_pt + [x_n_pt]
        # run ait
        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(inputs_pt))]
        input_names = [x._attrs["name"] for x in input_tensors]
        for x_name, x_pt in zip(input_names, inputs_pt):
            inputs[input_name_to_index[x_name]] = x_pt

        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_split_large_concat_with_strided_add(self):
        self._test_split_large_concat_with_strided_add(
            cat_dim=1,
            num_inputs=136,
            input_shape=(2, 3),
            test_name="split_large_concat_with_strided_add",
        )

    def _test_split_large_concat_with_strided_add_complex(
        self, cat_dim, num_inputs, input_shape, test_name, dtype="float16"
    ):
        # make a model like below:
        # a1 = add(x1, x2)
        # a2 = add(x3, x4)
        # ...
        # y = concat(a1, x1_1, a2, x1_2, ...)
        _LOGGER.info(f"test_split_large_concat with {num_inputs=}, {input_shape=}")
        add_input_tensor_names = [f"add_input_{i}" for i in range(num_inputs * 2)]
        add_input_tensors = self._make_tensors(
            num_inputs * 2, input_shape, dtype, add_input_tensor_names
        )
        add_output_tensors = []
        for i in range(num_inputs):
            a = ops.elementwise(FuncEnum.ADD)(
                add_input_tensors[i * 2], add_input_tensors[i * 2 + 1]
            )
            add_output_tensors.append(a)
        other_input_tensors = self._make_tensors(num_inputs, input_shape, dtype)
        concat_op = ops.concatenate()
        concat_input_tensors = []
        for i in range(num_inputs):
            concat_input_tensors.append(add_output_tensors[i])
            concat_input_tensors.append(other_input_tensors[i])
        Y = concat_op(concat_input_tensors, cat_dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        add_inputs_pt = [
            get_random_torch_tensor(input_shape, dtype) for _ in range(num_inputs * 2)
        ]
        add_outputs_pt = []
        for i in range(num_inputs):
            add_outputs_pt.append(add_inputs_pt[i * 2] + add_inputs_pt[i * 2 + 1])
        other_inputs_pt = [
            get_random_torch_tensor(input_shape, dtype) for _ in range(num_inputs)
        ]
        concat_inputs_pt = []
        for i in range(num_inputs):
            concat_inputs_pt.append(add_outputs_pt[i])
            concat_inputs_pt.append(other_inputs_pt[i])
        y_pt = torch.cat(concat_inputs_pt, cat_dim)

        input_tensors = add_input_tensors + other_input_tensors
        inputs_pt = add_inputs_pt + other_inputs_pt
        # run ait
        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(inputs_pt))]
        input_names = [x._attrs["name"] for x in input_tensors]
        for x_name, x_pt in zip(input_names, inputs_pt):
            inputs[input_name_to_index[x_name]] = x_pt

        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_split_large_concat_with_strided_add_complex(self):
        self._test_split_large_concat_with_strided_add_complex(
            cat_dim=1,
            num_inputs=136,
            input_shape=(2, 3),
            test_name="split_large_concat_with_strided_add_complex",
        )

    def _test_split_large_concat_with_reuse(
        self, cat_dim, num_inputs, input_shape, test_name, dtype="float16"
    ):
        # make a model like below:
        # a1 = add(x1, x2)
        # a2 = add(x3, x4)
        # ...
        # add_inputs = shuffle(x1,x2,x3...)
        # other_inputs = [o1, o2...]
        # concat_input = shuffle([a1, a2...] + add_inputs[0:10] + other_inputs)
        # y = concat(concat_input)
        _LOGGER.info(f"test_split_large_concat with {num_inputs=}, {input_shape=}")
        add_input_tensor_names = [f"add_input_{i}" for i in range(num_inputs * 2)]
        add_input_tensors = self._make_tensors(
            num_inputs * 2, input_shape, dtype, add_input_tensor_names
        )
        add_output_tensors = []
        for i in range(num_inputs):
            a = ops.elementwise(FuncEnum.ADD)(
                add_input_tensors[i * 2], add_input_tensors[i * 2 + 1]
            )
            add_output_tensors.append(a)
        other_input_tensors = self._make_tensors(num_inputs, input_shape, dtype)
        add_inputs_shuffle = list(range(len(add_input_tensors)))
        random.shuffle(add_inputs_shuffle)
        add_inputs_for_concat = [add_input_tensors[i] for i in add_inputs_shuffle[0:10]]
        concat_input_tensors = (
            add_output_tensors + other_input_tensors + add_inputs_for_concat
        )
        concat_inputs_shuffle = list(range(len(concat_input_tensors)))
        random.shuffle(concat_inputs_shuffle)
        real_concat_input_tensors = [
            concat_input_tensors[i] for i in concat_inputs_shuffle
        ]
        concat_op = ops.concatenate()
        Y = concat_op(real_concat_input_tensors, cat_dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        add_inputs_pt = [
            get_random_torch_tensor(input_shape, dtype) for _ in range(num_inputs * 2)
        ]
        add_outputs_pt = []
        for i in range(num_inputs):
            add_outputs_pt.append(add_inputs_pt[i * 2] + add_inputs_pt[i * 2 + 1])
        add_inputs_for_concat_pt = [add_inputs_pt[i] for i in add_inputs_shuffle[0:10]]
        other_inputs_pt = [
            get_random_torch_tensor(input_shape, dtype) for _ in range(num_inputs)
        ]
        concat_inputs_pt = add_outputs_pt + other_inputs_pt + add_inputs_for_concat_pt
        real_concat_inputs_pt = [concat_inputs_pt[i] for i in concat_inputs_shuffle]
        y_pt = torch.cat(real_concat_inputs_pt, cat_dim)

        input_tensors = add_input_tensors + other_input_tensors
        inputs_pt = add_inputs_pt + other_inputs_pt
        # run ait
        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(inputs_pt))]
        input_names = [x._attrs["name"] for x in input_tensors]
        for x_name, x_pt in zip(input_names, inputs_pt):
            inputs[input_name_to_index[x_name]] = x_pt

        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_split_large_concat_with_reuse(self):
        self._test_split_large_concat_with_reuse(
            cat_dim=1,
            num_inputs=136,
            input_shape=(2, 3),
            test_name="split_large_concat_with_reuse",
        )

    def _test_split_large_concat_with_slice(
        self,
        cat_dim,
        num_slice_inputs,
        slice_input_shape,
        start_indices,
        end_indices,
        num_add_inputs,
        add_input_shape,
        test_name,
        dtype="float16",
    ):
        # make a model like below:
        # s1 = t1[:, 0:10]
        # s2 = t1[:, 0:10]
        # ...
        # a1 = add(x1, x2)
        # a2 = add(x3, x4)
        # ...
        # concat_input = [s1, s2, ...] + [a1, a2...]
        # y = concat(concat_input)
        slice_input_tensor_names = [f"slice_input_{i}" for i in range(num_slice_inputs)]
        slice_input_tensors = self._make_tensors(
            num_slice_inputs, slice_input_shape, dtype, slice_input_tensor_names
        )
        slice_output_tensors = []
        for slice_input_tensor in slice_input_tensors:
            t = ops.dynamic_slice()(
                slice_input_tensor, start_indices=start_indices, end_indices=end_indices
            )
            slice_output_tensors.append(t)

        add_input_tensor_names = [f"add_input_{i}" for i in range(num_add_inputs * 2)]
        add_input_tensors = self._make_tensors(
            num_add_inputs * 2, add_input_shape, dtype, add_input_tensor_names
        )
        add_output_tensors = []
        for i in range(num_add_inputs):
            a = ops.elementwise(FuncEnum.ADD)(
                add_input_tensors[i * 2], add_input_tensors[i * 2 + 1]
            )
            add_output_tensors.append(a)

        concat_input_tensors = slice_output_tensors + add_output_tensors
        concat_op = ops.concatenate()
        Y = concat_op(concat_input_tensors, cat_dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        slice_inputs_pt = [
            get_random_torch_tensor(slice_input_shape, dtype)
            for _ in range(num_slice_inputs)
        ]
        slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
        slice_outputs_pt = [inp_pt[slice_indices] for inp_pt in slice_inputs_pt]

        add_inputs_pt = [
            get_random_torch_tensor(add_input_shape, dtype)
            for _ in range(num_add_inputs * 2)
        ]
        add_outputs_pt = []
        for i in range(num_add_inputs):
            add_outputs_pt.append(add_inputs_pt[i * 2] + add_inputs_pt[i * 2 + 1])
        concat_inputs_pt = slice_outputs_pt + add_outputs_pt
        y_pt = torch.cat(concat_inputs_pt, cat_dim)

        input_tensors = slice_input_tensors + add_input_tensors
        inputs_pt = slice_inputs_pt + add_inputs_pt
        # run ait
        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(inputs_pt))]
        input_names = [x._attrs["name"] for x in input_tensors]
        for x_name, x_pt in zip(input_names, inputs_pt):
            inputs[input_name_to_index[x_name]] = x_pt

        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_split_large_concat_with_slice(self):
        self._test_split_large_concat_with_slice(
            cat_dim=1,
            num_slice_inputs=161,
            slice_input_shape=(20, 20),
            start_indices=[0, 0],
            end_indices=[None, 10],
            num_add_inputs=5,
            add_input_shape=(20, 161 * 10),
            test_name="split_large_concat_with_dynamic_slice",
        )

    def _test_split_large_concat_with_reshape(
        self,
        num_inputs,
        input_shape,
        reshape_shape,
        cat_dim,
        test_name,
        dtype="float16",
    ):
        # make a model like below:
        # x = Tensor([10, 2, 20])
        # reshape_output = reshape(t1, [10, -1])
        # t1 = Tensor([10, 40])
        # ...
        # tn = Tensor([10, 40])
        # y = concat([x, t1, ..., tn])
        X = Tensor(
            shape=reshape_shape,
            dtype=dtype,
            name="x",
            is_input=True,
        )
        reshape_output = ops.reshape()(X, input_shape)
        normal_input_tensors = self._make_tensors(num_inputs, input_shape, dtype)
        concat_input_tensors = [reshape_output] + normal_input_tensors
        concat_op = ops.concatenate()
        Y = concat_op(concat_input_tensors, cat_dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1

        x_pt = get_random_torch_tensor(reshape_shape, dtype)
        reshape_output_pt = torch.reshape(x_pt, input_shape)
        normal_inputs_pt = [
            get_random_torch_tensor(input_shape, dtype) for _ in range(num_inputs)
        ]
        concat_inputs_pt = [reshape_output_pt] + normal_inputs_pt
        y_pt = torch.cat(concat_inputs_pt, cat_dim)

        # run ait
        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(concat_inputs_pt))]
        input_names = [X._attrs["name"]] + [
            i._attrs["name"] for i in normal_input_tensors
        ]
        for i_name, i_pt in zip(input_names, [x_pt] + normal_inputs_pt):
            inputs[input_name_to_index[i_name]] = i_pt

        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

    def test_split_large_concat_with_reshape(self):
        self._test_split_large_concat_with_reshape(
            num_inputs=180,
            input_shape=(10, 40),
            reshape_shape=(10, 2, 20),
            cat_dim=1,
            test_name="split_large_concat_with_reshape",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_split_large_concat_float(self):
        self._test_split_large_concat_simple(
            cat_dim=1,
            num_inputs=35,
            input_shape=(2, 3),
            split_count=2,
            test_name="split_large_concat_simple_float",
            dtype="float",
        )
        self._test_split_large_concat_with_add(
            cat_dim=1,
            num_inputs=136,
            input_shape=(2, 3, 4),
            test_name="split_large_concat_with_add_float",
            dtype="float",
        )
        self._test_split_large_concat_with_strided_add(
            cat_dim=1,
            num_inputs=136,
            input_shape=(2, 3),
            test_name="split_large_concat_with_strided_add_float",
            dtype="float",
        )
        self._test_split_large_concat_with_strided_add_complex(
            cat_dim=1,
            num_inputs=136,
            input_shape=(2, 3),
            test_name="split_large_concat_with_strided_add_complex_float",
            dtype="float",
        )
        self._test_split_large_concat_with_reuse(
            cat_dim=1,
            num_inputs=136,
            input_shape=(2, 3),
            test_name="split_large_concat_with_reuse_float",
            dtype="float",
        )
        self._test_split_large_concat_with_slice(
            cat_dim=1,
            num_slice_inputs=161,
            slice_input_shape=(20, 20),
            start_indices=[0, 0],
            end_indices=[None, 10],
            num_add_inputs=5,
            add_input_shape=(20, 161 * 10),
            test_name="split_large_concat_with_dynamic_slice_float",
            dtype="float",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
