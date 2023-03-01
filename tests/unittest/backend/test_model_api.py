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
import contextlib
import ctypes
import itertools
import json
import os
import tempfile
import unittest
from typing import Callable, Optional, Tuple

import numpy as np

import torch

from aitemplate.compiler import AIT_DEFAULT_NUM_RUNTIMES, compile_model, ops
from aitemplate.compiler.base import (
    _ConstantTensorData,
    _create_host_zero_tensor,
    _HostConstantTensorData,
    _NumpyConstantTensorData,
    _TorchConstantTensorData,
    get_dtype_size,
    IntVar,
)
from aitemplate.compiler.model import (
    AITData,
    AITemplateAllocatorKind,
    AITemplateMemcpyKind,
    Model,
    torch_to_ait_data,
)
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


class ModelAPITestCase(unittest.TestCase):
    def _get_simple_graph_and_output(
        self,
        test_name: str,
        dynamic_shape: bool = False,
        unsqueeze_output: bool = False,
    ) -> Tuple[
        Model, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ]:
        target = detect_target()
        input_0 = Tensor(shape=[1], dtype="float16", name="input_0", is_input=True)
        input_0_view = ops.reshape()(input_0, [1])
        input_1 = Tensor(
            shape=[IntVar([1, 1]) if dynamic_shape else 1],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        output = ops.elementwise(FuncEnum.MUL)(input_0_view, input_1)
        if unsqueeze_output:
            output = ops.unsqueeze(0)(output)

        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(output, target, "./tmp", test_name)
        in0_pt = torch.randn([1]).cuda().half()
        in1_pt = torch.randn([1]).cuda().half()
        output_pt = torch.mul(in0_pt, in1_pt)
        if unsqueeze_output:
            output_pt = output_pt.unsqueeze(0)
        output_storage = torch.randn(output_pt.shape).cuda().half()
        return (module, (in0_pt, in1_pt), (output_pt, output_storage))

    def test_set_unnamed_input(self):
        target = detect_target()

        input_0 = Tensor(shape=[1], dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=[1], dtype="float16", is_input=True)
        output = ops.elementwise(FuncEnum.SUB)(input_0, input_1)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(output, target, "./tmp", "test_set_unnamed_input")
        in0_pt = torch.randn([1]).cuda().half()
        in1_pt = torch.randn([1]).cuda().half()
        output_pt = in0_pt - in1_pt

        output_storage = torch.empty_like(output_pt)
        module.run_with_tensors(
            [in0_pt, in1_pt],
            [output_storage],
        )

        self.assertTrue(torch.allclose(output_storage, output_pt))

    def _test_param_name_to_index(self, output_is_view: bool, name: str):
        target = detect_target()

        input_0 = Tensor(shape=[1, 2], dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=[1, 2], dtype="float16", name="input_1", is_input=True)
        output = ops.elementwise(FuncEnum.SUB)(input_0, input_1)
        if output_is_view:
            output = ops.squeeze(0)(output)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(output, target, "./tmp", name)
        input_name_to_index = module.get_input_name_to_index_map()
        self.assertEqual(input_name_to_index, {"input_0": 0, "input_1": 1})
        output_name_to_index = module.get_output_name_to_index_map()
        self.assertEqual(output_name_to_index, {"output": 0})

    def test_get_param_name_to_index(self):
        self._test_param_name_to_index(
            output_is_view=False, name="test_get_param_name_to_index"
        )

    def test_get_param_name_to_index_output_is_view(self):
        self._test_param_name_to_index(
            output_is_view=True, name="test_get_param_name_to_index_output_is_view"
        )

    def test_error_handling_not_enough_inputs_outputs(self):
        module, (in0_pt, in1_pt), outputs = self._get_simple_graph_and_output(
            "test_error_handling_not_enough_inputs_outputs"
        )
        self.assertRaises(
            RuntimeError, module.run, [], [torch_to_ait_data(outputs[-1])]
        )
        self.assertRaises(
            RuntimeError,
            module.run_with_tensors,
            [in0_pt, in1_pt],
            [],
        )

    def test_error_handling_null_inputs_outputs(self):
        module, (in0_pt, in1_pt), outputs = self._get_simple_graph_and_output(
            "test_error_handling_null_inputs_outputs"
        )
        in0_pt_size = list(in0_pt.size())
        in1_pt_size = list(in1_pt.size())
        self.assertRaises(
            RuntimeError,
            module.run,
            [
                AITData(0, in0_pt_size, "float16"),
                AITData(0, in1_pt_size, "float16"),
            ],
            [torch_to_ait_data(outputs[-1])],
        )
        self.assertRaises(
            RuntimeError,
            module.run,
            [
                AITData(in0_pt.data_ptr(), in0_pt_size, "float16"),
                AITData(in1_pt.data_ptr(), in1_pt_size, "float16"),
            ],
            [AITData(0, list(outputs[-1].size()), "float16")],
        )

    def test_error_handling_wrong_param_dtypes(self):
        module, (in0_pt, in1_pt), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_error_handling"
        )
        in0_pt_size = list(in0_pt.size())
        in1_pt_size = list(in1_pt.size())
        self.assertRaises(
            RuntimeError,
            module.run,
            [
                AITData(in0_pt.data_ptr(), in0_pt_size, "float32"),
                AITData(in1_pt.data_ptr(), in1_pt_size, "float32"),
            ],
            [torch_to_ait_data(out_ait)],
        )

        self.assertRaises(
            RuntimeError,
            module.run,
            [
                torch_to_ait_data(in0_pt),
                torch_to_ait_data(in1_pt),
            ],
            [AITData(out_ait.data_ptr(), list(out_ait.size()), "float32")],
        )

        self.assertRaises(
            RuntimeError,
            module.run_with_tensors,
            [
                in0_pt,
                in1_pt.float(),
            ],
            [out_ait],
        )

        self.assertRaises(
            RuntimeError,
            module.run_with_tensors,
            [
                in0_pt,
                in1_pt,
            ],
            [out_ait.float()],
        )

    def test_one_input_many_constants(self):
        target = detect_target()

        input_0 = Tensor(shape=[1, 2], dtype="float16", name="input_0", is_input=True)
        constant_1 = Tensor(shape=[1, 2], dtype="float16", name="constant_1")
        constant_2 = Tensor(shape=[1, 2], dtype="float16", name="constant_2")
        x = ops.elementwise(FuncEnum.MUL)(input_0, constant_1)
        output = ops.elementwise(FuncEnum.MUL)(x, constant_2)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(output, target, "./tmp", "test_one_input_many_constants")
        in0_pt = torch.randn((1, 2)).cuda().half()
        const_1_pt = torch.randn((1, 2)).cuda().half()
        const_2_pt = torch.randn((1, 2)).cuda().half()

        module.set_constant_with_tensor("constant_1", const_1_pt)
        module.set_constant_with_tensor("constant_2", const_2_pt)

        output_data = torch.empty([1, 2]).cuda().half()
        module.run_with_tensors([in0_pt], [output_data])

        expected = in0_pt * const_1_pt.cuda() * const_2_pt.cuda()
        self.assertTrue(torch.allclose(output_data, expected))

    def test_get_param_maximum_shape(self):
        for dynamic_shape in (False, True):
            module, inputs, output_np = self._get_simple_graph_and_output(
                "test_get_param_maximum_shape",
                dynamic_shape=dynamic_shape,
            )
            names_to_index = module.get_output_name_to_index_map()
            output_shape = module.get_output_maximum_shape(names_to_index["output"])
            self.assertEqual(output_shape, [1])

            # Test str API
            output_shape = module.get_output_maximum_shape("output")
            self.assertEqual(output_shape, [1])

    def test_error_handling_maximum_shape(self):
        module, inputs, output_np = self._get_simple_graph_and_output(
            "test_get_param_maximum_shape",
        )
        self.assertRaises(ValueError, module.get_output_maximum_shape, "not_an_output")
        self.assertRaises(
            TypeError,
            module.get_output_maximum_shape,
            [],  # not a string or int
        )

    def test_get_param_maximum_shape_output_is_view(self):
        for dynamic_shape in (False, True):
            module, inputs, output_np = self._get_simple_graph_and_output(
                "test_get_param_maximum_shape",
                dynamic_shape=dynamic_shape,
                unsqueeze_output=True,
            )
            names_to_index = module.get_output_name_to_index_map()
            output_shape = module.get_output_maximum_shape(names_to_index["output"])
            self.assertEqual(output_shape, [1, 1])

    def test_dynamic_shape_api(self):
        target = detect_target()
        dynamic_dim = IntVar([1, 10], name="batch_size")
        input_0 = Tensor(
            shape=[dynamic_dim, 2],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        input_1 = Tensor(
            shape=[dynamic_dim, 2],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        output = ops.elementwise(FuncEnum.MUL)(input_1, input_0)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(output, target, "./tmp", "dynamic_shape_api")
        for batch_size in (1, 10):
            in0_pt = torch.randn([batch_size, 2]).cuda().half()
            in1_pt = torch.randn([batch_size, 2]).cuda().half()
            output_pt = torch.mul(in0_pt, in1_pt)
            output_storage = (
                torch.empty(module.get_output_maximum_shape("output")).cuda().half()
            )
            outputs_ait = module.run_with_tensors(
                [in0_pt, in1_pt],
                [output_storage],
            )
            self.assertTrue(torch.allclose(output_pt, outputs_ait["output"]))

    def _test_output_is_alias_of_input(self, view_of_view: bool):
        target = detect_target()
        input_0 = Tensor(
            shape=[2, 2],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        output = ops.reshape()(input_0, [4])
        if view_of_view:
            output = ops.reshape()(output, [4, 1])

        output._attrs["is_output"] = True
        output._attrs["name"] = "output"

        module = compile_model(output, target, "./tmp", "output_is_alias_of_input")

        in0_pt = torch.randn((2, 2)).cuda().half()
        out_shape = (4, 1) if view_of_view else (4,)
        out_pt = in0_pt.reshape(out_shape)
        out_ait = torch.empty(out_shape).cuda().half()

        module.run_with_tensors([in0_pt], [out_ait])
        self.assertTrue(torch.equal(out_pt, out_ait))

    def test_output_is_view_of_input(self):
        self._test_output_is_alias_of_input(False)

    def test_output_is_view_of_view_of_input(self):
        self._test_output_is_alias_of_input(True)

    def test_output_is_input(self):
        target = detect_target()
        input_0 = Tensor(
            shape=[2, 2], dtype="float16", name="input_0", is_input=True, is_output=True
        )

        module = compile_model(input_0, target, "./tmp", "output_is_input")

        in0_pt = torch.randn((2, 2)).cuda().half()
        out_ait = torch.empty((2, 2)).cuda().half()
        module.run_with_tensors([in0_pt], [out_ait])
        self.assertTrue(torch.equal(out_ait, in0_pt))

        inputs = module.get_input_name_to_index_map()
        self.assertEqual(inputs, {"input_0": 0})

        outputs = module.get_output_name_to_index_map()
        self.assertEqual(outputs, {"input_0": 0})

    def _test_output_is_view_of_constant(self, view_of_view: bool):
        target = detect_target()
        const = Tensor(shape=[2, 2], dtype="float16", name="constant")
        output = ops.reshape()(
            const,
            [
                4,
            ],
        )
        if view_of_view:
            output = ops.reshape()(output, [4, 1])
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(output, target, "./tmp", "output_is_view_of_constant")

        const_pt = torch.randn((2, 2)).cuda().half()
        out_shape = (4, 1) if view_of_view else (4,)
        out_pt = const_pt.reshape(out_shape)
        out_ait = torch.empty(out_shape).cuda().half()

        module.set_constant_with_tensor("constant", const_pt)
        module.run_with_tensors([], [out_ait])
        self.assertTrue(torch.equal(out_ait, out_pt))

    def test_output_is_view_of_constant(self):
        self._test_output_is_view_of_constant(False)

    def test_output_is_view_of_view_of_constant(self):
        self._test_output_is_view_of_constant(True)

    def test_output_is_constant(self):
        target = detect_target()
        const = Tensor(shape=[2, 2], dtype="float16", name="constant", is_output=True)
        module = compile_model(const, target, "./tmp", "output_is_constant")

        const_pt = torch.randn((2, 2)).cuda().half()
        out_ait = torch.empty((2, 2)).cuda().half()
        module.set_constant_with_tensor("constant", const_pt)
        module.run_with_tensors([], [out_ait])
        self.assertTrue(torch.equal(out_ait, const_pt))

    def _test_output_is_view_of_another_output(self, view_of_view: bool):
        target = detect_target()
        input_0 = Tensor(
            shape=[2, 2],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        output = ops.elementwise(FuncEnum.MUL)(input_0, input_0)
        output._attrs["is_output"] = True
        output._attrs["name"] = "output"

        view = ops.reshape()(output, (4,))
        if view_of_view:
            view = ops.reshape()(view, (4, 1))
        view._attrs["is_output"] = True
        view._attrs["name"] = "view"

        output1 = ops.elementwise(FuncEnum.MUL)(view, view)
        output1._attrs["is_output"] = True
        output1._attrs["name"] = "output1"

        outputs = [output, view, output1]
        module = compile_model(
            outputs, target, "./tmp", "output_is_alias_of_another_output"
        )

        out_shape = (4, 1) if view_of_view else (4,)
        in0_pt = torch.randn((2, 2)).cuda().half()
        out_pt = in0_pt * in0_pt
        view_pt = out_pt.reshape(out_shape)
        out1_pt = view_pt * view_pt

        out_ait = torch.empty((2, 2)).cuda().half()
        view_ait = torch.empty(out_shape).cuda().half()
        out1_ait = torch.empty(out_shape).cuda().half()
        module.run_with_tensors(
            [in0_pt],
            [out_ait, view_ait, out1_ait],
        )
        self.assertTrue(torch.equal(out_pt, out_ait))
        self.assertTrue(torch.equal(view_pt, view_ait))
        self.assertTrue(torch.equal(out1_pt, out1_ait))

    def test_output_is_view_of_another_output(self):
        self._test_output_is_view_of_another_output(False)

    def test_output_is_view_of_view_of_another_output(self):
        self._test_output_is_view_of_another_output(True)

    def test_output_is_alias_of_input_and_another_output(self):
        target = detect_target()
        input_0 = Tensor(
            shape=[2, 2],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        view1 = ops.reshape()(input_0, (1, 4))
        view1._attrs["is_output"] = True
        view1._attrs["name"] = "view1"

        view2 = ops.reshape()(view1, (4,))
        view2._attrs["is_output"] = True
        view2._attrs["name"] = "view2"

        module = compile_model(
            [view1, view2], target, "./tmp", "output_is_alias_of_another_output"
        )

        in0_pt = torch.randn((2, 2)).cuda().half()
        view1_pt = in0_pt.reshape((1, 4))
        view2_pt = in0_pt.reshape((4,))

        view1_ait = torch.empty((1, 4)).cuda().half()
        view2_ait = torch.empty((4,)).cuda().half()
        module.run_with_tensors(
            [in0_pt],
            [view1_ait, view2_ait],
        )
        self.assertTrue(torch.equal(view1_pt, view1_ait))
        self.assertTrue(torch.equal(view2_pt, view2_ait))

    def test_benchmark(self):
        module, (in0, in1), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_benchmark"
        )
        runtime_ms, _, outputs_ait = module.benchmark(
            [
                torch_to_ait_data(in0),
                torch_to_ait_data(in1),
            ],
            [torch_to_ait_data(out_ait)],
        )
        self.assertGreater(runtime_ms, 0)
        self.assertTrue(torch.equal(out_pt, out_ait))
        self.assertEqual(
            outputs_ait,
            {"output": AITData(out_ait.data_ptr(), [1], "float16")},
        )

        runtime_ms, _, tensors = module.benchmark_with_tensors(
            [in0, in1],
            [out_ait],
        )
        self.assertGreater(runtime_ms, 0)
        self.assertTrue(torch.equal(out_pt, out_ait))
        self.assertEqual(len(tensors), 1)
        self.assertTrue(torch.equal(tensors["output"], in0 * in1))

    def test_profile(self):
        module, (in0, in1), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_profile", False, True
        )
        with tempfile.TemporaryDirectory() as tmpdirname:
            profile_name = os.path.join(tmpdirname, "profile.json")
            module.profile(
                [
                    torch_to_ait_data(in0),
                    torch_to_ait_data(in1),
                ],
                [torch_to_ait_data(out_ait)],
                20,
                profile_name,
            )
            with open(profile_name) as f:
                report = json.load(f)
                self.assertTrue(len(report), 1)
                for _, elapsed in report.items():
                    self.assertGreater(elapsed["ms_per_iter"], 0)

    def test_get_output_dtype(self):
        module, inputs, output_np = self._get_simple_graph_and_output(
            "test_get_param_dtype"
        )
        self.assertEqual(module.get_output_dtype(0), 1)

    def test_dynamic_dims_out_of_bounds_error(self):
        target = detect_target()
        batch_size = IntVar([10, 1], name="batch_size")
        input_0 = Tensor(
            shape=[batch_size, 10], dtype="float16", name="input_0", is_input=True
        )
        output = ops.elementwise(FuncEnum.MUL)(input_0, input_0)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True
        module = compile_model(
            output, target, "./tmp", "test_dynamic_dim_out_of_bounds"
        )

        in0_pt = torch.randn((5, 10)).half().cuda()
        out_pt = torch.empty(module.get_output_maximum_shape("output")).cuda().half()

        self.assertRaises(
            RuntimeError,
            module.run,
            [AITData(in0_pt.data_ptr(), [0, 10], "float16")],
            [torch_to_ait_data(out_pt)],
        )

        self.assertRaises(
            RuntimeError,
            module.run,
            [AITData(in0_pt.data_ptr(), [11, 10], "float16")],
            [torch_to_ait_data(out_pt)],
        )

        # Make sure we can run with a valid batch size
        out = module.run_with_tensors(
            [in0_pt],
            [out_pt],
        )
        self.assertTrue(torch.equal(out["output"], in0_pt * in0_pt))

    def test_output_can_be_null_if_lower_bound_size_is_zero(self):
        target = detect_target()
        dynamic_dim = IntVar([0, 10], name="batch_size")
        input_0 = Tensor(
            shape=[dynamic_dim, 2],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        input_1 = Tensor(
            shape=[dynamic_dim, 2],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        output = ops.elementwise(FuncEnum.MUL)(input_1, input_0)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(
            output,
            target,
            "./tmp",
            "test_output_can_be_null_if_lower_bound_size_is_zero",
        )
        shape = [0, 2]
        module.run(
            [
                AITData(0, shape, "float16"),
                AITData(0, shape, "float16"),
            ],
            [AITData(0, [10, 2], "float16")],
        )

    def test_with_tensors_api_fails_on_cpu_inputs(self):
        module, (in0, in1), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_fail_on_cpu_inputs"
        )

        self.assertRaises(
            ValueError,
            module.run_with_tensors,
            [in0.cpu(), in1.cpu()],
            [out_ait],
        )
        self.assertRaises(
            ValueError,
            module.run_with_tensors,
            [in0, in1],
            [out_ait.cpu()],
        )
        self.assertRaises(
            ValueError,
            module.benchmark_with_tensors,
            [in0.cpu(), in1.cpu()],
            [out_ait],
        )
        self.assertRaises(
            ValueError,
            module.benchmark_with_tensors,
            [in0, in1],
            [out_ait.cpu()],
        )

    def test_with_tensors_api_fails_on_strided_inputs(self):
        target = detect_target()
        input_0 = Tensor(shape=[1, 2], dtype="float16", name="input_0", is_input=True)
        output = ops.elementwise(FuncEnum.MUL)(input_0, input_0)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(
            output, target, "./tmp", "test_with_tensors_api_fails_on_strided_inputs"
        )

        x = torch.randn((1, 1))
        in0_pt = x.expand((1, 2))
        out_pt = x.expand((1, 2))

        self.assertRaises(
            ValueError,
            module.run_with_tensors,
            [in0_pt],
            [out_pt.contiguous()],
        )
        self.assertRaises(
            ValueError, module.run_with_tensors, [x.contiguous()], [out_pt]
        )

    def _get_graph_three_inputs_three_outputs(self):
        target = detect_target()
        input_0 = Tensor(shape=[1], dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=[1], dtype="float16", name="input_1", is_input=True)
        input_2 = Tensor(shape=[1], dtype="float16", name="input_2", is_input=True)

        output_0 = ops.elementwise(FuncEnum.ADD)(input_0, input_1)
        output_1 = ops.elementwise(FuncEnum.ADD)(input_1, input_2)
        output_2 = ops.elementwise(FuncEnum.ADD)(input_0, input_2)

        output_0._attrs["name"] = "output_0"
        output_1._attrs["name"] = "output_1"
        output_2._attrs["name"] = "output_2"

        output_0._attrs["is_output"] = True
        output_1._attrs["is_output"] = True
        output_2._attrs["is_output"] = True

        module = compile_model(
            [output_0, output_1, output_2], target, "./tmp", "test_dict_api"
        )

        in0_pt = torch.randn((1,)).cuda().half()
        in1_pt = torch.randn((1,)).cuda().half()
        in2_pt = torch.randn((1,)).cuda().half()

        out0_pt = torch.empty((1,)).cuda().half()
        out1_pt = torch.empty((1,)).cuda().half()
        out2_pt = torch.empty((1,)).cuda().half()

        expected_out0 = in0_pt + in1_pt
        expected_out1 = in1_pt + in2_pt
        expected_out2 = in0_pt + in2_pt

        return (
            module,
            (in0_pt, in1_pt, in2_pt),
            (out0_pt, out1_pt, out2_pt),
            (expected_out0, expected_out1, expected_out2),
        )

    def test_dict_api(self):
        (
            module,
            (in0_pt, in1_pt, in2_pt),
            outputs,
            expected,
        ) = self._get_graph_three_inputs_three_outputs()
        out0_pt, out1_pt, out2_pt = outputs
        in_args = {
            "input_0": torch_to_ait_data(in0_pt),
            "input_1": torch_to_ait_data(in1_pt),
            "input_2": torch_to_ait_data(in2_pt),
        }
        out_args = {
            "output_0": torch_to_ait_data(out0_pt),
            "output_1": torch_to_ait_data(out1_pt),
            "output_2": torch_to_ait_data(out2_pt),
        }

        module.run(in_args, out_args)
        for out, expect in zip(outputs, expected):
            self.assertTrue(torch.equal(out, expect))
            out.zero_()
        module.benchmark(in_args, out_args)

        in_args_pt = {
            "input_0": in0_pt,
            "input_1": in1_pt,
            "input_2": in2_pt,
        }
        out_args_pt = {
            "output_0": out0_pt,
            "output_1": out1_pt,
            "output_2": out2_pt,
        }

        module.run_with_tensors(in_args_pt, out_args_pt)
        for out, expect in zip(outputs, expected):
            self.assertTrue(torch.equal(out, expect))
            out.zero_()
        module.benchmark_with_tensors(in_args_pt, out_args_pt)

    def test_error_handling_dict_api(self):
        (
            module,
            (in0_pt, in1_pt, in2_pt),
            outputs,
            expected,
        ) = self._get_graph_three_inputs_three_outputs()
        out0_pt, out1_pt, out2_pt = outputs
        in_args_pt = {
            "input_0": in0_pt,
            "input_1": in1_pt,
            "input_2": in2_pt,
        }
        out_args_pt = {
            "output_0": out0_pt,
            "output_1": out1_pt,
            "output_2": out2_pt,
            "not_an_output": torch.randn((3, 3)),
        }

        self.assertRaises(ValueError, module.run_with_tensors, {}, {})
        self.assertRaises(ValueError, module.run_with_tensors, in_args_pt, {})
        self.assertRaises(ValueError, module.run_with_tensors, in_args_pt, out_args_pt)

    def test_error_handling_model_init(self):
        for num_runtimes in (-1, 0):
            target = detect_target()
            input_0 = Tensor(
                shape=[1],
                dtype="float16",
                name="input_0",
                is_input=True,
                is_output=True,
            )
            with self.assertRaises(ValueError):
                compile_model(
                    input_0,
                    target,
                    "./tmp",
                    "test_error_handling_model_init",
                    num_runtimes=num_runtimes,
                )

    def test_bind_data_to_tensor_host_data(self):
        tensor = Tensor([10, 2], dtype="float16")
        self.assertRaises(
            ValueError,
            tensor._bind_data,
            _HostConstantTensorData(b"\x00" * 10),
        )
        # Make sure we can actually construct a constant tensor with the correct
        # size.
        for dtype in ("float16", "float32", "int32", "int64"):
            dtype_size = get_dtype_size(dtype)
            data = _HostConstantTensorData(b"\x00" * 20 * dtype_size, dtype=dtype)
            self.assertEqual(data.size(), len(data.to_bytes()))
            self.assertTrue(all(x == 0 for x in data.to_bytes()))
            tensor = Tensor([10, 2], dtype=dtype)
            tensor._bind_data(data)
            self.assertIsNotNone(tensor._attrs["data"])

            data_numpy = _NumpyConstantTensorData(np.zeros([10, 2], dtype))
            self.assertEqual(data_numpy.size(), len(data_numpy.to_bytes()))
            self.assertTrue(all(x == 0 for x in data_numpy.to_bytes()))
            tensor = Tensor([10, 2], dtype=dtype)
            tensor._bind_data(data_numpy)
            self.assertIsNotNone(tensor._attrs["data"])

    def test_bind_torch_tensor_data(self):
        small_tensor = torch.randn((5, 2)).cuda().half()
        tensor = Tensor([10, 2], dtype="float16")
        self.assertRaises(
            ValueError,
            tensor._bind_data,
            _TorchConstantTensorData(small_tensor),
        )
        dtype_to_torch = {
            "float16": torch.float16,
            "float32": torch.float32,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        for dtype in dtype_to_torch.keys():
            tensor = torch.ones((10, 2), dtype=dtype_to_torch[dtype]).cuda()
            data = _TorchConstantTensorData(tensor)
            self.assertEqual(data.size(), len(data.to_bytes()))

            data_np = np.frombuffer(data.to_bytes(), dtype=dtype).reshape((10, 2))
            np.testing.assert_equal(data_np, tensor.cpu().numpy())

            tensor = Tensor([10, 2], dtype=dtype)
            tensor._bind_data(data)
            self.assertIsNotNone(tensor._attrs["data"])

    def test_constant_tensor_construction_fails_mismatched_dtypes(self):
        torch_data = _TorchConstantTensorData(torch.randn((10, 2)).cuda())
        np_data = _NumpyConstantTensorData(np.random.rand(10, 2))
        host_data = _HostConstantTensorData("\x00" * 20 * 4, dtype="float32")
        bad_data = (torch_data, np_data, host_data)
        for data in bad_data:
            tensor = Tensor([10, 2], dtype="float16")
            self.assertRaises(
                ValueError,
                tensor._bind_data,
                data,
            )

    def _test_use_constant_tensor(
        self,
        make_data: Callable[[torch.Tensor], _ConstantTensorData],
        name: str,
        size: int = 3,
    ):
        target = detect_target()
        in0_pt = torch.randn((size,)).half()
        in1_pt = torch.randn((size,)).half()

        in0_data = make_data(in0_pt)
        in0 = Tensor(shape=[size], dtype="float16")
        in0._bind_data(in0_data)
        in1_data = make_data(in1_pt)
        in1 = Tensor(shape=[size], dtype="float16")
        in1._bind_data(in1_data)

        out = ops.elementwise(FuncEnum.MUL)(in0, in1)
        out._attrs["name"] = "output"
        out._attrs["is_output"] = True

        module = compile_model(out, target, "./tmp", name)

        output_ait = torch.randn((size,)).half().cuda()
        module.run_with_tensors([], [output_ait])

        self.assertTrue(torch.equal(output_ait.cpu(), in0_pt * in1_pt))

    def test_use_internal_constant_tensors_host(self):
        self._test_use_constant_tensor(
            lambda tensor: _HostConstantTensorData(tensor.cpu().numpy().tobytes()),
            "test_use_internal_constant_tensors_host",
        )

    def test_use_internal_constant_tensors_gpu(self):
        self._test_use_constant_tensor(
            lambda tensor: _TorchConstantTensorData(tensor),
            "test_use_internal_constant_tensors_gpu",
        )

    def test_use_internal_constant_tensors_huge(self):
        self._test_use_constant_tensor(
            lambda tensor: _TorchConstantTensorData(tensor),
            "test_use_internal_constant_tensors_huge",
            size=int(1e9 / 2),
        )

    def test_run_return_value_dynamic_batch(self):
        target = detect_target()

        input_0 = Tensor(
            shape=[IntVar([0, 2], name="out01"), IntVar([0, 2], name="out12")],
            dtype="float16",
            name="out0",
            is_input=True,
            is_output=True,
        )
        out = ops.elementwise(FuncEnum.MUL)(input_0, input_0)
        out._attrs["name"] = "out1"
        out._attrs["is_output"] = True

        module = compile_model(
            [input_0, out],
            target,
            "./tmp",
            "test_run_return_value_dynamic_batch",
        )

        for a in range(0, 2):
            for b in range(0, 2):
                in0 = torch.randn([a, b]).cuda().half()
                out0 = torch.empty_like(in0)
                out1 = torch.empty_like(in0)

                expected = {
                    "out0": AITData(out0.data_ptr(), [a, b], "float16"),
                    "out1": AITData(out1.data_ptr(), [a, b], "float16"),
                }
                actual = module.run(
                    {"out0": torch_to_ait_data(in0)},
                    {
                        "out0": torch_to_ait_data(out0),
                        "out1": torch_to_ait_data(out1),
                    },
                )
                self.assertEqual(expected, actual)

                out_tensors = module.run_with_tensors([in0], [out0, out1])
                self.assertEqual(len(out_tensors), 2)
                self.assertTrue(torch.equal(out_tensors["out0"], in0))
                self.assertTrue(torch.equal(out_tensors["out1"], in0 * in0))

    def test_run_return_value_static_shapes(self):
        target = detect_target()

        input_0 = Tensor(
            shape=[1, 2, 3, 4],
            dtype="float16",
            name="out0",
            is_input=True,
            is_output=True,
        )
        out = ops.elementwise(FuncEnum.MUL)(input_0, input_0)
        out._attrs["name"] = "out1"
        out._attrs["is_output"] = True

        module = compile_model(
            [input_0, out],
            target,
            "./tmp",
            "test_run_return_value_static_shapes",
        )

        in0 = torch.randn([1, 2, 3, 4]).cuda().half()
        out0 = torch.empty_like(in0)
        out1 = torch.empty_like(in0)
        expected = {
            "out0": AITData(out0.data_ptr(), [1, 2, 3, 4], "float16"),
            "out1": AITData(out1.data_ptr(), [1, 2, 3, 4], "float16"),
        }
        actual = module.run(
            {"out0": torch_to_ait_data(in0)},
            {
                "out0": torch_to_ait_data(out0),
                "out1": torch_to_ait_data(out1),
            },
        )
        self.assertEqual(expected, actual)

        out_tensors = module.run_with_tensors([in0], [out0, out1])
        self.assertEqual(len(out_tensors), 2)
        self.assertTrue(torch.equal(out_tensors["out0"], in0))
        self.assertTrue(torch.equal(out_tensors["out1"], in0 * in0))

    def test_run_return_value_dynamic_second_dim(self):
        target = detect_target()
        input_0 = Tensor(
            shape=[10, IntVar([0, 2], name="dim"), 2],
            dtype="float16",
            name="out0",
            is_input=True,
            is_output=True,
        )
        out = ops.elementwise(FuncEnum.MUL)(input_0, input_0)
        out._attrs["name"] = "out1"
        out._attrs["is_output"] = True

        module = compile_model(
            [input_0, out],
            target,
            "./tmp",
            "test_run_return_value_dynamic_second_dim",
        )

        for dim in range(0, 2):
            in0 = torch.randn([10, dim, 2]).cuda().half()
            out0 = torch.empty_like(in0)
            out1 = torch.empty_like(in0)

            expected = {
                "out0": AITData(out0.data_ptr(), [10, dim, 2], "float16"),
                "out1": AITData(out1.data_ptr(), [10, dim, 2], "float16"),
            }
            actual = module.run(
                {"out0": torch_to_ait_data(in0)},
                {
                    "out0": torch_to_ait_data(out0),
                    "out1": torch_to_ait_data(out1),
                },
            )
            self.assertEqual(expected, actual)

            out_tensors = module.run_with_tensors([in0], {"out0": out0, "out1": out1})
            self.assertEqual(len(out_tensors), 2)
            self.assertTrue(torch.equal(out_tensors["out0"], in0))
            self.assertTrue(torch.equal(out_tensors["out1"], in0 * in0))

    def test_many_threads_one_stream(self):
        module, (in0, in1), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_many_threads_one_stream"
        )
        runtime_ms, _, _ = module.benchmark_with_tensors(
            [in0, in1],
            [out_ait],
            num_threads=8,
            count=1000,
        )

    def test_many_threads_many_streams(self):
        module, (in0, in1), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_benchmark"
        )
        runtime_ms, _, _ = module.benchmark_with_tensors(
            [in0, in1],
            [out_ait],
            num_threads=8,
            count=1000,
            use_unique_stream_per_thread=True,
        )

    def test_compiled_module_preserves_output_order(self):
        input0 = Tensor(shape=[1], dtype="float16", name="input0", is_input=True)
        output0 = ops.elementwise(FuncEnum.MUL)(input0, input0)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output0"

        output1 = ops.elementwise(FuncEnum.ADD)(input0, input0)
        output1._attrs["is_output"] = True
        output1._attrs["name"] = "output1"

        output2 = ops.elementwise(FuncEnum.MUL)(output0, output1)
        output2._attrs["is_output"] = True
        output2._attrs["name"] = "output2"

        test_name = "test_compiled_module_preserves_output_order"

        for output_ordering in itertools.permutations((output0, output1, output2)):
            target = detect_target()
            with compile_model(
                output_ordering,
                target,
                "./tmp",
                test_name,
            ) as module:
                expected_ordering = {
                    tensor._attrs["name"]: idx
                    for idx, tensor in enumerate(output_ordering)
                }
                self.assertEqual(
                    module.get_output_name_to_index_map(),
                    expected_ordering,
                )

    def test_error_non_output_in_output_tensors_list(self):
        input0 = Tensor(shape=[1], dtype="float16", name="input0", is_input=True)
        intermediate = ops.elementwise(FuncEnum.ADD)(input0, input0)
        output0 = ops.elementwise(FuncEnum.MUL)(intermediate, intermediate)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output0"

        target = detect_target()
        self.assertRaises(
            (KeyError, ValueError),
            compile_model,
            [input0, output0],
            target,
            "./tmp",
            "test_error_non_output_in_output_tensors_list",
        )

    def test_error_missing_output_in_output_tensors_list(self):
        input0 = Tensor(shape=[1], dtype="float16", name="input0", is_input=True)
        intermediate = ops.elementwise(FuncEnum.ADD)(input0, input0)
        intermediate._attrs["is_output"] = True

        output0 = ops.elementwise(FuncEnum.MUL)(intermediate, intermediate)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output0"

        target = detect_target()
        self.assertRaises(
            ValueError,
            compile_model,
            [output0],
            target,
            "./tmp",
            "test_error_missing_output_in_output_tensors_list",
        )

    def test_error_duplicate_output_in_output_tensors_list(self):
        input0 = Tensor(shape=[1], dtype="float16", name="input0", is_input=True)
        intermediate = ops.elementwise(FuncEnum.ADD)(input0, input0)
        output0 = ops.elementwise(FuncEnum.MUL)(intermediate, intermediate)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output0"

        target = detect_target()
        self.assertRaises(
            ValueError,
            compile_model,
            [output0, output0],
            target,
            "./tmp",
            "test_error_duplicate_output_in_output_tensors_list",
        )

    def test_cannot_use_closed_model(self):
        (
            module,
            (in0_pt, in1_pt),
            (_, out_storage),
        ) = self._get_simple_graph_and_output("test_cannot_use_closed_model")

        module.close()

        self.assertRaises(
            RuntimeError, module.run_with_tensors, [in0_pt, in1_pt], [out_storage]
        )

    def test_cannot_use_closed_model_context_manager(self):
        (
            module,
            (in0_pt, in1_pt),
            (_, out_storage),
        ) = self._get_simple_graph_and_output("test_cannot_use_closed_model")

        with module as m:
            pass

        self.assertRaises(
            RuntimeError, m.run_with_tensors, [in0_pt, in1_pt], [out_storage]
        )

    def test_run_fails_with_unbound_constants(self):
        target = detect_target()

        constant_1 = Tensor(shape=[1, 2], dtype="float16", name="constant_1")
        constant_2 = Tensor(shape=[1, 2], dtype="float16", name="constant_2")
        x = ops.elementwise(FuncEnum.MUL)(constant_1, constant_1)
        output = ops.elementwise(FuncEnum.MUL)(x, constant_2)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(
            output, target, "./tmp", "test_run_fails_with_unbound_constants"
        )

        const_1_pt = torch.randn((1, 2)).cuda().half()
        const_2_pt = torch.randn((1, 2)).cuda().half()
        output_data = torch.empty([1, 2]).cuda().half()

        with self.assertRaises(RuntimeError):
            module.run_with_tensors([], [output_data])

        module.set_constant_with_tensor("constant_1", const_1_pt)

        with self.assertRaises(RuntimeError):
            module.run_with_tensors([], [output_data])
        module.set_constant_with_tensor("constant_2", const_2_pt)

        module.run_with_tensors([], [output_data])

        expected = const_1_pt * const_1_pt * const_2_pt
        self.assertTrue(torch.allclose(output_data, expected))

    def test_set_constant_fails_wrong_dtype(self):
        def _create_graph():
            constant_1 = Tensor(shape=[1, 2], dtype="float16", name="constant_1")
            output = ops.elementwise(FuncEnum.MUL)(constant_1, constant_1)
            output._attrs["name"] = "output"
            output._attrs["is_output"] = True
            return output

        for wrong_tensor in (
            torch.zeros([1, 2]).long().cuda(),
            torch.zeros([1, 2]).int().cuda(),
            torch.zeros([1, 2]).float().cuda(),
        ):
            target = detect_target()
            with compile_model(
                _create_graph(), target, "./tmp", "test_set_constant_fails_wrong_dtype"
            ) as module:
                self.assertRaises(
                    RuntimeError,
                    module.set_constant_with_tensor,
                    "constant_1",
                    wrong_tensor,
                )

    def test_set_constant_fails_wrong_shape(self):
        def _create_graph():
            constant_1 = Tensor(shape=[1, 2], dtype="float16", name="constant_1")
            output = ops.elementwise(FuncEnum.MUL)(constant_1, constant_1)
            output._attrs["name"] = "output"
            output._attrs["is_output"] = True
            return output

        for wrong_shape in (
            [2, 2],
            [3, 4],
            [0],
        ):
            wrong_tensor = torch.randn(wrong_shape).half().cuda()
            target = detect_target()
            output = _create_graph()
            with compile_model(
                output, target, "./tmp", "test_set_constant_fails_wrong_shape"
            ) as module:
                self.assertRaises(
                    RuntimeError,
                    module.set_constant_with_tensor,
                    "constant_1",
                    wrong_tensor,
                )

    def test_null_arguments_error(self):
        (
            module,
            (in0_pt, in1_pt),
            (_, out_storage),
        ) = self._get_simple_graph_and_output("test_null_arguments_error")

        old_handle = module.handle
        module.handle = None
        self.assertRaises(
            RuntimeError, module.run_with_tensors, [in0_pt, in1_pt], [out_storage]
        )
        self.assertRaises(RuntimeError, module.get_output_dtype, 0)

        # Put it back. Don't want to leak memory!
        module.handle = old_handle

    def test_memcpy(self):
        (
            module,
            _,
            _,
        ) = self._get_simple_graph_and_output("test_memcpy")

        # D2D
        torch_stream = torch.cuda.Stream().cuda_stream
        for stream_ptr in (None, torch_stream):
            expected = torch.randn((3, 2, 1)).half().cuda()
            actual = torch.empty_like(expected)
            module.memcpy(
                actual.data_ptr(),
                expected.data_ptr(),
                actual.numel() * actual.element_size(),
                AITemplateMemcpyKind.DeviceToDevice,
                stream_ptr,
            )
            self.assertTrue(torch.equal(expected, actual))

            # D2H
            expected = torch.randn((3, 2, 1)).half().cuda()
            actual = torch.empty_like(expected).cpu()
            module.memcpy(
                actual.data_ptr(),
                expected.data_ptr(),
                actual.numel() * actual.element_size(),
                AITemplateMemcpyKind.DeviceToHost,
                stream_ptr,
            )
            self.assertTrue(torch.equal(expected.cpu(), actual))

            # H2D
            expected = torch.randn((3, 2, 1)).half()
            actual = torch.empty_like(expected).cuda()
            module.memcpy(
                actual.data_ptr(),
                expected.data_ptr(),
                actual.numel() * actual.element_size(),
                AITemplateMemcpyKind.HostToDevice,
                stream_ptr,
            )
            self.assertTrue(torch.equal(expected, actual.cpu()))

    def test_alloc(self):
        (
            module,
            (in0_pt, in1_pt),
            (out_pt, out_ait),
        ) = self._get_simple_graph_and_output("test_memcpy")

        @contextlib.contextmanager
        def alloc_like(tensor: torch.Tensor, stream_ptr: Optional[int]):
            assert tensor.dtype == torch.half
            nbytes = tensor.numel() * tensor.element_size()
            ptr = module.allocate_gpu_memory(nbytes, stream_ptr)
            try:
                yield AITData(ptr, list(tensor.shape), "float16"), nbytes
            finally:
                module.free_gpu_memory(ptr, stream_ptr)

        torch_stream = torch.cuda.Stream().cuda_stream
        for stream_ptr in (None, torch_stream):
            with alloc_like(out_ait, stream_ptr) as (output, nbytes):
                module.run(
                    {
                        "input_0": torch_to_ait_data(in0_pt),
                        "input_1": torch_to_ait_data(in1_pt),
                    },
                    {"output": output},
                )
                module.memcpy(
                    out_ait.data_ptr(),
                    output.data_ptr,
                    nbytes,
                    AITemplateMemcpyKind.DeviceToDevice,
                    stream_ptr,
                )
                self.assertTrue(torch.equal(out_pt, out_ait))

    def test_get_num_runtimes(self):
        self.assertEqual(AIT_DEFAULT_NUM_RUNTIMES, 1)
        x = Tensor([1], dtype="float16", is_input=True, is_output=True)
        with compile_model(
            x, detect_target(), "./tmp", "test_get_num_runtimes_compile_module_default"
        ) as module:
            self.assertEqual(module.get_num_runtimes(), 1)

        with compile_model(
            x,
            detect_target(),
            "./tmp",
            "test_get_num_runtimes_compile_module_custom",
            num_runtimes=2,
        ) as module:
            self.assertEqual(module.get_num_runtimes(), 2)

    def test_ait_data_numpy_conversions(self):
        x = Tensor([1], dtype="float16", is_input=True, is_output=True)
        with compile_model(
            x, detect_target(), "./tmp", "test_ait_data_numpy_conversions"
        ) as module:
            x_shape = [1, 2, 3]
            x = np.ones(x_shape, dtype="float16")
            x_ait = module.numpy_to_ait_data(x)
            self.assertEqual(x_ait.dtype, "float16")
            self.assertEqual(x_ait.shape, x_shape)

            x_copied = module.ait_data_to_numpy(x_ait)
            np.testing.assert_equal(x, x_copied)

            y = torch.ones(x_shape, dtype=torch.float16).cuda()
            y_ait = AITData(y.data_ptr(), x_shape, "float16")
            y_np = module.ait_data_to_numpy(y_ait)

            np.testing.assert_equal(x, y_np)

    def test_numpy_to_ait_data_manual_free(self):
        x = Tensor([1], dtype="float16", is_input=True, is_output=True)
        with compile_model(
            x, detect_target(), "./tmp", "test_numpy_to_ait_data_manual_free"
        ) as module:
            x_shape = [1, 2, 3]
            x = np.ones(x_shape, dtype="float16")
            x_ait = module.numpy_to_ait_data(x)
            module.free_gpu_memory(x_ait.data_ptr)
            # Make sure we don't double-free when we exit.

    def test_custom_allocator(self):
        x = Tensor([1], dtype="float16", is_input=True)
        y = x * x
        z = y * y
        z._attrs["is_output"] = True
        for allocator_kind in (
            AITemplateAllocatorKind.DEFAULT,
            AITemplateAllocatorKind.TRACKING,
        ):
            with compile_model(
                z,
                detect_target(),
                "./tmp",
                f"test_custom_allocator_{allocator_kind.value}",
                allocator_kind=AITemplateAllocatorKind.TRACKING,
            ) as module:
                allocator = module.allocator_handle
                self.assertIsNotNone(allocator.value)

                if allocator_kind == AITemplateAllocatorKind.TRACKING:
                    num_bytes = ctypes.c_size_t()
                    module.DLL.AITemplateTrackingAllocatorGetNumBytes(
                        allocator, ctypes.byref(num_bytes)
                    )
                    self.assertGreater(num_bytes.value, 0)

                x_pt = (
                    torch.randn(
                        1,
                    )
                    .half()
                    .cuda()
                )
                y_pt = x_pt * x_pt
                z_pt = y_pt * y_pt

                z_ait = torch.empty_like(x_pt)
                module.run_with_tensors([x_pt], [z_ait])
                self.assertTrue(z_ait.equal(z_pt))

    def test_get_constant_names(self):
        target = detect_target()

        input_0 = Tensor(shape=[1, 2], dtype="float16", name="input_0", is_input=True)
        constant_0 = Tensor(shape=[1, 2], dtype="float16", name="constant_0")
        constant_1 = Tensor(shape=[1, 2], dtype="float16", name="constant_1")
        constant_2 = Tensor(shape=[1, 2], dtype="float16", name="constant_2")
        constant_3 = Tensor(shape=[1, 2], dtype="float16", name="constant_3")
        constant_4 = Tensor(shape=[1, 2], dtype="float16", name="constant_4")
        constants = {}

        # constant 0 and constant 1 are not folded.
        # constant 0 is unbounded, constant 1 is bounded.
        x = ops.elementwise(FuncEnum.MUL)(input_0, constant_0)
        x1 = ops.concatenate()([x, x, constant_1])
        constants["constant_1"] = get_random_torch_tensor((1, 2), "float16")

        # constants 2 and 3 and 4 are folded.
        # constants 2 and 4 are unbounded, constants 3 is bounded.
        y = ops.concatenate()([constant_2, constant_3, constant_4])
        constants["constant_3"] = get_random_torch_tensor((1, 2), "float16")

        output = ops.elementwise(FuncEnum.MUL)(x1, y)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(
            output, target, "./tmp", "test_get_constant_names", constants=constants
        )

        names_0 = module.get_constant_names(
            unbound_constants_only=True, constant_folding_only=False
        )
        self.assertEqual(set(names_0), {"constant_0", "constant_2", "constant_4"})

        names_1 = module.get_constant_names(
            unbound_constants_only=False, constant_folding_only=False
        )
        self.assertEqual(
            set(names_1),
            {"constant_0", "constant_1", "constant_2", "constant_3", "constant_4"},
        )

        names_2 = module.get_constant_names(
            unbound_constants_only=True, constant_folding_only=True
        )
        self.assertEqual(set(names_2), {"constant_2", "constant_4"})

        names_3 = module.get_constant_names(
            unbound_constants_only=False, constant_folding_only=True
        )
        self.assertEqual(set(names_3), {"constant_2", "constant_3", "constant_4"})

        names_4 = module.get_constant_folding_input_names(unbound_constants_only=True)
        self.assertEqual(set(names_4), {"constant_2", "constant_4"})

        names_5 = module.get_constant_folding_input_names(unbound_constants_only=False)
        self.assertEqual(set(names_5), {"constant_2", "constant_3", "constant_4"})

    def test_get_constant_names_with_ait_generated(self):
        target = detect_target()

        input_0 = Tensor(shape=[1, 2], dtype="float16", name="input_0", is_input=True)
        constant_0 = Tensor(shape=[1, 2], dtype="float16", name="constant_0")
        constant_1 = Tensor(shape=[1, 2], dtype="float16", name="constant_1")
        constant_2 = Tensor(shape=[1, 2], dtype="float16", name="constant_2")
        constant_3 = _create_host_zero_tensor(
            shape=[1, 2], name="constant_3", dtype="float16"
        )
        constant_4 = Tensor(shape=[1, 2], dtype="float16", name="constant_4")
        constants = {}

        # constant 0 and constant 1 are not folded.
        # constant 0 is unbounded, constant 1 is bounded.
        x = ops.elementwise(FuncEnum.MUL)(input_0, constant_0)
        x1 = ops.concatenate()([x, x, constant_1])
        constants["constant_1"] = get_random_torch_tensor((1, 2), "float16")

        # constants 2 and 3 and 4 are folded.
        # constants 2 and 4 are unbounded, constants 3 is bounded.
        y = ops.concatenate()([constant_2, constant_3, constant_4])

        output = ops.elementwise(FuncEnum.MUL)(x1, y)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(
            output,
            target,
            "./tmp",
            "test_get_constant_names_with_ait_generated",
            constants=constants,
        )

        names = module.get_constant_names(
            unbound_constants_only=False, constant_folding_only=False
        )
        self.assertEqual(
            set(names),
            {"constant_0", "constant_1", "constant_2", "constant_4"},
        )

    def test_set_many_constants(self):
        target = detect_target()

        input_0 = Tensor(shape=[1, 2], dtype="float16", name="input_0", is_input=True)
        constant_1 = Tensor(shape=[1, 2], dtype="float16", name="constant_1")
        constant_2 = Tensor(shape=[1, 2], dtype="float16", name="constant_2")
        x = ops.elementwise(FuncEnum.MUL)(input_0, constant_1)
        output = ops.elementwise(FuncEnum.MUL)(x, constant_2)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(output, target, "./tmp", "test_get_constant_names")

        input_0_pt = torch.randn((1, 2)).cuda().half()
        constant_1_pt = torch.randn((1, 2)).cuda().half()
        constant_2_pt = torch.randn((1, 2)).cuda().half()
        module.set_many_constants_with_tensors(
            {"constant_1": constant_1_pt, "constant_2": constant_2_pt}
        )
        output_pt = input_0_pt * constant_1_pt * constant_2_pt
        output_ait = torch.empty_like(input_0_pt)
        module.run_with_tensors([input_0_pt], [output_ait])
        self.assertTrue(torch.equal(output_pt, output_ait))

    def test_async_fold_constants(self):
        target = detect_target()

        input_0 = Tensor(
            shape=[10000, 2000], dtype="float16", name="input_0", is_input=True
        )
        constant_1 = Tensor(shape=[10000, 2000], dtype="float16", name="constant_1")
        constant_2 = Tensor(shape=[10000, 2000], dtype="float16", name="constant_2")
        x = ops.elementwise(FuncEnum.MUL)(input_0, constant_1)
        output = ops.elementwise(FuncEnum.MUL)(x, constant_2)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = compile_model(output, target, "./tmp", "test_get_constant_names")

        input_0_pt = torch.randn((10000, 2000)).cuda().half()
        constant_1_pt = torch.randn((10000, 2000)).cuda().half()
        constant_2_pt = torch.randn((10000, 2000)).cuda().half()
        output_pt = input_0_pt * constant_1_pt * constant_2_pt
        output_ait = torch.empty_like(input_0_pt)

        module.set_many_constants_with_tensors(
            {"constant_1": constant_1_pt, "constant_2": constant_2_pt}
        )
        module.fold_constants(sync=False)
        module.run_with_tensors([input_0_pt], [output_ait])

        self.assertTrue(torch.equal(output_pt, output_ait))


if __name__ == "__main__":
    unittest.main()
