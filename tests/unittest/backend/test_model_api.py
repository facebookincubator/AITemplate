# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import itertools
import unittest
from typing import Callable, Tuple

import numpy as np

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import (
    ConstantTensor,
    ConstantTensorData,
    get_dtype_size,
    HostConstantTensorData,
    IntVar,
    NumpyConstantTensorData,
    TorchConstantTensorData,
)
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.testing.model import AITemplateTensor, Model, torch_to_tensor_info


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

        module = gen_execution_module(output, target, "./tmp", test_name)
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

        module = gen_execution_module(output, target, "./tmp", "test_set_unnamed_input")
        in0_pt = torch.randn([1]).cuda().half()
        in1_pt = torch.randn([1]).cuda().half()
        output_pt = in0_pt - in1_pt

        output_storage = torch.empty_like(output_pt)
        module.RunWithTensors(
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

        module = gen_execution_module(output, target, "./tmp", name)
        input_name_to_index = module.GetInputNameToIndexMap()
        self.assertEqual(input_name_to_index, {"input_0": 0, "input_1": 1})
        output_name_to_index = module.GetOutputNameToIndexMap()
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
            RuntimeError, module.Run, [], [torch_to_tensor_info(outputs[-1])]
        )
        self.assertRaises(
            RuntimeError,
            module.RunWithTensors,
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
            module.Run,
            [
                AITemplateTensor(0, in0_pt_size, "float16"),
                AITemplateTensor(0, in1_pt_size, "float16"),
            ],
            [torch_to_tensor_info(outputs[-1])],
        )
        self.assertRaises(
            RuntimeError,
            module.Run,
            [
                AITemplateTensor(in0_pt.data_ptr(), in0_pt_size, "float16"),
                AITemplateTensor(in1_pt.data_ptr(), in1_pt_size, "float16"),
            ],
            [AITemplateTensor(0, list(outputs[-1].size()), "float16")],
        )

    def test_error_handling_wrong_param_dtypes(self):
        module, (in0_pt, in1_pt), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_error_handling"
        )
        in0_pt_size = list(in0_pt.size())
        in1_pt_size = list(in1_pt.size())
        self.assertRaises(
            RuntimeError,
            module.Run,
            [
                AITemplateTensor(in0_pt.data_ptr(), in0_pt_size, "float32"),
                AITemplateTensor(in1_pt.data_ptr(), in1_pt_size, "float32"),
            ],
            [torch_to_tensor_info(out_ait)],
        )

        self.assertRaises(
            RuntimeError,
            module.Run,
            [
                torch_to_tensor_info(in0_pt),
                torch_to_tensor_info(in1_pt),
            ],
            [AITemplateTensor(out_ait.data_ptr(), list(out_ait.size()), "float32")],
        )

        self.assertRaises(
            RuntimeError,
            module.RunWithTensors,
            [
                in0_pt,
                in1_pt.float(),
            ],
            [out_ait],
        )

        self.assertRaises(
            RuntimeError,
            module.RunWithTensors,
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

        module = gen_execution_module(output, target, "./tmp", "test_container")
        in0_pt = torch.randn((1, 2)).cuda().half()
        const_1_pt = torch.randn((1, 2)).half()
        const_2_pt = torch.randn((1, 2)).half()

        const_1_np = const_1_pt.numpy()
        const_2_np = const_2_pt.numpy()

        module.SetConstant("constant_1", const_1_np)
        module.SetConstant("constant_2", const_2_np)

        output_data = torch.empty([1, 2]).cuda().half()
        module.RunWithTensors([in0_pt], [output_data])

        expected = in0_pt * const_1_pt.cuda() * const_2_pt.cuda()
        self.assertTrue(torch.allclose(output_data, expected))

    def test_get_param_maximum_shape(self):
        for dynamic_shape in (False, True):
            module, inputs, output_np = self._get_simple_graph_and_output(
                "test_get_param_maximum_shape",
                dynamic_shape=dynamic_shape,
            )
            names_to_index = module.GetOutputNameToIndexMap()
            output_shape = module.GetOutputMaximumShape(names_to_index["output"])
            self.assertEqual(output_shape, [1])

            # Test str API
            output_shape = module.GetOutputMaximumShape("output")
            self.assertEqual(output_shape, [1])

    def test_error_handling_maximum_shape(self):
        module, inputs, output_np = self._get_simple_graph_and_output(
            "test_get_param_maximum_shape",
        )
        self.assertRaises(ValueError, module.GetOutputMaximumShape, "not_an_output")
        self.assertRaises(
            TypeError,
            module.GetOutputMaximumShape,
            [],  # not a string or int
        )

    def test_get_param_maximum_shape_output_is_view(self):
        for dynamic_shape in (False, True):
            module, inputs, output_np = self._get_simple_graph_and_output(
                "test_get_param_maximum_shape",
                dynamic_shape=dynamic_shape,
                unsqueeze_output=True,
            )
            names_to_index = module.GetOutputNameToIndexMap()
            output_shape = module.GetOutputMaximumShape(names_to_index["output"])
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

        module = gen_execution_module(output, target, "./tmp", "dynamic_shape_api")
        for batch_size in (1, 10):
            in0_pt = torch.randn([batch_size, 2]).cuda().half()
            in1_pt = torch.randn([batch_size, 2]).cuda().half()
            output_pt = torch.mul(in0_pt, in1_pt)
            output_storage = (
                torch.empty(module.GetOutputMaximumShape("output")).cuda().half()
            )
            outputs_ait = module.RunWithTensors(
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

        module = gen_execution_module(
            output, target, "./tmp", "output_is_alias_of_input"
        )

        in0_pt = torch.randn((2, 2)).cuda().half()
        out_shape = (4, 1) if view_of_view else (4,)
        out_pt = in0_pt.reshape(out_shape)
        out_ait = torch.empty(out_shape).cuda().half()

        module.RunWithTensors([in0_pt], [out_ait])
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

        module = gen_execution_module(input_0, target, "./tmp", "output_is_input")

        in0_pt = torch.randn((2, 2)).cuda().half()
        out_ait = torch.empty((2, 2)).cuda().half()
        module.RunWithTensors([in0_pt], [out_ait])
        self.assertTrue(torch.equal(out_ait, in0_pt))

        inputs = module.GetInputNameToIndexMap()
        self.assertEqual(inputs, {"input_0": 0})

        outputs = module.GetOutputNameToIndexMap()
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

        module = gen_execution_module(
            output, target, "./tmp", "output_is_view_of_constant"
        )

        const_pt = torch.randn((2, 2)).cuda().half()
        out_shape = (4, 1) if view_of_view else (4,)
        out_pt = const_pt.reshape(out_shape)
        out_ait = torch.empty(out_shape).cuda().half()

        module.SetConstant("constant", const_pt.cpu().numpy())
        module.RunWithTensors([], [out_ait])
        self.assertTrue(torch.equal(out_ait, out_pt))

    def test_output_is_view_of_constant(self):
        self._test_output_is_view_of_constant(False)

    def test_output_is_view_of_view_of_constant(self):
        self._test_output_is_view_of_constant(True)

    def test_output_is_constant(self):
        target = detect_target()
        const = Tensor(shape=[2, 2], dtype="float16", name="constant", is_output=True)
        module = gen_execution_module(const, target, "./tmp", "output_is_constant")

        const_pt = torch.randn((2, 2)).cuda().half()
        out_ait = torch.empty((2, 2)).cuda().half()
        module.SetConstant("constant", const_pt.cpu().numpy())
        module.RunWithTensors([], [out_ait])
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
        module = gen_execution_module(
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
        module.RunWithTensors(
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

        module = gen_execution_module(
            [view1, view2], target, "./tmp", "output_is_alias_of_another_output"
        )

        in0_pt = torch.randn((2, 2)).cuda().half()
        view1_pt = in0_pt.reshape((1, 4))
        view2_pt = in0_pt.reshape((4,))

        view1_ait = torch.empty((1, 4)).cuda().half()
        view2_ait = torch.empty((4,)).cuda().half()
        module.RunWithTensors(
            [in0_pt],
            [view1_ait, view2_ait],
        )
        self.assertTrue(torch.equal(view1_pt, view1_ait))
        self.assertTrue(torch.equal(view2_pt, view2_ait))

    def test_benchmark(self):
        module, (in0, in1), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_benchmark"
        )
        runtime_ms, _, shapes = module.Benchmark(
            [
                torch_to_tensor_info(in0),
                torch_to_tensor_info(in1),
            ],
            [torch_to_tensor_info(out_ait)],
        )
        self.assertGreater(runtime_ms, 0)
        self.assertTrue(torch.equal(out_pt, out_ait))
        self.assertEqual(shapes, {"output": [1]})

        runtime_ms, _, tensors = module.BenchmarkWithTensors(
            [in0, in1],
            [out_ait],
        )
        self.assertGreater(runtime_ms, 0)
        self.assertTrue(torch.equal(out_pt, out_ait))
        self.assertEqual(len(tensors), 1)
        self.assertTrue(torch.equal(tensors["output"], in0 * in1))

    def test_get_output_dtype(self):
        module, inputs, output_np = self._get_simple_graph_and_output(
            "test_get_param_dtype"
        )
        self.assertEqual(module.GetOutputDtype(0), 1)

    def test_dynamic_dims_out_of_bounds_error(self):
        target = detect_target()
        batch_size = IntVar([10, 1], name="batch_size")
        input_0 = Tensor(
            shape=[batch_size, 10], dtype="float16", name="input_0", is_input=True
        )
        output = ops.elementwise(FuncEnum.MUL)(input_0, input_0)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True
        module = gen_execution_module(
            output, target, "./tmp", "test_dynamic_dim_out_of_bounds"
        )

        in0_pt = torch.randn((5, 10)).half().cuda()
        out_pt = torch.empty(module.GetOutputMaximumShape("output")).cuda().half()

        self.assertRaises(
            RuntimeError,
            module.Run,
            [AITemplateTensor(in0_pt.data_ptr(), [0, 10], "float16")],
            [torch_to_tensor_info(out_pt)],
        )

        self.assertRaises(
            RuntimeError,
            module.Run,
            [AITemplateTensor(in0_pt.data_ptr(), [11, 10], "float16")],
            [torch_to_tensor_info(out_pt)],
        )

        # Make sure we can run with a valid batch size
        out = module.RunWithTensors(
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

        module = gen_execution_module(
            output,
            target,
            "./tmp",
            "test_output_can_be_null_if_lower_bound_size_is_zero",
        )
        shape = [0, 2]
        module.Run(
            [
                AITemplateTensor(0, shape, "float16"),
                AITemplateTensor(0, shape, "float16"),
            ],
            [AITemplateTensor(0, [10, 2], "float16")],
        )

    def test_with_tensors_api_fails_on_cpu_inputs(self):
        module, (in0, in1), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_fail_on_cpu_inputs"
        )

        self.assertRaises(
            ValueError,
            module.RunWithTensors,
            [in0.cpu(), in1.cpu()],
            [out_ait],
        )
        self.assertRaises(
            ValueError,
            module.RunWithTensors,
            [in0, in1],
            [out_ait.cpu()],
        )
        self.assertRaises(
            ValueError,
            module.BenchmarkWithTensors,
            [in0.cpu(), in1.cpu()],
            [out_ait],
        )
        self.assertRaises(
            ValueError,
            module.BenchmarkWithTensors,
            [in0, in1],
            [out_ait.cpu()],
        )

    def test_with_tensors_api_fails_on_strided_inputs(self):
        target = detect_target()
        input_0 = Tensor(shape=[1, 2], dtype="float16", name="input_0", is_input=True)
        output = ops.elementwise(FuncEnum.MUL)(input_0, input_0)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        module = gen_execution_module(
            output, target, "./tmp", "test_with_tensors_api_fails_on_strided_inputs"
        )

        x = torch.randn((1, 1))
        in0_pt = x.expand((1, 2))
        out_pt = x.expand((1, 2))

        self.assertRaises(
            ValueError,
            module.RunWithTensors,
            [in0_pt],
            [out_pt.contiguous()],
        )
        self.assertRaises(ValueError, module.RunWithTensors, [x.contiguous()], [out_pt])

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

        module = gen_execution_module(
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
            "input_0": torch_to_tensor_info(in0_pt),
            "input_1": torch_to_tensor_info(in1_pt),
            "input_2": torch_to_tensor_info(in2_pt),
        }
        out_args = {
            "output_0": torch_to_tensor_info(out0_pt),
            "output_1": torch_to_tensor_info(out1_pt),
            "output_2": torch_to_tensor_info(out2_pt),
        }

        module.Run(in_args, out_args)
        for out, expect in zip(outputs, expected):
            self.assertTrue(torch.equal(out, expect))
            out.zero_()
        module.Benchmark(in_args, out_args)

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

        module.RunWithTensors(in_args_pt, out_args_pt)
        for out, expect in zip(outputs, expected):
            self.assertTrue(torch.equal(out, expect))
            out.zero_()
        module.BenchmarkWithTensors(in_args_pt, out_args_pt)

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

        self.assertRaises(ValueError, module.RunWithTensors, {}, {})
        self.assertRaises(ValueError, module.RunWithTensors, in_args_pt, {})
        self.assertRaises(ValueError, module.RunWithTensors, in_args_pt, out_args_pt)

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
                gen_execution_module(
                    input_0,
                    target,
                    "./tmp",
                    "test_error_handling_model_init",
                    num_runtimes=num_runtimes,
                )

    def test_constant_tensor_construction_host_data(self):
        self.assertRaises(
            ValueError,
            ConstantTensor,
            HostConstantTensorData(b"\x00" * 10),
            shape=[10, 2],
            dtype="float16",
        )
        # Make sure we can actually construct a constant tensor with the correct
        # size.
        for dtype in ("float16", "float32", "int32", "int64"):
            dtype_size = get_dtype_size(dtype)
            data = HostConstantTensorData(b"\x00" * 20 * dtype_size, dtype=dtype)
            self.assertEqual(data.size(), len(data.to_bytes()))
            self.assertTrue(all(x == 0 for x in data.to_bytes()))
            constant_tensor = ConstantTensor(data, [10, 2], dtype=dtype)
            self.assertIn("data", constant_tensor._attrs)

            data_numpy = NumpyConstantTensorData(np.zeros([10, 2], dtype))
            self.assertEqual(data_numpy.size(), len(data_numpy.to_bytes()))
            self.assertTrue(all(x == 0 for x in data_numpy.to_bytes()))
            constant_tensor_numpy = ConstantTensor(data_numpy, [10, 2], dtype=dtype)
            self.assertIn("data", constant_tensor_numpy._attrs)

    def test_constant_tensor_construction_torch_tensor_data(self):
        small_tensor = torch.randn((5, 2)).cuda().half()
        self.assertRaises(
            ValueError,
            ConstantTensor,
            TorchConstantTensorData(small_tensor),
            shape=[10, 2],
            dtype="float16",
        )
        dtype_to_torch = {
            "float16": torch.float16,
            "float32": torch.float32,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        for dtype in dtype_to_torch.keys():
            tensor = torch.ones((10, 2), dtype=dtype_to_torch[dtype]).cuda()
            data = TorchConstantTensorData(tensor)
            self.assertEqual(data.size(), len(data.to_bytes()))

            data_np = np.frombuffer(data.to_bytes(), dtype=dtype).reshape((10, 2))
            np.testing.assert_equal(data_np, tensor.cpu().numpy())

            constant_tensor = ConstantTensor(data, [10, 2], dtype=dtype)
            self.assertIn("data", constant_tensor._attrs)

    def test_constant_tensor_construction_fails_mismatched_dtypes(self):
        torch_data = TorchConstantTensorData(torch.randn((10, 2)).cuda())
        np_data = NumpyConstantTensorData(np.random.rand(10, 2))
        host_data = HostConstantTensorData("\x00" * 20 * 4, dtype="float32")
        bad_data = (torch_data, np_data, host_data)
        for data in bad_data:
            self.assertRaises(
                ValueError,
                ConstantTensor,
                data,
                shape=[10, 2],
                dtype="float16",
            )

    def _test_use_constant_tensor(
        self,
        make_data: Callable[[torch.Tensor], ConstantTensorData],
        name: str,
        size: int = 3,
    ):
        target = detect_target()
        in0_pt = torch.randn((size,)).half()
        in1_pt = torch.randn((size,)).half()

        in0_data = make_data(in0_pt)
        in0 = ConstantTensor(in0_data, shape=[size], dtype="float16")
        in1_data = make_data(in1_pt)
        in1 = ConstantTensor(in1_data, shape=[size], dtype="float16")

        out = ops.elementwise(FuncEnum.MUL)(in0, in1)
        out._attrs["name"] = "output"
        out._attrs["is_output"] = True

        module = gen_execution_module(out, target, "./tmp", name)

        output_ait = torch.randn((size,)).half().cuda()
        module.RunWithTensors([], [output_ait])

        self.assertTrue(torch.equal(output_ait.cpu(), in0_pt * in1_pt))

    def test_use_internal_constant_tensors_host(self):
        self._test_use_constant_tensor(
            lambda tensor: HostConstantTensorData(tensor.cpu().numpy().tobytes()),
            "test_use_internal_constant_tensors_host",
        )

    def test_use_internal_constant_tensors_gpu(self):
        self._test_use_constant_tensor(
            lambda tensor: TorchConstantTensorData(tensor),
            "test_use_internal_constant_tensors_host",
        )

    def test_use_internal_constant_tensors_huge(self):
        self._test_use_constant_tensor(
            lambda tensor: TorchConstantTensorData(tensor),
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

        module = gen_execution_module(
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

                expected_shapes = {"out0": [a, b], "out1": [a, b]}
                out_shapes = module.Run(
                    {"out0": torch_to_tensor_info(in0)},
                    {
                        "out0": torch_to_tensor_info(out0),
                        "out1": torch_to_tensor_info(out1),
                    },
                )
                self.assertEqual(out_shapes, expected_shapes)

                out_tensors = module.RunWithTensors([in0], [out0, out1])
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

        module = gen_execution_module(
            [input_0, out],
            target,
            "./tmp",
            "test_run_return_value_static_shapes",
        )

        in0 = torch.randn([1, 2, 3, 4]).cuda().half()
        out0 = torch.empty_like(in0)
        out1 = torch.empty_like(in0)
        expected_shapes = {"out0": [1, 2, 3, 4], "out1": [1, 2, 3, 4]}
        out_shapes = module.Run(
            {"out0": torch_to_tensor_info(in0)},
            {
                "out0": torch_to_tensor_info(out0),
                "out1": torch_to_tensor_info(out1),
            },
        )
        self.assertEqual(out_shapes, expected_shapes)

        out_tensors = module.RunWithTensors([in0], [out0, out1])
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

        module = gen_execution_module(
            [input_0, out],
            target,
            "./tmp",
            "test_run_return_value_dynamic_second_dim",
        )

        for dim in range(0, 2):
            in0 = torch.randn([10, dim, 2]).cuda().half()
            out0 = torch.empty_like(in0)
            out1 = torch.empty_like(in0)

            expected_shapes = {"out0": [10, dim, 2], "out1": [10, dim, 2]}
            out_shapes = module.Run(
                {"out0": torch_to_tensor_info(in0)},
                {
                    "out0": torch_to_tensor_info(out0),
                    "out1": torch_to_tensor_info(out1),
                },
            )
            self.assertEqual(out_shapes, expected_shapes)

            out_tensors = module.RunWithTensors([in0], {"out0": out0, "out1": out1})
            self.assertEqual(len(out_tensors), 2)
            self.assertTrue(torch.equal(out_tensors["out0"], in0))
            self.assertTrue(torch.equal(out_tensors["out1"], in0 * in0))

    def test_many_threads_one_stream(self):
        module, (in0, in1), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_many_threads_one_stream"
        )
        runtime_ms, _, _ = module.BenchmarkWithTensors(
            [in0, in1],
            [out_ait],
            num_threads=8,
            count=1000,
        )

    def test_many_threads_many_streams(self):
        module, (in0, in1), (out_pt, out_ait) = self._get_simple_graph_and_output(
            "test_benchmark"
        )
        runtime_ms, _, _ = module.BenchmarkWithTensors(
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
            with gen_execution_module(
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
                    module.GetOutputNameToIndexMap(),
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
            gen_execution_module,
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
            gen_execution_module,
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
            gen_execution_module,
            [output0, output0],
            target,
            "./tmp",
            "test_error_duplicate_output_in_output_tensors_list",
        )

    def test_run_with_outputs_on_host(self):
        (
            module,
            (in0_pt, in1_pt),
            (out_pt, out_storage),
        ) = self._get_simple_graph_and_output("test_run_with_outputs_on_host")
        out_host = out_storage.cpu()
        out_pt_host = out_pt.cpu()
        module._RunWithOutputsOnHost(
            [
                torch_to_tensor_info(in0_pt),
                torch_to_tensor_info(in1_pt),
            ],
            [torch_to_tensor_info(out_host)],
        )

        self.assertTrue(torch.equal(out_pt_host, out_host))
        out_host.zero_()

        module._RunWithTensorsOutputsOnHost(
            {"input_0": in0_pt, "input_1": in1_pt}, {"output": out_host}
        )
        self.assertTrue(torch.equal(out_pt_host, out_host))

    def test_run_with_outputs_on_host_fails_with_outputs_on_device(self):
        (
            module,
            (in0_pt, in1_pt),
            (_, out_storage),
        ) = self._get_simple_graph_and_output(
            "test_run_with_outputs_on_host_fails_with_outputs_on_device"
        )

        self.assertRaises(
            ValueError,
            module._RunWithTensorsOutputsOnHost,
            {"input_0": in0_pt, "input_1": in1_pt},
            {"output": out_storage},
        )

    def test_cannot_use_closed_model(self):
        (
            module,
            (in0_pt, in1_pt),
            (_, out_storage),
        ) = self._get_simple_graph_and_output("test_cannot_use_closed_model")

        module.close()

        self.assertRaises(
            RuntimeError, module.RunWithTensors, [in0_pt, in1_pt], [out_storage]
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
            RuntimeError, m.RunWithTensors, [in0_pt, in1_pt], [out_storage]
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
            RuntimeError, module.RunWithTensors, [in0_pt, in1_pt], [out_storage]
        )
        self.assertRaises(RuntimeError, module.GetOutputDtype, 0)

        # Put it back. Don't want to leak memory!
        module.handle = old_handle


if __name__ == "__main__":
    unittest.main()
