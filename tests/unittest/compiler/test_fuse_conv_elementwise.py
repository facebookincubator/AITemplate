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
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import graph_has_op
from aitemplate.utils import shape_utils


@unittest.skipIf(
    detect_target().name() == "cuda" and detect_target()._arch < "80",
    "On CUDA, only supported on > SM80 arch.",
)
class FuseConvCase(unittest.TestCase):
    def _build_conv2d(
        self,
        batch_dim,
        CO,
        HH,
        WW,
        CI,
        filter_HW,
        stride=1,
        transpose=False,
    ):
        X = Tensor(
            shape=[batch_dim, HH, WW, CI],
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        W = Tensor(
            shape=[CO, filter_HW, filter_HW, CI],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        if transpose:
            conv2d = ops.transposed_conv2d(stride=stride, pad=0)(X, W)
        else:
            conv2d = ops.conv2d(stride=stride, pad=0)(X, W)

        return conv2d

    def test_do_not_fuse_with_add_not_1d(self):
        """
        We can't turn conv2d into conv2d_bias if the thing we do
        an add with is not 1d.
        """

        # Keep IntImm batch here just not to mess with profiling strategy
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B, name="batch_dim")
        CO, HH, WW, CI = 256, 28, 28, 128
        filter_HW = 3

        bias = Tensor(
            shape=[batch_dim, 26, 26, CO], dtype="float16", name="bias", is_input=True
        )
        conv2d = self._build_conv2d(batch_dim, CO, HH, WW, CI, filter_HW)
        output = ops.elementwise(FuncEnum.ADD)(bias, conv2d)
        output._attrs["is_output"] = True
        output._attrs["name"] = "output_0"

        target = detect_target()
        module = compile_model(
            output, target, "./tmp", "test_do_not_fuse_with_add_not_1d"
        )

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "output_0":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "fused_elementwise")

        for b in B:
            X_pt = torch.randn(b, CI, HH, WW).cuda().half()
            W_pt = torch.randn(CO, CI, filter_HW, filter_HW).cuda().half()
            Y_pt = torch.nn.functional.conv2d(X_pt, W_pt)
            B_pt = torch.randn(Y_pt.size()).cuda().half()
            Y_pt = Y_pt + B_pt

            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w = W_pt.permute((0, 2, 3, 1)).contiguous()
            b_pt = B_pt.permute((0, 2, 3, 1)).contiguous()
            inputs = {"input_0": x, "input_1": w, "bias": b_pt}

            y = torch.empty([b, 26, 26, CO]).cuda().half()
            module.run_with_tensors(inputs, [y])
            y_transpose = y.permute(0, 3, 1, 2)
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))

    def test_do_not_fuse_transpose_with_add_not_1d(self):
        """
        We can't turn transposed_conv2d into transposed_conv2d_bias if the thing we do
        an add with is not 1d.
        """
        B = [1]
        CO, HH, WW, CI = 256, 28, 28, 256
        filter_HW = 2

        batch_dim = shape_utils.gen_int_var_min_max(B, name="batch_dim")
        bias = Tensor(
            shape=[batch_dim, 56, 56, CO], dtype="float16", name="bias", is_input=True
        )
        conv2d = self._build_conv2d(
            batch_dim, CO, HH, WW, CI, filter_HW, stride=2, transpose=True
        )
        output = ops.elementwise(FuncEnum.ADD)(bias, conv2d)
        output._attrs["is_output"] = True
        output._attrs["name"] = "output_0"

        target = detect_target()
        module = compile_model(
            output, target, "./tmp", "test_do_not_fuse_with_add_not_1d"
        )

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "output_0":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "fused_elementwise")

        for b in B:
            X_pt = torch.randn(b, CI, HH, WW).cuda().half()
            W_pt = torch.randn(CO, CI, filter_HW, filter_HW).cuda().half()
            W_pt = torch.randn(CO, CI, filter_HW, filter_HW).cuda().half()
            Y_pt = torch.nn.functional.conv_transpose2d(X_pt, W_pt, stride=2)
            B_pt = torch.randn(b, CO, 56, 56).cuda().half()
            Y_pt = Y_pt + B_pt

            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w = W_pt.permute((0, 2, 3, 1)).contiguous()
            b_pt = B_pt.permute((0, 2, 3, 1)).contiguous()
            inputs = {"input_0": x, "input_1": w, "bias": b_pt}

            y = torch.empty([b, 56, 56, CO]).cuda().half()
            module.run_with_tensors(inputs, [y])
            y_transpose = y.permute(0, 3, 1, 2)
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))


class FuseConvBiasCase(unittest.TestCase):
    def _build_conv2d_bias(self, batch_dim, CO, HH, WW, CI, filter_HW, decomposed):
        X = Tensor(
            shape=[batch_dim, HH, WW, CI],
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        W = Tensor(
            shape=[CO, filter_HW, filter_HW, CI],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        B = Tensor(shape=[CO], dtype="float16", name="input_2", is_input=True)
        if decomposed:
            conv2d = ops.conv2d(stride=1, pad=1, dilate=1)(X, W)
            conv2d_bias = ops.elementwise(FuncEnum.ADD)(conv2d, B)
        else:
            conv2d_bias = ops.conv2d_bias(stride=1, pad=1, dilate=1)(X, W, B)

        return conv2d_bias

    def test_conv2d_bias(self):
        # Keep IntImm batch here just not to mess with profiling strategy
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B, name="batch_dim")
        CO, HH, WW, CI = 256, 28, 28, 128
        filter_HW = 3

        conv2d_bias = self._build_conv2d_bias(
            batch_dim, CO, HH, WW, CI, filter_HW, True
        )
        conv2d_bias._attrs["is_output"] = True
        conv2d_bias._attrs["name"] = "output_0"

        target = detect_target()
        module = compile_model(conv2d_bias, target, "./tmp", "test_conv2d_bias")

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "output_0":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "conv2d_bias")

        for b in B:
            X_pt = torch.randn(b, CI, HH, WW).cuda().half()
            W_pt = torch.randn(CO, CI, filter_HW, filter_HW).cuda().half()
            B_pt = torch.randn(1, CO, 1, 1).cuda().half()
            Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=1)
            Y_pt = Y_pt + B_pt

            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w = W_pt.permute((0, 2, 3, 1)).contiguous()
            inputs = {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}

            y = torch.empty([b, HH, WW, CO]).cuda().half()
            module.run_with_tensors(inputs, [y])
            y_transpose = y.permute(0, 3, 1, 2)
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))

    def test_conv2d_bias_add_relu(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B, name="batch_dim")
        CO, HH, WW, CI = 256, 28, 28, 128
        filter_HW = 3

        conv2d_bias = self._build_conv2d_bias(
            batch_dim, CO, HH, WW, CI, filter_HW, False
        )
        D = Tensor(
            shape=[batch_dim, HH, WW, CO],
            dtype="float16",
            name="input_3",
            is_input=True,
        )
        conv2d_bias_add = ops.elementwise(FuncEnum.ADD)(conv2d_bias, D)
        conv2d_bias_add_relu = ops.elementwise(FuncEnum.RELU)(conv2d_bias_add)
        conv2d_bias_add_relu._attrs["is_output"] = True
        conv2d_bias_add_relu._attrs["name"] = "output_0"

        target = detect_target()
        module = compile_model(
            conv2d_bias_add_relu, target, "./tmp", "test_conv2d_bias_add_relu"
        )

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "output_0":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "conv2d_bias_add_relu")

        for b in B:
            X_pt = torch.randn(b, CI, HH, WW).cuda().half()
            W_pt = torch.randn(CO, CI, filter_HW, filter_HW).cuda().half()
            B_pt = torch.randn(1, CO, 1, 1).cuda().half()
            D_pt = torch.randn(b, CO, HH, WW).cuda().half()
            Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=1)
            Y_pt = Y_pt + B_pt + D_pt
            Y_pt = torch.nn.functional.relu(Y_pt)

            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w = W_pt.permute((0, 2, 3, 1)).contiguous()
            d = D_pt.permute((0, 2, 3, 1)).contiguous()
            inputs = {
                "input_0": x,
                "input_1": w,
                "input_2": B_pt.squeeze(),
                "input_3": d,
            }

            y = torch.empty([b, HH, WW, CO]).cuda().half()
            module.run_with_tensors(inputs, [y])
            y_transpose = y.permute(0, 3, 1, 2)
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))

    def test_conv2d_bias_relu(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B, name="batch_dim")
        CO, HH, WW, CI = 256, 28, 28, 128
        filter_HW = 3

        conv2d_bias = self._build_conv2d_bias(
            batch_dim, CO, HH, WW, CI, filter_HW, False
        )
        conv2d_bias_relu = ops.elementwise(FuncEnum.RELU)(conv2d_bias)
        conv2d_bias_relu._attrs["is_output"] = True
        conv2d_bias_relu._attrs["name"] = "output_0"

        target = detect_target()
        module = compile_model(
            conv2d_bias_relu, target, "./tmp", "test_conv2d_bias_relu"
        )

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "output_0":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "conv2d_bias_relu")

        for b in B:
            X_pt = torch.randn(b, CI, HH, WW).cuda().half()
            W_pt = torch.randn(CO, CI, filter_HW, filter_HW).cuda().half()
            B_pt = torch.randn(1, CO, 1, 1).cuda().half()
            Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=1)
            Y_pt = Y_pt + B_pt
            Y_pt = torch.nn.functional.relu(Y_pt)

            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w = W_pt.permute((0, 2, 3, 1)).contiguous()
            inputs = {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}

            y = torch.empty([b, HH, WW, CO]).cuda().half()
            module.run_with_tensors(inputs, [y])
            y_transpose = y.permute(0, 3, 1, 2)
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))

    def test_conv2d_bias_sigmoid(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B, name="batch_dim")
        CO, HH, WW, CI = 256, 28, 28, 128
        filter_HW = 3

        conv2d_bias = self._build_conv2d_bias(
            batch_dim, CO, HH, WW, CI, filter_HW, False
        )
        conv2d_bias_sigmoid = ops.elementwise(FuncEnum.SIGMOID)(conv2d_bias)
        conv2d_bias_sigmoid._attrs["is_output"] = True
        conv2d_bias_sigmoid._attrs["name"] = "output_0"

        target = detect_target()
        module = compile_model(
            conv2d_bias_sigmoid, target, "./tmp", "test_conv2d_bias_sigmoid"
        )

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "output_0":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "conv2d_bias_sigmoid")

        for b in B:
            X_pt = torch.randn(b, CI, HH, WW).cuda().half()
            W_pt = torch.randn(CO, CI, filter_HW, filter_HW).cuda().half()
            B_pt = torch.randn(1, CO, 1, 1).cuda().half()
            Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=1)
            Y_pt = Y_pt + B_pt
            Y_pt = torch.sigmoid(Y_pt)

            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w = W_pt.permute((0, 2, 3, 1)).contiguous()
            inputs = {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}

            y = torch.empty([b, HH, WW, CO]).cuda().half()
            module.run_with_tensors(inputs, [y])
            y_transpose = y.permute(0, 3, 1, 2)
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))

    def test_conv2d_bias_add_fusion(self):
        target = detect_target()
        if target.name() == "rocm":
            return

        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B, name="batch_dim")
        CO, HH, WW, CI = 256, 28, 28, 128
        filter_HW = 3
        R = Tensor(
            shape=[batch_dim, HH, WW, CO],
            dtype="float16",
            name="residual",
            is_input=True,
        )

        conv2d_bias = self._build_conv2d_bias(
            batch_dim, CO, HH, WW, CI, filter_HW, False
        )
        conv2d_bias_add = ops.elementwise(FuncEnum.ADD)(conv2d_bias, R)
        conv2d_bias_add._attrs["is_output"] = True
        conv2d_bias_add._attrs["name"] = "output_0"

        module = compile_model(conv2d_bias_add, target, "./tmp", "test_conv2d_bias_add")

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "output_0":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "conv2d_bias_add_identity")

        for b in B:
            X_pt = torch.randn(b, CI, HH, WW).cuda().half()
            W_pt = torch.randn(CO, CI, filter_HW, filter_HW).cuda().half()
            B_pt = torch.randn(1, CO, 1, 1).cuda().half()
            R_pt = torch.randn(b, CO, HH, WW).cuda().half()
            Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=1)
            Y_pt = Y_pt + B_pt + R_pt

            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w = W_pt.permute((0, 2, 3, 1)).contiguous()
            r = R_pt.permute((0, 2, 3, 1)).contiguous()
            inputs = {
                "input_0": x,
                "input_1": w,
                "input_2": B_pt.squeeze(),
                "residual": r,
            }

            y = torch.empty([b, HH, WW, CO]).cuda().half()
            module.run_with_tensors(inputs, [y])
            y_transpose = y.permute(0, 3, 1, 2)
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))

    def test_conv2d_bias_add_do_not_fuse(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B, name="batch_dim")
        CO, HH, WW, CI = 256, 28, 28, 128
        filter_HW = 3
        R = Tensor(
            shape=[batch_dim, 1, WW, CO],
            dtype="float16",
            name="residual",
            is_input=True,
        )

        conv2d_bias = self._build_conv2d_bias(
            batch_dim, CO, HH, WW, CI, filter_HW, False
        )
        conv2d_bias_add = ops.elementwise(FuncEnum.ADD)(conv2d_bias, R)
        conv2d_bias_add._attrs["is_output"] = True
        conv2d_bias_add._attrs["name"] = "output_0"

        target = detect_target()
        module = compile_model(conv2d_bias_add, target, "./tmp", "test_conv2d_bias_add")

        graph = module.debug_sorted_graph

        self.assertFalse(graph_has_op(graph, "conv2d_bias_add_identity"))
        self.assertTrue(graph_has_op(graph, "conv2d_bias"))


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class FuseConvBiasFewChannelCase(unittest.TestCase):
    def test_conv2d_bias_relu_few_channels(self):
        HH, WW, CI, CO, batch = 224, 224, 4, 64, 4
        KK = 7
        stride = 2
        pad = 3
        target = detect_target()
        X = Tensor(
            shape=[batch, HH, WW, CI],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[CO, KK, KK, CI], dtype="float16", name="input_1", is_input=True
        )
        B = Tensor(shape=[CO], dtype="float16", name="input_2", is_input=True)
        OP = ops.conv2d_bias_few_channels(stride=stride, pad=pad, dilate=1)
        Y = OP(X, W, B)
        Y = ops.elementwise(FuncEnum.RELU)(Y)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", "test_conv_bias_relu_few_channels")

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "output_0":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "conv2d_bias_relu_few_channels")

        X_pt = torch.randn(batch, CI, HH, WW).cuda().half()
        W_pt = torch.randn(CO, CI, KK, KK).cuda().half()
        B_pt = torch.randn(1, CO, 1, 1).cuda().half()
        Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=pad, stride=stride)
        Y_pt = Y_pt + B_pt
        Y_pt = torch.nn.functional.relu(Y_pt)
        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        inputs = {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}
        y = torch.empty([batch, HH // stride, WW // stride, CO]).cuda().half()
        module.run_with_tensors(inputs, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
@unittest.skipIf(
    detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
    "Not supported by CUDA < SM80.",
)
class FuseTransposedConvCase(unittest.TestCase):
    def _build_transposedConv2d_bias_relu_chain(
        self, batch, HH, WW, CI, CO, filter_HW, stride, pad, dilate, depth, decomposed
    ):
        X = Tensor(
            shape=[batch, HH, WW, CI],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[CO, filter_HW, filter_HW, CI],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        B = Tensor(shape=[CO], dtype="float16", name="input_2", is_input=True)
        if decomposed:
            transposed_conv2d = ops.transposed_conv2d(
                stride=stride, pad=pad, dilate=dilate
            )(X, W)
            if depth == 0:
                return transposed_conv2d

            transposed_conv2d_bias = ops.elementwise(FuncEnum.ADD)(transposed_conv2d, B)
        else:
            transposed_conv2d_bias = ops.transposed_conv2d_bias(
                stride=stride, pad=pad, dilate=dilate
            )(X, W, B)
            if depth == 0:
                raise RuntimeError("depth == 0 needs to be decomposed.")
        if depth == 1:
            return transposed_conv2d_bias

        transposed_conv2d_bias_relu = ops.elementwise(FuncEnum.RELU)(
            transposed_conv2d_bias
        )
        if depth == 2:
            return transposed_conv2d_bias_relu

        raise RuntimeError(f"depth should be <= 2, unknown depth {depth}")

    def _test_transposed_conv2d_bias(self, decomposed):
        batch = 4
        HH, WW, CI, CO = 14, 14, 256, 256
        filter_HW = 2
        stride = 2
        pad = 0
        dilate = 1
        transposed_conv2d_bias = self._build_transposedConv2d_bias_relu_chain(
            batch,
            HH,
            WW,
            CI,
            CO,
            filter_HW,
            stride,
            pad,
            dilate,
            1,
            decomposed=decomposed,
        )
        transposed_conv2d_bias._attrs["is_output"] = True
        transposed_conv2d_bias._attrs["name"] = "output_0"

        target = detect_target()
        module = compile_model(
            transposed_conv2d_bias,
            target,
            "./tmp",
            f"fuse_transpose_conv2d_bias_{decomposed}",
        )

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "output_0":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "transposed_conv2d_bias")

        X_pt = torch.randn(batch, CI, HH, WW).cuda().half()
        W_pt = torch.randn(CO, CI, filter_HW, filter_HW).cuda().half()
        B_pt = torch.randn(1, CO, 1, 1).cuda().half()
        Y_pt = torch.nn.functional.conv_transpose2d(
            X_pt, W_pt, padding=pad, stride=stride
        )
        Y_pt = Y_pt + B_pt

        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        y = torch.empty([batch, 28, 28, CO]).cuda().half()
        module.run_with_tensors(
            {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}, [y]
        )
        y_transpose = y.permute((0, 3, 1, 2))
        self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))

    def test_transposed_conv2d_bias(self):
        self._test_transposed_conv2d_bias(True)
        self._test_transposed_conv2d_bias(False)

    def _test_transposed_conv2d_bias_relu(self, decomposed):
        batch = 4
        HH, WW, CI, CO = 14, 14, 256, 256
        filter_HW = 2
        stride = 2
        pad = 0
        dilate = 1
        transposed_conv2d_bias_relu = self._build_transposedConv2d_bias_relu_chain(
            batch,
            HH,
            WW,
            CI,
            CO,
            filter_HW,
            stride,
            pad,
            dilate,
            2,
            decomposed=decomposed,
        )
        transposed_conv2d_bias_relu._attrs["is_output"] = True
        transposed_conv2d_bias_relu._attrs["name"] = "output_0"

        target = detect_target()
        module = compile_model(
            transposed_conv2d_bias_relu,
            target,
            "./tmp",
            f"fuse_transpose_conv2d_bias_relu_{decomposed}",
        )

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "output_0":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "transposed_conv2d_bias_relu")

        X_pt = torch.randn(batch, CI, HH, WW).cuda().half()
        W_pt = torch.randn(CO, CI, filter_HW, filter_HW).cuda().half()
        B_pt = torch.randn(1, CO, 1, 1).cuda().half()
        Y_pt = torch.nn.functional.conv_transpose2d(
            X_pt, W_pt, padding=pad, stride=stride
        )
        Y_pt = Y_pt + B_pt
        Y_pt = torch.relu(Y_pt)

        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        y = torch.empty([batch, 28, 28, CO]).cuda().half()
        module.run_with_tensors(
            {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}, [y]
        )
        y_transpose = y.permute((0, 3, 1, 2))
        self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))

    def test_transposed_conv2d_bias_relu(self):
        self._test_transposed_conv2d_bias_relu(True)
        self._test_transposed_conv2d_bias_relu(False)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
