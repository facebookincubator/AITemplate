# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
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
from aitemplate.compiler.base import DynamicProfileStrategy
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ConvDynamicTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_conv_dynamic(
        self,
        test_name="conv_dynamic",
        dtype="float16",
    ):
        target = detect_target()
        batch_size = [2, 32]
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), 24, 24, 4],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[36, 3, 3, 4],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        OP = ops.conv2d(stride=2, pad=1, dilate=1)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
        )
        for batch in batch_size:
            print("Test batch: %d" % batch)
            X_pt = get_random_torch_tensor([batch, 4, 24, 24], dtype=dtype)
            W_pt = get_random_torch_tensor([36, 4, 3, 3], dtype=dtype)
            Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, stride=2, padding=1)
            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w = W_pt.permute((0, 2, 3, 1)).contiguous()
            y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
            module.run_with_tensors({"input_0": x, "input_1": w}, [y])
            y_transpose = y.permute((0, 3, 1, 2))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))

    def test_fp16(self):
        self._test_conv_dynamic(
            test_name="conv_dynamic_fp16",
            dtype="float16",
        )

    def test_fp32_sm80(self):
        self._test_conv_dynamic(
            test_name="conv_dynamic_fp32",
            dtype="float32",
        )

    def _test_conv2d_dynamic(
        self,
        test_name,
        dtype="float16",
    ):
        target = detect_target()
        batch_size = [2, 32]
        h_size = [3, 24]
        w_size = [3, 24]
        X = Tensor(
            shape=[
                IntVar(values=batch_size, name="input_batch"),
                IntVar(values=h_size, name="input_height"),
                IntVar(values=w_size, name="input_width"),
                4,
            ],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W1 = Tensor(
            shape=[12, 3, 3, 4],
            dtype=dtype,
            name="weight_1",
            is_input=True,
        )
        W2 = Tensor(
            shape=[36, 3, 3, 12],
            dtype=dtype,
            name="weight_2",
            is_input=True,
        )
        conv_op1 = ops.conv2d(stride=2, pad=1, dilate=1)
        Y1 = conv_op1(X, W1)
        conv_op2 = ops.conv2d(stride=2, pad=1, dilate=1)
        Y = conv_op2(Y1, W2)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
        )
        batches = [2, 5, 32]
        heights = [3, 11, 24]
        widths = [3, 8, 24]
        test_items = itertools.product(batches, heights, widths)
        for batch, height, width in test_items:
            print(f"Test {batch=}, {height=}, {width=}")
            X_pt = get_random_torch_tensor([batch, 4, height, width], dtype=dtype)
            W1_pt = get_random_torch_tensor([12, 4, 3, 3], dtype=dtype)
            W2_pt = get_random_torch_tensor([36, 12, 3, 3], dtype=dtype)
            Y1_pt = torch.nn.functional.conv2d(X_pt, W1_pt, stride=2, padding=1)
            Y_pt = torch.nn.functional.conv2d(Y1_pt, W2_pt, stride=2, padding=1)
            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            w1 = W1_pt.permute((0, 2, 3, 1)).contiguous()
            w2 = W2_pt.permute((0, 2, 3, 1)).contiguous()
            y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
            module.run_with_tensors({"input_0": x, "weight_1": w1, "weight_2": w2}, [y])
            y_transpose = y.permute((0, 3, 1, 2))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))

    def test_conv2d_fp16(self):
        self._test_conv2d_dynamic(
            test_name="conv2d_dynamic_fp16",
            dtype="float16",
        )

    def _test_conv3d_dynamic(
        self,
        test_name,
        dtype="float16",
    ):
        target = detect_target()
        batch_size = [1, 4]
        d_size = [1, 4]
        h_size = [3, 224]
        w_size = [3, 224]
        stride = (2, 4, 4)
        pad = (1, 2, 2)
        channel = 8
        X = Tensor(
            shape=[
                IntVar(values=batch_size, name="input_batch"),
                IntVar(values=d_size, name="input_depth"),
                IntVar(values=h_size, name="input_height"),
                IntVar(values=w_size, name="input_width"),
                channel,
            ],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W1 = Tensor(
            shape=[16, 3, 5, 5, channel],
            dtype=dtype,
            name="weight_1",
            is_input=True,
        )
        W2 = Tensor(
            shape=[36, 3, 5, 5, 16],
            dtype=dtype,
            name="weight_2",
            is_input=True,
        )
        conv_op1 = ops.conv3d(stride=stride, pad=pad, dilate=1)
        Y1 = conv_op1(X, W1)
        conv_op2 = ops.conv3d(stride=stride, pad=pad, dilate=1)
        Y = conv_op2(Y1, W2)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
        )
        depths = [1, 4]
        heights = [3, 78]
        widths = [3, 8, 224]
        test_items = itertools.product(batch_size, depths, heights, widths)
        for batch, depth, height, width in test_items:
            print(f"Test {batch=}, {depth=}, {height=}, {width=}")
            X_pt = get_random_torch_tensor(
                [batch, channel, depth, height, width], dtype=dtype
            )
            W1_pt = get_random_torch_tensor([16, channel, 3, 5, 5], dtype=dtype)
            W2_pt = get_random_torch_tensor([36, 16, 3, 5, 5], dtype=dtype)
            Y1_pt = torch.nn.functional.conv3d(X_pt, W1_pt, stride=stride, padding=pad)
            Y_pt = torch.nn.functional.conv3d(Y1_pt, W2_pt, stride=stride, padding=pad)
            x = X_pt.permute((0, 2, 3, 4, 1)).contiguous()
            w1 = W1_pt.permute((0, 2, 3, 4, 1)).contiguous()
            w2 = W2_pt.permute((0, 2, 3, 4, 1)).contiguous()
            y = torch.empty_like(Y_pt).permute((0, 2, 3, 4, 1)).contiguous()
            module.run_with_tensors({"input_0": x, "weight_1": w1, "weight_2": w2}, [y])
            y_transpose = y.permute((0, 4, 1, 2, 3))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=0.05, rtol=0.05))

    def test_conv3d_fp16_sm80(self):
        self._test_conv3d_dynamic(
            test_name="conv3d_dynamic_fp16",
            dtype="float16",
        )


filter_test_cases_by_test_env(ConvDynamicTestCase)


if __name__ == "__main__":
    unittest.main()
