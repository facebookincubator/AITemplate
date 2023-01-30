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
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_zeros_tensor,
)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class PadLastDim(unittest.TestCase):
    def _test_static_shape_4d(
        self,
        copy_op=False,
        test_name="static_shape_4d",
        dtype="float16",
    ):
        NN = 2
        HH = 7
        WW = 7
        CI = 262
        CO = 264
        X = Tensor(
            shape=[NN, HH, WW, CI],
            name="X",
            is_input=True,
            dtype=dtype,
        )
        op = ops.pad_last_dim(4, CO)
        if copy_op:
            op = ops.pad_last_dim(**op._get_op_attributes())
        Y = op(X)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = compile_model(Y, target, "./tmp", test_name)

        X_pt = get_random_torch_tensor([NN, HH, WW, CI], dtype=dtype)
        Pad_pt = get_torch_zeros_tensor([NN, HH, WW, CO - CI], dtype=dtype)
        Y_pt = torch.cat([X_pt, Pad_pt], dim=3)

        y = torch.empty_like(Y_pt)
        module.run_with_tensors([X_pt], [y])
        self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))

    def test_static_shape_4d_fp16(self):
        self._test_static_shape_4d(
            test_name="static_shape_4d_fp16",
            dtype="float16",
        )
        self._test_static_shape_4d(
            copy_op=True,
            test_name="static_shape_4d_fp16_copy_op",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_static_shape_4d_fp32(self):
        self._test_static_shape_4d(
            test_name="static_shape_4d_fp32",
            dtype="float32",
        )
        self._test_static_shape_4d(
            copy_op=True,
            test_name="static_shape_4d_fp32_copy_op",
            dtype="float32",
        )

    def _test_static_shape_2d(
        self,
        copy_op=False,
        test_name="static_shape_2d",
        dtype="float16",
    ):
        NN = 32
        CI = 259
        CO = 264
        X = Tensor(
            shape=[NN, CI],
            name="X",
            is_input=True,
            dtype=dtype,
        )
        op = ops.pad_last_dim(2, CO)
        if copy_op:
            op = ops.pad_last_dim(**op._get_op_attributes())
        Y = op(X)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = compile_model(Y, target, "./tmp", test_name)

        X_pt = get_random_torch_tensor([NN, CI], dtype=dtype)
        Pad_pt = get_torch_zeros_tensor([NN, CO - CI], dtype=dtype)
        Y_pt = torch.cat([X_pt, Pad_pt], dim=1)

        y = torch.empty_like(Y_pt)
        module.run_with_tensors([X_pt], [y])
        self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))

    def test_static_shape_2d_fp16(self):
        self._test_static_shape_2d(
            test_name="static_shape_2d_fp16",
            dtype="float16",
        )
        self._test_static_shape_2d(
            copy_op=True,
            test_name="static_shape_2d_fp16_copy_op",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_static_shape_2d_fp32(self):
        self._test_static_shape_2d(
            test_name="static_shape_2d_fp32",
            dtype="float32",
        )
        self._test_static_shape_2d(
            copy_op=True,
            test_name="static_shape_2d_fp32_copy_op",
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
