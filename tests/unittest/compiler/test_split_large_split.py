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
import unittest

import numpy as np
import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


class SplitLargeSplitTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SplitLargeSplitTestCase, self).__init__(*args, **kwargs)

    def _run_split(
        self,
        *,
        input_shape,
        split_size_or_sections,
        dim=None,
        input_type="float16",
        testname=None,
    ):
        logging.info(
            f"Test input shape {input_shape}, "
            f"split_size_or_sections={split_size_or_sections}, dim={dim}"
        )

        split_op = ops.split()
        # generate torch reference result
        X_pt = get_random_torch_tensor(input_shape, input_type)
        Ys_pt = (
            torch.split(X_pt, split_size_or_sections)
            if dim is None
            else torch.split(X_pt, split_size_or_sections, dim)
        )
        target = detect_target()
        X = Tensor(shape=input_shape, dtype=input_type, name="input_0", is_input=True)
        Ys = (
            split_op(X, split_size_or_sections)
            if dim is None
            else split_op(X, split_size_or_sections, dim)
        )
        np.testing.assert_equal(len(Ys_pt), len(Ys))

        y_shapes = []
        for idx, Y in enumerate(Ys):
            Y._attrs["name"] = f"output_{idx}"
            Y._attrs["is_output"] = True
            y_shape = [d._attrs["values"][0] for d in Y._attrs["shape"]]
            logging.info(f"AITemplate output_{idx} shape: {y_shape}")
            y_shapes.append(y_shape)

        module = compile_model(Ys, target, "./tmp", testname)

        outputs = {
            f"output_{idx}": get_torch_empty_tensor(y_shape, input_type)
            for idx, y_shape in enumerate(y_shapes)
        }
        module.run_with_tensors([X_pt], outputs)

        for idx, y_pt in enumerate(Ys_pt):
            y = outputs[f"output_{idx}"]
            self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

    def test_split(self):
        self._run_split(
            input_shape=[4096, 128, 64],
            split_size_or_sections=32,
            dim=0,
            testname="split_0",
        )
        self._run_split(
            input_shape=[128, 2048, 64],
            split_size_or_sections=3,
            dim=1,
            testname="split_1",
        )
        self._run_split(
            input_shape=[64, 128, 1024],
            split_size_or_sections=2,
            dim=2,
            testname="split_2",
        )
        self._run_split(
            input_shape=[64, 128, 1024],
            split_size_or_sections=7,
            dim=2,
            testname="split_3",
        )

    def test_split_with_strided_op(self):
        input_shape = [64, 128, 1024]
        split_size_or_sections = 3
        split_dim = 2
        strided_op_idx = [100, 200, 300]

        split_op = ops.split()
        # generate torch reference result
        X_pt = get_random_torch_tensor(input_shape)
        Ys_pt = list(torch.split(X_pt, split_size_or_sections, split_dim))
        for idx in strided_op_idx:
            Ys_pt[idx] = torch.relu(Ys_pt[idx])
        target = detect_target()
        X = Tensor(shape=input_shape, name="input_0", is_input=True)
        Ys = list(split_op(X, split_size_or_sections, split_dim))
        np.testing.assert_equal(len(Ys_pt), len(Ys))

        y_shapes = []
        for idx, Y in enumerate(Ys):
            if idx in strided_op_idx:
                Y = ops.elementwise(FuncEnum.RELU)(Y)
                Ys[idx] = Y
            Y._attrs["name"] = f"output_{idx}"
            Y._attrs["is_output"] = True

            y_shape = [d._attrs["values"][0] for d in Y._attrs["shape"]]
            logging.info(f"AITemplate output_{idx} shape: {y_shape}")
            y_shapes.append(y_shape)

        module = compile_model(Ys, target, "./tmp", "split_with_strided_ops")

        outputs = {
            f"output_{idx}": get_torch_empty_tensor(y_shape)
            for idx, y_shape in enumerate(y_shapes)
        }
        module.run_with_tensors([X_pt], outputs)

        for idx, y_pt in enumerate(Ys_pt):
            y = outputs[f"output_{idx}"]
            self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
