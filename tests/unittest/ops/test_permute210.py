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
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from parameterized import param, parameterized


class Permute210Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Permute210Test, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_permute_210(
        self,
        input_shape,
        test_name="permute210",
        dtype="float16",
    ):
        X = Tensor(
            shape=input_shape,
            name="X",
            dtype=dtype,
            is_input=True,
        )
        op = ops.permute210()
        Y = op(X)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = compile_model(Y, target, "./tmp", f"perm210_{self._test_id}")
        self._test_id += 1

        batch_dim = input_shape[0]
        if isinstance(batch_dim, IntVar):
            input_shapes = [(d, *input_shape[1:]) for d in batch_dim._attrs["values"]]
        else:
            input_shapes = [input_shape]

        for shape in input_shapes:
            X_pt = get_random_torch_tensor(shape, dtype=dtype)
            Y_pt = torch.permute(X_pt, [2, 1, 0])
            y = torch.empty_like(Y_pt).contiguous()
            module.run_with_tensors([X_pt], [y])
            self.assertTrue(torch.equal(y, Y_pt))

    @parameterized.expand(
        [
            param((2, 80, 300)),
            param((80, 300, 2)),
            param((300, 2, 80)),
            param((31, 7, 3)),
            param((128, 128, 63)),
            param((256, 256, 64)),
            param((IntVar([2, 3]), 256, 64)),
        ]
    )
    def test_permute210_fp16(self, input_shape):
        self._test_permute_210(
            input_shape=input_shape,
            test_name="permute210_fp16",
            dtype="float16",
        )

    @parameterized.expand(
        [
            param((2, 80, 300)),
            param((80, 300, 2)),
            param((300, 2, 80)),
            param((31, 7, 3)),
            param((128, 128, 63)),
            param((256, 256, 64)),
            param((IntVar([2, 3]), 256, 64)),
        ]
    )
    @unittest.skipIf(detect_target().name() == "rocm", "FP32 is not supported on ROCm")
    def test_permute210_fp32(self, input_shape):
        self._test_permute_210(
            input_shape=input_shape,
            test_name="permute210_fp32",
            dtype="float32",
        )

    @parameterized.expand(
        [
            param((2, 80, 300)),
            param((80, 300, 2)),
            param((300, 2, 80)),
            param((31, 7, 3)),
            param((128, 128, 63)),
            param((256, 256, 64)),
            param((IntVar([2, 3]), 256, 64)),
        ]
    )
    @unittest.skipIf(detect_target().name() == "rocm", "bf16 is not supported on ROCm")
    def test_permute210_bf16(self, input_shape):
        self._test_permute_210(
            input_shape=input_shape,
            test_name="permute210_bf16",
            dtype="bfloat16",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
