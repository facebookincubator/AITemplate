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


class Permute021Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Permute021Test, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_permute_021(
        self,
        input_shape,
        dims,
        test_name="permute021",
        dtype="float16",
    ):
        X = Tensor(
            shape=input_shape,
            name="X",
            dtype=dtype,
            is_input=True,
        )
        op = ops.permute021()
        Y = op(X)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = compile_model(Y, target, "./tmp", f"perm021_{self._test_id}")
        self._test_id += 1

        batch_dim = input_shape[0]
        if isinstance(batch_dim, IntVar):
            input_shapes = [(d, *input_shape[1:]) for d in batch_dim._attrs["values"]]
        else:
            input_shapes = [input_shape]

        for shape in input_shapes:
            X_pt = get_random_torch_tensor(shape, dtype=dtype)
            Y_pt = torch.permute(X_pt, dims)
            y = torch.empty_like(Y_pt).contiguous()
            module.run_with_tensors([X_pt], [y])
            self.assertTrue(torch.equal(y, Y_pt))

    @parameterized.expand(
        [
            param((2, 384, 262), (0, 2, 1)),
            param((2, 3, 384, 262), (0, 1, 3, 2)),
            param((2, 3, 4, 384, 262), (0, 1, 2, 4, 3)),
            param((IntVar([2, 3]), 384, 262), (0, 2, 1)),
            param((IntVar([2, 3, 4]), 5, 384, 262), (0, 1, 3, 2)),
        ]
    )
    def test_permute021_fp16(self, input_shape, dims):
        self._test_permute_021(
            input_shape=input_shape,
            dims=dims,
            test_name="permute021_fp16",
            dtype="float16",
        )

    @parameterized.expand(
        [
            param((2, 384, 262), (0, 2, 1)),
            param((2, 3, 384, 262), (0, 1, 3, 2)),
            param((2, 3, 4, 384, 262), (0, 1, 2, 4, 3)),
            param((IntVar([2, 3]), 384, 262), (0, 2, 1)),
            param((IntVar([2, 3, 4]), 5, 384, 262), (0, 1, 3, 2)),
        ]
    )
    @unittest.skipIf(detect_target().name() == "rocm", "FP32 is not supported on ROCm")
    def test_permute021_fp32(self, input_shape, dims):
        self._test_permute_021(
            input_shape=input_shape,
            dims=dims,
            test_name="permute021_fp32",
            dtype="float32",
        )

    @parameterized.expand(
        [
            param((2, 384, 262), (0, 2, 1)),
            param((2, 3, 384, 262), (0, 1, 3, 2)),
            param((2, 3, 4, 384, 262), (0, 1, 2, 4, 3)),
            param((IntVar([2, 3]), 384, 262), (0, 2, 1)),
            param((IntVar([2, 3, 4]), 5, 384, 262), (0, 1, 3, 2)),
        ]
    )
    @unittest.skipIf(detect_target().name() == "rocm", "bf16 is not supported on ROCm")
    def test_permute021_bf16(self, input_shape, dims):
        self._test_permute_021(
            input_shape=input_shape,
            dims=dims,
            test_name="permute021_bf16",
            dtype="bfloat16",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
