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
from parameterized import param, parameterized


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class PermuteTest(unittest.TestCase):
    @parameterized.expand(
        [
            param((80, 300, 2), (0, 2, 1), "permute_1"),
            param((80, 300, 2), (1, 0, 2), "permute_2"),
            param((80, 300, 2), (2, 1, 0), "permute_3"),
            param((5, 113, 15, 31), (0, 2, 1, 3), "permute_4"),
            param((3, 1, 113, 15, 64), (2, 0, 3, 1, 4), "permute_5"),
            param((8, 29, 100000, 3), (0, 2, 1, 3), "permute_6"),
            param((32, 12, 4096, 64), (0, 2, 1, 3), "permute_7"),
            param((1, 12, 128, 64), (0, 2, 1, 3), "permute_8"),
            param((2, 3, 4, 5), (3, 2, 1, 0), "permute_9"),
            param((3, 5, 128, 514), (2, 3, 0, 1), "permute_10"),
            param((128, 512), (1, 0), "permute_11"),
        ]
    )
    def test_static_shape_3d(self, input_shapes, dims, testname):
        X = Tensor(shape=input_shapes, name="X", is_input=True)
        op = ops.permute()
        Y = op(X, dims)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = compile_model(Y, target, "./tmp", testname)

        count = 1
        for dim in input_shapes:
            count *= dim
        X_pt = torch.randn(input_shapes).cuda().half()
        Y_pt = torch.permute(X_pt, dims)

        y = torch.empty(Y_pt.size()).cuda().half()
        module.run_with_tensors([X_pt], [y])

        # mean, _, _ = module.benchmark_with_tensors([X_pt], [y], count=1000)
        # mem = 1
        # for dim in input_shapes:
        #     mem *= dim
        # bw = 2 * 2 * mem / (mean * 1e-3 * 1e9)  # GB/s
        # print(f"bw: {bw} GB/s")

        self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
