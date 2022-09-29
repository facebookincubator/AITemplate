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
            param((0, 2, 1), "permute_1"),
            param((1, 0, 2), "permute_2"),
            param((2, 1, 0), "permute_3"),
        ]
    )
    def test_static_shape_3d(self, dims, testname):
        NN = 80
        WW = 300
        CI = 2
        X = Tensor(shape=[NN, WW, CI], name="X", is_input=True)
        op = ops.permute()
        Y = op(X, dims)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = compile_model(Y, target, "./tmp", testname)

        X_pt = torch.randn(NN, WW, CI).cuda().half()
        Y_pt = torch.permute(X_pt, dims)

        y = torch.empty(Y_pt.size()).cuda().half()
        module.run_with_tensors([X_pt], [y])
        self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
