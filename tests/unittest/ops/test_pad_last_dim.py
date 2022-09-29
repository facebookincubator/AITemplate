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


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class PadLastDim(unittest.TestCase):
    def test_static_shape_4d(self):
        NN = 2
        HH = 7
        WW = 7
        CI = 262
        CO = 264
        X = Tensor(shape=[NN, HH, WW, CI], name="X", is_input=True)
        op = ops.pad_last_dim(4, CO)
        Y = op(X)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = compile_model(Y, target, "./tmp", "pad_last_dim4d")

        X_pt = torch.randn(NN, HH, WW, CI).cuda().half()
        Pad_pt = torch.zeros(NN, HH, WW, CO - CI).cuda().half()
        Y_pt = torch.cat([X_pt, Pad_pt], dim=3)

        y = torch.empty([NN, HH, WW, CO]).cuda().half()
        module.run_with_tensors([X_pt], [y])
        self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))

    def test_static_shape_2d(self):
        NN = 32
        CI = 259
        CO = 264
        X = Tensor(shape=[NN, CI], name="X", is_input=True)
        op = ops.pad_last_dim(2, CO)
        Y = op(X)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"
        target = detect_target()
        module = compile_model(Y, target, "./tmp", "pad_last_dim2d")

        X_pt = torch.randn(NN, CI).cuda().half()
        Pad_pt = torch.zeros(NN, CO - CI).cuda().half()
        Y_pt = torch.cat([X_pt, Pad_pt], dim=1)

        y = torch.empty([NN, CO]).cuda().half()
        module.run_with_tensors([X_pt], [y])
        self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
