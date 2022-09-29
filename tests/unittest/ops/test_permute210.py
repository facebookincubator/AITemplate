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
import itertools
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target


class Permute210(unittest.TestCase):
    def test_static_shape_3d(self):
        for NWC in itertools.product([2, 80, 300], [2, 80, 300], [2, 80, 300]):
            with self.subTest(NWC=NWC):
                NN, WW, CI = NWC
                X = Tensor(shape=[NN, WW, CI], name="X", is_input=True)
                op = ops.permute210()
                Y = op(X)
                Y._attrs["is_output"] = True
                Y._attrs["name"] = "output"
                target = detect_target()
                module = compile_model(
                    Y, target, "./tmp", "perm210_{}_{}_{}".format(NN, WW, CI)
                )

                X_pt = torch.randn(NN, WW, CI).cuda().half()
                Y_pt = torch.permute(X_pt, [2, 1, 0])
                y = torch.empty([CI, WW, NN]).cuda().half()
                module.run_with_tensors([X_pt], [y])
                self.assertTrue(torch.allclose(y, Y_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
