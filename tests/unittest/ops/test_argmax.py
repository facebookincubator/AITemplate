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
"""
Unittests for argmax Operator.
"""
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target


class argmaxTestCase(unittest.TestCase):
    def _test_argmax(
        self, batch_size=1, shape=(2, 6), dim=0, test_name="argmax", copy_op=False
    ):

        o_shape = list(shape)[:-1]

        X1 = Tensor(
            shape=shape,
            dtype="float16",
            name="X",
            is_input=True,
        )
        X4_op = ops.argmax(dim=dim)
        if copy_op:
            X4_op = ops.argmax(**X4_op._get_op_attributes())
        X4 = X4_op(X1)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(X4, target, "./tmp", test_name)

        scores = torch.rand(shape).cuda().half()
        y_pt = torch.argmax(scores, dim=dim)
        y = torch.empty_like(y_pt, dtype=torch.int64)

        module.run_with_tensors([scores], [y])
        y_reshape = y.reshape(o_shape)
        self.assertTrue(torch.allclose(y_pt, y_reshape, atol=1e-2, rtol=1e-2))

    def test_argmax(self):
        self._test_argmax(shape=(300, 80), dim=1, test_name="argmax")
        self._test_argmax(
            shape=(300, 80), dim=1, test_name="argmax_copy_op", copy_op=True
        )


if __name__ == "__main__":
    torch.manual_seed(1024)
    unittest.main()
