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
from aitemplate.compiler.ops.common.epilogue import FuncEnum

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils, shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class RemoveUnusedOpsTestCase(unittest.TestCase):
    def _test_remove_unused_ops(
        self,
        batch_size=(1, 3),
        X_shape=(5, 10),
        test_name="test_remove_unused_ops",
        dtype="float16",
    ):
        target = detect_target()
        b_dim = shape_utils.gen_int_var_min_max(batch_size, name="input_batch")
        X = Tensor(
            shape=[b_dim, *X_shape],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        Y1 = ops.size()(X)
        Y2 = ops.getitem()(Y1, 1)
        CONST_X = Tensor(
            shape=[],
            dtype=dtype,
            name="input_1",
            is_input=True,
            value=Y2._attrs["int_var"].value(),
        )
        Y = ops.elementwise(FuncEnum.ADD)(X, CONST_X)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = get_random_torch_tensor(X_shape_pt, dtype)
            Y_pt = X_pt + X_pt.size(1)

            y = get_torch_empty_tensor(Y_pt.size(), dtype)
            module.run_with_tensors([X_pt], [y])

            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
            self.assertEqual(len(module.debug_sorted_graph), 2)
            self.assertEqual(
                len(graph_utils.get_sorted_ops(module.debug_sorted_graph)), 1
            )

    def test_remove_unused_ops_float16(self):
        self._test_remove_unused_ops()

    def test_remove_unused_ops_float32(self):
        self._test_remove_unused_ops(dtype="float32")


if __name__ == "__main__":
    unittest.main()
