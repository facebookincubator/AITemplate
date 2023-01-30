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


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class TupleListConstructTestCase(unittest.TestCase):
    def _test_construct(
        self,
        batch_size=(1, 3),
        X_shape=(16, 32, 64),
        test_op=ops.tuple_construct,
        test_name="tuple_construct",
        dtype="float16",
    ):
        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), *X_shape],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )

        X1 = ops.reshape()(X, [-1, X_shape[-1]])
        X2 = ops.flatten()(X)
        X3 = ops.unsqueeze(1)(X2)
        T = test_op()(X1, X2, X3)
        Y1 = ops.getitem()(T, 0)
        Y2 = ops.getitem()(T, 1)
        Y3 = ops.getitem()(T, 2)

        Y1._attrs["name"] = "output_0"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "output_1"
        Y2._attrs["is_output"] = True
        Y3._attrs["name"] = "output_2"
        Y3._attrs["is_output"] = True

        module = compile_model([Y1, Y2, Y3], target, "./tmp", test_name)

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = get_random_torch_tensor(X_shape_pt, dtype=dtype)
            Y1_pt = X_pt.reshape(-1, X_shape_pt[-1])
            Y2_pt = X_pt.flatten()
            Y3_pt = Y2_pt.unsqueeze(1)

            outputs = [
                torch.empty_like(Y1_pt),
                torch.empty_like(Y2_pt),
                torch.empty_like(Y3_pt),
            ]
            module.run_with_tensors([X_pt], outputs)

            self.assertTrue(torch.allclose(Y1_pt, outputs[0], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(Y2_pt, outputs[1], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(Y3_pt, outputs[2], atol=1e-2, rtol=1e-2))

    def test_construct_fp16(self):
        self._test_construct(
            test_op=ops.tuple_construct,
            test_name="construct_fp16_tuple",
            dtype="float16",
        )
        self._test_construct(
            test_op=ops.list_construct,
            test_name="construct_fp16_list",
            dtype="float16",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
