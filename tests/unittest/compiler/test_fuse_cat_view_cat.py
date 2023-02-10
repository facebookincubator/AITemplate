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

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.public import IntImm, IntVar

from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


class FuseCatViewCatTestCase(unittest.TestCase):
    def test_fuse_cat_view_cat(self):
        dtype = "float16"
        B = IntVar([1, 2048], name="batch_size")
        M1 = IntImm(16)
        M2 = IntImm(48)
        N = IntImm(18)
        K = IntImm(9)

        input_1 = Tensor(
            shape=[B, M1, N],
            name="input_1",
            is_input=True,
        )
        input_2 = Tensor(
            shape=[B, M2, N],
            name="input_2",
            is_input=True,
        )
        input_3 = Tensor(
            shape=[B, K],
            name="input_3",
            is_input=True,
        )
        concatenate_4 = ops.concatenate()([input_1, input_2], 1)
        reshape_5 = ops.reshape()(
            concatenate_4, [-1, (M1.value() + M2.value()) * N.value()]
        )
        concatenate_6 = ops.concatenate()([input_3, reshape_5], 1)

        # Set outputs
        concatenate_6._attrs["name"] = "output_0"
        concatenate_6._attrs["is_output"] = True
        # Compile
        mod = compile_model(
            concatenate_6,
            detect_target(),
            "./tmp",
            "test_fuse_cat_view_cat",
        )
        # Compare
        input_1_pt = get_random_torch_tensor((1024, M1.value(), N.value()), dtype)
        input_2_pt = get_random_torch_tensor((1024, M2.value(), N.value()), dtype)
        input_3_pt = get_random_torch_tensor((1024, K.value()), dtype)

        y_pt = torch.cat(
            [
                input_3_pt,
                torch.reshape(
                    torch.cat([input_1_pt, input_2_pt], dim=1),
                    (-1, (M1.value() + M2.value()) * N.value()),
                ),
            ],
            dim=1,
        )
        y_ait = torch.empty_like(y_pt)
        mod.run_with_tensors(
            {"input_1": input_1_pt, "input_2": input_2_pt, "input_3": input_3_pt},
            [y_ait],
        )
        torch.testing.assert_close(y_ait, y_pt, atol=0, rtol=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
