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
from aitemplate.compiler.base import JaggedDim, JaggedIntVar
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


class MakeJaggedTestCase(unittest.TestCase):
    def test_make_jagged(
        self,
    ):
        offsets1 = Tensor(
            shape=[
                IntVar(values=[1, 16]),
            ],
            name="off1",
            dtype="int32",
            is_input=True,
        )
        offsets2 = Tensor(
            shape=[
                IntVar(values=[1, 16]),
            ],
            name="off2",
            dtype="int32",
            is_input=True,
        )

        X = Tensor(
            shape=[
                IntVar(values=[1, 1024]),
                IntImm(value=128),
            ],
            name="X",
            dtype="float16",
            is_input=True,
        )
        W = Tensor(
            shape=[
                IntImm(value=128),
                IntImm(value=64),
            ],
            name="W",
            dtype="float16",
            is_input=True,
        )

        batch_dim = IntVar(values=[1, 128])
        jd0 = JaggedDim(min_value=0, max_value=10)
        jd1 = JaggedDim(min_value=0, max_value=15)
        Y = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=[jd0, jd1],
        )(X, [offsets1, offsets2])
        Z = ops.gemm_rrr()(Y, W)

        assert Y.is_jagged()
        assert Z.is_jagged()

        Y_dim_0 = Y._attrs["shape"][0]
        assert isinstance(Y_dim_0, JaggedIntVar)
        assert Y_dim_0.jagged_dims() == [jd0, jd1]
        assert jd0.offsets() == offsets1
        assert jd1.offsets() == offsets2

        Z_dim_0 = Z._attrs["shape"][0]
        assert Z_dim_0 == Y_dim_0

        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True
        Z._attrs["name"] = "Z"
        Z._attrs["is_output"] = True

        model = compile_model([Y, Z], detect_target(), "./tmp", "test_make_jagged")

        offsets1_pt = torch.tensor([0, 1, 3, 5], dtype=torch.int32).cuda()
        offsets2_pt = torch.tensor([0, 2, 4, 4, 9, 10], dtype=torch.int32).cuda()
        x_pt = get_random_torch_tensor([10, 128], "float16")
        w_pt = get_random_torch_tensor([128, 64], "float16")
        z_pt = torch.matmul(x_pt, w_pt)

        y = get_torch_empty_tensor([10, 128], "float16")
        z = get_torch_empty_tensor([10, 64], "float16")

        inputs = {"X": x_pt, "off1": offsets1_pt, "off2": offsets2_pt, "W": w_pt}
        model.run_with_tensors(inputs, [y, z])

        torch.testing.assert_close(y, x_pt)
        torch.testing.assert_close(z, z_pt)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
