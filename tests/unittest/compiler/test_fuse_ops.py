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
from aitemplate.testing.test_utils import get_random_torch_tensor


class TestFuseGroupnormSwish(unittest.TestCase):
    def test_fused(self):
        x_shape = [3, 3, 1, 4]
        num_groups = 2
        num_channels = x_shape[-1]
        dtype = "float16"
        eps = 1e-5

        X1 = Tensor(
            shape=x_shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )
        X2 = Tensor(
            shape=[num_channels],
            dtype=dtype,
            name="gamma",
            is_input=True,
        )
        X3 = Tensor(
            shape=[num_channels],
            dtype=dtype,
            name="beta",
            is_input=True,
        )

        op_name = "group_norm"
        OP = getattr(ops, op_name)(num_groups, num_channels)

        X4 = OP(X1, X2, X3, eps)
        X5 = ops.elementwise(FuncEnum.SIGMOID)(X4)
        X6 = ops.elementwise(FuncEnum.MUL)(X5)
        X6._attrs["is_output"] = True
        X6._attrs["name"] = "output"

        target = detect_target()
        dll_name = "test_0.so"
        module = compile_model(X6, target, "./tmp", op_name, dll_name=dll_name)

        x1_nhwc_pt = get_random_torch_tensor(x_shape, dtype)
        x1_nchw_pt = x1_nhwc_pt.permute(0, 3, 1, 2).contiguous()
        gamma_pt = get_random_torch_tensor((num_channels,), dtype)
        beta_pt = torch.randn_like(gamma_pt)

        x6_pt = torch.nn.functional.group_norm(
            x1_nchw_pt, num_groups, gamma_pt, beta_pt, eps=eps
        )

        x6_pt = torch.nn.SiLU()(x6_pt)

        inputs = {"X": x1_nhwc_pt}
        inputs["gamma"] = gamma_pt
        inputs["beta"] = beta_pt
        x6 = torch.empty_like(x1_nhwc_pt)
        module.run_with_tensors(inputs, [x6])

        torch.testing.assert_close(
            x6, x6_pt.permute(0, 2, 3, 1).contiguous(), atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
