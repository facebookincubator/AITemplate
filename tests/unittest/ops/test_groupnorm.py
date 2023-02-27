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
Unittests for group norm Operator.
"""
import logging
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


_LOGGER = logging.getLogger(__name__)


@unittest.skipIf(detect_target()._arch == "75", "Skip GN on sm75.")
class GroupnormTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GroupnormTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _test_groupnorm(
        self,
        x_shape=(4, 14, 14, 1024),
        num_groups=32,
        gamma_is_none=False,
        beta_is_none=False,
        use_size_op=False,
        eps=1e-5,
        use_swish=False,
        copy_op=False,
        atol=1e-2,
        rtol=1e-2,
        dtype="float16",
    ):
        test_name = "group_norm_swish" if use_swish else "group_norm"
        _LOGGER.info(f"Testing {test_name}: {x_shape}, num_groups: {num_groups}")
        num_channels = x_shape[-1]
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

        op_name = "group_norm_swish" if use_swish else "group_norm"
        OP = getattr(ops, op_name)(num_groups, num_channels)
        if copy_op:
            OP = getattr(ops, op_name)(**OP._get_op_attributes())
        X4 = OP(X1, X2, X3, eps)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(X4, target, "./tmp", op_name, dll_name=dll_name)

        x1_nhwc_pt = get_random_torch_tensor(x_shape, dtype)
        x1_nchw_pt = x1_nhwc_pt.permute(0, 3, 1, 2).contiguous()
        gamma_pt = get_random_torch_tensor((num_channels,), dtype)
        beta_pt = torch.randn_like(gamma_pt)

        x4_pt = torch.nn.functional.group_norm(
            x1_nchw_pt, num_groups, gamma_pt, beta_pt, eps=eps
        )
        if use_swish:
            x4_pt = torch.nn.SiLU()(x4_pt)

        inputs = {"X": x1_nhwc_pt}
        inputs["gamma"] = gamma_pt
        inputs["beta"] = beta_pt
        x4 = torch.empty_like(x1_nhwc_pt)
        module.run_with_tensors(inputs, [x4])

        # from aitemplate.testing.benchmark_pt import benchmark_torch_function
        # module.benchmark_with_tensors(inputs, [x4], count=100000)
        # t = benchmark_torch_function(
        #    100000,
        #    torch.nn.functional.group_norm,
        #    x1_nchw_pt,
        #    num_groups,
        #    gamma_pt,
        #    beta_pt,
        #    eps=eps,
        # )
        # print("pt: ", t)

        torch.testing.assert_close(
            x4, x4_pt.permute(0, 2, 3, 1).contiguous(), atol=atol, rtol=rtol
        )
        self.test_count += 1

    def test_groupnorm_float16(self):
        self._test_groupnorm()
        self._test_groupnorm(x_shape=[7, 13, 9, 12], num_groups=4, eps=1e-5)
        self._test_groupnorm(x_shape=[1, 16, 16, 8192], num_groups=32, eps=1e-3)
        self._test_groupnorm(x_shape=[3, 64, 64, 128], num_groups=16, eps=1e-5)
        self._test_groupnorm(x_shape=[3, 33, 64, 120], num_groups=10, eps=1e-5)
        self._test_groupnorm(x_shape=[8, 34, 10, 72], num_groups=6, eps=1e-5)
        self._test_groupnorm(x_shape=[1, 8, 1, 64], num_groups=32, eps=1e-5)
        self._test_groupnorm(x_shape=[1, 8, 1, 4], num_groups=2, eps=1e-5, copy_op=True)

    def test_groupnorm_swish(self):
        self._test_groupnorm(use_swish=True)
        self._test_groupnorm(
            x_shape=[3, 3, 1, 4], num_groups=2, eps=1e-5, use_swish=True
        )
        self._test_groupnorm(
            x_shape=[7, 13, 9, 12], num_groups=4, eps=1e-5, use_swish=True, copy_op=True
        )
        self._test_groupnorm(
            x_shape=[2, 8, 8, 1280], num_groups=32, eps=1e-5, use_swish=True
        )
        self._test_groupnorm(
            x_shape=[2, 32, 32, 320], num_groups=32, eps=1e-5, use_swish=True
        )
        self._test_groupnorm(
            x_shape=[1, 512, 512, 256], num_groups=32, eps=1e-5, use_swish=True
        )

        # For benchmark only.
        # shapes = [
        #     (2, 16, 16, 1280),
        #     (2, 16, 16, 1920),
        #     (2, 16, 16, 2560),
        #     (2, 16, 16, 640),
        #     (2, 32, 32, 1280),
        #     (2, 32, 32, 1920),
        #     (2, 32, 32, 320),
        #     (2, 32, 32, 640),
        #     (2, 32, 32, 960),
        #     (2, 64, 64, 320),
        #     (2, 8, 8, 1280),
        #     (2, 8, 8, 2560),
        #     (2, 64, 64, 640),
        #     (2, 64, 64, 960),
        #     (1, 256, 256, 128),
        #     (1, 512, 512, 256),
        # ]
        # for shape in shapes:
        #     self._test_groupnorm(x_shape=shape, num_groups=32, eps=1e-5, use_swish=True)

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_groupnorm_float32(self):
        # H % 8 != 0
        self._test_groupnorm(
            x_shape=[7, 13, 9, 12],
            num_groups=4,
            eps=1e-5,
            dtype="float32",
            use_swish=True,
        )
        # H % 8 == 0
        self._test_groupnorm(
            x_shape=[2, 16, 16, 640],
            num_groups=32,
            eps=1e-5,
            dtype="float32",
            use_swish=True,
        )

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "bf16 is supported with CUDA sm80+",
    )
    @unittest.skipIf(detect_target().name() == "rocm", "bf16 not supported in ROCm")
    def test_groupnorm_bfloat16(self):
        # H % 8 != 0
        self._test_groupnorm(
            x_shape=[7, 13, 9, 12],
            num_groups=4,
            eps=1e-5,
            atol=1e-1,
            rtol=1e-1,
            dtype="bfloat16",
            use_swish=True,
        )
        # H % 8 == 0
        self._test_groupnorm(
            x_shape=[2, 16, 16, 640],
            num_groups=32,
            eps=1e-5,
            atol=1e-1,
            rtol=1e-1,
            dtype="bfloat16",
            use_swish=True,
        )


if __name__ == "__main__":
    unittest.main()
