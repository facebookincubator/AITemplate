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
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


def hard_swish(x):
    # return x * F.relu6(x + 3) / 6
    return x * torch.clamp((x + 3), 0, 6) / 6


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ConvBiasHardswishTestCase(unittest.TestCase):
    def _test_conv_bias_hardswish(
        self,
        batch=4,
        copy_op=False,
        test_name="conv2d_bias_hardswish",
        dtype="float16",
    ):
        target = detect_target()
        X = Tensor(
            shape=[IntImm(batch), 28, 28, 128],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[256, 3, 3, 128],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B = Tensor(
            shape=[256],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        OP = ops.conv2d_bias_hardswish(stride=1, pad=1, dilate=1)
        if copy_op:
            OP = ops.conv2d_bias_hardswish(**OP._get_op_attributes())
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        X_pt = get_random_torch_tensor([batch, 128, 28, 28], dtype=dtype)
        W_pt = get_random_torch_tensor([256, 128, 3, 3], dtype=dtype)
        B_pt = get_random_torch_tensor([1, 256, 1, 1], dtype=dtype)
        Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=1)
        Y_pt = Y_pt + B_pt
        Y_pt = hard_swish(Y_pt)
        # np.savetxt("y.txt", Y_np.flatten())
        # module.SetDim("input_batch", batch)
        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        inputs = {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}
        # np.savetxt("x.txt", x.flatten())
        # np.savetxt("w.txt", w.flatten())
        # np.savetxt("b.txt", b.flatten())
        y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
        module.run_with_tensors(inputs, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        if dtype == "float32":
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=5e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))

    def test_fp16(self):
        self._test_conv_bias_hardswish(
            test_name="conv2d_bias_hardswish_fp16",
            dtype="float16",
        )
        self._test_conv_bias_hardswish(
            copy_op=True,
            test_name="conv2d_bias_hardswish_fp16_copy_op",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_fp32(self):
        self._test_conv_bias_hardswish(
            test_name="conv2d_bias_hardswish_fp32",
            dtype="float32",
        )
        self._test_conv_bias_hardswish(
            copy_op=True,
            test_name="conv2d_bias_hardswish_fp32_copy_op",
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
