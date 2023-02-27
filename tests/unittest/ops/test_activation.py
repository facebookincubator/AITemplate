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
Unittests for special activation Operator.
"""
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils.torch_utils import torch_dtype_to_string


TORCH_EQUIVALENTS = {
    FuncEnum.TANH: torch.tanh,
    FuncEnum.COS: torch.cos,
    FuncEnum.SIN: torch.sin,
    FuncEnum.SIGN: torch.sign,
    FuncEnum.ABS: torch.abs,
    FuncEnum.LOGE: torch.log,
    FuncEnum.EXP: torch.exp,
    FuncEnum.SQRT: torch.sqrt,
    FuncEnum.SIGMOID: torch.sigmoid,
    FuncEnum.RELU: torch.relu,
}

TORCH_FP_DTYPES = [torch.float16]
if detect_target().name() != "rocm":
    TORCH_FP_DTYPES.append(torch.float32)
    if int(detect_target()._arch) >= 80:
        TORCH_FP_DTYPES.append(torch.bfloat16)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class FusedElementwiseTestCase(unittest.TestCase):
    def _test_leaky_relu(
        self,
        input_size,
        negative_slope=0.01,
        test_name="leaky_relu",
        copy_op=False,
    ):
        for torch_dtype in TORCH_FP_DTYPES:
            dtype = torch_dtype_to_string(torch_dtype)
            assert len(input_size) == 2
            X1 = Tensor(
                shape=[IntImm(input_size[0]), IntImm(input_size[1])],
                dtype=dtype,
                name="input0",
                is_input=True,
            )
            slope = Tensor(
                shape=[],
                dtype=dtype,
                name="slope",
                value=negative_slope,
            )
            X2_op = ops.elementwise(FuncEnum.LRELU)
            if copy_op:
                X2_op = ops.elementwise(**X2_op._get_op_attributes())
            X2 = X2_op(X1, slope)
            X2._attrs["is_output"] = True
            X2._attrs["name"] = "output0"

            target = detect_target()
            module = compile_model(X2, target, "./tmp", f"{test_name}_{dtype}")

            x1_pt = torch.randn(input_size, dtype=torch_dtype).cuda()
            OP_pt = torch.nn.LeakyReLU(negative_slope)
            x2_pt = OP_pt(x1_pt)

            x2 = torch.empty_like(x2_pt)
            module.run_with_tensors([x1_pt], [x2])
            self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def _test_floor_div(
        self,
        input_size,
        test_name="floor_div",
        dividend=2,
        copy_op=False,
    ):
        for torch_dtype in TORCH_FP_DTYPES:
            dtype = torch_dtype_to_string(torch_dtype)
            assert len(input_size) == 2
            X1 = Tensor(
                shape=[IntImm(input_size[0]), IntImm(input_size[1])],
                dtype=dtype,
                name="input0",
                is_input=True,
            )
            slope = Tensor(
                shape=[],
                dtype=dtype,
                name="input1",
                value=dividend,
            )
            X2_op = ops.elementwise(FuncEnum.FLOOR_DIV)

            if copy_op:
                X2_op = ops.elementwise(**X2_op._get_op_attributes())
            X2 = X2_op(X1, slope)
            X2._attrs["is_output"] = True
            X2._attrs["name"] = "output0"

            target = detect_target()
            module = compile_model(X2, target, "./tmp", f"{test_name}_{dtype}")

            x1_pt = torch.randn(input_size, dtype=torch_dtype).cuda()
            x2_pt = torch.div(x1_pt, dividend, rounding_mode="floor")

            x2 = torch.empty_like(x2_pt)
            module.run_with_tensors([x1_pt], [x2])
            self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def _test_hardtanh(
        self,
        input_size,
        min_val=-1,
        max_val=1,
        test_name="hard_tanh",
        copy_op=False,
    ):
        for torch_dtype in TORCH_FP_DTYPES:
            dtype = torch_dtype_to_string(torch_dtype)
            assert len(input_size) == 2
            X1 = Tensor(
                shape=[IntImm(input_size[0]), IntImm(input_size[1])],
                dtype=dtype,
                name="input0",
                is_input=True,
            )
            X_min = Tensor(
                shape=[],
                dtype=dtype,
                name="min_val",
                value=min_val,
                is_input=True,
            )
            X_max = Tensor(
                shape=[],
                dtype=dtype,
                name="max_val",
                value=max_val,
                is_input=True,
            )
            X2_op = ops.elementwise(FuncEnum.HARDTANH)
            if copy_op:
                X2_op = ops.elementwise(**X2_op._get_op_attributes())
            X2 = X2_op(X1, X_min, X_max)
            X2._attrs["is_output"] = True
            X2._attrs["name"] = "output0"

            target = detect_target()
            module = compile_model(X2, target, "./tmp", f"{test_name}_{dtype}")

            x1_pt = torch.randn(input_size, dtype=torch_dtype).cuda()
            OP_pt = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)
            x2_pt = OP_pt(x1_pt)

            x2 = torch.empty_like(x2_pt)
            module.run_with_tensors([x1_pt], [x2])
            self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def _test_softplus(
        self,
        input_size,
        beta=1.0,
        threshold=20.0,
        test_name="softplus",
        copy_op=False,
    ):
        for torch_dtype in TORCH_FP_DTYPES:
            dtype = torch_dtype_to_string(torch_dtype)
            assert len(input_size) == 2
            X1 = Tensor(
                shape=[IntImm(input_size[0]), IntImm(input_size[1])],
                dtype=dtype,
                name="input0",
                is_input=True,
            )
            X_beta = Tensor(
                shape=[],
                dtype=dtype,
                name="beta",
                value=beta,
                is_input=True,
            )
            X_threshold = Tensor(
                shape=[],
                dtype=dtype,
                name="threshold",
                value=threshold,
                is_input=True,
            )
            X2_op = ops.elementwise(FuncEnum.SOFTPLUS)
            if copy_op:
                X2_op = ops.elementwise(**X2_op._get_op_attributes())
            X2 = X2_op(X1, X_beta, X_threshold)
            X2._attrs["is_output"] = True
            X2._attrs["name"] = "output0"

            target = detect_target()
            module = compile_model(X2, target, "./tmp", f"{test_name}_{dtype}")

            x1_pt = torch.randn(input_size, dtype=torch_dtype).cuda()
            OP_pt = torch.nn.Softplus(beta=beta, threshold=threshold)
            x2_pt = OP_pt(x1_pt)

            x2 = torch.empty_like(x2_pt)
            module.run_with_tensors([x1_pt], [x2])
            self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def _test_simple_function(self, input_size, function, test_name, copy_op=False):
        for torch_dtype in TORCH_FP_DTYPES:
            dtype = torch_dtype_to_string(torch_dtype)
            assert len(input_size) == 2
            X1 = Tensor(
                shape=[IntImm(input_size[0]), IntImm(input_size[1])],
                dtype=dtype,
                name="input0",
                is_input=True,
            )
            X2_op = ops.elementwise(function)
            if copy_op:
                X2_op = ops.elementwise(**X2_op._get_op_attributes())
            X2 = X2_op(X1)
            X2._attrs["is_output"] = True
            X2._attrs["name"] = "output0"

            target = detect_target()
            module = compile_model(X2, target, "./tmp", f"{test_name}_{dtype}")

            x1_pt = torch.randn(input_size, dtype=torch_dtype).cuda()
            x2_pt = TORCH_EQUIVALENTS[function](x1_pt)

            x2 = torch.empty_like(x2_pt)
            module.run_with_tensors([x1_pt], [x2])
            self.assertTrue(
                torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2, equal_nan=True)
            )

    def _test_elu(
        self,
        input_size,
        alpha=1.0,
        test_name="elu",
        copy_op=False,
    ):
        for torch_dtype in TORCH_FP_DTYPES:
            dtype = torch_dtype_to_string(torch_dtype)
            assert len(input_size) == 2
            X1 = Tensor(
                shape=[IntImm(input_size[0]), IntImm(input_size[1])],
                dtype=dtype,
                name="input0",
                is_input=True,
            )
            X_alpha = Tensor(
                shape=[],
                dtype=dtype,
                name="alpha",
                value=alpha,
            )
            X2_op = ops.elementwise(FuncEnum.ELU)
            if copy_op:
                X2_op = ops.elementwise(**X2_op._get_op_attributes())
            X2 = X2_op(X1, X_alpha)
            X2._attrs["is_output"] = True
            X2._attrs["name"] = "output0"

            target = detect_target()
            module = compile_model(X2, target, "./tmp", f"{test_name}_{dtype}")
            x1_pt = torch.randn(input_size, dtype=torch_dtype).cuda()
            OP_pt = torch.nn.ELU(alpha=alpha)
            x2_pt = OP_pt(x1_pt)

            x2 = torch.empty_like(x2_pt)
            module.run_with_tensors([x1_pt], [x2])
            self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def _test_softsign(
        self,
        input_shape,
        test_name="softsign",
        copy_op=False,
    ):
        for torch_dtype in TORCH_FP_DTYPES:
            dtype = torch_dtype_to_string(torch_dtype)
            X1 = Tensor(
                shape=[IntImm(dim) for dim in input_shape],
                dtype=dtype,
                name="input",
                is_input=True,
            )
            X2_op = ops.elementwise(FuncEnum.SOFTSIGN)
            if copy_op:
                X2_op = ops.elementwise(**X2_op._get_op_attributes())
            X2 = X2_op(X1)
            X2._attrs["is_output"] = True
            X2._attrs["name"] = "output"

            target = detect_target()
            module = compile_model(X2, target, "./tmp", f"{test_name}_{dtype}")

            x1_pt = torch.randn(input_shape, dtype=torch_dtype).cuda()
            OP_pt = torch.nn.Softsign()
            x2_pt = OP_pt(x1_pt)

            x2 = torch.empty_like(x2_pt)
            module.run_with_tensors([x1_pt], [x2])
            self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def test_lrelu(self):
        self._test_leaky_relu([512, 512], test_name="leaky_relu_1")
        self._test_leaky_relu(
            [1024, 1024],
            negative_slope=0.5,
            test_name="leaky_relu_2",
        )
        self._test_leaky_relu(
            [1024, 1024],
            negative_slope=0.5,
            test_name="leaky_relu_2_copy_op",
            copy_op=True,
        )
        self._test_leaky_relu([63, 63], test_name="leaky_relu_3")

    def test_htanh(self):
        self._test_hardtanh([511, 511], test_name="hard_tanh_1")
        self._test_hardtanh(
            [1024, 1024], min_val=-2, max_val=2, test_name="hard_tanh_2"
        )
        self._test_hardtanh(
            [1024, 1024],
            min_val=-2,
            max_val=2,
            test_name="hard_tanh_2_copy_op",
            copy_op=True,
        )

    def test_softplus(self):
        self._test_softplus([63, 63], test_name="softplus_1")
        self._test_softplus([128, 128], beta=1.0, threshold=1.5, test_name="softplus_2")
        self._test_softplus([128, 256], beta=2.0, threshold=0.5, test_name="softplus_3")
        self._test_softplus(
            [256, 128],
            beta=1.0,
            threshold=1.0,
            test_name="softplus_3_copy_op",
            copy_op=True,
        )

    def test_cos(self):
        self._test_simple_function([511, 511], FuncEnum.COS, test_name="cos_1")
        self._test_simple_function(
            [512, 512], FuncEnum.COS, test_name="cos_1_copy_op", copy_op=True
        )

    def test_sin(self):
        self._test_simple_function([511, 511], FuncEnum.SIN, test_name="sin_1")
        self._test_simple_function(
            [512, 512], FuncEnum.SIN, test_name="sin_1_copy_op", copy_op=True
        )

    def test_tanh(self):
        self._test_simple_function([512, 512], FuncEnum.TANH, test_name="tanh_1")
        self._test_simple_function([1, 1], FuncEnum.TANH, test_name="tanh_2")
        self._test_simple_function(
            [512, 512], FuncEnum.TANH, test_name="tanh_1_copy_op", copy_op=True
        )

    def test_sign(self):
        self._test_simple_function([511, 511], FuncEnum.SIGN, test_name="sign_1")
        self._test_simple_function(
            [512, 512], FuncEnum.SIGN, test_name="sign_1_copy_op", copy_op=True
        )

    def test_abs(self):
        self._test_simple_function([511, 511], FuncEnum.ABS, test_name="abs_1")
        self._test_simple_function(
            [512, 512], FuncEnum.ABS, test_name="abs_1_copy_op", copy_op=True
        )

    def test_loge(self):
        self._test_simple_function([511, 511], FuncEnum.LOGE, test_name="loge_1")
        self._test_simple_function(
            [512, 512], FuncEnum.LOGE, test_name="loge_1_copy_op", copy_op=True
        )

    def test_exp(self):
        self._test_simple_function([511, 511], FuncEnum.EXP, test_name="exp_1")
        self._test_simple_function(
            [512, 512], FuncEnum.EXP, test_name="exp_1_copy_op", copy_op=True
        )

    def test_sqrt(self):
        self._test_simple_function([511, 511], FuncEnum.SQRT, test_name="sqrt_1")
        self._test_simple_function(
            [512, 512], FuncEnum.SQRT, test_name="sqrt_1_copy_op", copy_op=True
        )

    def test_sigmoid(self):
        self._test_simple_function([511, 511], FuncEnum.SIGMOID, test_name="sigmoid_1")
        self._test_simple_function(
            [512, 512], FuncEnum.SIGMOID, test_name="sigmoid_1_copy_op", copy_op=True
        )

    def test_relu(self):
        self._test_simple_function([511, 511], FuncEnum.RELU, test_name="relu_1")
        self._test_simple_function(
            [512, 512], FuncEnum.RELU, test_name="relu_1_copy_op", copy_op=True
        )

    def test_elu(self):
        self._test_elu([63, 63], test_name="elu_1")
        self._test_elu([128, 128], alpha=4.0, test_name="elu_2")
        self._test_elu([128, 256], alpha=0.4, test_name="elu_3")
        self._test_elu(
            [256, 128],
            alpha=1.0,
            test_name="elu_3_copy_op",
            copy_op=True,
        )

    def test_softsign(self):
        self._test_softsign([63, 63], test_name="softsign_1")
        self._test_softsign([128], test_name="softsign_2")
        self._test_softsign([128], test_name="softsign_3", copy_op=True)
        self._test_softsign([121, 128], test_name="softsign_4")

    def test_floor_div(self):
        self._test_floor_div([511, 511], test_name="floor_div_1")
        self._test_floor_div(
            [1024, 1024],
            dividend=3,
            test_name="test_floor_div_2_copy_op",
            copy_op=True,
        )


if __name__ == "__main__":
    unittest.main()
