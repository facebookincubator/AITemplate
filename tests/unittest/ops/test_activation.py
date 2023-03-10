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
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)
from parameterized import parameterized


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
    FuncEnum.CELU: torch.celu,
}


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class FusedElementwiseTestCase(unittest.TestCase):
    def _test_leaky_relu(
        self,
        input_size,
        negative_slope=0.01,
        test_name="leaky_relu",
        copy_op=False,
        dtype="float16",
    ):
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

        x1_pt = get_random_torch_tensor(input_size, dtype=dtype)
        OP_pt = torch.nn.LeakyReLU(negative_slope)
        x2_pt = OP_pt(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        torch.testing.assert_close(x2, x2_pt, atol=1e-2, rtol=1e-2)

    def _test_floor_div(
        self,
        input_size,
        test_name="floor_div",
        dividend=2,
        copy_op=False,
        dtype="float16",
    ):
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

        x1_pt = get_random_torch_tensor(input_size, dtype)
        x2_pt = torch.div(x1_pt, dividend, rounding_mode="floor")

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        torch.testing.assert_close(x2, x2_pt, atol=1e-2, rtol=1e-2)

    def _test_hardtanh(
        self,
        input_size,
        min_val=-1,
        max_val=1,
        test_name="hard_tanh",
        copy_op=False,
        dtype="float16",
    ):
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

        x1_pt = get_random_torch_tensor(input_size, dtype)
        OP_pt = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)
        x2_pt = OP_pt(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        torch.testing.assert_close(x2, x2_pt, atol=1e-2, rtol=1e-2)

    def _test_softplus(
        self,
        input_size,
        beta=1.0,
        threshold=20.0,
        test_name="softplus",
        copy_op=False,
        dtype="float16",
    ):
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

        x1_pt = get_random_torch_tensor(input_size, dtype)
        OP_pt = torch.nn.Softplus(beta=beta, threshold=threshold)
        x2_pt = OP_pt(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        torch.testing.assert_close(x2, x2_pt, atol=1e-2, rtol=1e-2)

    def _test_simple_function(
        self, input_size, function, test_name, copy_op=False, dtype="float16"
    ):
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

        x1_pt = get_random_torch_tensor(input_size, dtype)
        x2_pt = TORCH_EQUIVALENTS[function](x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        torch.testing.assert_close(x2, x2_pt, atol=1e-2, rtol=1e-2, equal_nan=True)

    def _test_elu(
        self, input_size, alpha=1.0, test_name="elu", copy_op=False, dtype="float16"
    ):
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
        x1_pt = get_random_torch_tensor(input_size, dtype)
        OP_pt = torch.nn.ELU(alpha=alpha)
        x2_pt = OP_pt(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        torch.testing.assert_close(x2, x2_pt, atol=1e-2, rtol=1e-2)

    def _test_softsign(
        self,
        input_size,
        test_name="softsign",
        copy_op=False,
        dtype="float16",
    ):
        X1 = Tensor(
            shape=[IntImm(dim) for dim in input_size],
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

        x1_pt = get_random_torch_tensor(input_size, dtype)
        OP_pt = torch.nn.Softsign()
        x2_pt = OP_pt(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        torch.testing.assert_close(x2, x2_pt, atol=1e-2, rtol=1e-2)

    def _test_celu(
        self,
        input_size,
        alpha=1.0,
        test_name="celu",
        copy_op=False,
        dtype="float16",
    ):
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
        X2_op = ops.elementwise(FuncEnum.CELU)
        if copy_op:
            X2_op = ops.elementwise(**X2_op._get_op_attributes())
        X2 = X2_op(X1, X_alpha)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", f"{test_name}_{dtype}")
        x1_pt = get_random_torch_tensor(input_size, dtype)
        OP_pt = torch.nn.CELU(alpha=alpha)
        x2_pt = OP_pt(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        torch.testing.assert_close(x2, x2_pt, atol=1e-2, rtol=1e-2)

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_lrelu(self, dtype):
        self._test_leaky_relu([512, 512], test_name="leaky_relu_1", dtype=dtype)
        self._test_leaky_relu(
            [1024, 1024],
            negative_slope=0.5,
            test_name="leaky_relu_2",
            dtype=dtype,
        )
        self._test_leaky_relu(
            [1024, 1024],
            negative_slope=0.5,
            test_name="leaky_relu_2_copy_op",
            copy_op=True,
            dtype=dtype,
        )
        self._test_leaky_relu([63, 63], test_name="leaky_relu_3", dtype=dtype)

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_htanh(self, dtype):
        self._test_hardtanh([511, 511], test_name="hard_tanh_1", dtype=dtype)
        self._test_hardtanh(
            [1024, 1024], min_val=-2, max_val=2, test_name="hard_tanh_2", dtype=dtype
        )
        self._test_hardtanh(
            [1024, 1024],
            min_val=-2,
            max_val=2,
            test_name="hard_tanh_2_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_softplus(self, dtype):
        self._test_softplus([63, 63], test_name="softplus_1", dtype=dtype)
        self._test_softplus(
            [128, 128], beta=1.0, threshold=1.5, test_name="softplus_2", dtype=dtype
        )
        self._test_softplus(
            [128, 256], beta=2.0, threshold=0.5, test_name="softplus_3", dtype=dtype
        )
        self._test_softplus(
            [256, 128],
            beta=1.0,
            threshold=1.0,
            test_name="softplus_3_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_cos(self, dtype):
        self._test_simple_function(
            [511, 511], FuncEnum.COS, test_name="cos_1", dtype=dtype
        )
        self._test_simple_function(
            [512, 512],
            FuncEnum.COS,
            test_name="cos_1_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_sin(self, dtype):
        self._test_simple_function(
            [511, 511], FuncEnum.SIN, test_name="sin_1", dtype=dtype
        )
        self._test_simple_function(
            [512, 512],
            FuncEnum.SIN,
            test_name="sin_1_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_tanh(self, dtype):
        self._test_simple_function(
            [512, 512], FuncEnum.TANH, test_name="tanh_1", dtype=dtype
        )
        self._test_simple_function(
            [1, 1], FuncEnum.TANH, test_name="tanh_2", dtype=dtype
        )
        self._test_simple_function(
            [512, 512],
            FuncEnum.TANH,
            test_name="tanh_1_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_sign(self, dtype):
        self._test_simple_function(
            [511, 511], FuncEnum.SIGN, test_name="sign_1", dtype=dtype
        )
        self._test_simple_function(
            [512, 512],
            FuncEnum.SIGN,
            test_name="sign_1_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_abs(self, dtype):
        self._test_simple_function(
            [511, 511], FuncEnum.ABS, test_name="abs_1", dtype=dtype
        )
        self._test_simple_function(
            [512, 512],
            FuncEnum.ABS,
            test_name="abs_1_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_loge(self, dtype):
        self._test_simple_function(
            [511, 511], FuncEnum.LOGE, test_name="loge_1", dtype=dtype
        )
        self._test_simple_function(
            [512, 512],
            FuncEnum.LOGE,
            test_name="loge_1_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_exp(self, dtype):
        self._test_simple_function(
            [511, 511], FuncEnum.EXP, test_name="exp_1", dtype=dtype
        )
        self._test_simple_function(
            [512, 512],
            FuncEnum.EXP,
            test_name="exp_1_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_sqrt(self, dtype):
        self._test_simple_function(
            [511, 511], FuncEnum.SQRT, test_name="sqrt_1", dtype=dtype
        )
        self._test_simple_function(
            [512, 512],
            FuncEnum.SQRT,
            test_name="sqrt_1_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_sigmoid(self, dtype):
        self._test_simple_function(
            [511, 511], FuncEnum.SIGMOID, test_name="sigmoid_1", dtype=dtype
        )
        self._test_simple_function(
            [512, 512],
            FuncEnum.SIGMOID,
            test_name="sigmoid_1_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_relu(self, dtype):
        self._test_simple_function(
            [511, 511], FuncEnum.RELU, test_name="relu_1", dtype=dtype
        )
        self._test_simple_function(
            [512, 512],
            FuncEnum.RELU,
            test_name="relu_1_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_elu(self, dtype):
        self._test_elu([63, 63], test_name="elu_1", dtype=dtype)
        self._test_elu([128, 128], alpha=4.0, test_name="elu_2", dtype=dtype)
        self._test_elu([128, 256], alpha=0.4, test_name="elu_3", dtype=dtype)
        self._test_elu(
            [256, 128],
            alpha=1.0,
            test_name="elu_3_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_softsign(self, dtype):
        self._test_softsign(
            [63, 63],
            test_name="softsign_1",
            dtype=dtype,
        )
        self._test_softsign(
            [128],
            test_name="softsign_2",
            dtype=dtype,
        )
        self._test_softsign(
            [128],
            test_name="softsign_3",
            copy_op=True,
            dtype=dtype,
        )
        self._test_softsign(
            [121, 128],
            test_name="softsign_4",
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_floor_div(self, dtype):
        self._test_floor_div(
            [511, 511],
            test_name="floor_div_1",
            dtype=dtype,
        )
        self._test_floor_div(
            [1024, 1024],
            dividend=3,
            test_name="test_floor_div_2_copy_op",
            copy_op=True,
            dtype=dtype,
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_celu(self, dtype):
        self._test_celu([63, 63], alpha=1.0, test_name="celu_1", dtype=dtype)
        self._test_celu([128, 128], alpha=4.0, test_name="celu_2", dtype=dtype)
        self._test_celu([128, 256], alpha=0.4, test_name="celu_3", dtype=dtype)
        self._test_celu(
            [256, 128], alpha=1.0, test_name="celu_3_copy_op", copy_op=True, dtype=dtype
        )


if __name__ == "__main__":
    unittest.main()
