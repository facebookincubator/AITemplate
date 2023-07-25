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
Unittests for debug utils.
"""
import numpy as np
import pytest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm, IntVarTensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils import shape_utils
from aitemplate.utils.debug_settings import AITDebugSettings
from aitemplate.utils.torch_utils import string_to_torch_dtype


def _test_inf_and_nan(
    check_tensor, check_all, test_name, capfd: pytest.CaptureFixture[str]
):
    X1 = Tensor(
        shape=[IntImm(1), IntImm(3)],
        dtype="float16",
        name="input0",
        is_input=True,
    )
    X2_op = ops.elementwise(FuncEnum.DIV)
    X2 = X2_op(X1, 0.0)
    X2._attrs["is_output"] = True
    X2._attrs["name"] = "output0"
    X2._attrs["check_nan_and_inf"] = check_tensor

    target = detect_target()
    debug_settings = AITDebugSettings(check_all_nan_and_inf=check_all)
    module = compile_model(
        X2, target, "./tmp", test_name, debug_settings=debug_settings
    )

    x1_pt = torch.Tensor([[1.0, -2.0, 0.0]]).cuda().half()
    x2 = torch.empty_like(x1_pt)
    module.run_with_tensors([x1_pt], [x2])

    out, _ = capfd.readouterr()
    check_str = "Tensor (output0) contains NaN: 1, +INF: 1, -INF: 1, total elements: 3"
    assert check_str in out


def test_inf_and_nan(capfd):
    _test_inf_and_nan(True, False, "test_inf_and_nan_tensor", capfd)
    _test_inf_and_nan(False, True, "test_inf_and_nan_all", capfd)
    _test_inf_and_nan(True, True, "test_inf_and_nan_both", capfd)


def _test_outputs(
    check_tensor, check_all, test_name, dtype, capfd: pytest.CaptureFixture[str]
):
    X1 = Tensor(
        shape=[IntImm(1), IntImm(3)],
        dtype=dtype,
        name="input0",
        is_input=True,
    )
    X2_op = ops.elementwise(FuncEnum.MUL)
    X2 = X2_op(X1, 1.3)
    X2._attrs["is_output"] = True
    X2._attrs["name"] = "output0"
    X2._attrs["check_outputs"] = check_tensor

    target = detect_target()
    debug_settings = AITDebugSettings(check_all_outputs=check_all)
    module = compile_model(
        X2, target, "./tmp", test_name, debug_settings=debug_settings
    )

    x1_pt = (
        torch.Tensor([[1.0, 1.5, 2.0]])
        .to(dtype=string_to_torch_dtype(dtype))
        .to("cuda")
    )
    x2 = torch.empty_like(x1_pt)
    module.run_with_tensors([x1_pt], [x2])

    out, _ = capfd.readouterr()
    output_str = "Tensor (output0) output:"
    idx = out.find(output_str)
    assert idx != -1

    out = out[idx + len(output_str) :].strip()
    values = out.split(", ")
    assert len(values) == 3, f"Got {len(values)} outputs, expected 3"

    values = [float(value) for value in values]
    target_values = np.array([1.0, 1.5, 2.0]) * 1.3
    assert np.allclose(
        values, target_values, rtol=1e-2, atol=1e-2
    ), f"Expected {target_values}, got {values} instead"


def test_outputs(capfd):
    _test_outputs(True, False, "test_outputs_tensor", "float16", capfd)
    _test_outputs(False, True, "test_outputs_all", "float16", capfd)
    _test_outputs(True, True, "test_outputs_both_float16", "float16", capfd)
    _test_outputs(True, True, "test_outputs_both_float32", "float32", capfd)


@pytest.mark.skipif(
    detect_target().name == "rocm" or int(detect_target()._arch) < 80,
    reason="bfloat16 tests requires CUDA sm >= 80",
)
def test_outputs_bf16(capfd):
    _test_outputs(True, True, "test_outputs_both_bfloat16", "bfloat16", capfd)


def _test_with_int_var_tensor(test_name, dtype):
    target = detect_target()
    batch_size = (3, 5)
    x1_size = (2, 3)
    X_shape = (32, 64)
    b_dim = shape_utils.gen_int_var_min_max(batch_size, name="input_batch")
    x1_dim = shape_utils.gen_int_var_min_max(x1_size, name="input_size")
    X = Tensor(
        shape=[b_dim, x1_dim, *X_shape],
        dtype=dtype,
        name="input_0",
        is_input=True,
    )

    Y1 = ops.size()(X)
    Y2 = ops.getitem()(Y1, 0)
    Y3 = ops.getitem()(Y1, 1)
    Y4 = ops.getitem()(Y1, 2)
    Y5 = ops.getitem()(Y1, 3)
    f1 = ops.int_elementwise(FuncEnum.MUL)(Y4, Y5)
    f2 = IntVarTensor(IntImm(12))

    Y = ops.reshape()(X, [Y2 * Y3 * f1 / f2, f2])
    Y._attrs["name"] = "output_0"
    Y._attrs["is_output"] = True
    debug_settings = AITDebugSettings(
        check_all_outputs=True, check_all_nan_and_inf=True
    )
    module = compile_model(Y, target, "./tmp", test_name, debug_settings=debug_settings)

    for b, x1 in zip(batch_size, x1_size):
        X_shape_pt = (b, x1, *X_shape)
        X_pt = get_random_torch_tensor(X_shape_pt, dtype=dtype)
        Y_pt = X_pt.reshape(
            int(X_shape_pt[0] * X_shape_pt[1] * X_shape_pt[2] * X_shape_pt[3] / 12),
            12,
        )

        y = torch.empty_like(Y_pt)
        module.run_with_tensors([X_pt], [y])
        assert torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2)


def test_int_var_tensor(capfd):
    _test_with_int_var_tensor("test_outputs_int_var_tensor", "float16")


def _test_special_outputs(
    check_tensor, check_all, test_name, capfd: pytest.CaptureFixture[str]
):
    X1 = Tensor(
        shape=[IntImm(1), IntImm(3)],
        dtype="float16",
        name="input0",
        is_input=True,
    )
    X2_op = ops.elementwise(FuncEnum.DIV)
    X2 = X2_op(X1, 0.0)
    X2._attrs["is_output"] = True
    X2._attrs["name"] = "output0"
    X2._attrs["check_outputs"] = check_tensor

    target = detect_target()
    debug_settings = AITDebugSettings(check_all_outputs=check_all)
    module = compile_model(
        X2, target, "./tmp", test_name, debug_settings=debug_settings
    )

    x1_pt = torch.Tensor([[1.0, -2.0, 0.0]]).cuda().half()
    x2 = torch.empty_like(x1_pt)
    module.run_with_tensors([x1_pt], [x2])

    out, _ = capfd.readouterr()
    check_str = "inf, -inf, nan"
    assert check_str in out


def test_special_outputs(capfd):
    _test_special_outputs(True, False, "test_special_outputs_tensor", capfd)
    _test_special_outputs(False, True, "test_special_outputs_all", capfd)
    _test_special_outputs(True, True, "test_special_outputs_both", capfd)
