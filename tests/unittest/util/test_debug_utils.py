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
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils.debug_settings import AITDebugSettings


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
    check_tensor, check_all, test_name, capfd: pytest.CaptureFixture[str]
):
    X1 = Tensor(
        shape=[IntImm(1), IntImm(3)],
        dtype="float16",
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

    x1_pt = torch.Tensor([[1.0, 1.5, 2.0]]).cuda().half()
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
    _test_outputs(True, False, "test_outputs_tensor", capfd)
    _test_outputs(False, True, "test_outputs_all", capfd)
    _test_outputs(True, True, "test_outputs_both", capfd)


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
