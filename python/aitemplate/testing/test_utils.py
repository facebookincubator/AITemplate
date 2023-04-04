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
Utils for unit tests.
"""
import itertools
import unittest
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch

from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.compiler.dtype import normalize_dtype
from aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base import CausalType
from aitemplate.testing.detect_target import detect_target
from aitemplate.utils.graph_utils import get_sorted_ops
from aitemplate.utils.torch_utils import string_to_torch_dtype


class TestEnv(Enum):
    CUDA_LESS_THAN_SM80 = 1
    CUDA_SM80 = 2
    ROCM = 100


def _ROCM_filter(method_name: str) -> bool:
    return method_name.endswith("rocm")


def _SM80_filter(method_name: str) -> bool:
    return method_name.endswith("bf16") or method_name.endswith("sm80")


_TEST_ENV_TO_FILTER_METHOD: Dict[str, Callable[[str], bool]] = {
    TestEnv.CUDA_LESS_THAN_SM80: (
        lambda method_name: not (_SM80_filter(method_name) or _ROCM_filter(method_name))
    ),
    TestEnv.CUDA_SM80: _SM80_filter,
    TestEnv.ROCM: _ROCM_filter,
}


def _get_test_env(target) -> str:
    test_env = ""
    if target.name() == "cuda":
        if int(target._arch) < 80:
            test_env = TestEnv.CUDA_LESS_THAN_SM80
        elif int(target._arch) == 80:
            test_env = TestEnv.CUDA_SM80
        else:
            raise RuntimeError(
                f"Unknown test env, target: {target.name}, {target._arch}"
            )
    elif target.name() == "rocm":
        test_env = TestEnv.ROCM
    else:
        raise RuntimeError(f"Unknown test env, target: {target.name}, {target._arch}")
    if test_env not in _TEST_ENV_TO_FILTER_METHOD:
        raise RuntimeError(f"{test_env=} not defined in _TEST_ENV_TO_FILTER_METHOD")
    return test_env


def filter_test_cases_by_params(params: Dict[TestEnv, List[Tuple[Any]]]):
    """Filters test cases to run by given params. Only takes effect in CI env."""
    target = detect_target()
    test_env = _get_test_env(target)
    return (
        params.get(test_env, [])
        if target.in_ci_env()
        else list(itertools.chain.from_iterable(params.values()))
    )


def filter_test_cases_by_test_env(cls: Type[unittest.TestCase]):
    """Filters test cases to run by test case names implicitly. Only takes effect in CI env."""
    target = detect_target()
    test_env = _get_test_env(target)
    for attr in list(cls.__dict__.keys()):
        if (
            attr.startswith("test_")
            and target.in_ci_env()
            and (not _TEST_ENV_TO_FILTER_METHOD.get(test_env)(attr))
        ):
            delattr(cls, attr)


def _get_torch_tensor(torch_fn, shape, dtype):
    dtype = normalize_dtype(dtype)
    return torch_fn(shape, device="cuda", dtype=string_to_torch_dtype(dtype))


def get_random_torch_tensor(shape, dtype="float16"):
    return _get_torch_tensor(torch.randn, shape, dtype)


def get_torch_empty_tensor(shape, dtype="float16"):
    return _get_torch_tensor(torch.empty, shape, dtype)


def get_torch_zeros_tensor(shape, dtype="float16"):
    return _get_torch_tensor(torch.zeros, shape, dtype)


def get_torch_full_tensor(shape, fill_value, dtype="float16"):
    dtype = normalize_dtype(dtype)
    return torch.full(
        shape, fill_value, device="cuda", dtype=string_to_torch_dtype(dtype)
    )


def has_op(sorted_ops: List[Operator], op_name: str) -> bool:
    for op in sorted_ops:
        op_type = op._attrs["op"]
        if op_type == op_name:
            return True
    return False


def graph_has_op(graph: List[Tensor], op_name: str) -> bool:
    return has_op(get_sorted_ops(graph), op_name)


def count_ops(sorted_ops: List[Operator], op_name: str):
    count = 0
    for op in sorted_ops:
        op_type = op._attrs["op"]
        if op_type == op_name:
            count += 1
    return count


def gen_input_tensor(
    shape: List[Any], dtype: str = "float16", name: Optional[str] = None
) -> Tensor:
    tensor = Tensor(
        shape=shape,
        dtype=dtype,
        name=name,
        is_input=True,
    )
    return tensor


def get_src_op(tensor: Tensor) -> str:
    assert len(tensor._attrs["src_ops"]) == 1
    return list(tensor._attrs["src_ops"])[0]


def get_src_op_name(tensor: Tensor) -> str:
    return get_src_op(tensor)._attrs["op"]


def get_src_input(tensor: Tensor) -> str:
    src_op = get_src_op(tensor)
    assert len(src_op._attrs["inputs"]) >= 1
    return src_op._attrs["inputs"][0]


def get_shape(shape: List[IntVar], dim_to_value_dict: Dict[str, int]):
    res = [
        dim.value()
        if isinstance(dim, IntImm)
        else dim_to_value_dict[dim._attrs["name"]]
        for dim in shape
    ]
    return res


def epilogue_math_name_to_torch_fn(epilogue_math_name: str) -> Callable[[Any], Any]:
    if epilogue_math_name == "Identity":
        return lambda x: x
    elif epilogue_math_name == "Sigmoid":
        return torch.sigmoid
    elif epilogue_math_name == "SiLu":
        return torch.nn.functional.silu
    elif epilogue_math_name == "ReLu":
        return torch.nn.functional.relu
    elif epilogue_math_name == "Tanh":
        return torch.nn.functional.tanh
    else:
        raise NotImplementedError(f"Unsupported {epilogue_math_name=}!")


def get_attn_mask_per_causal_type(
    m: int, n: int, causal_type: CausalType, torch_dtype: str
) -> torch.Tensor:
    if causal_type == CausalType.NO_CAUSAL:
        invalid_attn_mask = torch.ones((m, n), dtype=torch_dtype, device="cuda")
    elif causal_type == CausalType.LOWER_LEFT_EMPTY:
        invalid_attn_mask: torch.Tensor = 1.0 - torch.tril(
            torch.ones(
                (m, n),
                dtype=torch.bool,
                device="cuda",
            )
        ).fill_diagonal_(False).to(torch_dtype)
    elif causal_type == CausalType.UPPER_RIGHT_EMPTY:
        invalid_attn_mask: torch.Tensor = torch.tril(
            torch.ones(
                (m, n),
                dtype=torch_dtype,
                device="cuda",
            )
        )
    else:
        raise NotImplementedError(f"Unsupported {causal_type=}!")
    return invalid_attn_mask
