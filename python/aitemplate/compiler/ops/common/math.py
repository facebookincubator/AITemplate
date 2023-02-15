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
Basic math functions.
"""

from typing import Any

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.op_registry import OP_REGISTRY


def tanh(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("TANH")(tensor)


def cos(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("COS")(tensor)


def sin(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("SIN")(tensor)


def sign(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("SIGN")(tensor)


def abs(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("ABS")(tensor)


def log(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("LOGE")(tensor)


def exp(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("EXP")(tensor)


def sqrt(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("SQRT")(tensor)


def max(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("MAX")(tensor)


def min(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("MIN")(tensor)


def sigmoid(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("SIGMOID")(tensor)


def leaky_relu(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("LRELU")(tensor)


def hardtanh(*args, **kwargs) -> Tensor:
    return OP_REGISTRY.get("HARDTANH")(*args, **kwargs)


def relu(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("RELU")(tensor)


def silu(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("SILU")(tensor)


def nan_to_num(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("NAN_TO_NUM")(tensor)


def pow(*args, **kwargs) -> Tensor:
    return OP_REGISTRY.get("POW")(*args, **kwargs)


def gelu(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("GELU")(tensor)


def fast_gelu(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("FASTGELU")(tensor)


def softplus(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("SOFTPLUS")(tensor)


def elu(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("ELU")(tensor)


def softsign(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("SOFTSIGN")(tensor)


def floor_div(tensor: Any) -> Tensor:
    return OP_REGISTRY.get("FLOOR_DIV")(tensor)
