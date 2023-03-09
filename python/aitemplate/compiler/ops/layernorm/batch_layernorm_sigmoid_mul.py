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
x: [b, m, n]
gamma: [b, n]
beta: [b, n]
"""
from typing import List

from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.layernorm.layernorm import layernorm

# pylint: disable=C0103,W0221,W0102,W0223


class batch_layernorm_sigmoid_mul(layernorm):
    """batch_layernorm_sigmoid_mul op.
    This op expects the normalized_shape to be 1D.
    """

    def __init__(self, normalized_shape: List[IntImm] = None) -> None:
        super().__init__(normalized_shape)
        self._attrs["op"] = "batch_layernorm_sigmoid_mul"

    def _sanity_check(self, x, gamma, beta):
        input_len = len(self._attrs["inputs"])
        if input_len < 1 or input_len > 4:
            raise NotImplementedError(
                f"Expect 1 ~ 4 inputs for Layernorm. Actual #inputs: {input_len}"
            )
        (x_shape, gamma_shape, beta_shape) = layernorm.get_input_shapes(x, gamma, beta)
        if len(x_shape) != 3:
            raise NotImplementedError(
                f"Layernorm input must be a 3-d matrix, current shape: {x_shape}"
            )
        if gamma_shape is not None:
            if len(gamma_shape) != 2:
                raise NotImplementedError(
                    f"Layernorm gamma must be a 2-d matrix, current shapes: {gamma_shape}"
                )
        if beta_shape is not None:
            if len(beta_shape) != 2:
                raise NotImplementedError(
                    f"Layernorm beta must be a 2-d matrix, current shapes: {beta_shape}"
                )

        # x: [b, m, n]
        # gamma: [b, n]
        # beta: [b, n]
        if gamma_shape is not None:
            if x_shape[2] != gamma_shape[1]:
                raise RuntimeError(
                    f"Layernorm inputs mismatch! x shape: {x_shape}, gamma shape: {gamma_shape}"
                )
            if x_shape[0] != gamma_shape[0]:
                raise RuntimeError(
                    f"Layernorm inputs mismatch! x shape: {x_shape}, gamma shape: {gamma_shape}"
                )

        if beta_shape is not None:
            if x_shape[2] != beta_shape[1]:
                raise RuntimeError(
                    f"Layernorm inputs mismatch! x shape: {x_shape}, beta shape: {beta_shape}"
                )
            if x_shape[0] != beta_shape[0]:
                raise RuntimeError(
                    f"Layernorm inputs mismatch! x shape: {x_shape}, beta shape: {beta_shape}"
                )

        normalized_shape = self._attrs["normalized_shape"]
        if len(normalized_shape) != 1:
            raise NotImplementedError(
                f"Layernorm normalized_shape length must be 1. Current normalized_shape: {normalized_shape}"
            )
        if normalized_shape[0]._attrs["values"] != x_shape[2]._attrs["values"]:
            raise RuntimeError(
                f"Layernorm normalized shape is not compatible with input shape. "
                f"Normalized shape: {normalized_shape}, input shape: {x_shape}"
            )
