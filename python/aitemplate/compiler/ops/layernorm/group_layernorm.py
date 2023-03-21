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
Operator definition for group_layernorm.
"""
from typing import Any, List

from aitemplate.compiler.base import IntImm, IntVarTensor, Tensor
from aitemplate.compiler.ops.layernorm.layernorm import layernorm
from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,W0102,W0223


class group_layernorm(layernorm):
    """group_layernorm.
    For each group, we expect each input to have shapes:
        Input shape: [M0, M1, ..., Mp, N1, N2, ..., ND]
        Normalized_shape: [N1, N2, ..., ND]
        Gamma/Beta, if not None, have the same shape as normalized_shape.
    Every input in the groups must have the same [M0, M1, ..., Mp] dims.
    """

    def __init__(self, normalized_shape: List[List[IntImm]] = None) -> None:
        super().__init__(normalized_shape[0] if normalized_shape is not None else None)
        self._attrs["op"] = "group_layernorm"
        self._attrs["has_profiler"] = False
        self._attrs["default_normalized_shape"] = normalized_shape

    def _sanity_check(self, all_inputs):
        input_len = len(all_inputs)
        if input_len % 3 != 0:
            raise NotImplementedError(
                "Expect multiples of 3 inputs for group layernorm. "
                "Actual #inputs: {}".format(input_len)
            )

        total = len(all_inputs)
        b = total // 3

        inputs, gammas, betas = (
            all_inputs[:b],
            all_inputs[b : 2 * b],
            all_inputs[2 * b :],
        )
        assert len(inputs) > 0
        assert (
            len(inputs)
            == len(gammas)
            == len(betas)
            == len(self._attrs["normalized_shape"])
        )

        for x, gamma, beta, normalized_shape in zip(
            inputs, gammas, betas, self._attrs["normalized_shape"]
        ):
            (x_shape, gamma_shape, beta_shape) = layernorm.get_input_shapes(
                x, gamma, beta
            )
            layernorm.check_shapes(x_shape, gamma_shape, beta_shape, normalized_shape)

        # check x in inputs have the same batch dims (Ms), it can be dynamic
        # x shape: B * [Ms, Ns], Ns can be different
        input_len = inputs[0]._rank()
        norm_len = len(self._attrs["normalized_shape"][0])
        for k in range(input_len - norm_len):
            M = inputs[0].shape()[k]
            for i, x in enumerate(inputs, 1):
                x_M = x.shape()[k]
                assert M == x_M, f"found shape mismatch for input_{i},"
                f"input_0 shape: {inputs[0].shape()}, input_{i} shape: {x.shape()}"

    def __call__(
        self,
        inputs: List[Tensor],
        gammas: List[Tensor],
        betas: List[Tensor],
        normalized_shapes: List[List[Any]] = None,
        eps: float = 1e-5,
    ) -> List[Tensor]:
        # inputs is flattend into a single list of tensors
        all_inputs = inputs + gammas + betas
        # 'real_inputs' only contains non-None tensors
        real_inputs = list(inputs)

        # FIXME: currently, only support two cases, either all gammas are None or
        # all gammas are non-None
        self._attrs["gamma_constant"] = "1.0"
        if gammas[0] is not None:
            if any(gamma is None for gamma in gammas):
                raise NotImplementedError(
                    f"expected beta not to be None, but got None: {gammas}"
                )
            self._attrs["gamma_constant"] = None
            real_inputs.extend(gammas)
        else:
            if any(gamma is not None for gamma in gammas):
                raise NotImplementedError(
                    f"expected all gammas to be None, but got {gammas}"
                )

        # FIXME: currently, only support two cases, either all betas are None or
        # all betas are non-None
        self._attrs["beta_constant"] = "0.0"
        if betas[0] is not None:
            if any(beta is None for beta in betas):
                raise NotImplementedError(
                    "expected beta not to be None, but got None: {betas}"
                )
            self._attrs["beta_constant"] = None
            real_inputs.extend(betas)
        else:
            if any(beta is not None for beta in betas):
                raise NotImplementedError(
                    f"expected all betas to be None, but got {betas}"
                )
        if normalized_shapes is not None:
            self._attrs["normalized_shape"] = []
            for normalized_shape in normalized_shapes:
                for shape in normalized_shape:
                    # Only add source of dynamic dim to inputs
                    if isinstance(shape, IntVarTensor) and not isinstance(
                        shape._attrs["int_var"], IntImm
                    ):
                        real_inputs.append(shape)
                self._attrs["normalized_shape"].append(
                    shape_utils.convert_shape_to_IntVar(normalized_shape)
                )
        else:
            self._attrs["normalized_shape"] = self._attrs["default_normalized_shape"]

        assert isinstance(eps, float), f"eps must be float, instead it is {type(eps)}"
        self._attrs["eps"] = eps
        self._attrs["inputs"] = real_inputs
        self._attrs["input_accessors"] = []
        self._sanity_check(all_inputs)
        self._set_depth()
        self._attrs["outputs"] = []
        self._attrs["output_accessors"] = []
        for x in inputs:
            output_shape = self._infer_shapes(x)
            output = Tensor(output_shape, src_ops={self}, dtype=x.dtype())
            self._attrs["outputs"].append(output)
            self._attrs["output_accessors"].append(TensorAccessor(output))
            self._attrs["input_accessors"].append(TensorAccessor(x))
        return self._attrs["outputs"]
