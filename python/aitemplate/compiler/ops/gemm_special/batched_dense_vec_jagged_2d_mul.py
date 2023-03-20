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
Define batched_dense_vec_jagged_2d_mul op
"""
from typing import List

from aitemplate.backend import registry

from aitemplate.backend.target import Target

from aitemplate.compiler.base import IntVar, Operator, Tensor


class batched_dense_vec_jagged_2d_mul(Operator):
    """
    Compute a dense tensor containing batched matrix
    multiplication of a batched dense vector and
    a batched jagged matrix.

    Args:
        vectors (Tensor): batched dense vector of shape [B, H, N].
        matrices (Tensor): batched jagged matrix of shape [sum_B(N_B), H, D].

    Returns:
        output (Tensor): dense tensor containing the batched vector /
        jagged matrix multiplication result of shape [B, H, D].
    """

    def __init__(
        self,
    ):
        super().__init__()
        self._attrs["op"] = "batched_dense_vec_jagged_2d_mul"

    def _infer_shape(self, vectors: Tensor, matrices: Tensor) -> List[IntVar]:
        jagged_int_var = matrices.shape()[0]
        return [jagged_int_var.batch_dim(), matrices.shape()[1], matrices.shape()[2]]

    def __call__(self, vectors: Tensor, matrices: Tensor) -> Tensor:
        if not matrices.is_jagged():
            raise TypeError(
                f"matrices must be a jagged Tensor, but got a dense Tensor {matrices}."
            )
        if vectors.is_jagged():
            raise TypeError(
                f"vectors must be a jagged Tensor, but got a jagged Tensor {vectors}."
            )

        if len(vectors.shape()) != 3:
            raise ValueError(f"vectors must be rank-3, but got {vectors}.")

        if len(matrices.shape()) != 3:
            raise ValueError(f"matrices must be rank-3, but got {matrices}.")

        jagged_int_var = matrices.shape()[0]
        if jagged_int_var.batch_dim() != vectors.shape()[0]:
            raise RuntimeError(
                "The batch dim B of the jagged matrices tensor and "
                "dense vectors tensor must be the same, but got "
                f"{jagged_int_var.batch_dim()=} != {vectors.shape()[0]=}."
            )

        if vectors.shape()[1] != matrices.shape()[1]:
            raise RuntimeError(
                f"The second dim H of the jagged matrices tensor and "
                "dense vectors tensor must be the same, but got "
                f"{matrices.shape()[1]=} != {vectors.shape()[1]}."
            )

        if vectors.dtype() != matrices.dtype():
            raise RuntimeError(
                "vectors and matrices must have the same type, but got "
                f"{vectors.dtype()=} != {matrices.dtype()=}."
            )

        if len(jagged_int_var.jagged_dims()) != 1:
            raise RuntimeError(
                "Jagged matrices tensor must have a "
                f"single JaggedDim, but got {matrices}."
            )
        else:
            max_value = jagged_int_var.jagged_dims()[0].max_value()
            if max_value != vectors.shape()[2]:
                raise RuntimeError(
                    "Upper bound (max_value) of the jagged dim in matrices "
                    "must be equal to the last dim N in vectors, but got "
                    f"{max_value=} != {vectors.shape()[2].value()=}."
                )

        self._attrs["inputs"] = [vectors, matrices]
        self._set_depth()
        output_shape = self._infer_shape(vectors, matrices)
        output = Tensor(output_shape, src_ops={self}, dtype=vectors.dtype())

        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = Target.current()
        func = registry.get(f"{target.name()}.{self._attrs['op']}.gen_function")
        return func(self._attrs)
