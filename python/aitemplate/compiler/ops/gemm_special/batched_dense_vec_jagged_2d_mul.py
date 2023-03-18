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
    Returns a dense tensor containing batched matrix multiplication of batched vector and batched jagged tensor.
    Args:
        vectors (Tensor): batched vector tensor
        matrices (Tensor): batched jagged tensor
    Returns:
        output (Tensor): a dense tensor containing the batched matrix multiplication result.
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
        # Check matrices is jagged tensor
        if not matrices.is_jagged():
            matrices_name = matrices._attrs["name"]
            raise RuntimeError(
                f"Input tensor {matrices_name} is expected to be jagged, but actually dense."
            )

        # Check input tensor's dimension is 3
        if len(vectors.shape()) != 3:
            vectors_name = vectors._attrs["name"]
            raise RuntimeError(f"Input tensor {vectors_name} dim should be 3.")

        if len(matrices.shape()) != 3:
            matrices_name = matrices._attrs["name"]
            raise RuntimeError(f"Input tensor {matrices_name} dim should be 3.")

        jagged_int_var = matrices.shape()[0]
        # Check first dim B
        if jagged_int_var.batch_dim() != vectors.shape()[0]:
            raise RuntimeError(
                f"Batch dim B of input tensors are expected to be the same, but actually first is {vectors.shape()[0]} and second is {jagged_int_var.batch_dim()}."
            )

        # Check second dim H
        if vectors.shape()[1] != matrices.shape()[1]:
            raise RuntimeError(
                f"Second dim H of input tensors are expected to be the same, but actually first is {vectors.shape()[1]} and second is {matrices.shape()[1]}."
            )

        # Check tensor types
        if vectors.dtype() != matrices.dtype():
            raise RuntimeError(
                f"Input tensors sare expected to have the same type, but actually first is {vectors.dtype()} and second is {matrices.dtype()}."
            )

        # Check Jagged dims
        num_jagged_dims = len(jagged_int_var.jagged_dims())
        if num_jagged_dims != 1:
            raise RuntimeError(
                f"Jagged dims for second jagged inputs should be 1, but actually is {num_jagged_dims}."
            )
        else:
            jagged_max_values = jagged_int_var.jagged_dims()[0].max_value()
            if jagged_max_values != vectors.shape()[2].value():
                raise RuntimeError(
                    f"max value is expected to be {vectors.shape()[2].value()} , but actually is {jagged_max_values}."
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
