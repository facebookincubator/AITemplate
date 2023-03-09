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
vector_norm op implementation that simulates pytorch's linalg.vector_norm.
Currently, we only support L2 norm.
"""
from aitemplate.compiler.ops.reduce.reduce_common import reduce_base

# pylint: disable=C0103


class vector_norm(reduce_base):
    """
    Vector_norm op implementation that simulates pytorch's linalg.vector_norm.
    Currently, we only support L2 norm.

    * .attr.:`ord_kind` (int or float or str), optional
      specifies the vector norm to be computed. (default: 2)

    * .attr.:`dim` (None or int or tuple of python:ints), optional
      the dimension or dimensions to be normalized.
      (default: None, in this case the input tensor will be treated as
      a 1-D tensor)

    * .attr.:`keepdim` (bool), optional
      keep the normalized dimensions if True, default is False

    * .attr.:`dtype` (str), optional
      the type of the return tensor. If it is not None,
      the input tensor is cast to dtype before reduction.

    Args:
        input (Tensor): the input tensor.

    Return:
        Tensor.
    """

    def __init__(self, ord_kind=2, dim=None, keepdim=False, dtype=None) -> None:
        """initialize the op"""
        if dim is None:
            raise NotImplementedError(
                "flattening input tensor before normalization is not supported yet"
            )
        super().__init__(dim, keepdim, dtype)
        self._attrs["op"] = "vector_norm"
        self._attrs["ord_kind"] = str(ord_kind)

    def _get_op_attributes(self):
        return {
            "dim": self._attrs["reduction_axes"],
            "dtype": self._attrs["output_type"],
            "keepdim": self._attrs["keepdim"],
            "ord_kind": self._attrs["ord_kind"],
        }
