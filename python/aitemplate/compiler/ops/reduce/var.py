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
var op implementation
"""
from aitemplate.compiler.ops.reduce.reduce_common import reduce_base

# pylint: disable=C0103


class var(reduce_base):
    """
    Calculates the variance of all elements in the input tensor.

    * .attr.:`dim` : int or tuple of python:ints
      the dimension or dimensions to reduce

    * .attr.:`unbiased` : bool
      specifying whether to use Bessel’s correction or not

    * .attr.:`keepdim` : bool, optional
      keep the reduced dimensions if True, default is False

    * .attr.:`dtype` : str, optional
      the type of the return tensor. If it is not None,
      the input tensor is cast to dtype before reduction.

    Args:
        input (Tensor): the input tensor.

    Return:
        Tensor.

    """

    def __init__(self, dim, unbiased, keepdim=False, dtype=None) -> None:
        """initialization routine of var op"""
        super().__init__(dim, keepdim, dtype)
        self._attrs["op"] = "var"
        self._attrs["unbiased"] = unbiased

    def _get_op_attributes(self):
        return {
            "dim": self._attrs["reduction_axes"],
            "dtype": self._attrs["output_type"],
            "keepdim": self._attrs["keepdim"],
            "unbiased": self._attrs["unbiased"],
        }
