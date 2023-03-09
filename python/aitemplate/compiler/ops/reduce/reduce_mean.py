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
Reduce_mean op implementation.
"""
from aitemplate.compiler.ops.reduce.reduce_common import reduce_base

# pylint: disable=C0103


class reduce_mean(reduce_base):
    """
    Implements the reduce_mean op.

    * .attr.:`dim` : int or tuple of python:ints
      the dimension or dimensions to reduce

    * .attr.:`keepdim` : bool, optional
      keep the reduced dimensions if True, default is False

    * .attr.:`dtype` : str, optional
      the type of the return tensor. If it is not None,
      the input tensor is cast to dtype before reduction.

    Args:
        input (Tensor): the input tensor.

    Return:
        Tensor that contains the mean value of all elements in the input tensor.
    """

    def __init__(self, dim, keepdim=False, dtype=None) -> None:
        super().__init__(dim, keepdim, dtype)
        self._attrs["op"] = "reduce_mean"
