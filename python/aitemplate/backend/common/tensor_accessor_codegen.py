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
Tensor accessor related codegens.
"""

import os
from typing import List

import jinja2

from ...compiler.tensor_accessor import TensorAccessor
from ..target import Target

# Template used to transform a Python TensorAccessor object
# to a C++ TensorAccessor struct.
TENSOR_ACCESSOR_TEMPLATE = jinja2.Template(
    """
    TensorAccessor {{name}} = {
      {{tensor_accessor.offset}},
      {% if tensor_accessor.is_contiguous %}
      true
      {% else %}
      false
      {% endif %}
      {% if not tensor_accessor.is_contiguous %}
      ,
      {{tensor_accessor.stride_dim}},
      {{tensor_accessor.original_total_elements_from_stride_dim}},
      {{tensor_accessor.actual_total_elements_from_stride_dim}}
      {% endif %}
    };
"""
)

STRIDED_ADDRESS_AT_IDX_FUNC_TEMPLATE = jinja2.Template(
    """
template <typename DATA_T, typename READ_T>
__device__ __forceinline__ READ_T* get_strided_address_at_idx(
    READ_T *data, int64_t data_idx) {
{%if output_accessor.is_contiguous %}
  return get_strided_address<DATA_T, READ_T, true>(
      data, data_idx, {{output_accessor.offset}}, 0, 0);
{% else %}
  return get_strided_address<DATA_T, READ_T, false>(
      data, data_idx,
      {{output_accessor.offset}},
      {{output_accessor.original_total_elements_from_stride_dim}},
      {{output_accessor.actual_total_elements_from_stride_dim}});
{% endif %}
}
"""
)


def get_libs() -> str:
    return Target.current().get_custom_libs(
        os.path.dirname(__file__), "tensor_accessor.cuh"
    )


# Currently read4, add2 is best for both backend, so two backend seems identical.
# They may diverge when we got deeper understanding / further optimization.
ALIGNMENTS = [
    8,
    4,
    2,
    1,
]


def _find_max_alignment(number: int) -> int:
    """
    Return the first alignment value that meets the alignment requirement
    for accessing the `number` of elements.
    """
    for alignment in ALIGNMENTS:
        if number % alignment == 0:
            return alignment
    return 1


def find_max_alignment_for_accessor(accessor: TensorAccessor) -> int:
    """the max alignment value that meets the requirement specified by
       the accessor

    Parameters
    ----------
    accessors: TensorAccessor

    Returns
    ----------
    int
        the max alignment value
    """
    alignment = _find_max_alignment(accessor.offset)
    if not accessor.is_contiguous:
        alignment = min(
            alignment,
            _find_max_alignment(accessor.original_total_elements_from_stride_dim),
        )
        alignment = min(
            alignment,
            _find_max_alignment(accessor.actual_total_elements_from_stride_dim),
        )
    return alignment


def find_max_alignment_for_accessors(accessors: List[TensorAccessor]) -> int:
    """the max alignment value that meets the requirement specified by
       the accessors

    Parameters
    ----------
    accessors: List[TensorAccessor]
        TensorAccessor(s) attached to the relevant tensor being accessed

    Returns
    ----------
    int
        the max alignment value
    """
    alignment = max(ALIGNMENTS)
    # Handle accessors
    for accessor in accessors:
        alignment = min(alignment, find_max_alignment_for_accessor(accessor))
    return alignment


def find_max_alignment(num_elements: int, accessors: List[TensorAccessor]) -> int:
    """find the max alignment value that meets the requirement of accessing
       num_elements of data with access patterns (strides and offsets)
       specified by accessors

    Parameters
    ----------
    num_elements: int
        specify the number of elements being accessed

    accessors: List[TensorAccessor]
        TensorAccessor(s) attached to the relevant tensor being accessed

    Returns
    ----------
    int
        the max alignment value
    """
    # get initial alignment based on the number of elements being accessed
    alignment = _find_max_alignment(num_elements)
    accessor_alignment = find_max_alignment_for_accessors(accessors)
    return min(alignment, accessor_alignment)
