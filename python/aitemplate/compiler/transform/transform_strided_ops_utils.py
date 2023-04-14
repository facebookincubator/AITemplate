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

import copy
import logging
from typing import List

from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.compiler.tensor_accessor import TensorAccessor


_LOGGER = logging.getLogger(__name__)


def _dynamic_shape_checker(shape: List[IntVar], dim: int) -> bool:
    has_dynamic_dim = False
    for idx in range(dim, len(shape)):
        if not isinstance(shape[idx], IntImm):
            has_dynamic_dim = True
    return not has_dynamic_dim


def cat_split_dim_is_static(op: Operator, cat_split_dim: int) -> bool:
    """
    To simplify, we only check either input or output. Assumption, if a certain input dim of any
    input/output tensor is dynamic, the corresponding output/input dim must be dynamic.

    Args:
        op (Operator): only supports split and concatenate ops
        cat_split_dim (int): cat or split dim

    Raises:
        RuntimeError: raises RuntimeError if unsupported op is passed

    Returns:
        bool: returns True if all the dims from cat_split_dim are static
    """
    tensor = None
    if op._attrs["op"] == "split":
        tensor = op._attrs["inputs"][0]
    elif op._attrs["op"] == "concatenate":
        tensor = op._attrs["outputs"][0]
    else:
        raise RuntimeError("Unsupported op:", op._attrs["op"])
    # check all dims after cat_split_dim
    return _dynamic_shape_checker(tensor._attrs["shape"], cat_split_dim)


def gemm_stride_checker(
    original_ta: TensorAccessor, dim: int, get_stride_at_dim: int = None
) -> bool:
    """
    Checks whether it's possible to get gemm stride correctly from original_ta
    given an input stride dim.

    This function should be called before actually invoking TensorAccessor.update_base_tensor()
    for gemm ops. If it returns False, we should avoid calling update_base_tensor().

    This is necessary because CUTLASS gemm ops have special "stride" params:
    batch_stride and row_stride. We need to make sure that after stride dim is updated
    in the TensorAccessor, these strides could still be populated successfully.
    """

    if not _dynamic_shape_checker(original_ta.original_shapes, dim):
        return False

    # Need to make sure that the new stride dim doesn't break
    # last dim's continuity.
    # This is because CUTLASS GEMM API assumes that GEMM stride
    # only operates on the last dim for row-major output.
    # For example, concatenations of GEMMs along dimensions to the right of the
    # original shape can't be fused. A particular case of this is when GEMM
    # output of shape (M, N) is unsqueezed to (M, N, 1) and concatenated with
    # another (M, N, 1).
    if not original_ta.is_rightmost_dim_contiguous(dim):
        return False

    if get_stride_at_dim is None:
        # The dim before the last dim
        get_stride_at_dim = len(original_ta.original_shapes) - 2

    tmp_ta = copy.deepcopy(original_ta)
    tmp_shape = copy.deepcopy(
        original_ta.actual_shapes
        if original_ta.actual_shapes is not None
        else original_ta.original_shapes
    )
    # Make sure new shape is different from the original shape.
    tmp_shape[dim] = IntImm(tmp_shape[dim].value() + 1)
    tmp_tensor = Tensor(shape=tmp_shape)
    tmp_ta.update_base_tensor(tmp_tensor, dim, stride_dim_offset=0)

    # TODO: Make this configurable for different gemms, bmms, etc.
    stride_strs = tmp_ta.try_get_stride_strs(get_stride_at_dim)
    if stride_strs is None:
        _LOGGER.debug(
            f"Failed in gemm_stride_checker: "
            f"dim: {dim}, "
            f"original_shapes length: {len(original_ta.original_shapes)}"
        )
        return False
    else:
        return True
