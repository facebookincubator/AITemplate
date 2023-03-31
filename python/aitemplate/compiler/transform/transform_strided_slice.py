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
Perform transformations on slice and strided ops.
"""
import math

from typing import List

from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.compiler.ops.tensor.dynamic_slice import dynamic_slice, MAX_INT32
from aitemplate.compiler.transform import transform_strided_ops_utils, transform_utils

from aitemplate.utils import alignment as utils_alignment, graph_utils, shape_utils


def _is_supported_gemm_or_bmm(gemm_or_bmm_op: Operator, slice_op: Operator) -> bool:
    if not gemm_or_bmm_op._attrs["op"].startswith(("gemm_rcr", "bmm")):
        return False
    slice_output_tensor = slice_op._attrs["outputs"][0]
    slice_output_rank = slice_output_tensor._rank()
    # TODO: support cases where slice_input_tensor is used by non-A/B
    # matrices, e.g. bias/d1/d2 in gemm_rcr_bias_add_add
    op_inputs = gemm_or_bmm_op._attrs["inputs"]
    if (
        op_inputs[0] is not slice_output_tensor
        and op_inputs[1] is not slice_output_tensor
    ):
        return False
    return slice_output_rank >= 2


def _sanity_check_concatenate(concat_op: Operator, slice_op: Operator) -> bool:
    # Although it cannot happen, just make sure we are not going to apply both
    # input and output accessors to a single input of the concat op
    slice_output_tensor = slice_op._attrs["outputs"][0]
    for idx, (input_tensor, input_mask) in enumerate(
        zip(concat_op._attrs["original_inputs"], concat_op._attrs["input_masks"])
    ):
        if input_tensor is slice_output_tensor:
            assert input_mask, (
                f"Expected input_mask to be True at {idx=} for "
                f'input_tensor {input_tensor._attrs["name"]} and '
                f'slice_op {slice_op._attrs["name"]}'
            )
    return True


def _is_supported_op(op: Operator, slice_op: Operator) -> bool:
    op_type = op._attrs["op"]
    if op_type.startswith(("bmm", "gemm")):
        return _is_supported_gemm_or_bmm(op, slice_op)
    if op_type == "concatenate":
        return _sanity_check_concatenate(op, slice_op)
    if op_type == "fused_elementwise" or op_type == "permute021":
        return True
    if op_type.startswith("layernorm") or op_type.startswith("group_layernorm"):
        return True
    return False


def _is_slice_full_range(dim: IntVar, start_idx: int, end_idx: int) -> bool:
    """
    return true if start_idx and end_idx covers the full range of a slice dim
    """
    if start_idx == 0 and end_idx == MAX_INT32:
        return True
    # if it's dynamic dimension, we don't know. So, just return False
    if not isinstance(dim, IntImm):
        return False
    start_idx, end_idx = dynamic_slice.normalize_start_end_indices(
        dim.value(), start_idx, end_idx
    )
    return end_idx - start_idx >= dim.value()


def _valid_alignment(
    op: Operator,
    slice_dim: int,
    slice_output_tensor: Tensor,
    slice_input_shape: List[IntVar],
    start_indices: List[int],
    end_indices: List[int],
) -> bool:
    op_type = op._attrs["op"]
    if (
        op_type in ("fused_elementwise", "concatenate", "permute021")
        or op_type.startswith("layernorm")
        or op_type.startswith("group_layernorm")
    ):
        return True

    dtype = slice_output_tensor.dtype()
    stride = shape_utils.get_static_stride(slice_input_shape, slice_dim)
    assert (
        stride is not None
    ), f"expected non-None stride for {slice_input_shape=} at {slice_dim=}"
    start_offset = start_indices[slice_dim] * stride
    if op_type.startswith("gemm_rcr"):
        # for n-d * 2-d cases, we are only able to support a special case
        # where we fully slice all axes except the last one (i.e. -1), because
        # it the only case where we can have a constant stride. To see why other
        # cases won't work, let's say we have the following inputs:
        #   slice_input_shape = [2, 3, 8]
        #   slice_start_indices = [0, 0, 0]
        #   slice_end_indices = [2, 2, 8]
        # for sliced output [2, 2, *], we end up with addresses 0, 8, 24, 32,
        # which cannot be represented by a single constant stride value.
        slice_output_rank = op._attrs["outputs"][0]._rank()
        if slice_output_rank > 2:
            for dim, s_i, e_i in zip(
                slice_input_shape[:-1], start_indices[:-1], end_indices[:-1]
            ):
                if not _is_slice_full_range(dim, s_i, e_i):
                    return False

        k_dim = slice_input_shape[-1]
        if not isinstance(k_dim, IntImm):
            return False
        alignment = math.gcd(k_dim.value(), start_offset)
        return utils_alignment.valid_alignment(alignment, dtype)

    if op_type.startswith("bmm"):
        bmm_inputs = op._attrs["inputs"]
        if bmm_inputs[0] is slice_output_tensor:
            # _get_a_leading_dim(m, k)
            leading_dim = op._get_a_leading_dim(
                slice_input_shape[op._get_m_idx_in_a(slice_input_shape)],
                slice_input_shape[op._get_k_idx_in_a(slice_input_shape)],
            )
        elif bmm_inputs[1] is slice_output_tensor:
            # _get_a_leading_dim(n, k)
            leading_dim = op._get_b_leading_dim(
                slice_input_shape[op._get_n_idx_in_b(slice_input_shape)],
                slice_input_shape[op._get_k_idx_in_b(slice_input_shape)],
            )
        else:
            # TODO: support strided access for other inputs
            return False
        if not isinstance(leading_dim, IntImm):
            return False
        alignment = math.gcd(leading_dim.value(), start_offset)
        return utils_alignment.valid_alignment(alignment, dtype)

    return False


def _process_one_slice_dst(
    slice_op: Operator, slice_output_tensor: Tensor, next_op: Operator
) -> bool:
    """
    Process one slice_output_tensor's dst op. Return True upon success.
    """
    if not _is_supported_op(next_op, slice_op):
        return False
    strided_op = next_op

    slice_input_tensor = slice_op._attrs["inputs"][0]
    slice_input_rank = slice_input_tensor._rank()
    slice_input_shape = slice_input_tensor._attrs["shape"]
    start_indices = slice_op._attrs["start_indices"]
    end_indices = slice_op._attrs["end_indices"]
    strided_op_name = strided_op._attrs["op"]

    # TODO: Currently, we only support a special case where all dimensions
    # are fully sliced except one. In such a case, we can use the same
    # TensorAccessor interface to update base tensors. We will revisit
    # this part once we refactor our TensorAccessor to support more general
    # senarios.
    slice_dim = None
    normalized_start_indices = []
    normalized_end_indices = []
    # Let's skip consecutive fully-sliced static dims starting from the
    # innermost dim
    for idx in reversed(range(slice_input_rank)):
        dim = slice_input_shape[idx]
        if not isinstance(dim, IntImm):
            break
        slice_dim = idx
        start_idx = start_indices[idx]
        end_idx = end_indices[idx]
        start_idx, end_idx = dynamic_slice.normalize_start_end_indices(
            dim.value(), start_idx, end_idx
        )
        normalized_start_indices.append(start_idx)
        normalized_end_indices.append(end_idx)
        if not _is_slice_full_range(dim, start_idx, end_idx):
            break
    # We encountered a dynamic dim before a valid slice_dim
    if slice_dim is None:
        return False

    invalid_slice = False
    # We got a valid slice_dim, but we need to keep checking if the
    # remaining slice indices are valid
    for idx in reversed(range(slice_dim)):
        dim = slice_input_shape[idx]
        start_idx = start_indices[idx]
        end_idx = end_indices[idx]
        if _is_slice_full_range(dim, start_idx, end_idx):
            start_idx = 0
            end_idx = MAX_INT32
            if isinstance(dim, IntImm):
                start_idx, end_idx = dynamic_slice.normalize_start_end_indices(
                    dim.value(), start_idx, end_idx
                )
            normalized_start_indices.append(start_idx)
            normalized_end_indices.append(end_idx)
        else:
            invalid_slice = True
            break

    if invalid_slice:
        return False
    offset = 0
    normalized_start_indices.reverse()
    normalized_end_indices.reverse()
    offset = normalized_start_indices[slice_dim]

    # Now let's check alignment
    if not _valid_alignment(
        strided_op,
        slice_dim,
        slice_output_tensor,
        slice_input_shape,
        normalized_start_indices,
        normalized_end_indices,
    ):
        return False

    to_be_updated_input_accessors = []
    for idx, input_tensor in enumerate(strided_op._attrs["inputs"]):
        if input_tensor is not slice_output_tensor:
            continue
        input_accessors = strided_op._attrs["input_accessors"]

        if any(strided_op_name.startswith(n) for n in ("gemm", "group_gemm", "bmm")):
            if not transform_strided_ops_utils.gemm_stride_checker(
                input_accessors[idx], slice_dim
            ):
                return False
        to_be_updated_input_accessors.append(input_accessors[idx])

    for tc in to_be_updated_input_accessors:
        tc.update_base_tensor(slice_input_tensor, slice_dim, offset)

    transform_utils.replace_tensor_for_op(
        strided_op, slice_output_tensor, slice_input_tensor
    )
    return True


def _fuse_slice_and_strided_op(
    sorted_graph: List[Tensor],
) -> List[Tensor]:
    """
    This pass detects patterns like below:
      x1 = slice(x, start_indices, end_indices)
      y = gemm_rcr(x1, w1)

    where gemm_rcr stands for a strided_op, which can be one of the followings:
    gemm-family ops, layernorm(s), elementwise ops, etc.
    When we detect such a pattern, we generate stride information for each input
    tensor with respect to its portion in slice. Later, the strided_op backend
    will generate strided accesses based on the stored stride information.
    """
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in sorted_ops:
        op_type = op._attrs["op"]
        if op_type != "dynamic_slice":
            continue
        slice_op = op

        slice_outputs = slice_op._attrs["outputs"]
        if len(slice_outputs) != 1:
            continue
        slice_output_tensor = slice_outputs[0]
        slice_dsts = slice_output_tensor.dst_ops()
        for next_op in list(slice_dsts):
            _process_one_slice_dst(slice_op, slice_output_tensor, next_op)
        # We remove the slice_op from the graph if all of its output dsts are
        # valid strided ops. In such a case, output_tensor's "dsts" is empty
        # already, so we can just remove the op from the graph.
        if (
            len(slice_output_tensor.dst_ops()) == 0
            and not slice_output_tensor._attrs["is_output"]
        ):
            transform_utils.remove_tensor_from_sorted_graph(slice_output_tensor)
    return transform_utils.sanitize_sorted_graph(sorted_graph)
