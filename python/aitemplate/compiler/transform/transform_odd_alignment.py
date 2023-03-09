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
Add permute for gemm/bmm if alignment is odd.
"""
from math import inf
from typing import Dict, List, Tuple

from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.compiler.ops.common.view_ops import unsqueeze
from aitemplate.compiler.ops.gemm_universal import bmm_ccr, bmm_crr, bmm_rcr, bmm_rrr
from aitemplate.compiler.ops.tensor import permute021

from aitemplate.compiler.transform.apply_padding import get_padding_length
from aitemplate.compiler.transform.fuse_utils import extract_only_one_op
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.compiler.transform.transform_strided_ops import (
    _is_supported_op as _is_supported_strided_op,
)
from aitemplate.compiler.transform.transform_strided_slice import (
    _is_supported_op as _is_supported_strided_slice,
)
from aitemplate.compiler.transform.transform_utils import (
    can_be_constant_folded,
    copy_src_op_attributes,
    copy_tensor_attributes,
    remove_dst_op_from_tensor,
    remove_tensor_from_sorted_graph,
    replace_tensor,
    sanitize_sorted_graph,
)

# pylint: disable=C0103,W0612


def _matrix_shape_prod(shapes: List[IntVar]) -> int:
    prod = 1
    for shape in shapes:
        if isinstance(shape, IntImm):
            prod *= shape.value()
        else:
            prod *= shape.upper_bound()

    return prod


def _compute_padding_flops(
    tensor: Tensor, shapes: List[IntVar], padding_idx: int
) -> int:
    if shapes[padding_idx].value() % 2 == 0:
        return 0

    if can_be_constant_folded(tensor):
        return 0
    elif _is_strided_tensor(tensor):
        return (
            _matrix_shape_prod(shapes)
            * get_padding_length(shapes[padding_idx].value(), tensor.dtype())
            / shapes[padding_idx].value()
        )
    else:
        return _matrix_shape_prod(shapes)


def _compute_slicing_flops(mm_op: Operator, slicing_dim: int, other_dim: int) -> int:
    can_be_fused = True

    if len(mm_op._attrs["outputs"][0].dst_ops()) == 0:
        can_be_fused = False
    for dst in mm_op._attrs["outputs"][0].dst_ops():
        # We can use mm_op here since the shape of post-slice would be the same.
        if not _is_strided_slice(mm_op, dst):
            can_be_fused = False

    if can_be_fused:
        return other_dim * get_padding_length(
            slicing_dim, mm_op._attrs["inputs"][0].dtype()
        )
    else:
        return other_dim * slicing_dim


def _get_K_index(op_type: str):
    k_mapping = {"ccr": [-2, -1], "crr": [-2, -2], "rcr": [-1, -1], "rrr": [-1, -2]}

    for k, v in k_mapping.items():
        if op_type.find(k) != -1:
            return v

    raise RuntimeError(f"Can't find K index mapping for {op_type}")


def _get_nonK_len(shape: List[IntVar], k_idx: int) -> int:
    nonK_shape = shape[-1] if k_idx == -2 else shape[-2]
    return (
        nonK_shape.value()
        if isinstance(nonK_shape, IntImm)
        else nonK_shape.upper_bound()
    )


def _is_strided_tensor(tensor: Tensor):
    src_op = extract_only_one_op(tensor.src_ops())
    if src_op is None:
        return False

    if src_op._attrs["op"] == "elementwise":
        # elementwise are not fused yet.
        return True
    return _is_supported_strided_op(src_op)


def _is_strided_slice(op: Operator, next_op: Operator):
    if next_op._attrs["op"] == "elementwise":
        return True
    return _is_supported_strided_slice(next_op, op)


def _compute_required_flops(mm_op: Operator, x_perm: bool, w_perm: bool) -> int:
    inputs = mm_op._attrs["inputs"]
    input_shapes = (inputs[0].shape(), inputs[1].shape())
    perm = [x_perm, w_perm]

    for idx in range(2):
        if not perm[idx]:
            continue
        if not (
            isinstance(input_shapes[idx][-1], IntImm)
            and input_shapes[idx][-1].value() % 2 == 1
            and isinstance(input_shapes[idx][-2], IntImm)
            and input_shapes[idx][-2].value() % 2 == 0
        ):
            # Make sure we are really permuting from odd to even alignment
            return inf

    k_idx = _get_K_index(mm_op._attrs["op"])

    count = 0
    pad_k = False
    for idx in range(2):
        if perm[idx]:
            count += (
                0
                if can_be_constant_folded(inputs[idx])
                else _matrix_shape_prod(input_shapes[idx])
            )
        else:
            count += _compute_padding_flops(inputs[idx], input_shapes[idx], -1)
            if k_idx[idx] == -1:
                pad_k = True

    for idx in range(2):
        # We add a k-padding if dimension k is being padded on other input
        if pad_k and not perm[idx] and k_idx[idx] != -1:
            count += _compute_padding_flops(inputs[idx], input_shapes[idx], -2)

    for idx in range(2):
        if (
            not perm[idx]
            and k_idx[idx] != -1
            and input_shapes[idx][-1].value() % 2 == 1
        ):
            nonk_len = _get_nonK_len(input_shapes[idx], k_idx[idx])

            other_idx = (idx + 1) % 2
            other_nonk_len = _get_nonK_len(input_shapes[other_idx], k_idx[other_idx])

            count += _compute_slicing_flops(mm_op, nonk_len, other_nonk_len)

    return count


def _transform_odd_alignment(
    sorted_graph: List[Tensor],
    permutable_pairs: Dict[str, Tuple[Operator, Operator, Operator]],
) -> List[Tensor]:
    """
    This function tries to insert new permute021() ops before gemms when applicable.

    For input tensors with odd alignments, either permutation or padding are needed
    to transform an odd alignment to an even alignment, so that SM80 cutlass kernels
    can be used.
    This function decides between permutation and padding by estimating cost of all
    the options, and selects the one with minimal cost. If adding permutation costs
    less, this function inserts new permute021 kernels before gemm ops. Otherwise
    this function does nothing and apply_padding pass kicks in.

    Cost of permutation is the total element of matrix.
    Cost of padding is computed with the following rules:
        1) If the connecting op cannot be fused with padding, the cost is the total
           elements of matrix.
        2) If the connecting op can be fused with padding, the cost would be the
           elements of the additional zeros padded.
    One special case is if the input is constant, both permutation and padding are
    free since we have constant folding pass.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        AIT graph to run fusion
    permutable_pairs: Dict[str, Tuple[Operator, Operator, Operator]]
        A dictionary where key is the op that might accept permuted inputs,
        and value is a tuple of len-3 which correspond to ops (permute first input,
        permute second input, permute both first and second inputs)
    """

    new_sorted_graph = []
    permuted_inputs = {}
    for tensor in sorted_graph:
        new_sorted_graph.append(tensor)

        src_op = extract_only_one_op(tensor._attrs["src_ops"])
        if src_op is None:
            continue

        op_type = src_op._attrs["op"]
        if op_type not in permutable_pairs:
            continue
        # FIXME: This pass only works for half type. We may need to change it to
        # work with other types such as int8 later. Note that for float type, it
        # is safe to skip, because gemm/bmm with float inputs always meet alignment
        # requirements.
        if src_op._attrs["inputs"][0].dtype() != "float16":
            continue

        perm_type = ([False, False], [False, True], [True, False], [True, True])
        permute_input = [False, False]
        best_cost = inf
        for p in perm_type:
            perm_cost = _compute_required_flops(src_op, p[0], p[1])
            if perm_cost < best_cost:
                permute_input = p
                best_cost = perm_cost

        if not permute_input[0] and not permute_input[1]:
            continue

        inputs = src_op._attrs["inputs"]
        new_inputs = list(inputs)
        for idx in range(2):
            if permute_input[idx]:
                if inputs[idx] in permuted_inputs:
                    permuted_input = permuted_inputs[inputs[idx]]
                else:
                    input_shape = inputs[idx].shape()
                    if len(input_shape) == 2:
                        expanded_input = unsqueeze(0)(inputs[idx])
                        new_sorted_graph.insert(-1, expanded_input)
                        permuted_input = permute021()(expanded_input)
                    else:
                        permuted_input = permute021()(inputs[idx])
                    new_sorted_graph.insert(-1, permuted_input)
                    permuted_inputs[inputs[idx]] = permuted_input
                new_inputs[idx] = permuted_input

        if permute_input[0] and permute_input[1]:
            new_tensor = permutable_pairs[op_type][2]()(*new_inputs)
        elif permute_input[0]:
            new_tensor = permutable_pairs[op_type][0]()(*new_inputs)
        elif permute_input[1]:
            new_tensor = permutable_pairs[op_type][1]()(*new_inputs)
        copy_tensor_attributes(new_tensor, tensor)
        copy_src_op_attributes(new_tensor, tensor)
        replace_tensor(tensor, new_tensor)

        remove_dst_op_from_tensor(inputs, src_op)
        remove_tensor_from_sorted_graph(tensor)

        new_sorted_graph[-1] = new_tensor

    new_sorted_graph = toposort(new_sorted_graph)
    return sanitize_sorted_graph(new_sorted_graph)


def transform_odd_alignment(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """Transform odd alignments to even alignments for bmm operators

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph
    workdir : str, optional
        workdir, by default None

    Returns
    -------
    List[Tensor]
        Optimized graph
    """
    permutable_pairs = {
        "bmm_ccr": (bmm_rcr, bmm_crr, bmm_rrr),
        "bmm_crr": (bmm_rrr, bmm_ccr, bmm_rcr),
        "bmm_rcr": (bmm_ccr, bmm_rrr, bmm_crr),
        "bmm_rrr": (bmm_crr, bmm_rcr, bmm_ccr),
    }

    sorted_graph = _transform_odd_alignment(sorted_graph, permutable_pairs)

    return sorted_graph
