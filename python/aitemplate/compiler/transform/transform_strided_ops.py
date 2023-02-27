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
Perform transformations on ops which support strided inputs / outputs.
"""
import functools

from typing import List

from aitemplate.testing import detect_target

from ...utils import graph_utils, shape_utils
from ..base import IntImm, Operator, Tensor
from ..ops.tensor.slice_reshape_scatter import slice_reshape_scatter
from ..ops.tensor.slice_scatter import slice_scatter
from . import transform_strided_ops_utils, transform_utils
from .fuse_split import _fuse_split_and_group_gemm, _fuse_split_and_strided_op
from .transform_strided_op_and_view_op import _fuse_strided_op_and_view_op
from .transform_strided_slice import _fuse_slice_and_strided_op

# pylint: disable=W0612


def _fuse_slices_concat(sorted_graph: List[Tensor]) -> List[Tensor]:
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op._attrs["op"] != "concatenate":
            continue
        concat_op = src_op
        if slice_scatter.is_valid(concat_op):
            slice_scatter(concat_op)

    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _fuse_slices_concat_reshape_concat(sorted_graph: List[Tensor]) -> List[Tensor]:
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op._attrs["op"] != "concatenate":
            continue

        concat_op = src_op
        # TODO: we simply lookahead one step for a reshape op.
        # Later, we may want to write a standalone pass that merge consecutive
        # view-like ops.
        concat_output = concat_op._attrs["outputs"][0]
        if len(concat_output.dst_ops()) != 1:
            continue

        next_op = list(concat_output.dst_ops())[0]
        if next_op._attrs["op"] != "reshape":
            continue

        reshape_op = next_op
        reshape_output = reshape_op._attrs["outputs"][0]
        if len(reshape_output.dst_ops()) != 1:
            continue

        next_op = list(reshape_output.dst_ops())[0]
        if not next_op._attrs["op"].startswith("concatenate"):
            continue

        concat_op_2 = next_op
        if slice_reshape_scatter.is_valid(concat_op, reshape_op, concat_op_2):
            slice_reshape_scatter.make_op(concat_op, reshape_op, concat_op_2)

    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _is_strided_gemm(op_type: str) -> bool:
    return op_type.startswith("gemm")


def _gemm_cat_checker(gemm_op: Operator, cat_op: Operator) -> bool:
    shapes = gemm_op._attrs["output_accessors"][0].original_shapes
    rank = len(shapes)
    cat_dim = cat_op._attrs["concat_dim"]
    # For > 2D gemms, the only cat_dim possible is the last dim
    # or cases like [m1, m2, 1, n] with cat_dim = -2 or -1
    if rank > 2 and cat_dim != rank - 1:
        for shape in shapes[cat_dim:-1]:
            if shape.value() != 1:
                return False

    # Only correct for row major in C (C = A @ B)
    return transform_strided_ops_utils.gemm_stride_checker(
        gemm_op._attrs["output_accessors"][0],
        cat_dim,
    )


def _is_strided_group_gemm(strided_op: Operator) -> bool:
    op_type = strided_op._attrs["op"]
    # make sure this op is a group_gemm op and the user doesn't
    # explicitly set output_stride_dim, which is used for creating
    # a target tensor holding all the outputs of the group_gemm.
    # That being said, we don't overwrite the user's intent.
    return (
        op_type.startswith("group_gemm")
        and "output_stride_dim" not in strided_op._attrs
    )


def _group_gemm_cat_checker(
    group_gemm_op: Operator, cat_op: Operator, out_idx: int
) -> bool:
    return transform_strided_ops_utils.gemm_stride_checker(
        group_gemm_op._attrs["output_accessors"][out_idx], cat_op._attrs["concat_dim"]
    )


def _is_bmm(op_type: str) -> bool:
    # TODO: support cutlass bmm ops
    return op_type.startswith(("bmm_rcr", "bmm_crr"))


def _bmm_checker(bmm_op: Operator, cat_op: Operator) -> bool:
    return transform_strided_ops_utils.gemm_stride_checker(
        bmm_op._attrs["output_accessors"][0], cat_op._attrs["concat_dim"]
    )


def _is_perm102_bmm(op_type: str) -> bool:
    # rcr/rrr family
    return op_type.startswith("perm102_bmm_r")


def _perm102_bmm_checker(bmm_op: Operator, cat_op: Operator) -> bool:
    # Only support fusion patterns like this:

    # x = perm102_bmm_xxx(a, b) # [m, b, n]
    # y = x.reshape()[x._size(0), -1] # [m, b * n]
    # z = cat()(y0, y1, ..., y, ..., yn, dim=-1)
    input = bmm_op._attrs["inputs"][0]
    output = bmm_op._attrs["outputs"][0]

    # Make sure reshape only flattens the last two dims
    if output._rank() != 2 or input._size(0) != output._size(0):
        return False

    cat_dim = cat_op._attrs["concat_dim"]

    return cat_dim == 1 and transform_strided_ops_utils.gemm_stride_checker(
        bmm_op._attrs["output_accessors"][0],
        cat_op._attrs["concat_dim"],
        get_stride_at_dim=0,
    )


def _is_layernorm(op_type: str) -> bool:
    return op_type.startswith("layernorm") or op_type.startswith("group_layernorm")


def _is_reduce_op(op_type: str) -> bool:
    return op_type in {"reduce_sum", "reduce_mean", "var", "vector_norm"}


def _reduce_cat_checker(op: Operator) -> bool:
    if op._attrs["op"] == "reduce_sum":
        # TODO: We only support output TensorAccessor for reduce_3d (and thus
        # the special reduce_small_axis kernels). reduce_sum invokes reduce_3d
        # only if the reduction_axis is -1, so we skip other reduction_axis values.
        x = op._attrs["inputs"][0]
        input_rank = x._rank()
        for axis in op._attrs["reduction_axes"]:
            assert (
                axis >= 0 and axis < input_rank
            ), f"axis ({axis}) is not in range of [0, {input_rank})"
        if axis != input_rank - 1:
            return False
    return True


def _is_supported_op(op: Operator) -> bool:
    op_type = op._attrs["op"]
    return (
        op_type == "fused_elementwise"
        or _is_strided_gemm(op_type)
        or _is_strided_group_gemm(op)
        or _is_layernorm(op_type)
        or _is_bmm(op_type)
        or _is_perm102_bmm(op_type)
        or _is_reduce_op(op_type)
    )


def _is_valid_for_fusion(strided_op: Operator, cat_op: Operator, out_idx: int):
    op_type = strided_op._attrs["op"]
    if _is_strided_gemm(op_type):
        return _gemm_cat_checker(strided_op, cat_op)
    if _is_strided_group_gemm(strided_op):
        return _group_gemm_cat_checker(strided_op, cat_op, out_idx)
    if _is_bmm(op_type):
        return _bmm_checker(strided_op, cat_op)
    if _is_perm102_bmm(op_type):
        return _perm102_bmm_checker(strided_op, cat_op)
    if _is_reduce_op(op_type):
        return _reduce_cat_checker(strided_op)
    return True


def get_tensor_index(tensors, tensor: Tensor) -> int:
    """
    Return the index for the tensor in the "tensors" list.
    """
    idx = None
    for input_idx, input_tensor in enumerate(tensors):
        if input_tensor is tensor:
            idx = input_idx
            # found the input to be removed
            break
    assert idx is not None, "Expected idx to be not None"
    return idx


def _fuse_strided_op_and_cat(sorted_graph: List[Tensor]) -> List[Tensor]:  # noqa: C901
    """
    Fuse strided_ops and cat op. One special case is group_gemm/group_layernorm + multiple cat.
    We can have
    y1, y2, y3, y4 = group_layernorm([x1, x2, x3, x4])
    cat1 = concatenate(y1, y2)
    cat2 = concatenate(y3, y4)
    """
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in sorted_ops:
        if op._attrs["op"] != "concatenate":
            continue
        cat_op = op
        # check cat_dim is static
        cat_dim = cat_op._attrs["concat_dim"]
        if not transform_strided_ops_utils.cat_split_dim_is_static(cat_op, cat_dim):
            continue

        cat_inputs = cat_op._attrs["inputs"]
        cat_output = cat_op._attrs["outputs"][0]

        # We can have output_list = some_op()(x, y, z); y = concatenate()(output_list).
        # Depending on whether we create a copy or not in concatenate(),
        # the cat inputs and some_op outputs may point to the same list.
        # This copy adds extra safety to make sure that it doesn't happen.
        cat_op._attrs["inputs"] = cat_op._attrs["inputs"].copy()

        cat_inputs_to_remove = []
        for idx, cat_input in enumerate(cat_inputs):
            src_ops = list(cat_input.src_ops())
            if len(src_ops) != 1 or len(cat_input.dst_ops()) != 1:
                continue
            if cat_input._attrs["is_output"]:
                continue
            strided_op = src_ops[0]
            if not _is_supported_op(strided_op):
                continue

            # get output_idx into strided_op
            strided_idx = get_tensor_index(strided_op._attrs["outputs"], cat_input)

            if not _is_valid_for_fusion(strided_op, cat_op, strided_idx):
                continue

            # Update "outputs" of the strided_op, and "input_masks" of the cat op.
            # Note that we do not update "inputs" of the cat op here, to make sure
            # that we could find input indexes correctly for other tensors.
            if cat_op._attrs["inputs"].count(cat_input) > 1:
                # We do not support the case that a strided op write to multiple outputs.
                # TODO: Add multi-output later.
                continue

            offset = 0

            # cat's inputs may have been updated for cases like view_op + cat.
            # So, we need to retrieve original shapes from its input accessors.
            cat_input_accessors = cat_op._attrs["input_accessors"]
            # This pass must run before any other pass that remove cat inputs, like
            # _fuse_strided_op_reshape_cat
            for orig_i in range(idx):
                input_accessor = cat_input_accessors[orig_i]
                # TODO: Add dynamic shape support.
                offset += input_accessor.original_shapes[cat_dim].value()

            cat_inputs_to_remove.append(idx)

            strided_op._attrs["output_accessors"][strided_idx].update_base_tensor(
                cat_output, cat_dim, offset
            )

            cat_output._attrs["src_ops"].add(strided_op)

            output_tensor = strided_op._attrs["outputs"][strided_idx]
            strided_op._attrs["outputs"][strided_idx] = cat_output
            transform_utils.remove_tensor_from_sorted_graph(output_tensor)

            # Note that we have to update strided_op's epilogue_alignment
            # in the backend codegen where we modify the generate gemm
            # instance string with the updated epilogue_alignment.
            # The reason is similar to the one with slice + gemm.
            # For a problem size, the max alignment value returned from
            # output_accessor may vary for different concat cases.
            # If we update the strided_op's epilogue_alignment at this point,
            # the updated epilogue_alignment value will be cached in our
            # profiler database. Next time, when we get a different
            # updated epilogue_alignment value for the same problem size,
            # we would end up with not being able to find a matching
            # config or hit misalignment failures at runtime with invalid
            # alignment values.

        cat_op.remove_input_at(cat_inputs_to_remove)

    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _fuse_group_gemm_reshape_cat(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    This pass fuses strided_op + view_op + concat patterns. It mainly performs the
    following transformations:
      (1) updates the strided_op's output_accessors with new stride information;
      (2) updates concat op's input_masks and inputs;
      (3) removes the view_op from the graph
    Crrently, group_gemm is the only supported strided_op, and reshape and
    unsqueeze are the supported view_op(s).
    """
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)

    visited_cat_ops = set()
    for op in sorted_ops:
        # TODO: add support to other strided ops such as elementwise ops
        if not _is_strided_group_gemm(op):
            continue
        strided_op = op
        for idx, output_tensor in enumerate(strided_op._attrs["outputs"]):
            # Find the next strided_op + reshape + cat.
            if len(output_tensor.dst_ops()) != 1:
                continue
            next_op = list(output_tensor.dst_ops())[0]

            # TODO: add flatten later
            if next_op._attrs["op"] not in ["reshape", "unsqueeze"]:
                continue

            view_op = next_op
            reshape_output_tensor = view_op._attrs["outputs"][0]
            # Let's keep it simple for now. We might be able to support
            # the case where reshape's output is used by multiple ops.
            if len(reshape_output_tensor.dst_ops()) > 1:
                continue
            # skip dynamic shape
            if not shape_utils.all_static_dimensions(
                reshape_output_tensor._attrs["shape"]
            ):
                continue

            if not reshape_output_tensor.dst_ops():
                continue

            next_op = list(reshape_output_tensor.dst_ops())[0]

            if next_op._attrs["op"] != "concatenate":
                continue
            cat_op = next_op

            if not _is_valid_for_fusion(strided_op, cat_op, idx):
                continue

            # Update "outputs" of the strided_op, and "input_masks" of the cat op.
            # Note that we do not update the "inputs" of the cat op here, to make sure
            # that we could find input indices correctly for other tensors.

            cat_dim = cat_op._attrs["concat_dim"]
            if not transform_strided_ops_utils.cat_split_dim_is_static(cat_op, cat_dim):
                continue

            reshape_to_shape = reshape_output_tensor._attrs["shape"]
            if cat_dim >= len(reshape_to_shape):
                continue

            if cat_op not in visited_cat_ops:
                # We can have output_list = some_op()(x, y, z); y = concatenate()(output_list).
                # Depending on whether we create a copy or not in concatenate(),
                # the cat inputs and some_op outputs may point to the same list.
                # This copy adds extra safety to make sure that it doesn't happen.
                cat_op._attrs["inputs"] = cat_op._attrs["inputs"].copy()
            visited_cat_ops.add(cat_op)

            cat_idx = cat_op.get_tensor_index(reshape_output_tensor)

            offset = 0
            orig_idx = cat_op.get_original_index(cat_idx)

            # We compute the offset with original inputs because the current inputs
            # may have been modified by other transformations. For example,
            # _fuse_strided_op_and_cat may remove some input tensors.
            for orig_i in range(orig_idx):
                input_tensor = cat_op._attrs["original_inputs"][orig_i]
                input_tensor_shape = input_tensor._attrs["shape"]
                assert cat_dim < len(input_tensor_shape), (
                    f"Expected cat_dim to be less than the length of input_tensor_shape, "
                    f"but got cat_dim: {cat_dim}, and input_tensor_shape: {input_tensor_shape}"
                )
                assert shape_utils.all_static_dimensions(input_tensor_shape, cat_dim), (
                    f"Expected input_tensor_shape[{cat_dim}:] are all static dimensions, "
                    f"but got: {input_tensor_shape}"
                )

                strided_dim_values = [
                    dim._attrs["values"][0] for dim in input_tensor_shape[cat_dim:]
                ]
                offset += functools.reduce(lambda t1, t2: t1 * t2, strided_dim_values)

            cat_op.remove_input_at(cat_idx)
            cat_output = cat_op._attrs["outputs"][0]
            cat_output_shape = cat_output._attrs["shape"]
            # cat_output_shape[cat_dim:] is guaranteed to be static by the
            # cat_split_dim_is_static above
            strided_dim_values = [
                dim._attrs["values"][0] for dim in cat_output_shape[cat_dim:]
            ]
            dim_value = functools.reduce(lambda t1, t2: t1 * t2, strided_dim_values)
            new_tensor_shape = cat_output_shape[:cat_dim] + [IntImm(dim_value)]
            new_tensor = Tensor(shape=new_tensor_shape, dtype=cat_output.dtype())
            strided_op._attrs["output_accessors"][idx].update_base_tensor(
                new_tensor, cat_dim, offset
            )

            cat_output._attrs["src_ops"].add(strided_op)

            transform_utils.remove_view_op_from_sorted_graph(view_op)

            output_tensor = strided_op._attrs["outputs"][idx]
            strided_op._attrs["outputs"][idx] = cat_output
            transform_utils.remove_tensor_from_sorted_graph(output_tensor)

    return transform_utils.sanitize_sorted_graph(sorted_graph)


def transform_strided_ops(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """
    Add strided inputs / outputs to ops to avoid unnecessary data movement.
    """
    if detect_target().name() == "cuda":
        funcs = [
            # TODO: Remove these passes after cat supports input_accessors.
            _fuse_slices_concat_reshape_concat,
            _fuse_split_and_group_gemm,
            # Common passes:
            _fuse_strided_op_and_view_op,
            _fuse_strided_op_and_cat,
            _fuse_split_and_strided_op,
            # make sure this pass runs after _fuse_strided_op_and_cat
            _fuse_slice_and_strided_op,
            # TODO: Remove this pass after we support general strides with input_accessors
            _fuse_slices_concat,
            # TODO: Remove group_gemm passes after group_gemm shape inference is fixed.
            _fuse_group_gemm_reshape_cat,
        ]
    else:
        funcs = [
            # Keep on ROCM
            _fuse_strided_op_and_view_op,
            _fuse_strided_op_and_cat,
            _fuse_split_and_strided_op,
            _fuse_slice_and_strided_op,
            _fuse_slices_concat,
        ]
    for func in funcs:
        sorted_graph = func(sorted_graph)
        graph_utils.dump_graph_debug_str_to_file(sorted_graph, workdir, func.__name__)
    return sorted_graph
