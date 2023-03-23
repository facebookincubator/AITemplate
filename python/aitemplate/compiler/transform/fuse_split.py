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
import logging
from typing import List

from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor

from aitemplate.compiler.stable_set import StableSet
from aitemplate.compiler.transform import transform_strided_ops_utils, transform_utils

from aitemplate.utils import alignment, graph_utils

# pylint: disable=W0612


_LOGGER = logging.getLogger(__name__)


def _can_fuse_split_op(split_op: Operator):
    split_dim = split_op._attrs["split_dim"]
    # FIXME: only support dim == 1 at the moment
    if split_dim != 1:
        return False
    if not transform_strided_ops_utils.cat_split_dim_is_static(split_op, split_dim):
        return False
    return True


def _fuse_split_and_group_gemm(  # noqa: C901
    sorted_graph: List[Tensor],
) -> List[Tensor]:
    """
    This pass detects patterns like below:
      [x1, x2, x3] = split(x, dim=1)
      [y1, y2, y3] = group_gemm([[x1, w1], [x2, w2], [x3, w2]], stride_dim=1)

    and we generate stride information for each input tensor with respect to
    its portion in split(x). Later, the group gemm backend will generate
    strided accesses based on the stored stride information.
    We remove the split op.
    """
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in sorted_ops:
        op_type = op._attrs["op"]
        if op_type != "split":
            continue
        split_op = op
        if not _can_fuse_split_op(split_op):
            continue

        def _optional_group_gemm_op(dst_ops):
            # skip cases where this output has multiple users
            if len(dst_ops) != 1:
                return None
            dst_op = list(dst_ops)[0]
            # FIXME: we only handle row-major A at the moment.
            # We might need to change our TensorAccessor code to support col-major
            # tensors.
            if dst_op._attrs["op"].startswith("group_gemm_r"):
                return dst_op
            return None

        split_outputs = split_op._attrs["outputs"]
        group_gemm_op = _optional_group_gemm_op(split_outputs[0]._attrs["dst_ops"])
        if group_gemm_op is None:
            continue
        if group_gemm_op._attrs["groups"] != len(split_outputs):
            continue

        all_as = []
        all_a_indices = {}
        # group_gemm "inputs" is like [a1, b1, a2, b2, ...]
        # group_gemm_bias "inputs" is like
        # [a1, b1, bias1, a2, b2, bias1, ...]
        stride = 3 if group_gemm_op._attrs["op"].endswith("bias") else 2
        group_gemm_inputs = group_gemm_op._attrs["inputs"]
        for i in range(group_gemm_op._attrs["groups"]):
            t = group_gemm_inputs[i * stride]
            all_as.append(t)
            all_a_indices[t] = (i, i * stride)

        # make sure we make transformation only if the targeting ops are valid
        def _valid_input(input_tensor):
            return (
                len(input_tensor._attrs["src_ops"])
                == len(input_tensor._attrs["dst_ops"])
                == 1
            )

        # let's make our life easier - we only handle two cases: either (1) split's
        # outputs all go into As, or (2) split's outputs all go into Bs.
        # FIXME: we only implement (1) and need to add support to (2).
        if set(all_as) != set(split_outputs):
            continue

        if all(_valid_input(x) for x in all_as):
            input_indices = all_a_indices
            input_accessors = group_gemm_op.input_a_accessors()
        else:
            continue

        split_input = split_op._attrs["inputs"][0]
        split_dim = split_op._attrs["split_dim"]
        split_dim_offset = 0
        for split_output_tensor in split_outputs:
            accessor_idx, input_idx = input_indices[split_output_tensor]
            input_accessors[accessor_idx].update_base_tensor(
                split_input, split_dim, split_dim_offset
            )
            group_gemm_op._attrs["inputs"][input_idx] = split_input
            split_dim_offset += split_output_tensor._attrs["shape"][split_dim]._attrs[
                "values"
            ][0]
            transform_utils.remove_tensor_from_sorted_graph(split_output_tensor)
        # sanity check
        assert (
            split_dim_offset
            == split_input._attrs["shape"][split_dim]._attrs["values"][0]
        )
        # some final updates
        split_input._attrs["dst_ops"] = StableSet([group_gemm_op])
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _is_supported_op(op_type: str):
    from aitemplate.backend.target import Target

    if Target.current().name() == "rocm":
        return op_type == "bmm_softmax_bmm_permute"
    else:
        return op_type in {"bmm_rcr_n1", "bmm_rcr", "bmm_rrr_permute"}


def get_stride(t: Tensor, dim: int):
    stride = 1
    for shape in t.shape()[dim + 1 :]:
        stride *= shape.value()
    return stride


def _check_dim_alignment(shape: List[IntVar], dim_idx: int, dtype: str) -> bool:
    k_dim = shape[dim_idx]
    # skip dynamic dim
    if not isinstance(k_dim, IntImm):
        return False
    k_dim_val = k_dim._attrs["values"][0]
    # We cannot have mis-aligned K
    return alignment.valid_alignment(k_dim_val, dtype)


def _check_alignment(op: Operator, offset: int):
    # ops that support align=1
    if op._attrs["op"] == "bmm_rcr_n1":
        return True

    dtype = op._attrs["inputs"][0].dtype()
    # ops that don't have valid alignments
    if not alignment.valid_alignment(offset, dtype):
        return False
    if op._attrs["op"] == "bmm_rrr_permute":
        a_shape = op._attrs["input_accessors"][0].original_shapes
        b_shape = op._attrs["input_accessors"][1].original_shapes
        # check K and N
        return _check_dim_alignment(
            a_shape, dim_idx=2, dtype=dtype
        ) and _check_dim_alignment(b_shape, dim_idx=2, dtype=dtype)
    if op._attrs["op"] == "bmm_rcr":
        a_shape = op._attrs["input_accessors"][0].original_shapes
        # check K
        return _check_dim_alignment(a_shape, dim_idx=2, dtype=dtype)
    if op._attrs["op"] == "bmm_softmax_bmm_permute":
        # a = (B, M, K), b = (B, N, K), c = (B, N, O)
        # t = bmm_rcr(a, b)
        # t' = softmax(t) # t' shape (B, M, N)
        # bmm_rrr(t', c)
        a_shape = op._attrs["input_accessors"][0].original_shapes
        c_shape = op._attrs["input_accessors"][2].original_shapes
        return (
            # check K for bmm_rcr((B, M, K), (B, N, K))
            _check_dim_alignment(a_shape, dim_idx=2, dtype=dtype)
            and
            # check N for bmm_rrr((B, M, N), (B, N, O))
            _check_dim_alignment(c_shape, dim_idx=1, dtype=dtype)
            and
            # check O for bmm_rrr((B, M, N), (B, N, O))
            _check_dim_alignment(c_shape, dim_idx=2, dtype=dtype)
        )

    raise RuntimeError(f'Unexpected op type: {op._attrs["op"]}')


def _fuse_split_and_strided_op(sorted_graph: List[Tensor]) -> List[Tensor]:
    """Fuse split and any op that supports strided inputs. This pass requires
    that all of the outputs can be fused into the next ops so that split op
    can be eliminated. Partial fusion is not supported yet.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph

    Returns
    -------
    List[Tensor]
        Fused graph
    """
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in sorted_ops:
        op_type = op._attrs["op"]
        if op_type != "split":
            continue
        split_op = op
        split_dim = split_op._attrs["split_dim"]

        split_input = split_op._attrs["inputs"][0]
        # split_dim must be static
        if not transform_strided_ops_utils.cat_split_dim_is_static(split_op, split_dim):
            continue

        outputs = split_op._attrs["outputs"]
        can_fuse_split = True

        stride = get_stride(split_input, split_dim)
        # offset on the split dim, which is different from the real offset
        dim_offset = 0
        output_offsets = []
        # We apply padding to bmm before this fuse_split pass. However, we may
        # still have mis-aligned accesses caused by offsets. This _check_alignment
        # filters out all bad cases.
        for output in outputs:
            can_fuse_split &= len(output.dst_ops()) > 0 and all(
                _is_supported_op(next_op._attrs["op"])
                # need to pass the real offset to alignment checker
                and _check_alignment(next_op, dim_offset * stride)
                and len(output.dst_ops()) == 1
                for next_op in output.dst_ops()
            )
            for next_op in output.dst_ops():
                for idx, input in enumerate(next_op._attrs["inputs"]):
                    if input == output:
                        can_fuse_split = can_fuse_split and (
                            transform_strided_ops_utils.gemm_stride_checker(
                                next_op._attrs["input_accessors"][idx], split_dim
                            )
                        )
            output_offsets.append(dim_offset)
            dim_offset += output._size(split_dim).value()

        if not can_fuse_split:
            continue
        _LOGGER.debug("Remove split from graph")
        split_input.dst_ops().remove(split_op)

        for output, offset in zip(outputs, output_offsets):
            for next_op in output.dst_ops():
                for idx, input in enumerate(next_op._attrs["inputs"]):
                    if input == output:
                        next_op._attrs["input_accessors"][idx].update_base_tensor(
                            split_input, split_dim, offset
                        )
                        # update the graph
                        next_op._attrs["inputs"][idx] = split_input
                        break
                split_input.dst_ops().add(next_op)

        # remove split op
        for output in outputs:
            transform_utils.remove_tensor_from_sorted_graph(output)
    return transform_utils.sanitize_sorted_graph(sorted_graph)
