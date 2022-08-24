# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] Perform transformations on ops which support strided inputs / outputs.
"""
from typing import List

from ...utils import graph_utils, logger
from ..base import IntImm, Operator, Tensor
from . import transform_utils

# pylint: disable=W0612


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
        split_dim = split_op._attrs["split_dim"]
        # FIXME: only support dim == 1 at the moment
        if split_dim != 1:
            continue
        if not transform_utils.cat_split_dim_is_static(split_op, split_dim):
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
        split_input._attrs["dst_ops"] = [group_gemm_op]
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _is_supported_op(op_type: str):
    return op_type in {"bmm_rcr_n1", "bmm_rcr"}


def _check_alignment(op: Operator, offset: int):
    # TODO: adjust alignment requirement based on dtype. 2-elem-alignment is
    # only required by fp16, because async.copy needs at least 32 bits.
    # For fp32 dtype values, 1-elem-alignment is valid.
    if op._attrs["op"] == "bmm_rcr_n1":
        return True
    if op._attrs["op"] == "bmm_rcr":
        if offset % 2 != 0:
            return False
        A = op._attrs["inputs"][0]
        k_dim = A._attrs["shape"][2]
        # skip dynamic dim
        if not isinstance(k_dim, IntImm):
            return False
        k_dim_val = k_dim._attrs["values"][0]
        # We cannot have mis-aligned K
        if k_dim_val % 2 == 0:
            return True
        else:
            raise RuntimeError(f"Unexpected misaligned K dimension: {k_dim_val}")
    raise RuntimeError(f'Unexpected op type: {op._attrs["op"]}')


def _fuse_split_and_strided_op(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Fuse split and any op that supports strided inputs. This pass requires
    that all of the outputs can be fused into the next ops so that split op
    can be eliminated. Partial fusion is not supported yet.
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
        if not transform_utils.cat_split_dim_is_static(split_op, split_dim):
            continue

        outputs = split_op._attrs["outputs"]
        can_fuse_split = True
        offset = 0
        output_offsets = []
        # We apply padding to bmm before this fuse_split pass. However, we may
        # still have mis-aligned accesses caused by offsets. This _check_alignment
        # filters out all bad cases.
        for output in outputs:
            can_fuse_split &= len(output.dst_ops()) > 0 and all(
                _is_supported_op(next_op._attrs["op"])
                and _check_alignment(next_op, offset)
                and len(output.dst_ops()) == 1
                for next_op in output.dst_ops()
            )
            output_offsets.append(offset)
            offset += output._size(split_dim).value()
        if not can_fuse_split:
            continue
        logger.debug(__file__, "Remove split from graph")
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
