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
Applies paddings to gemms based on alignment requirements.
"""
import logging
from typing import Callable, Dict, List

from aitemplate.compiler import ops

from aitemplate.compiler.base import _create_host_zero_tensor, IntImm, Operator, Tensor
from aitemplate.compiler.ops.gemm_universal.gemm_common import DimInfo, gemm, Source
from aitemplate.compiler.transform import transform_utils

from aitemplate.utils import alignment


_LOGGER = logging.getLogger(__name__)


def _extract_mnk_name(
    dim_info_dict: Dict[str, DimInfo], source: Source, tensor_idx: int, dim_idx: int
) -> str:
    for name, info_list in dim_info_dict.items():
        for info in info_list:
            if info == DimInfo(source, tensor_idx, [dim_idx]):
                return name
    return None


def get_padding_length(original_length: int, dtype: str) -> int:
    if alignment.valid_alignment(original_length, dtype):
        return 0

    # TODO(yingz): Tune padding strategy.
    if original_length < 16:
        return 1
    return int((original_length // 8 + 1) * 8) - original_length


def _pad_input_tensor(
    op: Operator,
    tensor_idx: int,
    f_extract_var_name: Callable[[int, int], str],
    alignment_var_to_padding_length: Dict[str, int],
    tensor_list: List[Tensor],
) -> None:
    original_shape = op._attrs["inputs"][tensor_idx]._attrs["shape"]
    for dim_idx, dim in enumerate(original_shape):
        tensor = op._attrs["inputs"][tensor_idx]
        original_tensor_debug_str = str(tensor)
        previous_shape = tensor._attrs["shape"]
        padding_shape = list(previous_shape)
        new_shape = list(previous_shape)

        var_name = f_extract_var_name(tensor_idx, dim_idx)
        if var_name is None or var_name not in alignment_var_to_padding_length:
            # This dim doesn't require alignment padding. Skipping.
            continue

        padding_length = alignment_var_to_padding_length.get(var_name)
        padding_shape[dim_idx] = IntImm(padding_length)
        new_shape[dim_idx] = IntImm(dim.value() + padding_length)
        tensor._attrs["dst_ops"].remove(op)

        padding_tensor = _create_host_zero_tensor(
            shape=padding_shape, dtype=tensor.dtype()
        )
        padded_tensor = ops.concatenate()(
            [tensor, padding_tensor],
            dim=dim_idx,
        )
        op._attrs["inputs"][tensor_idx] = padded_tensor
        padded_tensor._attrs["dst_ops"].add(op)
        tensor_list.append(padding_tensor)
        tensor_list.append(padded_tensor)

        _LOGGER.debug(
            "**** Apply padding ****, replace input tensor \n {} \n with \n {} \n".format(
                original_tensor_debug_str, padded_tensor
            ),
        )

    return


def _slice_output_tensor(
    new_output: Tensor, original_output: Tensor, tensor_list: List[Tensor]
) -> Tensor:
    new_shape = new_output._attrs["shape"]
    original_shape = original_output._attrs["shape"]
    if new_shape == original_shape:
        return new_output

    start_indicies = [0] * len(new_shape)
    end_indicies = [None] * len(new_shape)
    for i, (new_dim, old_dim) in enumerate(zip(new_shape, original_shape)):
        if new_dim != old_dim:
            assert isinstance(new_dim, IntImm) and isinstance(
                old_dim, IntImm
            ), f"new_shape: {new_shape}, old_shape: {original_shape}"
            assert (
                new_dim.value() > old_dim.value()
            ), f"new_shape: {new_shape}, old_shape: {original_shape}"
            end_indicies[i] = old_dim.value()
    sliced_tensor = ops.dynamic_slice()(new_output, start_indicies, end_indicies)
    tensor_list.append(sliced_tensor)
    sliced_tensor._attrs["is_output"] = new_output._attrs["is_output"]
    sliced_tensor._attrs["name"] = new_output._attrs["name"]
    new_output._attrs["name"] = None
    new_output._attrs["is_output"] = False
    return sliced_tensor


def apply_padding(sorted_graph: List[Tensor], workdir: str = None) -> List[Tensor]:
    """
    Applies padding to gemms to use SM80 kernels.
    SM80 kernels require min_alignment == 2.
    """

    visited_ops = set()
    new_sorted_graph = []
    for tensor in sorted_graph:
        new_tensor_list = [tensor]
        src_ops = tensor.src_ops()
        for op in src_ops:
            if op in visited_ops:
                continue

            # Exclude special gemm kernels.
            if (
                not isinstance(op, gemm)
                or isinstance(op, ops.gemm_rrr_small_nk)
                or isinstance(op, ops.bmm_rcr_n1)
                or isinstance(op, ops.bmm_rrr_k1_tanh)
                or "permute" in op._attrs["op"]
            ):
                continue

            # This pass only works for gemm or bmm. group_gemm is not supported.
            # We don't need to padd our special kernel bmm_rcr_n1, which does
            # not have any alignment constraint.
            op_name = op._attrs["name"]
            if op_name.startswith(("group_gemm", "bmm_rcr_n1")) or "softmax" in op_name:
                continue

            # Extract alignment var names and padding lengths.
            alignment_var_to_padding_length = {}
            dim_info_dict = op._extract_dims()
            for i, tensor in enumerate(op._attrs["inputs"]):
                alignment_var = _extract_mnk_name(
                    dim_info_dict, Source.INPUT, i, len(tensor._attrs["shape"]) - 1
                )
                if alignment_var is None:
                    # No alignment var is extracted. Skip padding.
                    continue
                alignment_dim = tensor._attrs["shape"][-1]
                if not isinstance(alignment_dim, IntImm):
                    raise NotImplementedError(
                        "Gemm does not support dynamic alignment dimensions "
                        "(i.e. alignment==1)! Gemm: {}".format(op)
                    )
                padding_length = get_padding_length(
                    alignment_dim.value(), tensor.dtype()
                )
                if padding_length > 0:
                    alignment_var_to_padding_length[alignment_var] = padding_length
            if len(alignment_var_to_padding_length) == 0:
                # No padding is necessary.
                continue

            _LOGGER.debug(
                "**** Apply padding ****, alignment_var_to_padding_length: \n {} \n".format(
                    alignment_var_to_padding_length
                ),
            )
            original_op_debug_str = str(op)

            # Pad A and B.
            for tensor_idx, _ in enumerate(op._attrs["inputs"][:2]):
                _pad_input_tensor(
                    op,
                    tensor_idx,
                    lambda tensor_idx, dim_idx: _extract_mnk_name(
                        dim_info_dict, Source.INPUT, tensor_idx, dim_idx
                    ),
                    alignment_var_to_padding_length,
                    new_tensor_list,
                )

            # Pad bias and extra sources if necessary.
            for tensor_idx, tensor in enumerate(op._attrs["inputs"][2:]):
                _pad_input_tensor(
                    op,
                    tensor_idx + 2,  # skip A and B
                    lambda _, dim_idx: _extract_mnk_name(
                        dim_info_dict,
                        Source.OUTPUT,  # bias alignment follows output alignment
                        0,  # always check output[0]
                        dim_idx
                        + len(op._attrs["outputs"][0]._attrs["shape"])
                        - len(tensor._attrs["shape"]),  # handle bias broadcast case
                    ),
                    alignment_var_to_padding_length,
                    new_tensor_list,
                )

            # Replaces the old op with the new op.
            for tensor_input in op._attrs["inputs"]:
                tensor_input._attrs["dst_ops"].discard(op)
            new_op = type(op)(**op._get_op_attributes())
            new_op._attrs["split_k"] = op._attrs["split_k"]
            if "alpha" in op._attrs:
                new_op._attrs["alpha"] = op._attrs["alpha"]
            new_output = new_op(*op._attrs["inputs"])
            new_tensor_list.append(new_output)
            original_output = op._attrs["outputs"][0]
            transform_utils.copy_tensor_attributes(new_output, original_output)

            # Slice output if necessary.
            new_output = _slice_output_tensor(
                new_output, original_output, new_tensor_list
            )
            transform_utils.replace_tensor(original_output, new_output)
            transform_utils.remove_tensor_from_sorted_graph(original_output)

            _LOGGER.debug(
                "**** Apply padding ****, replace op \n {} \n with \n {} \n".format(
                    original_op_debug_str, new_op
                ),
            )

        new_sorted_graph.extend(new_tensor_list)

    new_sorted_graph = transform_utils.sanitize_sorted_graph(new_sorted_graph)

    return new_sorted_graph
