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
This transformation splits a concat with a large number of inputs into multiple
concat ops, which share the same inputs with correct input_masks and the same
output.
"""
import copy
import logging

from typing import List

from aitemplate.compiler import ops
from aitemplate.compiler.base import Operator, Tensor

from aitemplate.compiler.stable_set import StableSet
from aitemplate.compiler.transform import transform_utils

from aitemplate.utils import graph_utils


_LOGGER = logging.getLogger(__name__)

CONCAT_INPUT_META_SIZE = 64
CONCAT_OUTPUT_META_SIZE = 16
MAX_CUDA_PARAM_BYTES = 4096


def _concat_kernel_single_input_output_param_size(op: Operator):
    """
    Return the total size (in bytes) of the concat's params.
    We need to adjust this if we change the concatenate op's params.
    """
    inputs = op._attrs["inputs"]
    rank = inputs[0]._rank()
    size_of_one_output_meta = CONCAT_OUTPUT_META_SIZE * rank
    # There are 3 more params, where each takes 8 bytes, so we add 24 more bytes
    total_params_size = CONCAT_INPUT_META_SIZE + size_of_one_output_meta + 24
    _LOGGER.debug(f'concat op {op._attrs["name"]}: {total_params_size=}')
    return total_params_size


def split_large_concat_ops(sorted_graph: List[Tensor], _: str) -> List[Tensor]:
    """
    Our concatenate CUDA kernel takes an input meta argument whose size
    is proportional to the number of inputs. In extreme cases, the total size
    of the params of a concatenate kernel may exceed the limit imposed by
    the CUDA compiler. In such cases, we split the concatenate op into separate
    ones, each of which takes the original output and inputs with correct
    input_masks values.
    """
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in sorted_ops:
        if not op._attrs["op"].startswith("concatenate"):
            continue
        concat_op = op
        # We create InputMeta for inputs that need to copy data.
        num_inputs = len([m for m in concat_op._attrs["input_masks"] if m is True])
        concat_inputs = concat_op._attrs["inputs"]
        assert num_inputs == len(
            concat_inputs
        ), f"expected {num_inputs=} and {len(concat_inputs)=} to be equal"
        if num_inputs == 0:
            continue
        concat_params_size = _concat_kernel_single_input_output_param_size(concat_op)
        if concat_params_size > MAX_CUDA_PARAM_BYTES:
            raise RuntimeError(
                f"cannot handle cases: {concat_params_size=} > {MAX_CUDA_PARAM_BYTES=}"
            )
        total_params_size = concat_params_size * num_inputs
        if total_params_size <= MAX_CUDA_PARAM_BYTES:
            continue
        num_inputs_per_split = MAX_CUDA_PARAM_BYTES // concat_params_size
        num_splits = (num_inputs + num_inputs_per_split - 1) // num_inputs_per_split
        split_sizes = [num_inputs_per_split] * num_splits
        if num_inputs % num_inputs_per_split:
            split_sizes[num_splits - 1] = num_inputs % num_inputs_per_split

        offset = 0
        all_new_concat_ops = []
        concat_outputs = concat_op._attrs["outputs"]
        input_accessors = concat_op._attrs["input_accessors"]
        for new_inputs_size in split_sizes:
            new_concat_op = ops.concatenate()
            new_concat_op._attrs["inputs"] = list(concat_inputs)
            new_concat_op._attrs["concat_dim"] = concat_op._attrs["concat_dim"]
            new_concat_op._attrs["outputs"] = concat_outputs.copy()
            new_concat_op._attrs["original_inputs"] = concat_op._attrs[
                "original_inputs"
            ].copy()
            new_concat_op._attrs["input_masks"] = concat_op._attrs["input_masks"].copy()
            new_concat_op._attrs["input_accessors"] = copy.deepcopy(input_accessors)
            new_concat_op._set_depth()

            indices_to_remove = list(range(offset)) + list(
                range(offset + new_inputs_size, num_inputs)
            )
            new_concat_op.remove_input_at(indices_to_remove)
            all_new_concat_ops.append(new_concat_op)
            offset += new_inputs_size
        # original inputs are distributed among new concats, so we need to adjust
        # their dst_ops
        for inp in concat_inputs:
            new_dst_ops = StableSet()
            for inp_dst_op in inp.dst_ops():
                if inp in inp_dst_op._attrs["inputs"]:
                    new_dst_ops.add(inp_dst_op)
            inp._attrs["dst_ops"] = new_dst_ops
        concat_output = concat_op._attrs["outputs"][0]
        concat_output._attrs["src_ops"].update(all_new_concat_ops)
        concat_output._attrs["src_ops"].remove(concat_op)
    sorted_graph = transform_utils.sanitize_sorted_graph(sorted_graph)
    return sorted_graph
