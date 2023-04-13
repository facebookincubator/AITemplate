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
This transformation splits a split with a large number of outputs into multiple
splitt ops, which share the same input with correct output_masks.
"""
import logging

from typing import List

from aitemplate.compiler import ops
from aitemplate.compiler.base import Operator, Tensor

from aitemplate.compiler.transform import toposort, transform_utils

from aitemplate.utils import graph_utils


_LOGGER = logging.getLogger(__name__)

SPLIT_INPUT_META_SIZE = 16
SPLIT_OUTPUT_META_SIZE = 32
MAX_CUDA_PARAM_BYTES = 4096


def _split_kernel_single_input_output_param_size(op: Operator):
    """
    Return the total size (in bytes) of the split's params.
    We need to adjust this if we change the split op's params.
    Note this is conservative by multiplying input_meta and constant 24 bytes.
    """
    outputs = op._attrs["outputs"]
    rank = outputs[0]._rank()
    size_of_input_meta = SPLIT_INPUT_META_SIZE * rank
    # There are 3 more params, where each takes 8 bytes, so we add 24 more bytes
    total_params_size = SPLIT_OUTPUT_META_SIZE + size_of_input_meta + 24
    _LOGGER.debug(f'split op op._attrs["name"]: {total_params_size=}')
    return total_params_size


def split_large_split_ops(sorted_graph: List[Tensor], _: str) -> List[Tensor]:
    """
    Our split CUDA kernel takes an output meta argument whose size
    is proportional to the number of outputs. In extreme cases, the total size
    of the params of a split kernel may exceed the limit imposed by the CUDA
    compiler. In such cases, we split the split op into separate ones.
    """
    modified = False
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in sorted_ops:
        if not op._attrs["op"].startswith("split"):
            continue
        split_op = op

        split_params_size = _split_kernel_single_input_output_param_size(split_op)
        if split_params_size > MAX_CUDA_PARAM_BYTES:
            raise RuntimeError(
                f"cannot handle cases: {split_params_size=} > {MAX_CUDA_PARAM_BYTES=}"
            )
        if split_params_size * len(split_op._attrs["outputs"]) <= MAX_CUDA_PARAM_BYTES:
            continue

        modified = True
        split_dim = split_op._attrs["split_dim"]
        split_sizes = split_op._attrs["split_sizes"]
        outputs = split_op._attrs["outputs"]
        num_outputs_per_split = MAX_CUDA_PARAM_BYTES // split_params_size
        # compute how many split ops we need to fix within MAX_CUDA_PARAM_BYTES
        num_split_ops = (
            len(outputs) + num_outputs_per_split - 1
        ) // num_outputs_per_split

        output_mapping = []
        for split_i in range(num_split_ops):
            start = split_i * num_outputs_per_split
            end = min(
                (split_i + 1) * num_outputs_per_split, len(split_op._attrs["outputs"])
            )

            remove_indices = list(range(start)) + list(
                range(end, len(split_op._attrs["outputs"]))
            )
            new_split = ops.split()
            new_outputs = new_split(
                split_op._attrs["inputs"][0], split_sizes, split_dim
            )
            new_split.remove_output_at(remove_indices)
            new_outputs = new_split._attrs["outputs"]
            sorted_graph += list(new_outputs)
            output_mapping += list(zip(outputs[start:end], new_outputs))

        for old_output, new_output in output_mapping:
            transform_utils.replace_tensor(old_output, new_output)

    if not modified:
        return sorted_graph

    new_output_tensors = [
        tensor for tensor in sorted_graph if tensor._attrs["is_output"]
    ]
    sorted_graph = toposort.toposort(new_output_tensors)
    return sorted_graph
