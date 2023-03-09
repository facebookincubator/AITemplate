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
This transformation splits a slice_scatter or slice_reshape_scatter with a large
number of inputs into multiple slice_scatter or slice_reshape_scatter ops.
"""
import copy
import logging

from typing import List

from aitemplate.compiler import ops
from aitemplate.compiler.base import Operator, Tensor

from aitemplate.compiler.ops.tensor.dynamic_slice import dynamic_slice
from aitemplate.compiler.transform import transform_utils

from aitemplate.utils import graph_utils, shape_utils


_LOGGER = logging.getLogger(__name__)

# slice_scatter and slice_reshape_scatter use the same kernel implementation
SLICE_SCATTER_INPUT_META_SIZE = 64  # bytes per input
SLICE_SCATTER_OUTPUT_META_SIZE = 16  # bytes per rank
MAX_CUDA_PARAM_BYTES = 4096  # bytes


def _slice_scatter_kernel_single_input_output_param_size(op: Operator):
    """
    Return the total size (in bytes) of the slice_scatter's params.
    We need to adjust this if we change its params.
    """
    inputs = op._attrs["inputs"]
    rank = inputs[0]._rank()
    size_of_output_meta = SLICE_SCATTER_OUTPUT_META_SIZE * rank
    # There are one more params, which takes 8 bytes.
    total_params_size = SLICE_SCATTER_INPUT_META_SIZE + size_of_output_meta + 8
    _LOGGER.debug(f'slice_scatter op {op._attrs["name"]}: {total_params_size=}')
    return total_params_size


def split_large_slice_scatter_ops(sorted_graph: List[Tensor], _: str) -> List[Tensor]:
    """
    Our slice_scatter CUDA kernel takes an input meta argument whose size
    is proportional to the number of inputs. In extreme cases, the total size
    of the kernel function params may exceed the limit imposed by the CUDA
    compiler. In such cases, we split the slice_scatter op into separate
    ones, each of which takes the original output and inputs with correct
    input_masks values.
    """
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in sorted_ops:
        # TODO: enable slice_scatter later
        if not op._attrs["op"].startswith("slice_reshape_scatter"):
            continue
        slice_scatter_op = op
        # We create InputMeta for inputs that need to copy data.
        inputs = slice_scatter_op._attrs["inputs"]
        num_inputs = len(inputs)
        if num_inputs == 0:
            continue
        params_size = _slice_scatter_kernel_single_input_output_param_size(
            slice_scatter_op
        )
        if params_size > MAX_CUDA_PARAM_BYTES:
            raise RuntimeError(
                f"cannot handle cases: {params_size=} > {MAX_CUDA_PARAM_BYTES=}"
            )
        total_params_size = params_size * num_inputs
        if total_params_size <= MAX_CUDA_PARAM_BYTES:
            continue
        num_inputs_per_split = MAX_CUDA_PARAM_BYTES // params_size
        num_splits = (num_inputs + num_inputs_per_split - 1) // num_inputs_per_split
        split_sizes = [num_inputs_per_split] * num_splits
        if num_inputs % num_inputs_per_split:
            split_sizes[num_splits - 1] = num_inputs % num_inputs_per_split

        inputs_offset = 0
        all_new_slice_scatter_ops = []
        outputs = slice_scatter_op._attrs["outputs"]
        output_accessors = slice_scatter_op._attrs["output_accessors"]
        scatter_dim = slice_scatter_op._attrs["scatter_dim"]
        has_profiler = slice_scatter_op._attrs["has_profiler"]
        local_output_offset = 0
        orig_name = slice_scatter_op._attrs["name"]
        element_func = slice_scatter_op._attrs["element_func"]
        slice_ops = slice_scatter_op._attrs["slice_ops"]
        for split_idx, new_inputs_size in enumerate(split_sizes):
            new_slice_scatter_op = ops.slice_reshape_scatter(scatter_dim, element_func)
            new_name = f"{orig_name}_split_{split_idx}"
            new_slice_scatter_op._attrs["name"] = new_name
            new_slice_scatter_op._attrs["original_name"] = new_name
            new_slice_scatter_op._attrs["has_profiler"] = has_profiler
            new_slice_scatter_op._attrs["outputs"] = outputs
            new_slice_scatter_op._attrs["output_accessors"] = copy.deepcopy(
                output_accessors
            )
            new_slice_scatter_op._set_depth()

            # import pdb; pdb.set_trace()
            new_inputs = list(inputs[inputs_offset : (inputs_offset + new_inputs_size)])
            new_slice_scatter_op._attrs["inputs"] = new_inputs
            new_slice_ops = slice_ops[inputs_offset : (inputs_offset + new_inputs_size)]
            new_slice_scatter_op._attrs["slice_ops"] = new_slice_ops

            # We also need to update the offset of the output tensor accessor.
            # Note that the strided information remains the same because the output
            # remains the same and we just shift the head offset for each new
            # slice scatter op.
            new_slice_scatter_op._attrs["output_accessors"][
                0
            ].offset += local_output_offset
            for input_tensor, slice_op in zip(new_inputs, new_slice_ops):
                input_tensor_shape = input_tensor._attrs["shape"]
                # This is enforced by slice_scatter op. Just ensure we didn't
                # violate the assumption somewhere.
                assert shape_utils.all_static_dimensions(
                    input_tensor_shape, scatter_dim
                ), (
                    f"Expected input_tensor_shape[{scatter_dim}:] are all static dimensions, "
                    f"but got: {input_tensor_shape}"
                )
                start_indices = slice_op._attrs["start_indices"]
                end_indices = slice_op._attrs["end_indices"]
                strided_dim_offset = 1
                for dim, start, end in zip(
                    input_tensor_shape[scatter_dim:],
                    start_indices[scatter_dim:],
                    end_indices[scatter_dim:],
                ):
                    n_start, n_end = dynamic_slice.normalize_start_end_indices(
                        dim.value(), start, end
                    )
                    assert n_start <= n_end, (
                        f"expected normalized {n_start=} <= {n_end=} for "
                        f"{dim=}, {start=}, {end=}"
                    )
                    strided_dim_offset *= n_end - n_start
                local_output_offset += strided_dim_offset
                input_tensor._attrs["dst_ops"].update([new_slice_scatter_op])
                input_tensor._attrs["dst_ops"].discard(slice_scatter_op)
            all_new_slice_scatter_ops.append(new_slice_scatter_op)
            inputs_offset += new_inputs_size
        output = outputs[0]
        output._attrs["src_ops"].update(all_new_slice_scatter_ops)
        output._attrs["src_ops"].remove(slice_scatter_op)
    sorted_graph = transform_utils.sanitize_sorted_graph(sorted_graph)
    return sorted_graph
