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
This pass performs the following fusion:
    t0 = tensor([1, M, N])
    x0 = expand(t0, [B, M, N])
    x1 = bmm(x0, t1) # or x1 = bmm(t1, x0)
==>
    x1 = bmm(t0, t1) # or x1 = bmm(t1, t0)

The basic idea behind the transformation is that we leverage bmm's
broadcasting capability to achieve the same functionality as expand.
"""
from typing import List

from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.ops.tensor.expand import ExpandDimensionType
from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.compiler.transform.transform_utils import (
    remove_single_tensor_op_from_sorted_graph,
    sanitize_sorted_graph,
)


def _can_fuse(expand_op: Operator, bmm_op: Operator) -> bool:
    """
    determine if expand_op and bmm_op can be fused
    """
    expand_output = expand_op._attrs["outputs"][0]
    if expand_output._attrs["is_output"]:
        return False
    expand_inputs = expand_op._attrs["inputs"]
    expand_input_shape = expand_inputs[0]._attrs["shape"]
    expand_output_shape = expand_output._attrs["shape"]
    # not valid for bmm
    if len(expand_output_shape) != 3:
        return False
    if len(expand_input_shape) == 2:
        # In this case, we are expanding the batch dim
        assert (
            expand_input_shape[0] == expand_output_shape[1]
            and expand_input_shape[1] == expand_output_shape[2]
        ), f"invalid {expand_input_shape=} and {expand_output_shape=}"
        return True
    # not valid for bmm
    if len(expand_input_shape) != 3:
        return False
    if expand_op._attrs["dim_types"][0] != ExpandDimensionType.EXPAND_DIM:
        return False
    bmm_inputs = bmm_op._attrs["inputs"]
    bmm_a = bmm_inputs[0]
    bmm_b = bmm_inputs[1]
    if expand_output is bmm_a:
        return expand_output_shape[0] == bmm_a._attrs["shape"][0]
    if expand_output is bmm_b:
        return expand_output_shape[0] == bmm_b._attrs["shape"][0]
    return False


def fuse_expand_bmm(sorted_graph: List[Tensor], workdir: str = None) -> List[Tensor]:
    """
    Transform expand + bmm into a single bmm op.

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
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if len(src_ops) != 1:
            continue
        op = list(src_ops)[0]
        if op._attrs["op"] != "expand":
            continue
        expand_op = op
        expand_output = expand_op._attrs["outputs"][0]
        dst_ops = expand_output._attrs["dst_ops"]
        if len(dst_ops) != 1:
            continue
        next_op = list(dst_ops)[0]
        if not next_op._attrs["op"].startswith("bmm_"):
            continue
        if not _can_fuse(expand_op, next_op):
            continue

        for int_var_tensor in expand_op._attrs["inputs"][1:]:
            int_var_tensor._attrs["dst_ops"].discard(expand_op)
        expand_op._attrs["inputs"] = [expand_op._attrs["inputs"][0]]
        remove_single_tensor_op_from_sorted_graph(expand_op)

        old_tensor_accessors = next_op._attrs["input_accessors"]
        assert (
            old_tensor_accessors[0].stride_dim is None
            and old_tensor_accessors[1].stride_dim is None
        ), f"next_op {next_op._attrs['name']} tensor accessors are expected to be None"
        bmm_inputs = next_op._attrs["inputs"]
        # refresh tensor accessors, which will be used by codegen
        next_op._attrs["input_accessors"] = [TensorAccessor(t) for t in bmm_inputs]

    sorted_graph = toposort(sorted_graph)
    return sanitize_sorted_graph(sorted_graph)
