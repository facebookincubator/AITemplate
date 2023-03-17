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
Fuse GEMM + reshape + permute0213
"""
from typing import List, Sequence

from aitemplate.compiler.base import IntImm, Operator, Tensor
from aitemplate.compiler.ops import gemm_rcr_permute
from aitemplate.compiler.transform import transform_utils
from aitemplate.compiler.transform.toposort import toposort

from aitemplate.utils import graph_utils


def _check_reshape(op: Operator) -> bool:
    """check reshape [M, N] -> [M/D1, D1, D2, N/D2]
    D1 and D2 must be static. Also checks alignment here.

    Args:
        op (Operator): reshape op

    Returns:
        bool: True if can fuse
    """
    input_shapes = op._attrs["inputs"][0].shape()
    output_shapes = op._attrs["outputs"][0].shape()

    if len(input_shapes) != 2 or len(output_shapes) != 4:
        return False

    m, n = input_shapes
    m_d1, d1, d2, n_d2 = output_shapes

    if not isinstance(n, IntImm) or not isinstance(n_d2, IntImm):
        return False

    if not isinstance(d1, IntImm) or not isinstance(d2, IntImm):
        return False

    d1 = d1.value()
    d2 = d2.value()

    if len(m._attrs["values"]) != len(m_d1._attrs["values"]):
        return False

    if n.value() != n_d2.value() * d2:
        return False

    # check alignment
    if n_d2.value() % 2 == 1:
        return False

    return True


def _check_permute(op: Operator, dims: Sequence[int]) -> bool:
    """Check permute dims match input dims

    Args:
        op (Operator): permute op
        dims (Sequence): permute dims

    Returns:
        bool: True if match
    """
    permute_dims = op._attrs["dims"]
    if len(dims) != len(permute_dims):
        return False
    for d0, d1 in zip(dims, permute_dims):
        if d0 != d1:
            return False
    return True


def _fuse_gemm_reshape_permute0213(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """Fuse GEMM + reshape + permute0213
    Fuse patterns like this together:

    y0 = gemm_rcr(a, b) # [M, N]
    y1 = reshape(y0, [M/D1, D1, D2, N/D2])
    y2 = permute(y1, [0, 2, 1, 3])

    into
    y2 = gemm_rcr_permute(a, b, shape=[D1, D2], layout="0213")

    fusion condition:
    N/D2 must meet alignment condition: align > 1 for fp16
    Otherwise, it causes perf regression to gemm.
    Must run before any pass that modifies Tensor Accessor or fuses reshape

    Args:
        sorted_graph (List[Tensor]): input graph
        workdir (str, optional): current workdir for dumping debug info. Defaults to None.

    Returns:
        List[Tensor]: optimized graph
    """

    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)

    for op in sorted_ops:
        if op._attrs["op"] != "gemm_rcr":
            continue

        outputs = op._attrs["outputs"]
        assert len(outputs) == 1

        gemm_output = outputs[0]
        if len(gemm_output.dst_ops()) != 1:
            continue

        reshape_op = list(gemm_output.dst_ops())[0]

        if reshape_op._attrs["op"] != "reshape":
            continue

        reshape_output = reshape_op._attrs["outputs"][0]
        if len(reshape_output.dst_ops()) != 1:
            continue

        permute_op = list(reshape_output.dst_ops())[0]

        if permute_op._attrs["op"] not in ("permute", "permute0213"):
            continue

        permute_output = permute_op._attrs["outputs"][0]

        # check reshape [M, N] -> [M/D1, D1, D2, N/D2]
        if not _check_reshape(reshape_op):
            continue

        # check permute dims match [0, 2, 1, 3]: either
        # permute0213 or generic permute with those dims
        if permute_op._attrs["op"] != "permute0213" and not _check_permute(
            permute_op, [0, 2, 1, 3]
        ):
            continue

        # fuse ops together
        _, d1, d2, _ = reshape_output.shape()
        d1_v = d1.value()
        d2_v = d2.value()
        gemm_permute_op = gemm_rcr_permute(shape=(d1_v, d2_v), layout="0213")
        a, b = op._attrs["inputs"]
        transform_utils.remove_dst_op_from_tensor(a, op)
        transform_utils.remove_dst_op_from_tensor(b, op)

        new_output = gemm_permute_op(a, b)

        transform_utils.replace_tensor(permute_output, new_output)
        sorted_graph.append(new_output)

        transform_utils.remove_tensor_from_sorted_graph(gemm_output)
        transform_utils.remove_tensor_from_sorted_graph(reshape_output)

    sorted_graph = toposort(sorted_graph)
    transform_utils.sanitize_sorted_graph(sorted_graph)
    return sorted_graph


def fuse_mm_reshape_permute(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """Fuse GEMM/BMM + reshape + permute into a single op

    Args:
        sorted_graph (List[Tensor]): input graph
        workdir (str, optional): current workdir for dumping debug info. Defaults to None.

    Returns:
        List[Tensor]: optimized graph
    """

    funcs = [
        _fuse_gemm_reshape_permute0213,
    ]
    for func in funcs:
        sorted_graph = func(sorted_graph)
    return sorted_graph
