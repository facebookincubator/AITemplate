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
Perform fusions for bmm + permute021 operators:
    bmm_xxc + permute021 -> bmm_xxr
    bmm_xxr + permute021 -> bmm_xxc
"""
from typing import List

from aitemplate.compiler.base import Tensor

from aitemplate.compiler.ops.gemm_universal import (
    bmm_ccc,
    bmm_ccr,
    bmm_crc,
    bmm_crr,
    bmm_rcc,
    bmm_rcr,
    bmm_rrc,
    bmm_rrr,
)

from aitemplate.compiler.ops.tensor import permute021

from aitemplate.compiler.transform.fuse_utils import transform_simple_fusion_patterns


def fuse_bmm_permute(sorted_graph: List[Tensor], _: str) -> List[Tensor]:
    """
    Fuse bmm + permute021 ops. The second argument is unused, it's only
    here to make the type of this function the same as the others called in optimize_graph.
    """
    ops_r = [
        bmm_ccr,
        bmm_crr,
        bmm_rcr,
        bmm_rrr,
    ]

    ops_c = [
        bmm_ccc,
        bmm_crc,
        bmm_rcc,
        bmm_rrc,
    ]
    patterns_cr = [((c_op(), permute021()), r_op) for c_op, r_op in zip(ops_c, ops_r)]
    patterns_rc = [((r_op(), permute021()), c_op) for c_op, r_op in zip(ops_c, ops_r)]

    sorted_graph = transform_simple_fusion_patterns(
        sorted_graph, patterns_cr + patterns_rc
    )

    return sorted_graph
