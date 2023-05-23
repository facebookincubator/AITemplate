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
Remove id ops from a sorted_graph.
"""
from typing import List

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.transform import transform_utils


def remove_id_ops(sorted_graph: List[Tensor], workdir: str = None) -> List[Tensor]:
    """Remove id ops from the input sorted_graph."""
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op._attrs["op"] != "identity":
            continue
        id_op = src_op
        input_tensor = id_op._attrs["inputs"][0]
        # skip a very special case where id takes an input and produces an output
        if tensor._attrs["is_output"] and input_tensor._attrs["is_input"]:
            continue
        transform_utils.remove_single_tensor_op_from_sorted_graph(id_op)

    sorted_graph = transform_utils.sanitize_sorted_graph(sorted_graph)
    return transform_utils.sanitize_sorted_graph(sorted_graph)
