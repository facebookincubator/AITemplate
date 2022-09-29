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
import os
from typing import Any, List

from aitemplate.utils import logger


def get_sorted_ops(tensors) -> List[Any]:
    from aitemplate.compiler.base import Tensor

    visited = set()
    sorted_ops = []
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    for tensor in tensors:
        for src_op in tensor._attrs["src_ops"]:
            if src_op in visited:
                continue
            visited.add(src_op)
            sorted_ops.append(src_op)
    return sorted_ops


def sorted_graph_debug_str(tensors) -> str:
    from aitemplate.compiler.base import Tensor

    if isinstance(tensors, Tensor):
        tensors = [tensors]
    tensor_str = "\n\n".join([str(tensor) for tensor in tensors])
    op_str = "\n\n".join([str(op) for op in get_sorted_ops(tensors)])
    return "Tensors: {}\n\nOperators: {}\n\n".format(tensor_str, op_str)


def sorted_graph_pseudo_code(tensors, with_shape=True) -> str:
    from aitemplate.compiler.base import Tensor

    if isinstance(tensors, Tensor):
        tensors = [tensors]
    op_str = "\n".join([op.pseudo_code(with_shape) for op in get_sorted_ops(tensors)])
    return op_str


def sorted_op_pseudo_code(ops, with_shape=True) -> str:
    from aitemplate.compiler.base import Operator

    if isinstance(ops, Operator):
        ops = [ops]
    op_str = "\n".join([op.pseudo_code(with_shape) for op in ops])
    return op_str


def dump_graph_debug_str_to_file(tensors, workdir, name):
    prefix = os.path.join(workdir, name)
    graph_path = prefix + "_graph.txt"
    pseudo_code_path = prefix + "_pseudo_code.txt"
    with open(graph_path, "w") as f:
        f.write(sorted_graph_debug_str(tensors))
        logger.info(__file__, f"Dumped {name} graph to {graph_path}")
    with open(pseudo_code_path, "w") as f:
        f.write(sorted_graph_pseudo_code(tensors))
        logger.info(__file__, f"Dumped {name} pseudo code to {pseudo_code_path}")
