# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import os
from typing import List, Union

from aitemplate.compiler.base import Operator, Tensor
from aitemplate.utils import logger


def get_sorted_ops(tensors: Union[Tensor, List[Tensor]]) -> List[Operator]:
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


def sorted_graph_debug_str(tensors: Union[Tensor, List[Tensor]]) -> str:
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    tensor_str = "\n\n".join([str(tensor) for tensor in tensors])
    op_str = "\n\n".join([str(op) for op in get_sorted_ops(tensors)])
    return "Tensors: {}\n\nOperators: {}\n\n".format(tensor_str, op_str)


def sorted_graph_pseudo_code(
    tensors: Union[Tensor, List[Tensor]], with_shape=True
) -> str:
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    op_str = "\n".join([op.pseudo_code(with_shape) for op in get_sorted_ops(tensors)])
    return op_str


def dump_graph_debug_str_to_file(tensors: Union[Tensor, List[Tensor]], workdir, name):
    prefix = os.path.join(workdir, name)
    graph_path = prefix + "_graph.txt"
    pseudo_code_path = prefix + "_pseudo_code.txt"
    with open(graph_path, "w") as f:
        f.write(sorted_graph_debug_str(tensors))
        logger.info(__file__, f"Dumped {name} graph to {graph_path}")
    with open(pseudo_code_path, "w") as f:
        f.write(sorted_graph_pseudo_code(tensors))
        logger.info(__file__, f"Dumped {name} pseudo code to {pseudo_code_path}")
