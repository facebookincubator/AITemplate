# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Graph visualization tool for AITemplate
"""

import json
from typing import List

from aitemplate.compiler.base import IntImm, Operator, Tensor

from aitemplate.utils.visualization import op_attr_factory, pydot
from aitemplate.utils.visualization.web_template import (
    INDEX_TEMPLATE,
    MODAL_TEMPLATE,
    TABLE_TEMPLATE,
)


COLOR_SCHEME = {
    "default_tensor": "lightskyblue1",
    "view": "plum1",
    "output": "violetred",
    "param": "x11gray",
    "default_op": "mediumpurple1",
}


def _get_tensor_shape_str(tensor: Tensor) -> str:
    shape = []
    tensor_shape = tensor.shape()
    for s in tensor_shape:
        if isinstance(s, IntImm):
            shape.append(s.value())
            continue

        # s is IntVar
        s = s._attrs["values"]
        if len(s) == 1:
            shape.append(s[0])
        elif len(s) >= 2:
            shape.append((min(s), max(s)))
        else:
            raise RuntimeError()
    return str(shape)


def _gen_tensor_modal(tensor: Tensor) -> str:
    content = {}
    content["shape"] = _get_tensor_shape_str(tensor)
    content["is_view_of"] = (
        "None"
        if tensor._attrs["is_view_of"] is None
        else tensor._attrs["is_view_of"]._attrs["name"]
    )
    content["is_output"] = str(tensor._attrs["is_output"])
    content["is_param"] = str(tensor._attrs["is_param"])
    content["dtype"] = str(tensor._attrs["dtype"])
    table_src = TABLE_TEMPLATE.render(table_data=content)
    modal_src = MODAL_TEMPLATE.render(
        modal_id=tensor._attrs["name"] + "_modal",
        modal_label=tensor._attrs["name"] + "_label",
        modal_title=tensor._attrs["name"],
        modal_content=table_src,
    )
    return modal_src


def _gen_op_modal(op: Operator) -> str:
    content = op_attr_factory.op_to_content(op)
    table_src = TABLE_TEMPLATE.render(table_data=content)
    modal_src = MODAL_TEMPLATE.render(
        modal_id=op._attrs["name"] + "_modal",
        modal_label=op._attrs["name"] + "_label",
        modal_title=op._attrs["name"],
        modal_content=table_src,
    )
    return modal_src


def plot_graph(
    sorted_graph: List[Tensor], file_path: str, network_name: str = ""
) -> None:
    """Plot a sorted graph into an interactive HTML page.

    The sorted graph must be named.
    The HTML can be opened in Chrome directly.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        output of sorted graph / other optimization pass.
    file_path : str
        output HTML path
    network_name : str, optional
        the name of network, will appear in navigation bar, by default ""
    """
    dot_graph = pydot.Dot(graph_type="digraph")

    op_set = {}
    tensor_set = {}
    modal_set = []
    items = []
    popover_data = {}
    for tensor in sorted_graph:
        tensor_node = None
        tensor_name = tensor._attrs["name"]
        if tensor_name is None:
            raise RuntimeError(
                "Input sorted_graph must be named. Try to run name_graph pass on it."
            )
        if tensor in tensor_set:
            tensor_node = tensor_set[tensor]
        else:
            color = COLOR_SCHEME["default_tensor"]
            if tensor._attrs["is_view_of"] is not None:
                color = COLOR_SCHEME["view"]
            if tensor._attrs["is_output"] is True:
                color = COLOR_SCHEME["output"]
            if tensor._attrs["is_param"] is True:
                color = COLOR_SCHEME["param"]
            tensor_node = pydot.Node(
                name=tensor_name,
                shape="note",
                id=tensor_name,
                color=color,
            )
            tensor_set[tensor] = tensor_node
            dot_graph.add_node(tensor_node)
            modal_set.append(_gen_tensor_modal(tensor))
            items.append(tensor_name)
            popover_data[tensor_name] = "shape: " + _get_tensor_shape_str(tensor)

        for src_op in tensor.src_ops():
            op_node = None
            op_name = src_op._attrs["name"]
            if op_name is None:
                raise RuntimeError(
                    "Input sorted_graph must be named. Try to run name_graph pass on it."
                )
            if src_op in op_set:
                op_node = op_set[src_op]
            else:
                op_node = pydot.Node(
                    name=op_name,
                    shape="folder",
                    id=op_name,
                    color="mediumpurple1",
                )
                op_set[src_op] = op_node
                dot_graph.add_node(op_node)
                modal_set.append(_gen_op_modal(src_op))
                items.append(op_name)
                popover_data[op_name] = "op: " + src_op._attrs["op"]
            dot_graph.add_edge(pydot.Edge(op_node, tensor_node))

        for dst_op in tensor.dst_ops():
            op_node = None
            op_name = dst_op._attrs["name"]
            if op_name is None:
                raise RuntimeError(
                    "Input sorted_graph must be named. Try to run name_graph pass on it."
                )
            if dst_op in op_set:
                op_node = op_set[dst_op]
            else:
                op_node = pydot.Node(
                    name=op_name,
                    shape="folder",
                    id=op_name,
                    color="mediumpurple1",
                )
                op_set[dst_op] = op_node
                dot_graph.add_node(op_node)
                items.append(op_name)
                popover_data[op_name] = "op: " + dst_op._attrs["op"]
                # add modal
                modal_set.append(_gen_op_modal(dst_op))
            dot_graph.add_edge(pydot.Edge(tensor_node, op_node))
    dot_src = dot_graph.to_string()
    modal_src = "\n".join(modal_set)
    items_src = [f'"{item}"' for item in items]
    popover_src = json.dumps(popover_data)
    index = INDEX_TEMPLATE.render(
        dot_src=dot_src,
        modals=modal_src,
        network_name=network_name,
        items=items_src,
        popover_data=popover_src,
    )

    with open(file_path, "w") as fo:
        fo.write(index)
