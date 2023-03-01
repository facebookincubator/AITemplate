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
Graph visualization tool for AITemplate
"""
import json
import os

from aitemplate import compiler
from aitemplate.utils.visualization import op_attr_factory, pydot
from aitemplate.utils.visualization.op_attr_factory import op_to_content
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


def _get_tensor_shape_str(tensor) -> str:
    from aitemplate.compiler.base import IntImm

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


def _gen_tensor_modal(tensor) -> str:
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
        modal_id=f'{tensor._attrs["name"]}_modal',
        modal_label=f'{tensor._attrs["name"]}_label',
        modal_title=tensor._attrs["name"],
        modal_content=table_src,
    )
    return modal_src


def _gen_op_modal(op) -> str:
    content = op_attr_factory.op_to_content(op)
    table_src = TABLE_TEMPLATE.render(table_data=content)
    modal_src = MODAL_TEMPLATE.render(
        modal_id=f'{op._attrs["name"]}_modal',
        modal_label=f'{op._attrs["name"]}_label',
        modal_title=op._attrs["name"],
        modal_content=table_src,
    )
    return modal_src


def plot_graph(tensors, file_path: str) -> None:
    """
    Plot AIT graph.

    Parameters
    ----------
    tensors : Union[Tensor, List[Tensor]]
        An output Tensor, or a list of output Tensors of AIT graph.
    file_path : str
        Output file path, currently we support the following extension:
            - html
            - format supported by graphviz
    """
    dot_graph = pydot.Dot(graph_type="digraph")
    _, ext = os.path.splitext(file_path)
    if ext == "":
        raise ValueError("Please provide a file extension in path to plot.")

    ext = ext[1:]
    if ext != "html" and ext not in dot_graph.formats:
        raise ValueError(f"Unsupported extension '{ext}' to plot!")

    sorted_graph = compiler.transform.toposort(tensors)
    compiler.transform.name_graph(sorted_graph)

    op_set = {}
    tensor_set = {}
    modal_set = []
    items = []
    popover_data = {}
    for tensor in sorted_graph:
        tensor_node = None
        tensor_name = tensor._attrs["name"]
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
                popover_data[op_name] = ", ".join(
                    [f"{x}: {y}" for x, y in op_to_content(src_op).items()]
                )
            dot_graph.add_edge(pydot.Edge(op_node, tensor_node))

        for dst_op in tensor.dst_ops():
            op_node = None
            op_name = dst_op._attrs["name"]
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
                popover_data[op_name] = ", ".join(
                    [f"{x}: {y}" for x, y in op_to_content(dst_op).items()]
                )
                # add modal
                modal_set.append(_gen_op_modal(dst_op))
            dot_graph.add_edge(pydot.Edge(tensor_node, op_node))

    file_dir = os.path.dirname(file_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    if ext == "html":
        basename = os.path.splitext(os.path.basename(file_path))[0]
        dot_src = dot_graph.to_string()
        modal_src = "\n".join(modal_set)
        items_src = [f'"{item}"' for item in items]
        popover_src = json.dumps(popover_data)
        index = INDEX_TEMPLATE.render(
            dot_src=dot_src,
            modals=modal_src,
            network_name=basename,
            items=items_src,
            popover_data=popover_src,
        )

        with open(file_path, "w") as fo:
            fo.write(index)
    else:
        dot_graph.write(file_path, format=ext)
