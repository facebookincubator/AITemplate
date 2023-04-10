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

from typing import Optional

from aitemplate import compiler
from aitemplate.utils.environ import shorten_tensor_names_for_plots
from aitemplate.utils.misc import short_str
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
    content["is_input"] = str(tensor._attrs["is_input"])
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


def _highlight_op_node(op_node, op, time_stats):
    if op in time_stats.op_durations:
        perf_op = time_stats.op_durations[op]
        scale_factor = float(perf_op) / float(time_stats.total_duration)

        if perf_op > time_stats.duration_p95:
            op_node.set("color", "maroon1")
            op_node.set("penwidth", 9)
            op_node.set("width", 1 + scale_factor * 100)
            op_node.set("height", 1 + scale_factor * 50)
        elif perf_op > time_stats.duration_p90:
            op_node.set("color", "magenta1")
            op_node.set("penwidth", 6)
            op_node.set("width", 1 + scale_factor * 100)
            op_node.set("height", 1 + scale_factor * 50)
        elif perf_op > time_stats.duration_p70:
            op_node.set("color", "mediumorchid1")
            op_node.set("penwidth", 3)
            op_node.set("width", 1 + scale_factor * 100)
            op_node.set("height", 1 + scale_factor * 50)


def plot_graph(
    tensors, file_path: str, file_with_time_profiles: Optional[str] = None
) -> None:
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
    file_with_time_profile : Optional[str]
        Adds time for every node, if provided

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

    # Before doing the further processing, it is needed
    # to find whether there is an Operator instance with the same
    # name like 'fused_elementwise_123' that is used
    # several times, but with different input and/or outputs.
    # In such a case, every Operator instance should get its unique
    # name.
    #
    # The following dict will be used to store such unique names,
    # such as 'fused_elementwise_123 0' and 'fused_elementwise_123 1'.
    from aitemplate.utils.json_utils import gen_unique_op_names

    op_names = gen_unique_op_names(sorted_graph)

    from aitemplate.utils.graph_utils import ProfiledTimeStatistics, track_graph_timings

    time_stats = ProfiledTimeStatistics()
    if file_with_time_profiles is not None:
        time_stats = track_graph_timings(sorted_graph, file_with_time_profiles)

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

            label = tensor_name

            if shorten_tensor_names_for_plots():
                if tensor_name is not None and len(tensor_name) > 30:
                    label = short_str(tensor_name)

            # add a label with time
            label_with_time = ""
            seq_tracker = time_stats.tensor_sequential_trackers.get(tensor, None)
            if seq_tracker is not None and seq_tracker.execution_end != 0:
                label_with_time += f"{seq_tracker.execution_end:.3f} ms"

            par_tracker = time_stats.tensor_parallel_trackers.get(tensor, None)
            if par_tracker is not None and par_tracker.execution_end != 0:
                if label_with_time:
                    label_with_time += " / "
                label_with_time += f"{par_tracker.execution_end:.3f} ms"

            if label_with_time:
                label = f"{tensor_name}\\n{label_with_time}"

            # add a node
            tensor_node = pydot.Node(
                name=tensor_name,
                shape="note",
                id=tensor_name,
                label=label,
                color=color,
            )
            tensor_set[tensor] = tensor_node
            dot_graph.add_node(tensor_node)
            modal_set.append(_gen_tensor_modal(tensor))
            items.append(tensor_name)

            popover_data[tensor_name] = f"shape: {_get_tensor_shape_str(tensor)}"

        for src_op in tensor.src_ops():
            op_node = None
            op_name = src_op._attrs["name"]

            # replace op_name with a unique name, if provided
            if op_name is not None:
                op_name = op_names.get(src_op, op_name)

            if src_op in op_set:
                op_node = op_set[src_op]
            else:
                label = (
                    f"{op_name}\\n{str(time_stats.op_durations[src_op])} ms"
                    if src_op in time_stats.op_durations
                    else op_name
                )
                op_node = pydot.Node(
                    name=op_name,
                    shape="folder",
                    id=op_name,
                    label=label,
                    color="mediumpurple1",
                )
                _highlight_op_node(op_node, src_op, time_stats)

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

            # replace op_name with a unique name, if provided
            if op_name is not None:
                op_name = op_names.get(dst_op, op_name)

            if dst_op in op_set:
                op_node = op_set[dst_op]
            else:
                label = (
                    f"{op_name}\\n{str(time_stats.op_durations[dst_op])} ms"
                    if dst_op in time_stats.op_durations
                    else op_name
                )
                op_node = pydot.Node(
                    name=op_name,
                    shape="folder",
                    id=op_name,
                    label=label,
                    color="mediumpurple1",
                )
                _highlight_op_node(op_node, dst_op, time_stats)

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
        items_src = json.dumps(items, indent=2)
        popover_src = json.dumps(popover_data, indent=2)
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
