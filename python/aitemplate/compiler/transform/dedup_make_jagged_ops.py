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
Deduplicate make_jagged ops in the graph.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Set

from aitemplate.compiler.base import IntVar, JaggedIntVar, Operator, Tensor

from aitemplate.compiler.ops import make_jagged
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.compiler.transform.transform_utils import (
    remove_dst_op_from_tensor,
    replace_tensor,
    replace_tensor_for_op,
    sanitize_sorted_graph,
)
from aitemplate.utils.graph_utils import get_sorted_ops


_LOGGER = logging.getLogger(__name__)


@dataclass
class MakeJaggedMetaData:
    op: Operator
    sources_list: List[Tensor]
    offsets_list: List[Tensor]
    outputs: List[Tensor]
    jagged_int_var: JaggedIntVar


def _get_make_jagged_metadata(
    sorted_graph: List[Tensor],
) -> Dict[IntVar, List[MakeJaggedMetaData]]:
    """Collect metadata about the existing make_jagged ops in the graph.

    The MakeJaggedMetaData instances, one per make_jagged op, are grouped
    by the total_length dimension in the source input Tensors of the ops.
    In case of multiple inputs, total_length dimension is the same in
    every input. The metadata is used further to inform the transformation.
    """
    metadata = {}
    for op in get_sorted_ops(sorted_graph):
        if op._attrs["op"] == "make_jagged":
            outputs = op._attrs["outputs"]
            jagged_int_var = outputs[0]._attrs["shape"][0]
            total_length = jagged_int_var.total_length()
            num_sources = op._attrs["num_sources"]
            if total_length not in metadata:
                metadata[total_length] = []
            metadata[total_length].append(
                MakeJaggedMetaData(
                    op=op,
                    sources_list=op._attrs["inputs"][:num_sources],
                    offsets_list=op._attrs["inputs"][num_sources:],
                    outputs=outputs,
                    jagged_int_var=jagged_int_var,
                )
            )

    return metadata


def _remove_make_jagged_ops(
    make_jagged_metadata: Dict[IntVar, List[MakeJaggedMetaData]],
    graph_inputs: Set[Tensor],
    graph_outputs: Set[Tensor],
):
    """Remove the make_jagged ops from the graph where possible.

    The individual make_jagged ops scattered over the graph are removed,
    to be further replaced by a single make_jagged instance, per total_length
    dimension, applied to all inputs with the total_length dimension at once.
    The ops are considered group by group, where group is formed from
    the ops with the same total_length dimension in the source Tensors.

    The make_jagged ops in the group are not removed (and the respective
    total_length key is popped from the make_jagged_metadata) if:

        1. There is only one make_jagged op in the group.

        2. There is a make_jagged op in the group connecting a
           graph input to a graph output: can't be eliminated.

        3. The total_length dimension representing the group is
           not present in any of the graph inputs' shape.

    In other cases, all make_jagged ops in the grpup are removed from the graph
    (and the respective total_length key is kept in the make_jagged_metadata).
    """
    for total_length in list(make_jagged_metadata.keys()):
        make_jagged_group = make_jagged_metadata[total_length]
        assert len({d.jagged_int_var for d in make_jagged_group}) == 1, (
            "All make_jagged ops applied to the sources with the "
            "same total_length must produce the same jagged_int_var."
        )  # this includes offsets identity check internally

        if len(make_jagged_group) == 1:
            _LOGGER.debug(
                "There is only one make_jagged op in the group "
                f"with {total_length=}: skipping the group."
            )
            make_jagged_metadata.pop(total_length)
            continue

        has_input_to_output_op = False
        for data in make_jagged_group:
            if any(s in graph_inputs for s in data.sources_list) and any(
                o in graph_outputs for o in data.outputs
            ):
                has_input_to_output_op = True
                break
        if has_input_to_output_op:
            _LOGGER.debug(
                "There is a make_jagged op in the group with "
                f"{total_length=} that maps a graph input to "
                "a graph output: skipping the group."
            )
            make_jagged_metadata.pop(total_length)
            continue

        graph_input_with_total_length = False
        for inp in graph_inputs:
            shape = inp._attrs["shape"]
            if shape and shape[0] == total_length:
                graph_input_with_total_length = True
                break
        if not graph_input_with_total_length:
            _LOGGER.debug(
                "None of the graph inputs has the first dimension "
                f"equal to {total_length=}: skipping the group."
            )
            make_jagged_metadata.pop(total_length)
            continue

        _LOGGER.debug(
            f"Removing {len(make_jagged_group)} make_jagged ops "
            f"in the group with {total_length=} from the graph."
        )
        for data in make_jagged_group:
            for source, output in zip(data.sources_list, data.outputs):
                replace_tensor(output, source)
                remove_dst_op_from_tensor(source, data.op)


def _apply_make_jagged_to_inputs(
    make_jagged_metadata: Dict[IntVar, List[MakeJaggedMetaData]],
    sorted_graph: List[Tensor],
    graph_inputs: Set[Tensor],
) -> Dict[IntVar, JaggedIntVar]:
    """Apply new make_jagged ops to the (bundled) input source Tensors.

    For each group of make_jagged ops that removed from the graph,
    a new make_jagged op is applied to all graph inputs with the
    corresponding total_length dimension. This way, the source Tensors
    are converted to jagged Tensors right from the "beginning" of the
    graph and can be used as jagged Tensors downstream.

    Two points are worth mentioning:

        1. Due to the fact that the new make_jagged op is applied to
           *all* source inputs with the total_length dimension, it is
           guaranteed that the offsets validation performed by the
           make_jagged op's back-end will run before any of the
           resulting jagged Tensors can be used downstream.

        2. Because a single make_jagged op is applied to multiple
           graph inputs, the make_jagged op's back-end kernel will
           be launched only once to validate the offsets (the latter
           are the same for every source input). This optimizes out
           redundant validation of the same offsets.

    The mapping of each total_length to the new JaggedIntVar (produced
    by the corresponding new make_jagged op) is returned.
    """
    new_jagged_int_vars = {}
    for total_length, make_jagged_group in make_jagged_metadata.items():
        sources_list = []
        for inp in graph_inputs:
            shape = inp._attrs["shape"]
            if shape and shape[0] == total_length:
                sources_list.append(inp)

        _LOGGER.debug(
            "Adding a single make_jagged op for the source inputs "
            f"{[source._attrs['name'] for source in sources_list]}."
        )

        data = make_jagged_group[0]
        new_make_jagged_op = make_jagged(
            batch_dim=data.jagged_int_var.batch_dim(),
            jagged_dims=data.jagged_int_var.jagged_dims(),
            check_sequence_lengths=all(
                d.op._attrs["check_sequence_lengths"] for d in make_jagged_group
            ),
        )
        jagged_tensors = new_make_jagged_op(
            source=sources_list,
            offsets_list=data.offsets_list,
        )
        jagged_int_var = jagged_tensors[0]._attrs["shape"][0]
        new_jagged_int_vars[total_length] = jagged_int_var

        for source, jagged in zip(sources_list, jagged_tensors):
            for op in source._attrs["dst_ops"]:
                if op is not new_make_jagged_op:
                    replace_tensor_for_op(op, source, jagged)

        sorted_graph.extend(jagged_tensors)

    return new_jagged_int_vars


def _replace_total_length_with_jagged_int_var(
    new_jagged_int_vars: Dict[IntVar, JaggedIntVar],
    sorted_graph: List[Tensor],
    graph_inputs: Set[Tensor],
):
    """Replace total_length dimensions by the new JaggedIntVars.

    As we've removed the internal make_jagged ops from the graph and
    replaced their output jagged Tensors by the input source Tensors,
    the latter have lost their JaggedIntVars. Here we replace the
    total_length dimension in *every* non-input Tensor in the graph
    by the corresponding new JaggedIntVar (produced by the new
    make_jagged op applied to the bundled source inputs). This includes,
    but is not limited to, the source inputs of the make_jagged ops
    removed from within the graph in the beginning of the pass.
    """
    for total_length, new_jagged_int_var in new_jagged_int_vars.items():
        for tensor in sorted_graph:
            if tensor not in graph_inputs:
                shape = tensor._attrs["shape"]
                if shape and shape[0] == total_length:
                    shape[0] = new_jagged_int_var


def dedup_make_jagged_ops(
    sorted_graph: List[Tensor],
    workdir: str = None,
) -> List[Tensor]:
    """Deduplicate make_jagged ops in the graph.

    The rationale is to eliminate redundant offset validation as
    well as make the implicit jagged Tensors (sources) in the graph
    explicit, by replacing their total_length dimension with the
    corresponding JaggedIntVar.

    The pass is performed in the following steps:

        1. Collect the metadata of the existing make_jagged ops.
        2. Remove make_jagged ops from the graph where possible.
        3. Apply new make_jagged ops to the (bundled) source inputs.
        4. Replace total_length dimensions with new JaggedIntVars.

    See the docstrings of the individual steps' helper functions
    above for more details.
    """
    make_jagged_metadata = _get_make_jagged_metadata(sorted_graph)

    if not make_jagged_metadata:
        _LOGGER.debug("No make_jagged ops in the graph: skipping.")
        return sorted_graph

    graph_inputs = {t for t in sorted_graph if t._attrs["is_input"]}
    graph_outputs = {t for t in sorted_graph if t._attrs["is_output"]}

    _remove_make_jagged_ops(
        make_jagged_metadata,
        graph_inputs,
        graph_outputs,
    )

    if not make_jagged_metadata:
        _LOGGER.debug(
            "There are make_jagged ops in the graph, "
            "but nothing to deduplicate: skipping."
        )
        return sorted_graph

    # drop the removed make_jagged outputs
    sorted_graph = sanitize_sorted_graph(sorted_graph)

    new_jagged_int_vars = _apply_make_jagged_to_inputs(
        make_jagged_metadata,
        sorted_graph,
        graph_inputs,
    )
    _replace_total_length_with_jagged_int_var(
        new_jagged_int_vars,
        sorted_graph,
        graph_inputs,
    )

    # sort the new make_jagged outputs
    sorted_graph = toposort(sorted_graph)
    # name the new tensors + do sanity check
    sorted_graph = sanitize_sorted_graph(sorted_graph)

    return sorted_graph
