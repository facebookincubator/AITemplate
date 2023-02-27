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
import logging
import os
from typing import Dict, List, Tuple

from aitemplate import backend, compiler

from aitemplate.compiler.base import IntVarTensor, Tensor
from aitemplate.compiler.transform.memory_planning import Workspace
from aitemplate.compiler.transform.transform_utils import replace_tensor
from aitemplate.utils import graph_utils


_LOGGER = logging.getLogger(__name__)


def _create_dummy_constant_folder():
    model_container_generator = backend.codegen.ModelContainerGenerator(
        max_blob_size=0,
        max_constant_blob_size=0,
        workspace=Workspace(0, 0),
        constants_data_file=None,
        graph=[],
        output_tensors=[],
        model_name=backend.codegen.CONSTANT_FOLDER_MODEL_NAME,
    )
    return model_container_generator.generate_model()


def _make_op_names_unique(graph: List[Tensor]) -> Dict[str, str]:
    """
    To avoid ODR issues, we rename all ops in the constant folding subgraph.
    ODR issues can arise if two ops end up sharing the same name & implementation (which
    can actualy happen, e.g. in the proposal op).
    """
    new_name_to_old = {}
    for tensor in graph:
        for op in tensor._attrs["src_ops"]:
            if op._attrs["name"] not in new_name_to_old:
                new_name = f"{op._attrs['name']}_constant_folding"
                new_name_to_old[new_name] = op._attrs["name"]
                op._attrs["name"] = new_name
    return new_name_to_old


def _rename_ops(graph: List[Tensor], new_name_to_old: Dict[str, str]) -> None:
    for tensor in graph:
        for op in tensor._attrs["src_ops"]:
            if op._attrs["name"] in new_name_to_old:
                op._attrs["name"] = new_name_to_old[op._attrs["name"]]


def _non_output_from_tensor(tensor: Tensor) -> Tensor:
    new_tensor = Tensor(
        shape=tensor._attrs["shape"],
        name=tensor._attrs["name"],
        src_ops=tensor._attrs["src_ops"].copy(),
        dst_ops=tensor._attrs["dst_ops"].copy(),
        dtype=tensor._attrs["dtype"],
        is_view_of=tensor._attrs["is_view_of"],
        is_internal_constant=tensor._attrs["is_internal_constant"],
    )
    new_tensor._attrs["is_param"] = tensor._attrs["is_param"]
    new_tensor._attrs["data"] = tensor._attrs["data"]
    new_tensor._attrs["external_tensor"] = tensor._attrs["external_tensor"]
    return new_tensor


def _output_from_tensor(tensor: Tensor) -> Tensor:
    new_tensor = _non_output_from_tensor(tensor)
    new_tensor._attrs["is_output"] = True
    return new_tensor


def _fix_op_inputs_outputs(
    subgraph: List[Tensor], name_to_new_tensor: Dict[str, Tensor]
) -> None:
    """
    This is an unfortunate hack made necessary by the following:

    1) When constructing the constant folding subgraph, the most understandable
       thing to do is create *new* tensors so we can modify their attributes without
       affecting the original graph.
    2) However, the inputs of each tensor's src and dst ops need to be wired up to
       the new tensors since the memory planning pass will traverse the graph through those attributes.

    So, we store the mapping from tensor name to its corresponding subgraph tensor and the tensor in
    original graph.

    Before we do memory planning for constant folding, we call:
      _fix_op_inputs_outputs(subgraph, name_to_constant_folding_tensor)

    And then afterwards we restore everything with:
      _fix_op_inputs_outputs(subgraph, name_to_original_tensor)

    It would be nice if we could deep copy the src and dst ops when we create new tensors so we can
    skip the restoration step. But this is not implemented and not trivial. Thankfully, this function
    is not too hard to understand once the rationale behind it is understood.
    """
    ops = graph_utils.get_sorted_ops(subgraph)
    for op in ops:
        op._attrs["inputs"] = [
            name_to_new_tensor[tensor._attrs["name"]] for tensor in op._attrs["inputs"]
        ]

        op._attrs["outputs"] = [
            name_to_new_tensor[tensor._attrs["name"]] for tensor in op._attrs["outputs"]
        ]


def _extract_foldable_subgraph(
    sorted_graph: List[Tensor],
) -> List[Tensor]:
    """
    Extract a list of foldable nodes. A node is foldable if:
    * It has bound data, or
    * All of its inputs are foldable.

    The subgraph returned is just a list of Tensors. All foldable
    tensors that do not have bound data are marked as outputs in
    the subgraph. The original graph is not modified.

    All tensors that do not have bound data are marked as outputs.
    This is because we want to execute the subgraph and get all
    of the new constants. Only the ones that are actually needed are put
    back into the final graph.
    """
    foldable_node_names = set()
    foldable_ops = set()
    subgraph = []

    for tensor in sorted_graph:
        if tensor._attrs["is_input"]:
            continue

        name = tensor._attrs["name"]
        if tensor._attrs["data"] is not None or tensor._attrs["is_param"]:
            foldable_node_names.add(name)
            subgraph.append(tensor)
            continue
        elif isinstance(tensor, IntVarTensor):
            continue
        foldable = all(
            inp._attrs["name"] in foldable_node_names
            for op in tensor._attrs["src_ops"]
            for inp in op._attrs["inputs"]
        )

        if foldable:
            foldable_node_names.add(name)
            subgraph.append(tensor)
            for op in tensor._attrs["src_ops"]:
                foldable_ops.add(op)

    def _is_used_by_non_foldable_op(tensor: Tensor) -> bool:
        for op in tensor._attrs["dst_ops"]:
            if op not in foldable_ops:
                return True
        return False

    def _is_used_by_foldable_op(tensor: Tensor) -> bool:
        for op in tensor._attrs["dst_ops"]:
            if op in foldable_ops:
                return True
        return False

    # Now figure out which tensors can be marked as outputs.
    filtered_subgraph = []
    name_to_new_tensor = {}
    name_to_old_tensor = {}
    constant_folding_inputs = []

    for tensor in subgraph:
        name = tensor._attrs["name"]
        new_tensor = None

        if not tensor._attrs["is_param"] and (
            _is_used_by_non_foldable_op(tensor) or tensor._attrs["is_output"]
        ):
            # Tensor is required outside of the subgraph, make it an output.
            # Parameters don't need to be marked as outputs in the
            # subgraph, we already know their values.
            new_tensor = _output_from_tensor(tensor)

        elif _is_used_by_foldable_op(tensor):
            # No need to append constants that are not used by any foldable ops.
            new_tensor = _non_output_from_tensor(tensor)
            if new_tensor._attrs["is_param"]:
                constant_folding_inputs.append(new_tensor)

        if new_tensor is not None:
            name_to_new_tensor[name] = new_tensor
            name_to_old_tensor[name] = tensor
            filtered_subgraph.append(new_tensor)

    _fix_op_inputs_outputs(filtered_subgraph, name_to_new_tensor)
    return filtered_subgraph, name_to_old_tensor, constant_folding_inputs


def _constant_folding_impl(
    sorted_graph: List[Tensor],
    workdir: str,
    model_name: str,
) -> Tuple[Dict[str, Tensor], List[Tuple[str, str]], List[Tensor]]:
    model_dir = os.path.join(workdir, model_name)

    # Collect the set of output names before we do any transformations. We'll need this
    # if we end up turning outputs into constants. _extract_foldable_subgraph marks *all*
    # folded constants as outputs, so we can't just query attrs["is_output"] (see
    # extract_foldable_subgraph for more info on why that happens)
    original_output_tensors = {
        tensor._attrs["name"] for tensor in sorted_graph if tensor._attrs["is_output"]
    }

    (
        subgraph,
        name_to_old_tensor,
        constant_folding_inputs,
    ) = _extract_foldable_subgraph(sorted_graph)
    output_tensors = [tensor for tensor in subgraph if tensor._attrs["is_output"]]
    if not output_tensors:
        _LOGGER.info("No constants to fold, skipping constant folding.")
        # Write a dummy constant folder so everything still compiles.
        with open(os.path.join(model_dir, "constant_folder-generated.h"), "w") as f:
            f.write(_create_dummy_constant_folder())
        _fix_op_inputs_outputs(subgraph, name_to_old_tensor)
        return {}, [], []

    blob, constant_blob, workspace = compiler.transform.memory_planning(subgraph)
    new_name_to_old = _make_op_names_unique(subgraph)
    file_pairs = backend.codegen.gen_function_src(subgraph, workdir, model_name)
    model_container_generator = backend.codegen.ModelContainerGenerator(
        blob,
        constant_blob,
        workspace,
        constants_data_file=None,
        graph=subgraph,
        output_tensors=output_tensors,
        model_name=backend.codegen.CONSTANT_FOLDER_MODEL_NAME,
    )
    model_container_generator.append_all_tensors()
    constant_folding_model_def = model_container_generator.generate_model()
    with open(os.path.join(model_dir, "constant_folder-generated.h"), "w") as f:
        f.write(constant_folding_model_def)

    _fix_op_inputs_outputs(subgraph, name_to_old_tensor)
    _rename_ops(subgraph, new_name_to_old)
    new_tensors = {}
    for tensor in subgraph:
        if not tensor._attrs["is_param"]:
            name = tensor._attrs["name"]
            new_tensor = Tensor(
                shape=tensor._attrs["shape"],
                name=name,
                dtype=tensor._attrs["dtype"],
                is_output=name in original_output_tensors,
            )
            if name in model_container_generator.output_name_to_idx:
                new_tensor._attrs[
                    "constant_folding_output_idx"
                ] = model_container_generator.output_name_to_idx[name]
            new_tensors[name] = new_tensor

    return new_tensors, file_pairs, constant_folding_inputs


def constant_folding(
    sorted_graph: List[Tensor],
    workdir: str,
    model_name: str,
) -> Tuple[List[Tensor], Tuple[str, str]]:
    """
    Fold and propagate constants.

    This pass looks for ops that have inputs which can be determined
    at compile time. It evaluates them, then puts the new constants
    back into the graph with bound data. The old ops are eliminated.

    This pass actually compiles and runs an AIT runtime. If there are
    any problems (e.g. due to buggy ops), the constant folding is
    aborted and the graph is returned unchanged. All generated code
    is stored in workdir/constant_folding.
    """
    new_constants, file_pairs, constant_folding_inputs = _constant_folding_impl(
        sorted_graph, workdir, model_name
    )

    # Replace ops with their folded values.
    for idx, tensor in enumerate(sorted_graph):
        name = tensor._attrs["name"]
        if name in new_constants:
            new_tensor = new_constants[name]
            replace_tensor(tensor, new_tensor)
            sorted_graph[idx] = new_tensor

    # Eliminate constants that are no longer used
    compiler.transform.remove_unused_ops(sorted_graph)
    return (
        compiler.transform.transform_utils.sanitize_sorted_graph(sorted_graph),
        file_pairs,
        constant_folding_inputs,
    )
