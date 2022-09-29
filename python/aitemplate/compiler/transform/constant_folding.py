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
from typing import Dict, List

import numpy as np

from aitemplate import backend, compiler

from aitemplate.compiler.base import _NumpyConstantTensorData, Tensor
from aitemplate.compiler.model import AITData, Model
from aitemplate.compiler.transform.transform_utils import replace_tensor
from aitemplate.utils import logger


def _output_from_tensor(tensor: Tensor) -> Tensor:
    new_tensor = Tensor(
        shape=tensor._attrs["shape"],
        name=tensor._attrs["name"],
        src_ops=tensor._attrs["src_ops"].copy(),
        dst_ops=tensor._attrs["dst_ops"].copy(),
        dtype=tensor._attrs["dtype"],
        is_output=True,
        is_view_of=tensor._attrs["is_view_of"],
    )
    if new_tensor._attrs["is_view_of"] is not None:
        # If this tensor is a view, we need to set external_tensor
        # so codegen handles the "output is view of output" case
        # correctly.
        new_tensor._attrs["external_tensor"] = new_tensor._attrs["is_view_of"]
    return new_tensor


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
    subgraph = []

    for tensor in sorted_graph:
        if tensor._attrs["is_input"]:
            continue

        name = tensor._attrs["name"]
        if tensor._attrs["data"] is not None:
            foldable_node_names.add(name)
            subgraph.append(tensor)
            continue
        elif tensor._attrs["is_param"]:
            # Params that do not have bound data cannot be folded.
            continue

        foldable = all(
            inp._attrs["name"] in foldable_node_names
            for op in tensor._attrs["src_ops"]
            for inp in op._attrs["inputs"]
        )

        if foldable:
            foldable_node_names.add(name)
            subgraph.append(_output_from_tensor(tensor))

    return subgraph


def _constant_folding_impl(
    sorted_graph: List[Tensor], workdir: str
) -> Dict[str, Tensor]:

    # Collect the set of output names before we do any transformations. We'll need this
    # if we end up turning outputs into constants. _extract_foldable_subgraph marks *all*
    # folded constants as outputs, so we can't just query attrs["is_output"] (see
    # extract_foldable_subgraph for more info on why that happens)
    original_output_tensors = {
        tensor._attrs["name"] for tensor in sorted_graph if tensor._attrs["is_output"]
    }

    subgraph = _extract_foldable_subgraph(sorted_graph)
    output_tensors = [tensor for tensor in subgraph if tensor._attrs["is_output"]]
    if not output_tensors:
        logger.info(__file__, "No constants to fold, skipping constant folding.")
        return {}

    blob, constant_blob, workspace = compiler.transform.memory_planning(subgraph)

    constant_folding_workdir = os.path.join(workdir, "constant_folding")
    os.makedirs(constant_folding_workdir, exist_ok=True)
    file_pairs = backend.codegen.gen_function_src(subgraph, workdir, "constant_folding")
    main_pairs = backend.codegen.gen_library_src(
        subgraph,
        blob,
        constant_blob,
        workspace,
        workdir,
        output_tensors,
        "constant_folding",
    )
    file_pairs.extend(main_pairs)
    compile_engine = backend.builder.Builder()
    compile_engine.build_objs(
        file_pairs,
        backend.target.Target.current().compile_cmd(False),
        backend.target.Target.current().binary_compile_cmd(),
    )

    so_name = os.path.join(constant_folding_workdir, "test.so")
    compile_engine.build_so(so_name, [p[1] for p in file_pairs])

    module = Model(so_name, num_runtimes=1)

    outputs = {}
    new_tensors = {}
    for tensor in subgraph:
        if tensor._attrs["data"] is None:
            name = tensor._attrs["name"]
            shape = module.get_output_maximum_shape(tensor._attrs["name"])
            arr = np.empty(shape, dtype=tensor._attrs["dtype"])
            new_tensor = Tensor(
                shape=tensor._attrs["shape"],
                name=name,
                # copy dst_ops so we can modify the original tensor without affecting this one.
                dst_ops=tensor._attrs["dst_ops"].copy(),
                dtype=tensor._attrs["dtype"],
                is_output=name in original_output_tensors,
            )
            new_tensor._bind_data(_NumpyConstantTensorData(arr))
            new_tensors[name] = new_tensor
            outputs[name] = AITData(arr.ctypes.data, shape, tensor._attrs["dtype"])

    module._run_with_outputs_on_host({}, outputs)
    return new_tensors


def constant_folding(sorted_graph: List[Tensor], workdir: str) -> List[Tensor]:
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
    try:
        new_constants = _constant_folding_impl(sorted_graph, workdir)
    except Exception as e:
        logger.warning(
            __file__,
            f"Constant folding encountered an error: {e}. The graph will not be modified.",
        )
        return sorted_graph

    # Replace ops with their folded values.
    for idx, tensor in enumerate(sorted_graph):
        name = tensor._attrs["name"]
        if name in new_constants:
            new_tensor = new_constants[name]
            replace_tensor(tensor, new_tensor)
            sorted_graph[idx] = new_tensor

    # Eliminate constants that are no longer used
    compiler.transform.remove_unused_ops(sorted_graph)
    return compiler.transform.transform_utils.sanitize_sorted_graph(sorted_graph)
