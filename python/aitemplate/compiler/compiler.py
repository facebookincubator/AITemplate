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
build a test module from a tensor
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from aitemplate import backend, compiler
from aitemplate.compiler.model import AITemplateAllocatorKind
from aitemplate.compiler.transform.profile import elapsed_dt_sec
from aitemplate.utils import graph_utils
from aitemplate.utils.debug_settings import AITDebugSettings
from aitemplate.utils.serialization.serdes_code import dump_program

from .base import DynamicProfileStrategy, Tensor

from .model import AIT_DEFAULT_NUM_RUNTIMES, Model, TorchTensor

# pylint: disable=W0102


_LOGGER = logging.getLogger(__name__)


def _validate_tensor_args(sorted_graph: List[Tensor], output_tensors: List[Tensor]):
    """
    Validate the user's desired output name -> index ordering.

    Errors if:
    1) The given ordering has duplicates
    2) The given ordering has non-outputs
    3) The given ordering is missing outputs that are reachable

    Note that we have to do this before any optimizations. It is legal to replace output tensors
    with new Tensor objects of the same name, so the user-provided tensors might not be in
    the graph after optimizations (replacing a Tensor sets is_output=False).
    """
    seen_tensors = set()
    for tensor in output_tensors:
        name = tensor._attrs["name"]
        if not tensor._attrs["is_output"]:
            raise ValueError(f"Got non-output tensor in output_tensors list: {name}")
        if name in seen_tensors:
            raise ValueError(f"Got duplicate name {name} in output_tensors list.")
        seen_tensors.add(name)

    given_tensors = {tensor._attrs["name"] for tensor in output_tensors}
    for tensor in reversed(sorted_graph):
        name = tensor._attrs["name"]
        if tensor._attrs["is_output"] and name not in given_tensors:
            raise ValueError(f"Output {name} was not passed into output_tensors")


def _verify_outputs_still_in_graph(sorted_graph: List[Tensor], outputs: List[Tensor]):
    seen = {tensor._attrs["name"]: False for tensor in outputs}
    for tensor in sorted_graph:
        name = tensor._attrs["name"]
        if name not in seen:
            continue

        if seen[name]:
            raise ValueError(
                f"Output {name} appears in the graph twice after optimizations."
            )

        seen[name] = True

    for tensor, was_seen in seen.items():
        if not was_seen:
            raise ValueError(
                f"Output {tensor} was not found in the graph after opitmizations."
            )


_DEBUG_SETTINGS = AITDebugSettings()


def compile_model(
    tensor: Union[Tensor, List[Tensor]],
    target: backend.target.Target,
    workdir: str,
    test_name: str,
    profile_devs: List[int] = None,
    dynamic_profiling_strategy: DynamicProfileStrategy = DynamicProfileStrategy.MAX,
    dll_name: str = "test.so",
    num_runtimes: int = AIT_DEFAULT_NUM_RUNTIMES,
    profile_dir: str = None,
    constants: Optional[Dict[str, TorchTensor]] = None,
    allocator_kind: Optional[AITemplateAllocatorKind] = None,
    debug_settings: AITDebugSettings = _DEBUG_SETTINGS,
) -> Model:
    """Compiles a model and generates a .so file.

    Parameters
    ----------
    tensor : Union[Tensor, List[Tensor]]
        An output Tensor, or a list of output Tensors.
        The compiled module will preserve the ordering of the outputs in its
        internal ordering.
    target : Target
        A compilation target. See comments for Target.
    workdir : str
        A workdir to store profiling and execution source codes, as well as the result .so file.
    test_name : str
        Name of the test. Used as the name of the subdir which stores the generated .so file.
    profile_devs : List[int], optional
        A list of profiling devices, by default device 0 will be used.
    dynamic_profiling_strategy: DynamicProfileStrategy, optional
        A DynamicProfileStrategy used for profiling. See comments for DynamicProfileStrategy.
    dll_name: str
        The output .so name.
    num_runtimes: int
        How many runtimes should be stored in the internal pool. This
        determines how many inferences can happen concurrently. By
        default, set to 2. Must be positive.
    allocator_kind: AITemplateAllocatorKind, optional
        The GPU allocator to use. If none is specified, use the default allocator.
    debug_settings: AITDebugSettings
        specify debug settings such as where to dump AITemplate model Python file, etc.

    Returns
    -------
    Model
        A model object.
    """
    if constants is None:
        constants = {}

    recompile = os.getenv("AIT_RECOMPILE", "1")
    graph = None
    # Super important: we cannot have commas in the test name.
    # We want to add a -Iworkdir/test_name flag to nvcc, but
    # if the name has a comma in it, it will be parsed as two
    # arguments (even if we put quotes around it)!!
    test_name = test_name.replace(",", "_")
    test_dir = os.path.join(workdir, test_name)
    profile_dir = workdir if profile_dir is None else profile_dir

    if debug_settings.dump_ait_to_py:
        dump_program(tensor, debug_settings.dump_ait_to_py)

    if int(recompile) == 1:
        os.makedirs(test_dir, exist_ok=True)
        with target:
            graph = compiler.transform.toposort(tensor)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "toposort")

            output_tensors = [tensor] if isinstance(tensor, Tensor) else tensor
            _validate_tensor_args(graph, output_tensors)

            compiler.transform.bind_constants(graph, constants)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "bind_constants")

            compiler.transform.remove_unused_ops(graph)
            graph_utils.dump_graph_debug_str_to_file(
                graph, test_dir, "remove_unused_ops"
            )

            compiler.transform.remove_no_ops(graph)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "remove_no_ops")

            compiler.transform.name_graph(graph)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "name_graph")

            compiler.transform.mark_param_tensor(graph)
            graph_utils.dump_graph_debug_str_to_file(
                graph, test_dir, "mark_param_tensor"
            )

            start_t = datetime.now()
            graph = compiler.transform.optimize_graph(graph, test_dir)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "optimize_graph")
            _LOGGER.info(f"optimized graph elapsed time: {elapsed_dt_sec(start_t)}")

            compiler.transform.mark_special_views(graph)
            compiler.transform.refine_graph(graph)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "refine_graph")

            if profile_devs is None:
                device_env = os.getenv(target.dev_select_flag(), None)
                if device_env is None:
                    profile_devs = [0]
                else:
                    profile_devs = device_env.split(",")
            compiler.transform.profile(
                graph, profile_dir, profile_devs, dynamic_profiling_strategy
            )
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "profile")

            start_t = datetime.now()
            constant_folding_workdir = os.path.join(workdir, test_name)
            os.makedirs(constant_folding_workdir, exist_ok=True)
            (
                graph,
                constant_folding_file_pairs,
                constant_folding_inputs,
            ) = compiler.transform.constant_folding(graph, workdir, test_name)
            graph_utils.dump_graph_debug_str_to_file(
                graph, test_dir, "constant_folding"
            )
            _LOGGER.info(f"folded constants elapsed time: {elapsed_dt_sec(start_t)}")

            (
                max_blob,
                max_constant_blob,
                workspace,
            ) = compiler.transform.memory_planning(graph)
            _verify_outputs_still_in_graph(graph, output_tensors)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "memory_planning")

            file_pairs = backend.codegen.gen_function_src(graph, workdir, test_name)
            file_pairs.extend(constant_folding_file_pairs)

            # It's possible that the original output tensor has been replaced with a new tensor.
            # Preserve original output tensors' orders but use the new tensors.
            new_output_tensor_dict = {
                tensor._attrs["name"]: tensor
                for tensor in graph
                if tensor._attrs["is_output"]
            }
            output_tensors = [tensor] if isinstance(tensor, Tensor) else tensor
            output_tensors = [
                new_output_tensor_dict[tensor._attrs["name"]]
                for tensor in output_tensors
            ]

            main_pairs = backend.codegen.gen_library_src(
                graph,
                max_blob,
                max_constant_blob,
                workspace,
                workdir,
                output_tensors,
                test_name,
                additional_unbound_constants=constant_folding_inputs,
                debug_settings=debug_settings,
            )
            file_pairs.extend(main_pairs)

            start_t = datetime.now()
            compile_engine = backend.builder.Builder()
            compile_engine.make(
                file_pairs, dll_name, workdir, test_name, debug_settings
            )
            _LOGGER.info(
                f"compiled the final .so file elapsed time: {elapsed_dt_sec(start_t)}",
            )

    module = Model(
        os.path.join(workdir, test_name, dll_name), num_runtimes, allocator_kind
    )
    module.debug_sorted_graph = graph
    return module
