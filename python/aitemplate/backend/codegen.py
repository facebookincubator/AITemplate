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
This module is for generating the final C++
source code in files from Tensor and Operators.
Functions in this module will be used for generating
function source code files, profiler source code files,
and model driver source code files.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Tuple

from aitemplate.backend.main_templates import MODEL_CONTAINER_TEMPLATE, MODEL_TEMPLATE
from aitemplate.compiler.base import Operator
from aitemplate.compiler.tensor_accessor import TensorAccessor

from aitemplate.compiler.transform.memory_planning import Workspace

from ..compiler.base import get_dtype_size, IntImm, IntVar, Tensor
from . import registry
from .target import Target

# pylint: disable=C0103,W0613,C0301

DTYPE_TO_POINTERTYPE: Dict[str, str] = {
    "float32": "float*",
    "float": "float*",
    "int": "int32_t*",
    "int32": "int32_t*",
    "int64": "int64_t*",
}


def gen_profiler(sorted_graph: list[Tensor], workdir: str, dynamic_profiling_strategy):
    """Generate operator profiler source code files for the given graph

    Parameters
    ----------
    sorted_graph : list[Tensor]
        The network after running toposort transformation
    workdir : str
        Target directory for generated C++ source code files
    dynamic_profiling_strategy: DynamicProfileStrategy, optional
        A dynamic profiling strategy, used to filter generated profiles at compile time.
        Pass-through to gen_profiler kernels of nodes in the graph.
        See also: :func:`~aitemplate.compiler.transform.profile.profile`
    """
    for node in sorted_graph:
        for func in node.src_ops():
            if "has_profiler" in func._attrs and func._attrs["has_profiler"]:
                func.gen_profiler(workdir, dynamic_profiling_strategy)


def gen_function_src(
    sorted_graph: list[Tensor], workdir: str, model_name: str = ""
) -> list[Tuple[str, str]]:
    """Generate functions source code files for the given graph

    Parameters
    ----------
    sorted_graph : list[Tensor]
        The network after running toposort transformation
    workdir : str
        Target directory for generated C++ source code files
    model_name : str, optional
        Sub working directory in the workdir for the given model, by default ""

    Returns
    -------
    list[Tuple[str, str]]
        List of tuple (source file path, object file path)
    """
    target = Target.current()
    file_pairs = []
    exist_func = set()
    prefix = os.path.join(workdir, model_name)
    for node in sorted_graph:
        for func in node.src_ops():
            fname = func._attrs["name"]
            if fname not in exist_func:
                src_path = os.path.join(prefix, fname + target.src_extension())
                obj_path = os.path.join(prefix, fname + ".obj")
                file_pairs.append((src_path, obj_path))
                with open(src_path, "w") as fo:
                    fo.write(func.gen_function())
                exist_func.add(fname)
    return file_pairs


def map_set(
    map_name: str,
    key_name: str,
    value_name: Optional[str] = None,
    indent: str = "    ",
) -> str:
    """Generate a string setting a value in a map.

    If value name is given, sets map_name["key_name"] = value_name. Else, sets
    map_name["key_name"] = key_name. Special maps like dim_map may make
    additional modificiations to the LHS of this expression.

    Parameters
    ----------
    map_name : str
        The map to use
    key_name : str
        The key to set. Will be put into quotes.
    value_name : Optional[str]
        If set, force map_name["key_name"] = value_name
    indent : str
        For formatting

    Returns
    -------
    str
        The formatted map set statement.
    """
    if value_name is not None:
        value = value_name
    else:
        value = key_name
        if map_name == "dim_map":
            # Because ROCM backend uses int64_t while CUDA uses int,
            # this is a temporary workaround to cast int64_t* to int*.
            # FIXME: After we unified the two backends,
            # reinterpret_cast<int *> should be removed.
            value = f"reinterpret_cast<int64_t *>(&{value})"

    return f'{indent}{map_name}["{key_name}"] = {value};'


def set_value(lhs: Any, rhs: Any, indent: str = "     ") -> str:
    return f"{indent}{lhs} = {rhs};"


def set_value_from_map(map_name: Any, var_name: Any, indent: str = "    ") -> str:
    """Generate a string that sets a value to something stored in a map.

    Parameters
    ----------
    map_name : str
        The map to use
    var_name : str
        The var_name, used as the name of the value and the key.
    indent : str
        For formatting

    Returns
    -------
    str
        The formatted statement.
    """
    key = var_name
    value = var_name
    return f'{indent}{value} = static_cast<decltype({value})>({map_name}["{key}"]);'


def dtype_to_enumerator(dtype):
    def _impl(dtype):
        if dtype == "float16":
            return "kHalf"
        elif dtype == "float32" or dtype == "float":
            return "kFloat"
        elif dtype == "int32" or dtype == "int":
            return "kInt"
        elif dtype == "int64":
            return "kLong"
        else:
            raise AssertionError(f"unknown dtype {dtype}")

    return f"AITemplateDtype::{_impl(dtype)}"


def count_inputs_outputs(graph):
    n_inputs = n_outputs = 0
    for node in graph:
        if node._attrs["is_input"]:
            n_inputs += 1
        if node._attrs["is_output"]:
            n_outputs += 1
    return n_inputs, n_outputs


def check_not_null(
    tensor: Tensor,
    tensor_idx: Optional[int] = None,
    skip_if_lower_bound_is_zero: bool = False,
) -> str:
    """
    Generate a nullptr check to be used by pointer initialization code.

    If skip_if_lower_bound_is_zero == True, no code will be generated
    when the Tensor has at least one dynamic dim with a lower bound
    of zero. This is most useful for outputs; we put the nullptr
    checks at the start of the inference, but we won't know output
    shapes until after Run() finishes. We therefore just relax the check
    for these outputs - only allow them to be null if their lower bound
    is zero, otherwise never allow them to be null.
    """
    name = tensor._attrs["name"]
    if tensor_idx is None:
        check = name
    else:
        check = f"params[{tensor_idx}].ptr"

    shape = ["1"]
    lower_bound_is_zero = False
    for dim in tensor._attrs["shape"]:
        lower_bound_is_zero |= dim.lower_bound() == 0
        if skip_if_lower_bound_is_zero and lower_bound_is_zero:
            return ""
        if isinstance(dim, IntImm):
            shape.append(str(dim._attrs["values"][0]))
        else:
            shape.append(dim._attrs["name"])

    nullptr_check = f"{check} == nullptr"
    condition = (
        nullptr_check
        # If the lower bound of the shape is positive, never allow
        # the tensor to be null.
        if not lower_bound_is_zero
        # Otherwise, allow it to be null only if the (possibly dynamic)
        # size is zero.
        else f"{nullptr_check} && {'*'.join(shape)} != 0"
    )
    return f"""
if ({condition}) {{
    throw std::runtime_error("Constant {name} was not set! Set the value with set_constant.");
}}
    """


def device_copy(dst_tensor: Tensor, src_tensor: Tensor, dst_idx: int) -> str:
    src_name = src_tensor._attrs["name"]
    dst_ptr = f"params[{dst_idx}].ptr"
    shape = ["1"]
    for dim in dst_tensor._attrs["shape"]:
        if isinstance(dim, IntImm):
            shape.append(str(dim._attrs["values"][0]))
        else:
            shape.append(dim._attrs["name"])
    shape = "*".join(shape)
    size = f"{shape} * {get_dtype_size(dst_tensor._attrs['dtype'])}"
    return f"DEVICE_CHECK(DeviceToDeviceCopy({dst_ptr}, {src_name}, {size}, stream));"


class ModelContainerGenerator:
    def __init__(
        self,
        max_blob_size: int,
        max_constant_blob_size: int,
        workspace: Workspace,
        num_inputs: int,
        num_outputs: int,
        constants_data_file: io.BytesIO,
        output_name_to_idx: Dict[str, int],
    ):
        self.target = Target.current()
        self.f_var_decl = registry.get(self.target.name() + ".lib.var_decl")
        self.f_ptr_decl = registry.get(self.target.name() + ".lib.ptr_decl")

        self.constants_data_file = constants_data_file

        self.exist_funcs = set()
        self.func_decl = []
        self.tensor_slice = []
        self.tensor_map_set = []
        self.set_inputs = []
        self.func_seq = []
        self.tensor_decl = []
        self.dim_decl = []
        self.device_to_device_copies = []
        self.function_state = []
        self.set_up_constants = []
        self.set_up_param_names = []
        self.set_up_param_dtypes = []
        self.set_up_output_shapes = []
        self.set_up_param_dynamic_shapes = []
        self.state_record = set()
        self.visited_func = set()
        self.visited_dims = set()
        self.set_up_constant_names = []
        self.param_name_to_ptr_idx = {}

        self.num_constants = 0
        self.constants_data_size = 0
        self.owned_constants_init = []

        self.input_idx = 0
        self.unbound_constant_idx = 0
        self.output_name_to_idx = output_name_to_idx

        (
            self.max_blob_size,
            self.max_constant_blob_size,
            self.workspace,
            self.num_inputs,
            self.num_outputs,
        ) = (
            max_blob_size,
            max_constant_blob_size,
            workspace,
            num_inputs,
            num_outputs,
        )

    def _tensor_slice_func(
        self,
        node: Tensor,
        blob_name: str,
        indent="    ",
    ) -> str:
        offset = node._attrs["offset"]
        name = node._attrs["name"]
        return f"{indent}{name} = reinterpret_cast<decltype({name})>({blob_name} + {offset});"

    def _record_param_tensor_info(self, tensor: Tensor, idx: int) -> None:
        def max_value(var_or_imm):
            if isinstance(var_or_imm, IntImm):
                return var_or_imm.value()
            else:
                assert isinstance(var_or_imm, IntVar)
                return var_or_imm.upper_bound()

        shape_init = ", ".join(str(max_value(dim)) for dim in tensor._attrs["shape"])
        param_shape_init = ", ".join(
            f'&{dim._attrs["name"]}' for dim in tensor._attrs["shape"]
        )
        self.set_up_output_shapes.append(
            set_value(f"max_param_shapes_[{idx}]", f"{{{shape_init}}}")
        )
        param_shape_init = ", ".join(
            f'ParamDim({dim.lower_bound()}, {dim.upper_bound()}, &{dim._attrs["name"]})'
            for dim in tensor._attrs["shape"]
        )
        self.set_up_param_dynamic_shapes.append(
            set_value(f"params[{idx}].shape_ptrs", f"{{{param_shape_init}}}")
        )
        name = tensor._attrs["name"]
        self.set_up_param_names.append(set_value(f"param_names_[{idx}]", f'"{name}"'))
        self.set_up_param_dtypes.append(
            set_value(
                f"param_dtypes_[{idx}]",
                dtype_to_enumerator(tensor.dtype()),
            )
        )

    def _codegen_param_setup(
        self,
        tensor: Tensor,
    ) -> None:
        """
        Generate code needed for setting up a constant in Model/ModelContainer.
        """
        name = tensor._attrs["name"]
        data = tensor._attrs["data"]
        if data is not None:
            # Owned constant. Set up logic for copying the constant in from *.so.
            assert (
                tensor._attrs["offset"] >= 0
            ), f"Constant node '{name}' must have non-negative offset"
            self.set_up_constants.append(self._tensor_slice_func(tensor, "constants"))
            num_bytes = len(data)
            self.constants_data_file.write(data.to_bytes())

            constant_info = f'ConstantInfo{{"{name}", {self.constants_data_size}, {tensor._attrs["offset"]}, {num_bytes}}}'
            self.owned_constants_init.append(constant_info)
            self.constants_data_size += num_bytes
            self.num_constants += 1
        else:
            # Unbound constant. We will expect the user to set this via SetConstant.
            self.set_up_constant_names.append(
                set_value(
                    f'unbound_constant_name_to_idx_["{name}"]',
                    self.unbound_constant_idx,
                )
            )
            self._record_param_tensor_info(
                tensor, self.unbound_constant_idx + self.num_inputs + self.num_outputs
            )
            self.unbound_constant_idx += 1
            self.set_inputs.append(check_not_null(tensor))
            self.set_up_constants.append(
                set_value(
                    f'constant_name_to_ptr_["{name}"]',
                    f"const_cast<const void**>(reinterpret_cast<void**>(&{name}))",
                )
            )

    def _codegen_input_tensor(self, tensor: Tensor) -> None:
        name = tensor._attrs["name"]
        view = tensor._attrs["is_view_of"]
        assert (
            view is None
        ), f"_codegen_input_tensor cannot be called with a view; expected a non-view tensor with is_input=True, got: {tensor}"
        self.set_inputs.append(
            set_value(
                name,
                f"static_cast<decltype({name})>(params[{self.input_idx}].ptr)",
            )
        )
        self.set_inputs.append(check_not_null(tensor))
        self.param_name_to_ptr_idx[name] = self.input_idx
        self._record_param_tensor_info(tensor, self.input_idx)
        self.input_idx += 1

    def _get_output_idx(self, name: str) -> int:
        assert (
            name in self.output_name_to_idx
        ), f"Tensor {name} was marked as an output, but its index was not found in output_name_to_index"
        # Add num_inputs since we internally store outputs in the same array as inputs w/
        # inputs first
        return self.output_name_to_idx[name] + self.num_inputs

    def _codegen_output_aliases_tensor(self, tensor: Tensor) -> None:
        name = tensor._attrs["name"]
        view = tensor._attrs["is_view_of"]
        if tensor._attrs["external_tensor"] is not None:
            self.set_inputs.append(set_value(name, view._attrs["name"]))
            return
        is_view = view is not None
        if is_view:
            ptr_idx = self.param_name_to_ptr_idx[view._attrs["name"]]
            self.set_inputs.append(set_value(name, view._attrs["name"]))
        else:
            ptr_idx = self._get_output_idx(name)
            self.set_inputs.append(
                set_value(
                    name,
                    f"static_cast<decltype({name})>(params[{ptr_idx}].ptr)",
                )
            )

        self.param_name_to_ptr_idx[name] = ptr_idx
        if tensor._attrs["is_output"]:
            self._record_param_tensor_info(tensor, ptr_idx)
            self.set_inputs.append(
                check_not_null(tensor, skip_if_lower_bound_is_zero=True)
            )

    def _codegen_output_tensor(self, tensor: Tensor) -> None:
        is_param = tensor._attrs["is_param"]
        is_input = tensor._attrs["is_input"]
        view = tensor._attrs["is_view_of"]
        is_view = view is not None
        external_tensor = tensor._attrs["external_tensor"]
        name = tensor._attrs["name"]

        output_idx = self._get_output_idx(name)

        if is_param:
            self._codegen_param_setup(tensor)
            self._record_param_tensor_info(tensor, output_idx)
            self.device_to_device_copies.append(device_copy(tensor, tensor, output_idx))
        elif external_tensor is not None:
            # Special view cases for outputs; we can hit this case if the output
            # is a view of a constant, input, or another output.
            assert (
                is_view
            ), f"orig_tensor is not None, but node {name} is not marked as a view! Node: {tensor}"
            self.set_inputs.append(
                check_not_null(tensor, output_idx, skip_if_lower_bound_is_zero=True)
            )
            self.set_inputs.append(set_value(name, view._attrs["name"]))
            self.device_to_device_copies.append(
                device_copy(tensor, external_tensor, output_idx)
            )
            self._record_param_tensor_info(tensor, output_idx)
        elif is_input:
            # Inputs that are also outputs require an extra copy
            self.set_inputs.append(
                set_value(
                    name,
                    f"static_cast<decltype({name})>(params[{self.input_idx}].ptr)",
                )
            )
            self._record_param_tensor_info(tensor, self.input_idx)
            self._record_param_tensor_info(tensor, output_idx)
            self.device_to_device_copies.append(device_copy(tensor, tensor, output_idx))
            self.input_idx += 1
        else:
            self._codegen_output_aliases_tensor(tensor)

    def _process_dims(self, shape: List[IntVar]) -> None:
        for dim in shape:
            if dim._attrs["name"] in self.visited_dims:
                continue
            intimm = 0
            if len(dim._attrs["values"]) == 1:
                intimm = dim._attrs["values"][0]
            self.dim_decl.append(self.f_var_decl(dim._attrs["name"], intimm))
            self.visited_dims.add(dim._attrs["name"])

    def _process_dims_for_tensor(self, node: Tensor) -> None:
        self._process_dims(node._attrs["shape"])

    def _process_dims_for_tensor_accessors(
        self, tensor_accessors: List[TensorAccessor]
    ) -> None:
        if tensor_accessors is None:
            return
        for accessor in tensor_accessors:
            self._process_dims(accessor.original_shapes)

    def _process_dims_for_op(self, node: Operator) -> None:
        self._process_dims_for_tensor_accessors(node._attrs.get("input_accessors"))
        self._process_dims_for_tensor_accessors(node._attrs.get("output_accessors"))

    def _process_src_ops(self, node: Tensor) -> None:
        funcs = node.src_ops()
        for func in funcs:
            f_func_decl = registry.get(
                ".".join((self.target.name(), func._attrs["op"], "func_decl"))
            )
            f_func_call = registry.get(
                ".".join((self.target.name(), func._attrs["op"], "func_call"))
            )
            if func._attrs["name"] not in self.exist_funcs:
                self.func_decl.append(f_func_decl(func._attrs))
                self.exist_funcs.add(func._attrs["name"])

            # Only code gen func once for ops with multiple outputs
            # The func can get renamed during refine_graph pass.
            # We use original_name here because it's unique.
            if func._attrs["original_name"] not in self.visited_func:
                self.visited_func.add(func._attrs["original_name"])
                self.func_seq.append(f_func_call(func._attrs, indent="    "))
            if "int_state_flag" in func._attrs:
                if func._attrs["name"] not in self.state_record:
                    self.function_state.append(
                        f'  int64_t {func._attrs["name"]}_state {{0}};'
                    )
                    self.state_record.add(func._attrs["name"])
            self._process_dims_for_op(func)

    def append_tensor(self, node: Tensor) -> None:
        if node._attrs["nop"]:
            return
        name = node._attrs["name"]
        dtype = node._attrs["dtype"]
        self.tensor_decl.append(self.f_ptr_decl(name=name, dtype=dtype))

        is_param = node._attrs["is_param"]
        is_output = node._attrs["is_output"]
        has_output_aliases = node._attrs["has_output_aliases"]
        is_input = node._attrs["is_input"]
        view = node._attrs["is_view_of"]
        is_view = view is not None

        if is_output:
            # Outputs have a ton of special cases that depend on
            # is_input, is_view, etc, so this condition needs to
            # be checked before all the others
            self._codegen_output_tensor(node)
        elif is_param:
            self._codegen_param_setup(node)
        elif is_input:
            self._codegen_input_tensor(node)
        elif has_output_aliases:
            # Special case: internal tensor that aliases an output.
            self._codegen_output_aliases_tensor(node)
        elif not is_view:
            # Normal, internal tensor that is not a view: point it to the
            # internal blob of memory
            assert (
                node._attrs["offset"] >= 0
            ), f"Non-parameter node '{name}' must have non-negative offset"
            self.tensor_slice.append(self._tensor_slice_func(node, "blob_ptr"))
        else:
            # Normal view, point it to the same memory as whatever it
            # aliases
            self.set_inputs.append(set_value(name, view._attrs["name"]))

        self._process_dims_for_tensor(node)
        self._process_src_ops(node)

    def generate_source(self) -> Dict[str, str]:
        """
        Perform the codegen after adding all tensors.
        The dictionary returned is a map from filename -> contents.
        """
        device_functions_header_name = f"{self.target.name()}_device_functions.h"
        result = {}
        result[
            "device_functions-generated.h"
        ] = f'#include "{device_functions_header_name}"'

        # Disable graph mode on ROCM because the updating operations
        # are not supported
        target_has_graph_mode = "true" if self.target.name() == "cuda" else "false"

        model_def = MODEL_TEMPLATE.render(
            function_decl="\n".join(self.func_decl),
            device_functions_header=device_functions_header_name,
            set_inputs="\n".join(self.set_inputs),
            tensor_slice="\n".join(self.tensor_slice),
            tensor_map_set="\n".join(self.tensor_map_set),
            set_up_constants="\n".join(self.set_up_constants),
            device_to_device_copies="\n".join(self.device_to_device_copies),
            set_up_param_dynamic_shapes="\n".join(self.set_up_param_dynamic_shapes),
            function_seq=self.func_seq,
            tensor_decl="\n".join(self.tensor_decl),
            dim_decl="\n".join(self.dim_decl),
            function_state="\n".join(self.function_state),
            target_has_graph_mode=target_has_graph_mode,
            unique_workspace_size=self.workspace.unique_size,
        )

        result["model-generated.h"] = model_def

        model_container_src_fname = f"model_container_base{self.target.src_extension()}"
        model_container_base_src = MODEL_CONTAINER_TEMPLATE.render(
            blob_size=self.max_blob_size,
            workspace_size=self.workspace.total_size(),
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            param_size=self.max_constant_blob_size,
            set_up_constant_names="\n".join(self.set_up_constant_names),
            set_up_param_dtypes="\n".join(self.set_up_param_dtypes),
            set_up_output_shapes="\n".join(self.set_up_output_shapes),
            set_up_param_names="\n".join(self.set_up_param_names),
            num_constants=self.num_constants,
            num_unbound_constants=self.unbound_constant_idx,
            owned_constants_init=",".join(self.owned_constants_init),
        )
        result[model_container_src_fname] = model_container_base_src
        return result


def _construct_output_name_to_index_map(
    sorted_graph: List[Tensor], output_tensors: List[Tensor]
) -> Dict[str, int]:
    """
    Use the given output ordering to construct a name -> index map
    to be used for constructing an internal ordering during codegen.

    The indices in the map are propagated to an output's entire alias set.
    If two outputs are part of the same alias set, only one of them propagates
    its output index.
    """
    result = {tensor._attrs["name"]: i for i, tensor in enumerate(output_tensors)}

    # Mark alias sets
    for tensor in reversed(sorted_graph):
        name = tensor._attrs["name"]
        orig = tensor._attrs["is_view_of"]
        if orig is None:
            continue
        orig_name = orig._attrs["name"]
        if name in result and orig_name not in result:
            result[orig_name] = result[name]

    return result


def gen_library_src(  # noqa: C901
    sorted_graph: list[Tensor],
    max_blob_size: int,
    max_constant_blob_size: int,
    workspace: Workspace,
    workdir: str,
    output_tensors: List[Tensor],
    model_name: str = "",
) -> list[Tuple[str, str]]:
    """Generate model driver source code files for the given graph

    Parameters
    ----------
    sorted_graph : list[Tensor]
        The network after running toposort transformation
    max_blob_size : int
        Total memory for input/output tensor and intermediate results,
        calculated by memory planning transformation
    workspace : Workspace
        Workspace sizes, computed by memory planning
    workdir : str
        Target directory for generated C++ source code files
    model_name : str, optional
        Sub working directory in the workdir for the given model, by default ""

    Returns
    -------
    list[Tuple[str, str]]
        List of tuple (source file path, object file path)
    """

    def to_obj_name(name: str):
        name, _ = os.path.splitext(name)
        return f"{name}.obj"

    num_inputs, num_outputs = count_inputs_outputs(sorted_graph)
    prefix = os.path.join(workdir, model_name)
    constants_fname = os.path.join(prefix, "constants.bin")
    constants_data_file = open(constants_fname, "wb")

    output_name_to_index = _construct_output_name_to_index_map(
        sorted_graph, output_tensors
    )

    model_container_generator = ModelContainerGenerator(
        max_blob_size,
        max_constant_blob_size,
        workspace,
        num_inputs,
        num_outputs,
        constants_data_file,
        output_name_to_index,
    )
    for node in sorted_graph:
        model_container_generator.append_tensor(node)
    constants_data_file.close()

    files = model_container_generator.generate_source()
    to_build = [(constants_fname, to_obj_name(constants_fname))]
    for fname, contents in files.items():
        fname_full = os.path.join(prefix, fname)
        with open(fname_full, "w") as fo:
            fo.write(contents)
        if not fname_full.endswith(".h"):
            to_build.append((fname_full, to_obj_name(fname_full)))

    # Copy over static csrc/headers
    sources = model_container_generator.target.copy_headers_and_csrc_to_workdir(prefix)
    for fname in sources:
        to_build.append((fname, to_obj_name(fname)))

    return to_build
