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
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import jinja2

from aitemplate.backend.main_templates import MODEL_CONTAINER_TEMPLATE, MODEL_TEMPLATE
from aitemplate.compiler.base import Operator
from aitemplate.compiler.dtype import dtype_to_enumerator, get_dtype_size
from aitemplate.compiler.tensor_accessor import TensorAccessor

from aitemplate.compiler.transform.memory_planning import Workspace
from aitemplate.utils.debug_settings import AITDebugSettings

from ..compiler.base import IntImm, IntVar, IntVarTensor, Tensor
from . import registry
from .target import Target

# pylint: disable=C0103,W0613,C0301


_LOGGER = logging.getLogger(__name__)

DTYPE_TO_POINTERTYPE: Dict[str, str] = {
    "float32": "float*",
    "float": "float*",
    "int": "int32_t*",
    "int32": "int32_t*",
    "int64": "int64_t*",
    "bool": "bool*",
}


CONSTANT_FOLDER_MODEL_NAME = "ConstantFolder"
MODEL_NAME = "Model"


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
    results = []
    for node in sorted_graph:
        for func in node.src_ops():
            if "has_profiler" in func._attrs and func._attrs["has_profiler"]:
                results.append(func.gen_profiler(workdir, dynamic_profiling_strategy))
    return results


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
    _LOGGER.info(f"generated {len(file_pairs)} function srcs")
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
        check = f"params_[{tensor_idx}].ptr"

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


def extract_input_output_shapes(func_attrs):
    if "input_accessors" in func_attrs:
        input_shape = [
            [v.pseudo_code() for v in acc.original_shapes]
            for acc in func_attrs["input_accessors"]
        ]
    else:
        input_shape = [
            [v.pseudo_code() for v in t.shape()] for t in func_attrs["inputs"]
        ]

    if "output_accessors" in func_attrs:
        output_shape = [
            [v.pseudo_code() for v in acc.original_shapes]
            for acc in func_attrs["output_accessors"]
        ]

    else:
        output_shape = [
            [v.pseudo_code() for v in t.shape()] for t in func_attrs["outputs"]
        ]
    return input_shape, output_shape


def device_copy(dst_tensor: Tensor, src_tensor: Tensor, dst_idx: int) -> str:
    src_name = src_tensor._attrs["name"]
    dst_ptr = f"params_[{dst_idx}].ptr"
    shape = ["1"]
    for dim in dst_tensor._attrs["shape"]:
        if isinstance(dim, IntImm):
            shape.append(str(dim._attrs["values"][0]))
        else:
            shape.append(dim._attrs["name"])
    shape = "*".join(shape)
    size = f"{shape} * {get_dtype_size(dst_tensor._attrs['dtype'])}"
    return f"DEVICE_CHECK(DeviceToDeviceCopy({dst_ptr}, {src_name}, {size}, stream));"


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


class ModelContainerGenerator:
    def __init__(
        self,
        max_blob_size: int,
        max_constant_blob_size: int,
        workspace: Workspace,
        constants_data_file: Optional[io.BytesIO],
        graph: List[Tensor],
        output_tensors: List[Tensor],
        model_name: str = MODEL_NAME,
        additional_unbound_constants: Optional[list[Tensor]] = None,
        debug_settings: Optional[AITDebugSettings] = None,
    ):
        self.target = Target.current()
        self.f_var_decl = registry.get(self.target.name() + ".lib.var_decl")
        self.f_ptr_decl = registry.get(self.target.name() + ".lib.void_ptr_decl")

        self.constants_data_file = constants_data_file

        self.exist_funcs = set()
        self.func_decl = []
        self.tensor_slice = []
        self.tensor_map_set = []
        self.set_inputs = []
        self.func_name_seq = []
        self.func_seq = []
        self._input_shape_seq = []
        self._output_shape_seq = []
        self.tensor_decl = []
        self.dim_decl = []
        self.jagged_decl = []
        self.device_to_device_copies = []
        self.function_state = []
        self.set_up_constants = []
        self.set_up_param_names = []
        self.set_up_param_dtypes = []
        self.set_up_bound_constant_dtypes = []
        self.set_up_bound_constant_size = []
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
        self.reset_constants = []

        self.set_up_bound_constant_offsets = []
        self.set_up_constant_folding_outputs_offsets = []

        self.input_idx = 0
        self.bound_constant_idx = 0
        self.unbound_constant_idx = 0
        self.output_name_to_idx = _construct_output_name_to_index_map(
            graph, output_tensors
        )
        self.graph = graph

        self.num_inputs, self.num_outputs = count_inputs_outputs(graph)
        (self.max_blob_size, self.max_constant_blob_size, self.workspace,) = (
            max_blob_size,
            max_constant_blob_size,
            workspace,
        )

        self.debug_settings = (
            AITDebugSettings() if debug_settings is None else debug_settings
        )

        # This records whether or not we should debug header.
        self.debug_header = False

        self.model_name = model_name

        # additional_unbound_constants stores tensors that are used in constant folding
        # but are not used in the main graph. We need this info so we can codegen SetConstant
        # correctly; when we call SetConstant for one of these special names, we want to forward
        # to constant_folder_->SetConstant().
        self.additional_unbound_constants = additional_unbound_constants
        self.set_up_constant_folding_inputs = []

        # This is used to handle a corner case; if we have an owned tensor that is used as an input
        # for constant folding, we need to allocate space for it in our constant buffer, but its
        # size won't be found during memory planning.
        self.extra_owned_constant_size = 0

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
            set_value(f"params_[{idx}].shape_ptrs", f"{{{param_shape_init}}}")
        )
        name = tensor._attrs["name"]
        self.set_up_param_names.append(set_value(f"param_names_[{idx}]", f'"{name}"'))
        self.set_up_param_dtypes.append(
            set_value(
                f"param_dtypes_[{idx}]",
                dtype_to_enumerator(tensor.dtype()),
            )
        )

    def _add_owned_constant(self, tensor: Tensor) -> None:
        """
        Add an owned constant, e.g. one with a bound "data" attribute.
        Here, we codegen some extra logic to load it into memory from the .so.
        """
        assert (
            self.constants_data_file is not None
        ), "Cannot add owned constants without a data file"

        name = tensor._attrs["name"]
        data = tensor._attrs["data"]
        assert (
            tensor._attrs["offset"] >= 0
        ), f"Constant node '{name}' must have non-negative offset"
        num_bytes = len(data)
        self.constants_data_file.write(data.to_bytes())

        constant_info = f'ConstantInfo{{"{name}", {self.constants_data_size}, {tensor._attrs["offset"]}, {num_bytes}}}'
        self.owned_constants_init.append(constant_info)
        self.constants_data_size += num_bytes
        self.num_constants += 1

    def _codegen_bound_constant(self, tensor: Tensor) -> None:
        if tensor._attrs.get("is_internal_constant", False):
            return

        name = tensor._attrs["name"]
        self.set_up_constant_names.append(
            set_value(
                f'bound_constant_name_to_idx_["{name}"]',
                self.bound_constant_idx,
            )
        )
        self.set_up_bound_constant_dtypes.append(
            set_value(
                f"bound_constant_dtypes_[{self.bound_constant_idx}]",
                dtype_to_enumerator(tensor.dtype()),
            )
        )
        self.set_up_bound_constant_size.append(
            set_value(
                f"bound_constant_size_[{self.bound_constant_idx}]",
                len(tensor._attrs["data"]),
            )
        )
        self.set_up_bound_constant_offsets.append(
            set_value(
                f"bound_constant_offsets_[{self.bound_constant_idx}]",
                tensor._attrs["offset"],
            )
        )
        self.bound_constant_idx += 1

    def _codegen_param_setup(
        self,
        tensor: Tensor,
    ) -> None:
        """
        Generate code needed for setting up a constant in Model/ModelContainer.
        """
        name = tensor._attrs["name"]
        data = tensor._attrs["data"]
        const_slice = self._tensor_slice_func(tensor, "constants")
        if data is not None:
            # Owned constant. Set up logic for copying the constant in from *.so.
            self.set_up_constants.append(const_slice)
            self.set_up_constants.append(
                set_value(
                    f'constant_name_to_ptr_["{name}"]',
                    f"const_cast<const void**>(reinterpret_cast<void**>(&{name}))",
                )
            )
            self._codegen_bound_constant(tensor)
            self.reset_constants.append(const_slice)
            if self.constants_data_file is not None:
                self._add_owned_constant(tensor)
        elif tensor._attrs["constant_folding_output_idx"] is not None:
            self.set_up_constant_folding_outputs_offsets.append(
                set_value(
                    f'constant_folding_outputs_offsets_[{tensor._attrs["constant_folding_output_idx"]}]',
                    tensor._attrs["offset"],
                )
            )
            self.tensor_slice.append(const_slice)
            self.reset_constants.append(const_slice)
        elif not isinstance(tensor, IntVarTensor):
            # Unbound constant. We will expect the user to set this via SetConstant.
            self.set_up_constant_names.append(
                set_value(
                    f'unbound_constant_name_to_idx_["{name}"]',
                    self.unbound_constant_idx,
                )
            )
            self._record_param_tensor_info(
                tensor,
                self.unbound_constant_idx + self.num_inputs + self.num_outputs,
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
                f"static_cast<decltype({name})>(params_[{self.input_idx}].ptr)",
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
        if is_view and len(self.param_name_to_ptr_idx) > 0:
            ptr_idx = self.param_name_to_ptr_idx[view._attrs["name"]]
            self.set_inputs.append(set_value(name, view._attrs["name"]))
        else:
            ptr_idx = self._get_output_idx(name)
            self.set_inputs.append(
                set_value(
                    name,
                    f"static_cast<decltype({name})>(params_[{ptr_idx}].ptr)",
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
                    f"static_cast<decltype({name})>(params_[{self.input_idx}].ptr)",
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

    def _process_jagged_dims(self, node: Tensor) -> None:
        # JaggedIntVars are processed separately here (besides being processed
        # like normal IntVars in _process_dims above), as they require adding
        # the offset structure declaration into the Model codegen, as well as
        # the batch_dim if it's not set when processing other tensors that
        # directly contain the batch_dim it in their shapes
        jagged_int_var = node._attrs["shape"][0]
        name = jagged_int_var._attrs["name"]

        # we use the key with a prefix here, as the JaggedIntVar's name
        # is identical to the name of the total_length it is based on,
        # which might have been traversed already
        key = f"jagged_int_var_{name}"
        if key not in self.visited_dims:
            for i, jagged_dim in enumerate(jagged_int_var.jagged_dims()):
                if jagged_dim.offsets() is None:
                    raise RuntimeError(
                        f"No offsets Tensor is associated with the JaggedDim {i} in "
                        f"the JaggedIntVar {name}: can't generate offset-related code."
                    )
            self.jagged_decl.append(
                f"   {jagged_int_var.offsets_struct_type()} "
                f"{jagged_int_var.offsets_var_name()};"
            )
            self.visited_dims.add(key)

        batch_dim_name = jagged_int_var.batch_dim()._attrs["name"]
        if batch_dim_name not in self.visited_dims:
            self.dim_decl.append(self.f_var_decl(batch_dim_name, 0))
            self.visited_dims.add(batch_dim_name)

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
        if len(funcs) == 0:
            return

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
                seq = f_func_call(func._attrs, indent="    ")
                if self.debug_settings.gen_profiler_annotation:
                    seq = f'  {{\n  RAII_ProfilerRange _raiiOpProfilerRange("{func._attrs["outputs"][0]._attrs["name"]}");\n{seq}\n  }}'
                self.func_name_seq.append(func._attrs["original_name"])
                self.func_seq.append(seq)
                input_shape, output_shape = extract_input_output_shapes(func._attrs)
                self._input_shape_seq.append(input_shape)
                self._output_shape_seq.append(output_shape)

            if "int_state_flag" in func._attrs:
                if func._attrs["name"] not in self.state_record:
                    self.function_state.append(
                        f'  int64_t {func._attrs["name"]}_state {{0}};'
                    )
                    self.state_record.add(func._attrs["name"])
            self._process_dims_for_op(func)

        if self.debug_settings.check_all_nan_and_inf or node._attrs.get(
            "check_nan_and_inf", False
        ):
            self._append_check_nan_and_inf(node)
        if self.debug_settings.check_all_outputs or node._attrs.get(
            "check_outputs", False
        ):
            self._append_check_outputs(node)

    def _append_check_nan_and_inf(self, node: Tensor):
        self.debug_header = True
        tensor_name = node._attrs["name"]
        elem_cnt = "*".join([shape.pseudo_code() for shape in node.shape()])
        self.func_name_seq.append("nan_and_inf_check")
        self.func_seq.append(
            f'    InvokeInfAndNanChecker(reinterpret_cast<half*>({tensor_name}), "{tensor_name}", {elem_cnt}, stream);\n'
        )

    def _append_check_outputs(self, node: Tensor):
        self.debug_header = True
        tensor_name = node._attrs["name"]
        elem_cnt = "*".join([shape.pseudo_code() for shape in node.shape()])
        self.func_name_seq.append("output_check")
        self.func_seq.append(
            f'    InvokeOutputsChecker(reinterpret_cast<half*>({tensor_name}), "{tensor_name}", {elem_cnt}, stream);\n'
        )

    def append_tensor(self, node: Tensor) -> None:
        if node._attrs["nop"]:
            return
        name = node._attrs["name"]
        dtype = node._attrs["dtype"]
        if isinstance(node, IntVarTensor):
            int_var = node._attrs["int_var"]
            if isinstance(int_var, IntImm):
                self.tensor_decl.append(
                    self.f_var_decl(name=name, value=int_var._attrs["values"][0])
                )
            else:
                self.tensor_decl.append(self.f_var_decl(name=name))
        else:
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
        elif not is_view and not isinstance(node, IntVarTensor):
            # Normal, internal tensor that is not a view: point it to the
            # internal blob of memory
            assert (
                node._attrs["offset"] >= 0
            ), f"Non-parameter node '{name}' must have non-negative offset"
            self.tensor_slice.append(self._tensor_slice_func(node, "blob_ptr"))
        elif not isinstance(node, IntVarTensor):
            # Normal view, point it to the same memory as whatever it
            # aliases
            self.set_inputs.append(set_value(name, view._attrs["name"]))

        self._process_dims_for_tensor(node)
        self._process_src_ops(node)

        if node.is_jagged():
            self._process_jagged_dims(node)

    def generate_model(self) -> str:
        # Disable graph mode on ROCM because the updating operations
        # are not supported
        target_has_graph_mode = "true" if self.target.name() == "cuda" else "false"

        per_op_profiler_seq = zip(
            self.func_name_seq,
            self.func_seq,
            self._input_shape_seq,
            self._output_shape_seq,
        )
        return MODEL_TEMPLATE.render(
            model_name=self.model_name,
            function_decl="\n".join(self.func_decl),
            set_inputs="\n".join(self.set_inputs),
            tensor_slice="\n".join(self.tensor_slice),
            tensor_map_set="\n".join(self.tensor_map_set),
            set_up_constants="\n".join(self.set_up_constants),
            device_to_device_copies="\n".join(self.device_to_device_copies),
            set_up_param_dynamic_shapes="\n".join(self.set_up_param_dynamic_shapes),
            function_seq=self.func_seq,
            per_op_profiler_seq=per_op_profiler_seq,
            tensor_decl="\n".join(self.tensor_decl),
            dim_decl="\n".join(self.dim_decl),
            jagged_decl="\n".join(self.jagged_decl),
            function_state="\n".join(self.function_state),
            target_has_graph_mode=target_has_graph_mode,
            unique_workspace_size=self.workspace.unique_size,
            debug_header=self.debug_header,
            blob_size=self.max_blob_size,
            workspace_size=self.workspace.total_size(),
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            param_size=self.max_constant_blob_size + self.extra_owned_constant_size,
            num_unbound_constants=self.unbound_constant_idx,
            reset_constants="\n".join(self.reset_constants),
            profiler_annotation=self.debug_settings.gen_profiler_annotation,
        )

    def _create_set_up_constant_offsets(self) -> str:
        """
        bound_constant_offsets_ stores a map for each constant to the offset in constant buffer,
        constant_folding_outputs_offsets_ stores a map from each output of constant folding
        to its offset inside the constant buffer.


        When the model is loaded, we use these offsets to wire up the constant folding output
        pointers to the outputs of the constant folder.
        """
        constant_offsets = ""
        if self.set_up_constant_folding_outputs_offsets:
            constant_offsets = jinja2.Template(
                """
    constant_folding_outputs_offsets_.resize({{num_constant_folding_outputs}});
    {{set_up_statements}}
    """
            ).render(
                num_constant_folding_outputs=len(
                    self.set_up_constant_folding_outputs_offsets
                ),
                set_up_statements="\n".join(
                    self.set_up_constant_folding_outputs_offsets
                ),
            )
            constant_offsets += "\n"
        if self.set_up_bound_constant_offsets:
            constant_offsets += jinja2.Template(
                """
    bound_constant_offsets_.resize({{num_bound_constant_offsets}});
    {{set_up_statements}}
    """
            ).render(
                num_bound_constant_offsets=len(self.set_up_bound_constant_offsets),
                set_up_statements="\n".join(self.set_up_bound_constant_offsets),
            )
        return constant_offsets

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

        result["model-generated.h"] = self.generate_model()

        model_container_src_fname = f"model_container_base{self.target.src_extension()}"

        model_container_base_src = MODEL_CONTAINER_TEMPLATE.render(
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            param_size=self.max_constant_blob_size + self.extra_owned_constant_size,
            set_up_constant_names="\n".join(self.set_up_constant_names),
            set_up_param_dtypes="\n".join(self.set_up_param_dtypes),
            set_up_bound_constant_dtypes="\n".join(self.set_up_bound_constant_dtypes),
            set_up_bound_constant_size="\n".join(self.set_up_bound_constant_size),
            set_up_output_shapes="\n".join(self.set_up_output_shapes),
            set_up_param_names="\n".join(self.set_up_param_names),
            num_constants=self.num_constants,
            num_bound_constants=self.bound_constant_idx,
            num_unbound_constants=self.unbound_constant_idx,
            owned_constants_init=",".join(self.owned_constants_init),
            set_up_constant_offsets=self._create_set_up_constant_offsets(),
            set_up_constant_folding_inputs="\n".join(
                self.set_up_constant_folding_inputs
            ),
        )
        result[model_container_src_fname] = model_container_base_src
        return result

    def add_constant_folding_input(self, tensor: Tensor):
        """
        Handle an input to constant fold
        Handle an input to constant folding, e.g. a constant that is
        no longer part of the main graph
        """
        name = tensor._attrs["name"]

        if tensor._attrs["data"] is None:
            self.set_up_constant_names.append(
                set_value(
                    f'unbound_constant_name_to_idx_["{name}"]',
                    self.unbound_constant_idx,
                )
            )
            self._record_param_tensor_info(
                tensor,
                self.unbound_constant_idx + self.num_inputs + self.num_outputs,
            )
            self.unbound_constant_idx += 1
            self.set_up_constant_folding_inputs.append(
                f'constant_folding_inputs_.insert("{name}");'
            )
        else:
            self._add_owned_constant(tensor)
            self._codegen_bound_constant(tensor)
            self.set_up_constant_folding_inputs.append(
                f'constant_folding_optional_inputs_.insert("{name}");'
            )

        self._process_dims_for_tensor(tensor)

    def append_all_tensors(self) -> None:
        if self.additional_unbound_constants is not None:
            for tensor in self.additional_unbound_constants:
                self.add_constant_folding_input(tensor)
                self.extra_owned_constant_size += tensor.size_bytes(alignment=64)

        for tensor in self.graph:
            if tensor._attrs["is_param"] and tensor._attrs["offset"] is not None:
                # Make sure we leave room for the tensors that constant folding
                # needs. These have been excluded from the final graph, so
                # the memory planning pass will not have known about them.
                tensor._attrs["offset"] += self.extra_owned_constant_size

            self.append_tensor(tensor)


_DEBUG_SETTINGS = AITDebugSettings()


def gen_library_src(  # noqa: C901
    sorted_graph: list[Tensor],
    max_blob_size: int,
    max_constant_blob_size: int,
    workspace: Workspace,
    workdir: str,
    output_tensors: List[Tensor],
    model_name: str = "",
    debug_settings: AITDebugSettings = _DEBUG_SETTINGS,
    additional_unbound_constants: Optional[list[Tensor]] = None,
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
    debug_settings : AITDebugSettings
        specify debug settings such as where to dump AITemplate model Python file, etc.

    Returns
    -------
    list[Tuple[str, str]]
        List of tuple (source file path, object file path)
    """

    def to_obj_name(name: str):
        name, _ = os.path.splitext(name)
        return f"{name}.obj"

    prefix = os.path.join(workdir, model_name)
    constants_fname = os.path.join(prefix, "constants.bin")
    constants_data_file = open(constants_fname, "wb")

    model_container_generator = ModelContainerGenerator(
        max_blob_size,
        max_constant_blob_size,
        workspace,
        constants_data_file,
        sorted_graph,
        output_tensors,
        additional_unbound_constants=additional_unbound_constants,
        debug_settings=debug_settings,
    )
    model_container_generator.append_all_tensors()
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

    _LOGGER.info(f"generated {len(to_build)} library srcs")
    return to_build
