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
Dump/Read sorted_graph to/from python code.
"""
import os

from typing import Dict, List, Optional, Tuple, Union

import jinja2

from aitemplate.compiler.base import IntImm, IntVar, IntVarTensor, Operator, Tensor

from aitemplate.compiler.transform import mark_param_tensor, name_graph, toposort

PROGRAM_TEMPLATE = jinja2.Template(
    """import numpy as np

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntImm, IntVar, _HostConstantTensorData, _NumpyConstantTensorData
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor

from aitemplate.utils.serialization.ait_program import AITBasicProgram

class AITProgram(AITBasicProgram):
{{indent}}def __init__(self):
{{indent}}{{indent}}# Inputs of model
{% for input in inputs -%}
{{indent}}{{indent}}{{input}}
{% endfor %}
{{indent}}{{indent}}# End of inputs
{{indent}}{{indent}}# Constants of model
{% for const in consts -%}
{{indent}}{{indent}}{{const}}
{% endfor %}
{{indent}}{{indent}}# End of Constants
{{indent}}{{indent}}self.set_default_constants()
{{indent}}{{indent}}return


{{indent}}def get_constants(self):
{{indent}}{{indent}}ret = {}
{% for k, v in consts_info.items() -%}
{{indent}}{{indent}}ret["{{k}}"] = {{v}}
{% endfor %}
{{indent}}{{indent}}return ret


{{indent}}def get_inputs(self):
{{indent}}{{indent}}ret = {}
{% for k, v in inputs_info.items() -%}
{{indent}}{{indent}}ret["{{k}}"] = {{v}}
{% endfor %}
{{indent}}{{indent}}return ret


{{indent}}def set_default_constants(self):
{{indent}}{{indent}}super().set_default_constants()
{% for const_val in default_const_vals -%}
{{indent}}{{indent}}{{const_val}}
{% endfor %}
{{indent}}{{indent}}# End of set_default_constants
{{indent}}{{indent}}return


{{indent}}def model(self):
{% for op in ops -%}
{{indent}}{{indent}}{{op}}
{% endfor %}
{{indent}}{{indent}}# Set outputs
{% for output in outputs -%}
{{indent}}{{indent}}{{output}}._attrs["name"] = "{{output}}"
{{indent}}{{indent}}{{output}}._attrs["is_output"] = True
{% endfor %}
{{indent}}{{indent}}# End of setting outputs
{{indent}}{{indent}}return {{", ".join(outputs)}}
"""
)

OPS_TEMPLATE = jinja2.Template(
    "{{op_name}} = ops.{{op_type}}({{op_attrs}})({{op_inputs}})"
)
PARAMS_TEMPLATE = jinja2.Template(
    'self.{{input_name}} = Tensor(shape={{tensor_shape}}, name="{{input_name}}", is_input={{is_input}})'
)
DEFAULT_CONST_VAL_TEMPLATE = jinja2.Template(
    "self.{{const_name}}._bind_data(_HostConstantTensorData({{bytes_data}}, '{{dtype}}'))"
)


def _shape_to_str(shapes: List[Union[IntVar, Tensor]], intimm_to_int=False):
    shape_str = "["
    for idx, shape in enumerate(shapes):
        if idx != 0:
            shape_str += ", "
        if isinstance(shape, IntImm):
            if intimm_to_int:
                shape_str += f"{shape.value()}"
            else:
                shape_str += f"IntImm({shape.value()})"
        elif isinstance(shape, IntVar):
            shape_str += (
                f"IntVar({shape._attrs['values']}, name='{shape._attrs['name']}')"
            )
        elif isinstance(shape, Tensor):
            shape_str += shape._attrs["name"]
    shape_str += "]"

    return shape_str


def _retrieve_op_info(op: Operator, params_set) -> Tuple[List, Dict]:
    op_inputs = list(op._attrs["inputs"])
    op_attrs = op._get_op_attributes()

    if op._attrs["op"] == "elementwise":
        # Elementwise might have constants as inputs.
        args = op._attrs["args"]
        tmp_inputs = []
        for arg in args:
            if not arg.is_a_const_num():
                tmp_inputs.append(arg)
            else:
                tmp_inputs.append(str(arg._attrs["value"]))
        op_inputs = tmp_inputs
    elif op._attrs["op"] == "layernorm":
        # normalized_shape in _attrs are Optional[List[IntImm]], we serialize them here.
        default_normalized_shape = op._attrs["default_normalized_shape"]
        normalized_shape = op._attrs["normalized_shape"]
        if default_normalized_shape == normalized_shape:
            op_attrs["normalized_shape"] = default_normalized_shape
        else:
            op_inputs = op_inputs[:3]

            norm_shapes_input = []
            curr_idx = 3
            for s in normalized_shape:
                if isinstance(s, IntImm):
                    norm_shapes_input.append(f"IntImm({s.value()})")
                else:
                    if isinstance(op_inputs[curr_idx], IntVarTensor):
                        input_name = op_inputs[curr_idx]._attrs["name"]
                        if input_name in params_set:
                            input_name = "self." + input_name
                        norm_shapes_input.append(input_name)
                    elif isinstance(op_inputs[curr_idx], IntVar):
                        norm_shapes_input.append(
                            f'IntVar(values={s._attrs["values"]}, name="{s._attrs["name"]}")'
                        )
                    curr_idx += 1

            op_inputs.append(f'[{", ".join(norm_shapes_input)}]')
            op_inputs.append(str(op._attrs["eps"]))
    elif op._attrs["op"] == "split":
        # split has size and dim provided as inputs.
        op_inputs.append(str(op._attrs["split_sizes"]))
        op_inputs.append(str(op._attrs["split_dim"]))
    elif op._attrs["op"].startswith("concatenate"):
        # concatenate takes list as input
        tmp_inputs = []
        for input_ in op_inputs:
            input_name = input_._attrs["name"]
            if input_name in params_set:
                input_name = "self." + input_name
            tmp_inputs.append(input_name)
        op_inputs = [
            f'[{", ".join(tmp_inputs)}]',
            str(op._attrs["concat_dim"]),
        ]
    elif op._attrs["op"] == "reshape":
        # reshape take shape as inputs
        op_inputs = op_inputs[:1]
        shape_str = _shape_to_str(op._attrs["shape"], intimm_to_int=True)

        op_inputs.append(shape_str)
    elif op._attrs["op"].startswith("group_gemm_rcr"):
        # group_gemm takes bundled X,W,(B) as inputs.
        diff = 2
        if op._attrs["op"].startswith("group_gemm_rcr_bias"):
            diff = 3
        inputs_str = "["
        for i in range(0, len(op_inputs), diff):
            if i != 0:
                inputs_str += ", "
            inputs_str += "["
            input_group = op_inputs[i : i + diff]
            input_group_names = []
            for input_ in input_group:
                input_name = input_._attrs["name"]
                if input_name in params_set:
                    input_name = "self." + input_name
                input_group_names.append(input_name)
            inputs_str += ", ".join(input_group_names)
            inputs_str += "]"
        inputs_str += "]"
        op_inputs = [inputs_str]
    elif op._attrs["op"] == "dynamic_slice":
        # dynamic slice provides start/end indices as inputs
        op_inputs.append(str(op._attrs["start_indices"]))
        op_inputs.append(str(op._attrs["end_indices"]))
    elif op._attrs["op"] == "permute":
        # permute takes permuted dimensions as input,
        # but can forward to static shape permute ops
        # that don't (e.g., permute021 or permute102)
        if "dims" in op._attrs:
            op_inputs.append(str(op._attrs["dims"]))

    return op_inputs, op_attrs


def convert_to_default_const_val_str(tensor: Tensor) -> str:
    const_name = tensor._attrs["name"]
    assert const_name is not None, "const name cannot be none."

    return DEFAULT_CONST_VAL_TEMPLATE.render(
        const_name=const_name,
        bytes_data=tensor._attrs["data"].to_bytes(),
        dtype=tensor._attrs["data"].dtype,
    )


def convert_to_param_str(tensor: Tensor) -> str:
    input_name = tensor._attrs["name"]
    assert input_name is not None, "input name cannot be none."

    return PARAMS_TEMPLATE.render(
        input_name=input_name,
        tensor_shape=_shape_to_str(tensor.shape()),
        is_input=tensor._attrs["is_input"],
    )


def convert_to_info_str(shapes: List[Union[IntImm, IntVar]], is_constant=False) -> str:
    info_str_shapes = []
    for shape in shapes:
        if is_constant:
            if not isinstance(shape, IntImm):
                raise RuntimeError(
                    f"Constant got type {type(shape)} can't have non-IntImm input!"
                )
            info_str_shapes.append(str(shape.value()))
        elif isinstance(shape, IntImm):
            info_str_shapes.append(
                f'IntImm(value={shape.value()}, name="{shape._attrs["name"]}")'
            )
        else:
            info_str_shapes.append(
                f'IntVar(values={shape._attrs["values"]}, name="{shape._attrs["name"]}")'
            )
    return f"[{', '.join(info_str_shapes)}]"


def _str_val(v):
    return f'"{v}"' if isinstance(v, str) else v


def convert_to_op_str(op: Operator, params_set) -> str:
    op_inputs, op_attrs = _retrieve_op_info(op, params_set)

    serialized_op_inputs = []
    for input_ in op_inputs:
        if isinstance(input_, Tensor):
            input_name = input_._attrs["name"]
            if input_name in params_set:
                input_name = "self." + input_name
            serialized_op_inputs.append(input_name)
        else:
            # If done being processed as string
            serialized_op_inputs.append(input_)

    return OPS_TEMPLATE.render(
        op_name=", ".join([o._attrs["name"] for o in op._attrs["outputs"]]),
        op_type=op._attrs["op"],
        op_attrs=", ".join([f"{k}={_str_val(v)}" for k, v in op_attrs.items()]),
        op_inputs=", ".join(serialized_op_inputs),
    )


def dump_program(
    sorted_graph: Union[Tensor, List[Tensor]],
    file_path: str,
    indent: str = "    ",
    random_constants: bool = False,
):
    """This function dumps out an AIT sorted graph to an executable python code.

    Parameters
    ----------
    sorted_graph : Union[Tensor, List[Tensor]]
        Final tensor(s) that are associated to the AIT graph.
    file_path: str
        Location for the python file to be dumped.
    indent: str, optional
        The indentation to be used in python code, default is 4 spaces.
    random_constants: bool, optional
        Assign random values for constants, default is False.
    """
    if isinstance(sorted_graph, Tensor):
        sorted_graph = [sorted_graph]

    # Make sure the graph is in correct order and has names and param set correctly.
    sorted_graph = toposort(sorted_graph)
    mark_param_tensor(sorted_graph)
    name_graph(sorted_graph)

    params_set = set()
    inputs_str = []
    consts_str = []
    default_const_vals = []
    op_str = []
    inputs_info = {}
    consts_info = {}
    outputs = []
    visited_ops = set()
    for tensor in sorted_graph:
        if tensor._attrs["is_input"]:
            inputs_str.append(convert_to_param_str(tensor))
            inputs_info[tensor._attrs["name"]] = convert_to_info_str(tensor.shape())
            params_set.add(tensor._attrs["name"])
            continue
        if tensor._attrs["is_param"]:
            # This is the case that the tensor is some constant.
            consts_str.append(convert_to_param_str(tensor))
            consts_info[tensor._attrs["name"]] = convert_to_info_str(
                tensor.shape(), is_constant=True
            )
            if tensor._attrs["data"] is not None and not random_constants:
                default_const_vals.append(convert_to_default_const_val_str(tensor))
            params_set.add(tensor._attrs["name"])
            continue

        if tensor._attrs["is_output"]:
            outputs.append(tensor._attrs["name"])

        src_ops = tensor.src_ops()
        for src_op in src_ops:
            if src_op in visited_ops:
                continue
            visited_ops.add(src_op)
            op_str.append(convert_to_op_str(src_op, params_set))

    program = PROGRAM_TEMPLATE.render(
        indent=indent,
        inputs=inputs_str,
        inputs_info=inputs_info,
        consts_info=consts_info,
        consts=consts_str,
        default_const_vals=default_const_vals,
        ops=op_str,
        outputs=outputs,
    )

    if file_path != "":
        dirs = os.path.dirname(file_path)
        if dirs != "":
            os.makedirs(dirs, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(program)

    return program


def _get_class(file_path: str, class_name: str = "AITProgram"):
    import importlib.util

    spec = importlib.util.spec_from_file_location("AITProgram", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, class_name)()


def get_inputs_from_graph(file_path: str):
    program = _get_class(file_path)
    return program.get_inputs()


def get_program(file_path: str) -> Tuple[Tuple[Tensor], Union[Tensor, List[Tensor]]]:
    program = _get_class(file_path)

    outputs = program.model()
    sorted_graph = toposort(outputs)

    return outputs, sorted_graph


def strip_hardcoded_constants(file_path: str, new_file: Optional[str] = None) -> None:
    program = _get_class(file_path)
    outputs = program.model()
    if new_file:
        file_path = new_file
    dump_program(outputs, file_path, random_constants=True)
