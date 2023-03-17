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
Codegen functions for view ops.
"""
import jinja2

from aitemplate.backend import registry

SRC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>

void {{function_name}} (
    {{input_args}}
    {{output_args}}
) {
  {{shape_functions}}
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)
# indent: 4 spaces
INPUT_ARGS_TEMPLATE = jinja2.Template(
    """
{% for idx in range(input_ndim) %}
    int64_t* in_{{idx}},
{% endfor %}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)
# indent: 4 spaces
OUTPUT_ARGS_TEMPLATE = jinja2.Template(
    """
{% for idx in range(output_ndim - 1) %}
    int64_t* out_{{idx}},
{% endfor %}
    int64_t* out_{{output_ndim - 1}}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
{% for idx in range(input_ndim + output_ndim - 1) %}
  int64_t*,
{% endfor %}
  int64_t*
);
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{% for name in input_names %}
{{indent}}    &{{name}},
{% endfor %}
{% for name in output_names_except_last %}
{{indent}}    &{{name}},
{% endfor %}
{{indent}}    &{{last_output}}
{{indent}});
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


def _is_intvar(func_attrs):
    return func_attrs["is_intvar"] if "is_intvar" in func_attrs else False


@registry.reg("cuda.reshape.gen_function")
@registry.reg("cuda.flatten.gen_function")
def reshape_gen_function(func_attrs, shape_eval_template):
    func_name = func_attrs["name"]
    unknown_idx = func_attrs["unknown_idx"]
    input_ndim = len(func_attrs["inputs"][0]._attrs["shape"])
    if _is_intvar(func_attrs):
        input_ndim = len(func_attrs["inputs"]) - 1
    output_ndim = len(func_attrs["outputs"][0]._attrs["shape"])
    input_args = INPUT_ARGS_TEMPLATE.render(input_ndim=input_ndim)
    output_args = OUTPUT_ARGS_TEMPLATE.render(output_ndim=output_ndim)

    shape_functions = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        input_ndim=input_ndim,
        output_ndim=output_ndim,
        unknown_idx=unknown_idx,
    )

    return SRC_TEMPLATE.render(
        function_name=func_name,
        shape_functions=shape_functions.strip(),
        input_args=input_args.strip(),
        output_args=output_args.strip(),
    )


@registry.reg("cuda.reshape.func_decl")
@registry.reg("cuda.flatten.func_decl")
def reshape_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndim = len(func_attrs["inputs"][0]._attrs["shape"])
    if _is_intvar(func_attrs):
        input_ndim = len(func_attrs["inputs"]) - 1
    output_ndim = len(func_attrs["outputs"][0]._attrs["shape"])

    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name, input_ndim=input_ndim, output_ndim=output_ndim
    )


@registry.reg("cuda.reshape.func_call")
@registry.reg("cuda.flatten.func_call")
def reshape_gen_function_call(func_attrs, indent="  "):
    func_name = func_attrs["name"]
    input_names = []
    if _is_intvar(func_attrs):
        for i, inp in enumerate(func_attrs["inputs"]):
            if i == 0:
                continue
            input_names.append(inp._attrs["int_var"]._attrs["name"])
    else:
        input_names = [
            shape._attrs["name"] for shape in func_attrs["inputs"][0]._attrs["shape"]
        ]

    output_names = [
        shape._attrs["name"] for shape in func_attrs["outputs"][0]._attrs["shape"]
    ]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_name,
        input_names=input_names,
        output_names_except_last=output_names[:-1],
        last_output=output_names[-1],
        indent=indent,
    )


@registry.reg("cuda.squeeze.gen_function")
@registry.reg("cuda.unsqueeze.gen_function")
def squeeze_gen_function(func_attrs, shape_eval_template):
    """Generate the function body squeeze/unsqueeze.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        The _attrs dict from the original op.
    shape_eval_template : jinja2.Template
        The template that implements the logic for writing to dynamic shapes.
    """
    func_name = func_attrs["name"]
    out_dim_to_in = func_attrs["out_dim_to_in"]

    input_ndim = len(func_attrs["inputs"][0]._attrs["shape"])
    output_ndim = len(func_attrs["outputs"][0]._attrs["shape"])

    input_args = INPUT_ARGS_TEMPLATE.render(input_ndim=input_ndim)
    output_args = OUTPUT_ARGS_TEMPLATE.render(output_ndim=output_ndim)

    shape_functions = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        input_ndim=input_ndim,
        output_ndim=output_ndim,
        out_dim_to_in=out_dim_to_in,
    )

    return SRC_TEMPLATE.render(
        function_name=func_name,
        shape_functions=shape_functions.strip(),
        input_args=input_args.strip(),
        output_args=output_args.strip(),
    )


@registry.reg("cuda.squeeze.func_decl")
@registry.reg("cuda.unsqueeze.func_decl")
def squeeze_gen_function_decl(func_attrs):
    """Generate the function declaration for squeeze/unsqueeze.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        The _attrs dict from the original op.
    """
    func_name = func_attrs["name"]
    input_ndim = len(func_attrs["inputs"][0]._attrs["shape"])
    output_ndim = len(func_attrs["outputs"][0]._attrs["shape"])

    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name, input_ndim=input_ndim, output_ndim=output_ndim
    )


@registry.reg("cuda.squeeze.func_call")
@registry.reg("cuda.unsqueeze.func_call")
def squeeze_gen_function_call(func_attrs, indent="  "):
    """Generate the function invocation for squeeze/unsqueeze.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        The _attrs dict from the original op.
    ident : str
        Sequence to use to generate the indentations in the CUDA code
    """
    func_name = func_attrs["name"]
    input_names = [
        shape._attrs["name"] for shape in func_attrs["inputs"][0]._attrs["shape"]
    ]
    output_names = [
        shape._attrs["name"] for shape in func_attrs["outputs"][0]._attrs["shape"]
    ]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_name,
        input_names=input_names,
        output_names_except_last=output_names[:-1],
        last_output=output_names[-1],
        indent=indent,
    )
