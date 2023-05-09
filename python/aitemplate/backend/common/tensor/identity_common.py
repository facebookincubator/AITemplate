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
identity kernel codegen.
"""

from typing import Any, Dict

import jinja2
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.dtype import get_dtype_size

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
{{func_signature}}
{
{% if is_copy %}
    {{prefix}}MemcpyAsync(*output, input, size, {{prefix}}MemcpyDeviceToDevice, stream);
{% else %}
    *output = input;
{% endif %}
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void** output, void* input, size_t size, {{prefix}}Stream_t stream)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   &{{output}},
{{indent}}   {{input}},
{{indent}}   {{size}},
{{indent}}   stream
{{indent}});
    """
)


def gen_function(func_attrs: Dict[str, Any], backend_spec) -> str:
    """Generates function.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    header_files : str
        Includes the header files for a backend.
    backend_spec : class
        Specifies the backend configurations.

    Returns
    -------
    str
        Rendered function.
    """
    is_copy = func_attrs["outputs"][0]._attrs["is_output"]

    return FUNC_TEMPLATE.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
        ),
        prefix=backend_spec.prefix,
        is_copy=is_copy,
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    """Generates function decl.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec : class
        Specifies the backend configurations.

    Returns
    -------
    str
        Rendered function decl.
    """
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
        ),
    ).strip()


def gen_function_call(func_attrs: Dict[str, Any], backend_spec, indent="  ") -> str:
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec : class
        Specifies the backend configurations.
    indent : str, optional
        Indent for template, by default "  ".

    Returns
    -------
    str
        Rendered function call.
    """
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 1

    input_name = func_attrs["inputs"][0]._attrs["name"]

    output_node = func_attrs["outputs"][0]
    output_name = output_node._attrs["name"]
    shape = ["1"]
    for dim in output_node._attrs["shape"]:
        if isinstance(dim, IntImm):
            shape.append(str(dim._attrs["values"][0]))
        else:
            shape.append(dim._attrs["name"])
    shape = "*".join(shape)
    size = f"{shape} * {get_dtype_size(output_node._attrs['dtype'])}"

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        input=input_name,
        size=size,
        indent=indent,
    )
