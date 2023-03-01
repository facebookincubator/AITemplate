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
Slice reshape backend common implementation.
"""
import functools

import jinja2

from . import slice_common

OUTPUT_DIM_DEF_TEMPLATE = jinja2.Template(
    """
{{indent}}int64_t {{dim_name}} = {{dim_value}};
"""
)

OUTPUT_SHAPE_DEF_TEMPLATE = jinja2.Template(
    """
{{dim_defs}}
{{indent}}  int64_t *{{output_name}}_shape[] = {
{{indent}}    {{output_dim_refs}}
{{indent}}  };
"""
)


def gen_function_decl(func_attrs, backend_spec):
    """Generate function declaration.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec: dataclass
        Backend specification.

    Returns
    -------
    str
        Rendered function declaration.
    """
    return slice_common.gen_function_decl(func_attrs, backend_spec=backend_spec)


def gen_function(
    func_attrs, backend_spec, tanh_def, element_func=None, extra_header_template=None
):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec: dataclass
        Backend specification.
    element_func: str
        Attributes for ease of tanh concatenate fusion, default is None.
    extra_header_template: str
        Header for fast_tanh, default is None.


    Returns
    -------
    str
        Rendered function body.
    """
    # TODO: consider to profile elems_per_thread
    elems_per_thread = 8 if len(func_attrs["inputs"]) == 1 else 256
    element_func_def = None if element_func is None else tanh_def.render()
    # slice_reshape_scatter is a temporary solution for a special fusion pattern.
    # It will be replaced with a more general slice + concat pass once it's
    # ready. Second, the constrains of slice_reshape_scatter ensure that its
    # output_accessor's stride is actually linear offset in the output tensor.
    # So, let's not to pollute a common slice kernel with output TensorAccessors
    # at the moment since we do not support output TensorAccessors for slice
    # op yet, which may have perf implication to the kernel as well.
    output_accessor = func_attrs["output_accessors"][0]
    output_offset = output_accessor.offset
    return slice_common.gen_function(
        func_attrs,
        backend_spec=backend_spec,
        output_offset=output_offset,
        elems_per_thread=elems_per_thread,
        update_output_shape=False,
        element_func=element_func,
        element_func_def=element_func_def,
        extra_header_template=extra_header_template,
    )


def gen_function_call(func_attrs, backend_spec, indent="  "):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec: dataclass
        Backend specification.
    indent : str, optional
        Indent for template, by default "  ".

    Returns
    -------
    str
        Rendered function call.
    """
    slice_ops = func_attrs["slice_ops"]
    assert len(slice_ops) >= 1
    start_indices = [op._attrs["start_indices"] for op in slice_ops]
    end_indices = [op._attrs["end_indices"] for op in slice_ops]

    y = func_attrs["outputs"][0]
    dims = [d._attrs["values"][0] for d in y._attrs["shape"]]
    scatter_dim = func_attrs["scatter_dim"]
    output_shape_dims = []
    output_shape_dim_defs = []
    new_dims = dims[:scatter_dim]
    remaining_dim = functools.reduce(lambda a, b: a * b, dims[scatter_dim:])
    new_dims.append(remaining_dim)
    for i, dim in enumerate(new_dims):
        dim_name = "output_dim_{}".format(i)
        output_shape_dims.append(dim_name)
        dim_def = OUTPUT_DIM_DEF_TEMPLATE.render(
            indent=indent, dim_name=dim_name, dim_value=dim
        )
        output_shape_dim_defs.append(dim_def)
    y_dim_refs = ", ".join(["&" + dim for dim in output_shape_dims])
    output_shape_def = OUTPUT_SHAPE_DEF_TEMPLATE.render(
        indent=indent,
        dim_defs="".join(output_shape_dim_defs),
        output_name=y._attrs["name"],
        output_dim_refs=y_dim_refs,
    )

    return slice_common.gen_function_call(
        backend_spec,
        func_attrs["name"],
        func_attrs["inputs"],
        func_attrs["outputs"],
        start_indices,
        end_indices,
        dim=scatter_dim,
        indent=indent,
        output_shape_def=output_shape_def,
    )
