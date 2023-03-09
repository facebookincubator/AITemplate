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
A kernel that implements vector_norm
"""

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cuda.reduce import reduce_3d


L2_NORM_PROLOGUE_TEMPLATE = jinja2.Template(
    """
{{indent}}using MultipliesOp = cutlass::multiplies<FragmentCompute>;
{{indent}}MultipliesOp multiplies;
{{indent}}return multiplies(fragment, fragment);
"""
)


L2_NORM_EPILOGUE_SCALAR_TEMPLATE = jinja2.Template(
    """
{{indent}}cutlass::NumericConverter<ElementCompute, float> local_converter;
{{indent}}return local_converter(cutlass::fast_sqrt(reduced_result));
"""
)


@registry.reg("cuda.vector_norm.func_decl")
def vector_norm_gen_function_decl(func_attrs) -> str:
    """the registered function for generating vector_norm function declaration

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this vector_norm op

    Returns
    -------
    str
        returns the rendered function declaration with appropriate replacements
    """
    return reduce_3d.gen_function_decl(func_attrs)


@registry.reg("cuda.vector_norm.gen_function")
def vector_norm_gen_function(func_attrs) -> str:
    """the registered function for generating vector_norm kernel and all of
    its auxiliary functions

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this vector_norm op

    Returns
    -------
    str
        returns the rendered code for the complete implementation of this reduce mean op
    """
    ord_kind = func_attrs["ord_kind"]
    if ord_kind == "2":
        prologue_template = L2_NORM_PROLOGUE_TEMPLATE
        epilogue_scalar_template = L2_NORM_EPILOGUE_SCALAR_TEMPLATE
    else:
        raise RuntimeError("unsupported ord_kind {} for vector_norm".format(ord_kind))

    return reduce_3d.gen_function(
        func_attrs, "cutlass::plus", prologue_template, epilogue_scalar_template
    )


@registry.reg("cuda.vector_norm.func_call")
def vector_norm_gen_function_call(func_attrs, indent="  ") -> str:
    """the registered function for generating a function call to vector_norm

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this vector_norm op
    indent : str, optional
        indentation for each line of the rendered code (default "  ")

    Returns
    -------
    str
        returns rendered code for invoking the reduce op
    """
    return reduce_3d.gen_function_call(func_attrs, indent)
