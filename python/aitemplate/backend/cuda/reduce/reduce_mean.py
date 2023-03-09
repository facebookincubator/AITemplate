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
A reduce_mean kernel implementation
"""

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cuda.reduce import reduce_3d


EPILOGUE_SCALAR_TEMPLATE = jinja2.Template(
    """
{{indent}}return (reduced_result / ElementCompute(num_reduced_elems));
"""
)


@registry.reg("cuda.reduce_mean.func_decl")
def reduce_mean_gen_function_decl(func_attrs) -> str:
    """the registered function for generating reduce_mean function declaration

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce_mean op

    Returns
    -------
    str
        returns the rendered function declaration with appropriate replacements
    """
    return reduce_3d.gen_function_decl(func_attrs)


@registry.reg("cuda.reduce_mean.gen_function")
def reduce_mean_gen_function(func_attrs) -> str:
    """the registered function for generating reduce_mean kernel and all of
    its auxiliary functions

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce_mean op

    Returns
    -------
    str
        returns the rendered code for the complete implementation of this reduce mean op
    """
    return reduce_3d.gen_function(
        func_attrs,
        "cutlass::plus",
        reduce_3d.DEFAULT_PROLOGUE_TEMPLATE,
        EPILOGUE_SCALAR_TEMPLATE,
    )


@registry.reg("cuda.reduce_mean.func_call")
def reduce_mean_gen_function_call(func_attrs, indent="  ") -> str:
    """the registered function for generating a function call to reduce_mean

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce_mean op
    indent : str, optional
        indentation for each line of the rendered code (default "  ")

    Returns
    -------
    str
        returns rendered code for invoking the reduce op
    """
    return reduce_3d.gen_function_call(func_attrs, indent)
