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
CUDA concatenate function
"""

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common import concatenate_common
from aitemplate.backend.cuda.tensor import concatenate_fast


def _is_valid_fast_cat(func_attrs):
    """
    Checks whether the call is acceptable for the concatenate
    kernel in concatenate_fast.py
    """

    if "fast_cat" not in func_attrs:
        return False
    if not func_attrs["fast_cat"]:
        return False
    if len(func_attrs["inputs"]) == 0:
        return False
    return True


@registry.reg("cuda.concatenate.func_decl")
def gen_function_decl(func_attrs):
    """Generate function declaration.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    Returns
    -------
    str
        Rendered function declaration.
    """
    # get dtype from orig_x in case actual "inputs" is turned into empty
    # by some transformation
    return concatenate_common.gen_function_decl(
        func_attrs=func_attrs,
        backend_spec=CUDASpec(),
    )


@registry.reg("cuda.concatenate.gen_function")
def gen_function(func_attrs, element_func=None, element_func_def=None):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.

    Returns
    -------
    str
        Rendered function body.
    """
    if _is_valid_fast_cat(func_attrs):
        return concatenate_fast.gen_function(
            func_attrs,
            concatenate_common.SRC_TEMPLATE,
            element_func=element_func,
            element_func_def=element_func_def,
        )
    else:
        return concatenate_common.gen_function(
            func_attrs=func_attrs,
            backend_spec=CUDASpec(),
            element_func=element_func,
            element_func_def=element_func_def,
        )


@registry.reg("cuda.concatenate.func_call")
def gen_function_call(func_attrs, indent="  "):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    indent : str, optional
        Indent for template, by default "  ".

    Returns
    -------
    str
        Rendered function call.
    """
    return concatenate_common.gen_function_call(
        func_attrs=func_attrs,
        backend_spec=CUDASpec(),
        indent="  ",
    )
