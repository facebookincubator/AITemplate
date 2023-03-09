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
Dynamic slice CUDA implementation.
"""

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common.tensor import slice_common


@registry.reg("cuda.dynamic_slice.func_decl")
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
    return slice_common.gen_function_decl(func_attrs, backend_spec=CUDASpec())


@registry.reg("cuda.dynamic_slice.gen_function")
def gen_function(func_attrs):
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
    return slice_common.gen_function(
        func_attrs, backend_spec=CUDASpec(), elems_per_thread=8
    )


@registry.reg("cuda.dynamic_slice.func_call")
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
    return slice_common.gen_function_call(
        backend_spec=CUDASpec(),
        func_name=func_attrs["name"],
        inputs=func_attrs["inputs"],
        outputs=func_attrs["outputs"],
        start_indices=[func_attrs["start_indices"]],
        end_indices=[func_attrs["end_indices"]],
        dim=0,
        indent=indent,
    )
