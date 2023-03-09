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
permute for cuda
"""
import os
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.target import Target

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  const void*,
{% for _ in range(input_rank) %}
  int64_t*,
{% endfor %}
  const int*,
  cudaStream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{
{{indent}}int dims[] = {{permutation}};
{{indent}}{{func_name}}(
{{indent}}    {{dst}},
{{indent}}    {{src}},
{% for dim in input_dims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    dims,
{{indent}}    stream
{{indent}});
}
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include <limits>
#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "logging.h"

using bfloat16 = __nv_bfloat16;

namespace {

{{custom_libs}}

} // namespace

void {{func_name}}(
  void* dst,
  const void* src,
{% for i in range(input_rank) %}
  int64_t* dim_{{i}},
{% endfor %}
  const int* permutation,
  cudaStream_t stream
){
    // invoke permute kernel
    int64_t src_dims[] = {
{% for i in range(input_rank - 1) %}
  *dim_{{i}},
{% endfor %}
  *dim_{{input_rank - 1}}
    };
    invokePermute<{{input_rank}}, {{elem_type}}>(dst, src, src_dims, permutation, stream);
}

  """
)


@registry.reg("cuda.permute.gen_function")
def gen_function(func_attrs: Dict[str, Any]) -> str:
    """
    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Attributes from Operator

    Returns
    -------
    str
        Source code for function generated.
    """

    func_name = func_attrs["name"]
    x = func_attrs["inputs"][0]
    rank = x._rank()

    custom_libs = Target.current().get_custom_libs(
        os.path.dirname(__file__), "permute.cuh"
    )
    dtype = x.dtype()
    assert dtype in (
        "float16",
        "bfloat16",
        "float32",
        "float",
    ), "permute is only tested for floating point type"
    backend_type = CUDASpec().dtype_to_backend_dtype[dtype]
    return SRC_TEMPLATE.render(
        func_name=func_name,
        custom_libs=custom_libs,
        input_rank=rank,
        elem_type=backend_type,
    )


@registry.reg("cuda.permute.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    """
    Parameters
    ----------
    func_attrs : dict
        Attributes from Operator
    backend_spec : class
        specifies backend configs

    Returns
    -------
    str
        Function declaration
    """

    func_name = func_attrs["name"]
    x = func_attrs["inputs"][0]
    rank = x._rank()
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_rank=rank,
    )


@registry.reg("cuda.permute.func_call")
def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    """
    Parameters
    ----------
    func_attrs : dict
        Attributes from Operator
    indent : str, optional
        Indentation for function call template, by default "  "

    Returns
    -------
    str
        Driver code for invoking call
    """

    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    input_dims = [f"&{dim._attrs['name']}" for dim in xshape]

    y = func_attrs["outputs"][0]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        dst=y._attrs["name"],
        src=x._attrs["name"],
        input_dims=input_dims,
        permutation="{" + ",".join(str(dim) for dim in func_attrs["dims"]) + "}",
        indent=indent,
    )
