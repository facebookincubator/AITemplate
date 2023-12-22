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
Define index_select codegen and ROCM kernel

Example input:
 - tensor of shape (6,5,4,3,2)
 - dim = 2 (0->6, 1->5, 2->4, 3->3, 4->2)
 - dim_len = 4
 - dim_idxs = [1,2] (numbers taken from interval [0,3])
 - dim_idx_len = 2
 - num_before = 6*5
 - num_after = 3*2

Output tensor has dim (6,5,2,3,2) i.e.
it has 6*5 (num_before) sets of 2 (dim_idx_len) sets of  3*2 (num_after) elements.

Assuming contiguous memory layout of the original tensor (which seems like a base check for bad_tensor),
the first few elements to be selected are at positions [6-11], [12-17] corresponding to dim_idxs values 1 and 2.
Generalized to:
    - Divide global thread_idx by num_after and calculate start of innermost set as the remainder
    - Further divide by dim_idx_len and calculate start of next outer set as the remainder
    - Use the final value as the offset for the outer most set
    - Compute offset and assign to the element denoted by thread idx
    - increment idx by grid stride

Num threads = 256.
Blocks are(N + threads - 1) / threads;

"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import ROCMSpec


header_files = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
"""

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    {{input_type}}* /*output*/,
    const {{input_type}}* /*input*/,
    const {{index_type}} /*dim*/,
    const {{index_type}} /*dim_len*/,
    const {{index_type}}* /*dim_idxs*/,
    const {{index_type}} /*dim_idxs_len*/,
    const {{index_type}} /*num_before*/,
    const {{index_type}} /*num_after*/,
    hipStream_t /*stream*/
    );
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

__global__ void index_select_kernel(
    {{input_type}}* output,
    const {{input_type}}* input,
    const {{index_type}} dim,
    const {{index_type}} dim_len,
    const {{index_type}}* dim_idxs,
    const {{index_type}} dim_idxs_len,
    const {{index_type}} num_before,
    const {{index_type}} num_after,
    const {{index_type}} N
) {
    auto idx = blockIdx.x*blockDim.x + threadIdx.x;
    #pragma unroll
    for(auto i = idx; i<N; i+=gridDim.x*blockDim.x) {
        auto res = i;
        auto k = i%num_after;
        res = res/num_after;
        auto j = res%dim_idxs_len;
        res = res/dim_idxs_len;
        auto skip = res*dim_len*num_after + (dim_idxs[j]*num_after) + k;
        output[i] = input[skip];
    }

}

void {{func_name}}(
    {{input_type}}* output,
    const {{input_type}}* input,
    const {{index_type}} dim,
    const {{index_type}} dim_len,
    const {{index_type}}* dim_idxs,
    const {{index_type}} dim_idxs_len,
    const {{index_type}} num_before,
    const {{index_type}} num_after,
    hipStream_t stream
    ) {

    {{index_type}} N =  num_before*dim_idxs_len*num_after;
    const {{index_type}} threads  = 256;
    auto blocks = (N + threads - 1) / threads;

    index_select_kernel<<<blocks, threads, 0, stream>>>(output, input, dim, dim_len, dim_idxs,
        dim_idxs_len, num_before, num_after, N);
}
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  {{index_type}} x_dims[] = {
{{indent}}      {{x_dims}}
{{indent}}  };
{{indent}}  {{index_type}} num_before = 1;
{{indent}}  {{index_type}} num_after = 1;
{{indent}}  {{index_type}} dim_len = x_dims[{{dim}}];
{{indent}}  for(auto i=0;i<{{dim}};i++) {
{{indent}}   num_before *= x_dims[i];
{{indent}}  }
{{indent}}  for(auto i={{dim}}+1;i<sizeof(x_dims)/sizeof(x_dims[0]);i++) {
{{indent}}   num_after *= x_dims[i];
{{indent}}  }
{{indent}}  {{func_name}}(
{{indent}}      {{output}},
{{indent}}      {{input}},
{{indent}}      {{dim}},
{{indent}}      dim_len,
{{indent}}      {{dim_idxs}},
{{indent}}      {{dim_idxs_len}},
{{indent}}      num_before,
{{indent}}      num_after,
{{indent}}      stream
{{indent}}  );
{{indent}}}
"""
)


@registry.reg("rocm.index_select.gen_function")
def gen_function(func_attrs) -> str:
    """
    Generate function body

    Returns
    -------
    str
        The function body string
    """
    backend_spec = ROCMSpec()
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]

    input_type = backend_spec.dtype_to_backend_type(x._attrs["dtype"])
    output_type = backend_spec.dtype_to_backend_type(y._attrs["dtype"])
    if input_type != output_type:
        raise TypeError("input type must equal to output type")

    return SRC_TEMPLATE.render(
        input_type=input_type,
        index_type=backend_spec.index_type,
        func_name=func_attrs["name"],
        header_files=header_files,
    )


@registry.reg("rocm.index_select.func_decl")
def gen_function_decl(func_attrs) -> str:
    """
    Generate function declaration.

    Returns
    -------
    str
        The function declaration string
    """
    backend_spec = ROCMSpec()
    x = func_attrs["inputs"][0]
    input_type = backend_spec.dtype_to_backend_type(x._attrs["dtype"])
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input_type=input_type,
        index_type=backend_spec.index_type,
    )


@registry.reg("rocm.index_select.func_call")
def gen_function_call(func_attrs, indent="  ") -> str:
    """
    Generate function call.

    Returns
    -------
    str
        The function call string
    """
    backend_spec = ROCMSpec()
    x = func_attrs["inputs"][0]
    dim_idxs = func_attrs["inputs"][1]
    y = func_attrs["outputs"][0]
    dim = func_attrs["dim"]

    dtype = backend_spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])

    dim_idxs_ptr = backend_spec.cast_to_ptr_template.render(
        name=dim_idxs._attrs["name"],
        dtype=backend_spec.index_type,
    )
    input_ptr = backend_spec.cast_to_ptr_template.render(
        name=x._attrs["name"],
        dtype=dtype,
    )

    output_ptr = backend_spec.cast_to_ptr_template.render(
        name=y._attrs["name"],
        dtype=dtype,
    )

    x_dims = ", ".join(dim._attrs["name"] for dim in x._attrs["shape"])
    dim_idxs_len = dim_idxs._attrs["shape"][0]._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        index_type=backend_spec.index_type,
        x_dims=x_dims,
        input_type=dtype,
        func_name=func_attrs["name"],
        output=output_ptr,
        input=input_ptr,
        dim=dim,
        dim_idxs=dim_idxs_ptr,
        dim_idxs_len=dim_idxs_len,
    )
