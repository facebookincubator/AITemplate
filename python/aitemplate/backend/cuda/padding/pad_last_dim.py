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
Codegen functions for pad_last_dim.
"""
import jinja2

from ... import registry

# pylint: disable=C0301,W0613,W0612

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  cutlass::half_t*,
  cutlass::half_t*,
  {%for i in range(ndim)%}
  int64_t*,
  {% endfor %}
  {%for i in range(ndim)%}
  int64_t*,
  {% endfor %}
  int out_dim,
  cudaStream_t stream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{out_ptr}},
{% for dim in xshape %}
{{indent}}{{dim}},
{% endfor %}
{% for dim in yshape %}
{{indent}}{{dim}},
{% endfor %}
{{indent}}  {{out_dim}},
{{indent}}  stream
{{indent}});
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}padding4d_launcher(
{{indent}}    in_ptr,
{{indent}}    out_ptr,
{%for i in range(4 - ndim)%}
1,
{% endfor %}
{%for i in range(ndim)%}
{{indent}}    *x_dim{{i}},
{% endfor %}
{{indent}}    out_dim,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are not permitted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

namespace {
template <typename T>
__global__ void padding4d_kernel(const T* input,
                                 T* output,
                                 const int32_t x_dim0,
                                 const int32_t x_dim1,
                                 const int32_t x_dim2,
                                 const int32_t x_dim3,
                                 const int32_t out_dim,
                                 const T zero){

  const int32_t idx_jump       = blockDim.x * gridDim.x;
  const int32_t total_elements = x_dim0 * x_dim1 * x_dim2 * out_dim;

  int32_t dim3_idx = 0;
  int32_t dim2_idx = 0;
  int32_t dim1_idx = 0;
  int32_t dim0_idx = 0;
  int32_t residual = 0;

  T value;
  for (int32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += idx_jump) {

    dim3_idx = idx % out_dim;
    if (dim3_idx >= x_dim3){
      value = zero;
    }
    else{
      residual = idx / out_dim;
      dim2_idx = residual % x_dim2;
      residual = residual / x_dim2;
      dim1_idx = residual % x_dim1;
      dim0_idx = residual / x_dim1;
      residual = ((dim0_idx * x_dim1 + dim1_idx) * x_dim2 + dim2_idx) * x_dim3 + dim3_idx;
      value = input[residual];
    }
    output[idx] = value;
  }
}



void padding4d_launcher(cutlass::half_t* in_ptr,
                        cutlass::half_t* out_ptr,
                        const int32_t x_dim0,
                        const int32_t x_dim1,
                        const int32_t x_dim2,
                        const int32_t x_dim3,
                        const int32_t out_dim,
                        cudaStream_t stream) {
  const int block_size = 256;
  if ((out_dim % 2) == 0 && (x_dim3 % 2) == 0) {
    int32_t total_elements = x_dim0 * x_dim1 * x_dim2 * x_dim3 / 2;
    dim3 grid((total_elements + 255) /  block_size);
    dim3 block(block_size);
    const __half2 zero  = {0.0f, 0.0f};
    padding4d_kernel<__half2><<<grid, block, 0, stream>>>(
        (const __half2*)in_ptr, (__half2*)out_ptr,
        x_dim0, x_dim1, x_dim2, x_dim3 / 2,
        out_dim / 2,
        zero
    );
  } else {
    int32_t total_elements = x_dim0 * x_dim1 * x_dim2 * x_dim3;
    dim3 grid((total_elements + 255) /  block_size);
    dim3 block(block_size);
    const __half zero = static_cast<__half>(0.f);
    padding4d_kernel<__half><<<grid, block, 0, stream>>>(
        (const __half*)in_ptr, (__half*)out_ptr,
        x_dim0, x_dim1, x_dim2, x_dim3,
        out_dim,
        zero
    );
  }
}

} // namespace

void {{function_name}} (
    cutlass::half_t* in_ptr,
    cutlass::half_t* out_ptr,
    {%for i in range(ndim)%}
    int64_t* x_dim{{i}},
    {% endfor %}
    {%for i in range(ndim)%}
    int64_t* y_dim{{i}},
    {% endfor %}
    int out_dim,
    cudaStream_t stream
) {
  {{shape_function}}
  {{exec_paths}}
}

"""
)


@registry.reg("cuda.pad_last_dim.gen_function")
def gen_function(func_attrs, template_path, shape_eval_template, shape_save_template):
    """

    Parameters
    ----------
    func_attrs : [type]
        [description]
    template_path : [type]
        [description]
    shape_eval_template : [type]
        [description]
    shape_save_template : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    func_name = func_attrs["name"]
    ndim = func_attrs["ndim"]
    xshape = ["*x_dim%d" % i for i in range(ndim)]
    shape_eval_func = shape_eval_template.render(
        indent="  ", dtype="int64_t ", shape=xshape, out_dim="out_dim"
    )
    yshape = ["*y_dim%d" % i for i in range(ndim - 1)]
    shape_save_func = shape_save_template.render(
        indent="  ", shape=yshape, last_dim="*y_dim%d" % (ndim - 1)
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = EXEC_TEMPLATE.render(ndim=func_attrs["ndim"], indent="  ")
    return SRC_TEMPLATE.render(
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
        ndim=func_attrs["ndim"],
    )


@registry.reg("cuda.pad_last_dim.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return FUNC_DECL_TEMPLATE.render(func_name=func_name, ndim=func_attrs["ndim"])


@registry.reg("cuda.pad_last_dim.func_call")
def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    xshape_args = ["&" + dim._attrs["name"] for dim in xshape]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    yshape_args = ["&" + dim._attrs["name"] for dim in yshape]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        xshape=xshape_args,
        yshape=yshape_args,
        out_dim=func_attrs["out_dim"],
        indent=indent,
    )
