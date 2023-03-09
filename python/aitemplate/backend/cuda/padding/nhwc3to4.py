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
CUDA codegen for nhwc3to4 op
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec

# pylint: disable=C0301,W0613,W0612

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  cudaStream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{out_ptr}},
{{indent}}    {{p_batch}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    stream
{{indent}});
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}nhwc3to4_launcher<{{elem_input_type}}>(
{{indent}}    static_cast<const {{elem_input_type}}*>(in_ptr),
{{indent}}    static_cast<{{elem_input_type}}*>(out_ptr),
{{indent}}    NI,
{{indent}}    HI,
{{indent}}    WI,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include "logging.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

// fast kernel for c_in = 3 & c_out = 4
template <typename Tio, typename Telement, int element_in_Tio>
__global__ void nhwc_padding_channel_3To4_kernel(const int32_t n,
                                                 const int32_t h,
                                                 const int32_t w,
                                                 const Tio *input,
                                                 Tio *output,
                                                 const int32_t max_output_element,
                                                 const int32_t max_input_element,
                                                 const Tio zero_io,
                                                 const Telement zero_element){
  __shared__ Tio shm[192];
  const int tidx = blockIdx.x * 192 + threadIdx.x;
  const int threadidx = threadIdx.x;

  shm[threadIdx.x] = tidx >= max_input_element ? zero_io : input[tidx];
  __syncthreads();

  const int ouput_offset = blockIdx.x * 256;
  const int lower_bound = max_output_element < ouput_offset + 256 ? max_output_element : ouput_offset + 256;
  for (int i = ouput_offset + threadidx, j = threadidx ; i < lower_bound ; i+=192, j+=192)
  {
    const Telement* shm_element = (const Telement*)shm + j*3*element_in_Tio/4;
    Telement array[element_in_Tio];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0 ; k < element_in_Tio ; k++)
      array[k] = ((k+1)%4 == 0) ? zero_element : shm_element[(k > 3) ? (k - 1) : k];
    output[i] = *((const Tio *)array);
  }
}

template <typename ElemT>
void nhwc3to4_launcher(const ElemT* in_ptr,
                       ElemT* out_ptr,
                       int NI,
                       int HI,
                       int WI,
                       cudaStream_t stream) {
  dim3 block(192);
  const int nhw = NI * HI * WI;
  const int nhwc = nhw * 3;
  CHECK_EQ(nhw % 8, 0);
  const int element_in_Tio = sizeof(int4) / sizeof(ElemT);
  const int max_input_element = nhwc / element_in_Tio;
  const int max_output_element = nhw * 4 / element_in_Tio;
  const int4 zero_io = {0, 0, 0, 0};
  const ElemT zero_element = static_cast<ElemT>(0.0f);
  dim3 grid((nhwc + 192 * element_in_Tio - 1)/(192 * element_in_Tio));
  nhwc_padding_channel_3To4_kernel<int4, ElemT, element_in_Tio><<<grid, block, 0, stream>>>
          (NI, HI, WI,
          (const int4 *)in_ptr,
          (int4 *)out_ptr,
          max_output_element,
          max_input_element,
          zero_io,
          zero_element);
}

void {{function_name}} (
    void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    cudaStream_t stream
) {
  {{shape_function}}
  {{exec_paths}}
}

"""
)


@registry.reg("cuda.nhwc3to4.gen_function")
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
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = EXEC_TEMPLATE.render(elem_input_type=elem_input_type)
    return SRC_TEMPLATE.render(
        function_name=func_name,
        elem_input_type=elem_input_type,
        shape_function=shape_func,
        exec_paths=exec_paths,
    )


@registry.reg("cuda.nhwc3to4.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.nhwc3to4.func_call")
def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        indent=indent,
    )
