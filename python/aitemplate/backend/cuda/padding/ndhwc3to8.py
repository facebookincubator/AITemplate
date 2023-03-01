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
CUDA codegen for ndhwc3to8 op
"""
import jinja2

from ... import registry
from ...backend_spec import CUDASpec

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
{{indent}}    {{p_in_d}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_d}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    stream
{{indent}});
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}ndhwc3to8_launcher<{{elem_input_type}}>(
{{indent}}    static_cast<const {{elem_input_type}}*>(in_ptr),
{{indent}}    static_cast<{{elem_input_type}}*>(out_ptr),
{{indent}}    NI,
{{indent}}    DI,
{{indent}}    HI,
{{indent}}    WI,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

// load 128 bit every time (n ElemT = 4 float)
// use as many as thread with factor of 3:
// each time load num_thread * n ElemT = num_thread / 3 * n ElemT * 3ch ->
// num_thread / 3 * n ElemT * n ElemT ch

template<typename ElemT, int num_thread>
__global__ void ndhwc3to8_kernel(const float4* input,
                                float4* output,
                                const int NI,
                                const int DI,
                                const int HI,
                                const int WI,
                                const int max_in_elements,
                                const int max_out_elements) {
  constexpr int num_elem_t_in_float4 = sizeof(float4) / sizeof(ElemT);
  __shared__ float4 shared_mem[num_thread];
  const int out_offset = num_thread * num_elem_t_in_float4 / 3;
  const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};
  const ElemT zero = static_cast<ElemT>(0.f);
  const int in_idx = blockIdx.x * num_thread + threadIdx.x;
  const int tid = threadIdx.x;

  shared_mem[tid] = in_idx >= max_in_elements ? zero4 : __ldg(input + in_idx);
  __syncthreads();

  const int out_start_idx = blockIdx.x * out_offset;
  const int boundary = out_start_idx + out_offset > max_out_elements ? max_out_elements : out_start_idx + out_offset;
  for (int i = out_start_idx + tid, j = tid; i < boundary; i += num_thread, j += num_thread) {
    const ElemT* smem_element = (const ElemT*)shared_mem + j * 3;
    ElemT tmp[num_elem_t_in_float4];

    #pragma unroll
    for (int k = 0; k < num_elem_t_in_float4; ++k) {
      tmp[k] = k < 3 ? smem_element[k] : zero;
    }
    output[i] = *((const float4*)tmp);
  }
}

template <typename ElemT>
void ndhwc3to8_launcher(const ElemT* in_ptr,
                       ElemT* out_ptr,
                       int NI,
                       int DI,
                       int HI,
                       int WI,
                       cudaStream_t stream) {
  constexpr int num_elem_t_in_float4 = sizeof(float4) / sizeof(ElemT);
  constexpr int nthread = 240;
  const int NDHW = NI * DI * HI * WI;
  if (NDHW % num_elem_t_in_float4 != 0) {
    throw std::runtime_error(
        "NDHW (" + std::to_string(NDHW) + ") mod num_elem_t_in_float4 (" +
        std::to_string(num_elem_t_in_float4) + ") is not 0"
    );
  }
  static_assert(nthread % 3 == 0);
  const int max_in_elements = NDHW * 3 / num_elem_t_in_float4;
  const int max_out_elements = NDHW * num_elem_t_in_float4 / num_elem_t_in_float4;
  dim3 thread_block(nthread);
  dim3 grid((NDHW * 3 + nthread * num_elem_t_in_float4 -1) / (nthread * num_elem_t_in_float4));
  ndhwc3to8_kernel<ElemT, nthread><<<grid, thread_block, 0, stream>>>(
    (const float4*)in_ptr,
    (float4*) out_ptr,
    NI,
    DI,
    HI,
    WI,
    max_in_elements,
    max_out_elements
  );
}

void {{function_name}} (
    void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_d,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_d,
    int64_t* out_h,
    int64_t* out_w,
    cudaStream_t stream
) {
  {{shape_function}}
  {{exec_paths}}
}

"""
)


@registry.reg("cuda.ndhwc3to8.gen_function")
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
        x_dim1="*in_d",
        x_dim2="*in_h",
        x_dim3="*in_w",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_d",
        y_dim2="*out_h",
        y_dim3="*out_w",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = EXEC_TEMPLATE.render(elem_input_type=elem_input_type)
    return SRC_TEMPLATE.render(
        function_name=func_name,
        elem_input_type=elem_input_type,
        shape_function=shape_func,
        exec_paths=exec_paths,
    )


@registry.reg("cuda.ndhwc3to8.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.ndhwc3to8.func_call")
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
        p_in_d="&" + xshape[1]._attrs["name"],
        p_in_h="&" + xshape[2]._attrs["name"],
        p_in_w="&" + xshape[3]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_d="&" + yshape[1]._attrs["name"],
        p_out_h="&" + yshape[2]._attrs["name"],
        p_out_w="&" + yshape[3]._attrs["name"],
        indent=indent,
    )
