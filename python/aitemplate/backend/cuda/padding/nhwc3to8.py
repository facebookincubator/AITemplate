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
CUDA codegen for nhwc3to8 op
"""
import jinja2

from ... import registry

# pylint: disable=C0301,W0613,W0612

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  cutlass::half_t*,
  cutlass::half_t*,
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
{{indent}}nhwc3to8_launcher(
{{indent}}    in_ptr,
{{indent}}    out_ptr,
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

// load 128 bit every time (8 half = 4 float)
// use as many as thread with factor of 3:
// each time load num_thread * 8 half = num_thread / 3 * 8 * 3ch -> num_thread / 3 * 8 * 8ch

template<int num_thread>
__global__ void nhwc3to8_kernel(const float4* input,
                                float4* output,
                                const int NI,
                                const int HI,
                                const int WI,
                                const int max_in_elements,
                                const int max_out_elements) {
  __shared__ float4 shared_mem[num_thread];
  const int out_offset = num_thread * 8 / 3;
  const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};
  const half zero = static_cast<half>(0.f);
  const int in_idx = blockIdx.x * num_thread + threadIdx.x;
  const int tid = threadIdx.x;

  shared_mem[tid] = in_idx >= max_in_elements ? zero4 : __ldg(input + in_idx);
  __syncthreads();

  const int out_start_idx = blockIdx.x * out_offset;
  const int boundary = out_start_idx + out_offset > max_out_elements ? max_out_elements : out_start_idx + out_offset;
  for (int i = out_start_idx + tid, j = tid; i < boundary; i += num_thread, j += num_thread) {
    const half* smem_element = (const half*)shared_mem + j * 3;
    half tmp[8];

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      tmp[k] = k < 3 ? smem_element[k] : zero;
    }
    output[i] = *((const float4*)tmp);
  }
}

void nhwc3to8_launcher(cutlass::half_t* in_ptr,
                       cutlass::half_t* out_ptr,
                       int NI,
                       int HI,
                       int WI,
                       cudaStream_t stream) {
  const int nthread = 240;
  const int NHW = NI * HI * WI;
  // assert NHW % 8 == 0
  // assert nthread % 3 == 0
  const int max_in_elements = NHW * 3 / 8;
  const int max_out_elements = NHW * 8 / 8;
  dim3 thread_block(nthread);
  dim3 grid((NHW * 3 + nthread * 8 -1) / (nthread * 8));
  nhwc3to8_kernel<nthread><<<grid, thread_block, 0, stream>>>(
    (const float4*)in_ptr,
    (float4*) out_ptr,
    NI,
    HI,
    WI,
    max_in_elements,
    max_out_elements
  );
}

void {{function_name}} (
    cutlass::half_t* in_ptr,
    cutlass::half_t* out_ptr,
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


@registry.reg("cuda.nhwc3to8.gen_function")
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
    exec_paths = EXEC_TEMPLATE.render()
    return SRC_TEMPLATE.render(
        function_name=func_name, shape_function=shape_func, exec_paths=exec_paths
    )


@registry.reg("cuda.nhwc3to8.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.nhwc3to8.func_call")
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
