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
Codegen functions for avg_pool2d.
"""

import jinja2

from ... import registry
from . import pool2d

# pylint: disable=C0103,C0415,W0613,C0301,W0612


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}avg_pool_launcher<{{kernel_size}}, {{stride}}, {{padding}}>(
{{indent}}    in_ptr,
{{indent}}    out_ptr,
{{indent}}    NI,
{{indent}}    HI,
{{indent}}    WI,
{{indent}}    CI,
{{indent}}    HO,
{{indent}}    WO,
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

namespace {

template <int kernel_size, int stride, int padding>
__global__ void avg_pool_f16_nhwc_kernel(const half2* input,
                                         half2* output,
                                         const int N,
                                         const int H,
                                         const int W,
                                         const int C,
                                         const int HO,
                                         const int WO) {
  const int tid = threadIdx.x;
  const int n_idx = blockIdx.x;
  const int out_h_idx = blockIdx.y;
  const int out_w_idx = blockIdx.z;

  int h_start_idx = out_h_idx * stride - padding;
  int h_end_idx = h_start_idx + kernel_size;
  h_start_idx = (h_start_idx < 0) ? 0 : h_start_idx;
  h_end_idx = h_end_idx > H ? H : h_end_idx;

  int w_start_idx = out_w_idx * stride - padding;
  int w_end_idx = w_start_idx + kernel_size;
  w_start_idx = (w_start_idx < 0) ? 0 : w_start_idx;
  w_end_idx = w_end_idx > W ? W : w_end_idx;

  input += n_idx * H * W * C;
  output += ((n_idx * HO + out_h_idx) * WO + out_w_idx) * C;
  const float norm_factor =
      static_cast<float>(1.0f / (kernel_size * kernel_size));
  for (int c_idx = tid; c_idx < C; c_idx += blockDim.x) {
    float2 avg = {0.f, 0.f};
    for (int h = h_start_idx; h < h_end_idx; h++) {
      #pragma unroll
      for (int w = w_start_idx; w < w_end_idx; w++) {
        const int idx = (h * W + w) * C;
        const half2 tmp = __ldg(input + (idx + c_idx));
        avg.x += __half2float(tmp.x);
        avg.y += __half2float(tmp.y);
      }
    }

    avg.x *= norm_factor;
    avg.y *= norm_factor;
    output[c_idx] = __float22half2_rn(avg);
  }
}

template <int kernel_size, int stride, int padding>
void avg_pool_launcher(cutlass::half_t* input,
                      cutlass::half_t* output,
                      const int N,
                      const int H,
                      const int W,
                      const int C,
                      const int HO,
                      const int WO,
                      cudaStream_t stream) {
  int num_thread = (C / 2) < 256 ? C / 2 : 256;
  dim3 grid(N, HO, WO);
  dim3 block(num_thread);
  avg_pool_f16_nhwc_kernel<kernel_size, stride, padding><<<grid, block, 0, stream>>>(
      (const half2*)input, (half2*)output, N, H, W, C / 2, HO, WO);
}
} // namespace

void {{function_name}} (
    cutlass::half_t* in_ptr,
    cutlass::half_t* out_ptr,
    int64_t* batch,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* in_ch,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    cudaStream_t stream
) {
  {{shape_function}}
  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this avg pool2d specialization."
  );
}
"""
)


@registry.reg("cuda.avg_pool2d.gen_function")
def gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]

    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
        x_dim3="*in_ch",
        kernel_h=func_attrs["kernel_size"],
        kernel_w=func_attrs["kernel_size"],
        stride=func_attrs["stride"],
        pad=func_attrs["pad"],
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = ""
    for key in exec_path:
        program = EXEC_TEMPLATE.render(
            indent="    ",
            kernel_size=func_attrs["kernel_size"],
            padding=func_attrs["pad"],
            stride=func_attrs["stride"],
        )
        exec_inst = exec_cond_remplate.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return SRC_TEMPLATE.render(
        function_name=func_name, shape_function=shape_func, exec_paths=exec_paths
    )


@registry.reg("cuda.avg_pool2d.func_decl")
def avg_pool2d_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return pool2d.gen_function_decl(func_name)


@registry.reg("cuda.avg_pool2d.func_call")
def avg_pool2d_gen_function_call(func_attrs, indent="  "):
    return pool2d.gen_function_call(func_attrs, indent)
