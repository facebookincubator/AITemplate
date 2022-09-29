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
Codegen functions for max_pool2d.
"""
import jinja2

from ... import registry
from . import pool2d

# pylint: disable=C0103,C0415,W0613,C0301,W0612


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}max_pooling_launcher<{{kernel_size}}, {{stride}}, {{padding}}>(
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
extern __shared__ char* shared_mem[];

template <int kernel_size,
          int stride,
          int padding,
          int block_ch,
          int block_h,
          int block_w>
__global__ void max_pool_f16_nhwc_kernel(const half2* input,
                                         half2* output,
                                         const int N,
                                         const int H,
                                         const int W,
                                         const int C,
                                         const int HO,
                                         const int WO) {
  half2* shm = (half2*)shared_mem;
  const int ldg_h = (block_h - 1) * stride + kernel_size;
  const int ldg_w = (block_w - 1) * stride + kernel_size;
  const int ldg_hw_num = ldg_h * ldg_w;

  const int n_idx = blockIdx.x;
  const int out_h_start_idx = blockIdx.y * block_h;
  const int out_w_start_idx = blockIdx.z * block_w;

  int ldg_h_start_idx = out_h_start_idx * stride - padding;

  int ldg_w_start_idx = out_w_start_idx * stride - padding;

  input += n_idx * H * W * C;

  const int hw_start_idx_of_thread = threadIdx.y;
  const int ch_thread_idx = threadIdx.x;

  const half2 min = {static_cast<half>(-65503.0f),
                     static_cast<half>(-65503.0f)};

  for (int i = hw_start_idx_of_thread; i < ldg_hw_num; i += block_ch) {
    const int shm_h_idx = i / ldg_w;
    const int shm_w_idx = i % ldg_w;
    const int input_h_idx = ldg_h_start_idx + shm_h_idx;
    const int input_w_idx = ldg_w_start_idx + shm_w_idx;
    const int input_idx = (input_h_idx * W + input_w_idx) * C + ch_thread_idx;
    const int shm_idx = i * C + ch_thread_idx;
    if (input_h_idx >= 0 && input_h_idx < H && input_w_idx >= 0 &&
        input_w_idx < W) {
      shm[shm_idx] = __ldg(input + input_idx);
    } else {
      shm[shm_idx] = min;
    }
  }

  __syncthreads();

  for (int i = hw_start_idx_of_thread; i < block_h * block_w; i += block_ch) {
    const int out_h_offset = i / block_w;
    const int out_w_offset = i % block_w;
    const int out_h_idx = out_h_start_idx + out_h_offset;
    const int out_w_idx = out_w_start_idx + out_w_offset;
    if (out_h_idx >= 0 && out_h_idx < HO && out_w_idx >= 0 &&
        out_w_idx < WO) {
      half2 max = min;

      const int shm_h_start_idx = out_h_offset * stride;
      const int shm_h_end_idx = shm_h_start_idx + kernel_size;
      const int shm_w_start_idx = out_w_offset * stride;
      const int shm_w_end_idx = shm_w_start_idx + kernel_size;

      for (int shm_h_idx = shm_h_start_idx; shm_h_idx < shm_h_end_idx;
           shm_h_idx++) {
        #pragma unroll
        for (int shm_w_idx = shm_w_start_idx; shm_w_idx < shm_w_end_idx;
             shm_w_idx++) {
          const int shm_idx =
              (shm_h_idx * ldg_w + shm_w_idx) * C + ch_thread_idx;
          const half2 tmp = shm[shm_idx];
          max.x = (tmp.x > max.x) ? tmp.x : max.x;
          max.y = (tmp.y > max.y) ? tmp.y : max.y;
        }
      }
      output[((n_idx * HO + out_h_idx) * WO + out_w_idx) * C +
             ch_thread_idx] = max;
    }
  }
}

template <int kernel_size, int stride, int pad>
void max_pooling_launcher(cutlass::half_t* input,
                          cutlass::half_t* output,
                          int NI,
                          int HI,
                          int WI,
                          int CI,
                          int HO,
                          int WO,
                          cudaStream_t stream) {
  const int block_ch = 4;
  const int block_w = 4;
  const int block_h = 4;
  const size_t shm_size = ((block_h - 1) * stride + kernel_size) *
                          ((block_w - 1) * stride + kernel_size) * CI *
                          sizeof(half);
  dim3 grid(NI, (HO + block_h - 1) / block_h,
            (WO + block_w - 1) / block_w);
  dim3 block(CI / 2, block_ch);
  max_pool_f16_nhwc_kernel<kernel_size, stride, pad, 4, 4, 4>
      <<<grid, block, shm_size, stream>>>((const half2*)input, (half2*)output, NI, HI,
                                  WI, CI / 2, HO, WO);
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
      "Unsupported workload for this max pool2d specialization."
  );
}
"""
)


@registry.reg("cuda.max_pool2d.gen_function")
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


@registry.reg("cuda.max_pool2d.func_decl")
def avg_pool2d_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return pool2d.gen_function_decl(func_name)


@registry.reg("cuda.max_pool2d.func_call")
def avg_pool2d_gen_function_call(func_attrs, indent="  "):
    return pool2d.gen_function_call(func_attrs, indent)
