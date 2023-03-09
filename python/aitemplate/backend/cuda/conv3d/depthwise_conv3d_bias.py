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
Codegen functions for depthwise_conv3d_bias.
"""
import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.conv3d import common_bias

# pylint: disable=C0103,C0415,W0613,C0301,W0612

SRC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

#include <algorithm>
#include <limits>
#include <assert.h>

namespace {
#define CUDA_KERNEL_LOOP(i, n)                                                                          \\
    int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;                                         \\
    for (int64_t i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

template <typename scalar_t, typename accscalar_t, typename Telement, int element_in_Tio, int kernel_k, int dil_d>
__global__ void conv_depthwise3d_cuda_kernel(
    const scalar_t * input,
    const {{dtype}}* kernel,
    {% if has_bias %}
    const {{dtype}}* bias,
    {% endif %}
    scalar_t * output,
    int _kT, int _kH, int _kW,
    int strideT, int strideH, int strideW,
    int paddingT, int paddingH, int paddingW,
    int _dilationT, int _dilationH, int _dilationW,
    int iC, int iT, int iH, int iW,
    int oT, int oH, int oW,
    int num_outputs)
{
  int kT = kernel_k > 0? kernel_k: _kT;
  int kH = kernel_k > 0? kernel_k: _kH;
  int kW = kernel_k > 0? kernel_k: _kW;

  int dilationT = dil_d > 0? dil_d: _dilationT;
  int dilationH = dil_d > 0? dil_d: _dilationH;
  int dilationW = dil_d > 0? dil_d: _dilationW;

  const int oC = iC;
  const int channel_multiplier = 1;

  CUDA_KERNEL_LOOP(index, num_outputs) {
    const int out_channel = index  % oC;
    const int out_col = (index / oC) % oW;
    const int out_row = (index / oC / oW) % oH;
    const int out_frame = (index / oC / oW / oH) % oT;
    const int batch = index / oC / oW / oH / oT;

    const int in_channel = out_channel / channel_multiplier;

    const int in_col_start = out_col * strideW - paddingW;
    const int in_row_start = out_row * strideH - paddingH;
    const int in_frame_start = out_frame * strideT - paddingT;

    const int in_offset = in_channel + iC * (in_col_start + iW * (in_row_start + iH * (in_frame_start + iT* batch)));
    const int out_offset = out_channel + oC * (out_col + oW * (out_row + oH * (out_frame + oT* batch)));

    accscalar_t sum[element_in_Tio];
    for (int tk = 0; tk < element_in_Tio; tk++){
        sum[tk] = 0;
    }
    const {{dtype}} *kernel_ptr = kernel + out_channel * element_in_Tio * kT * kH * kW;
    const scalar_t *input_ptr = input + in_offset;
    for (int k_frame = 0; k_frame < kT; ++k_frame) {
      const int in_frame = in_frame_start + k_frame * dilationT;
      for (int k_row = 0; k_row < kH; ++k_row) {
        const int in_row = in_row_start + k_row * dilationH;
        for (int k_col = 0; k_col < kW; ++k_col) {
          const int in_col = in_col_start + k_col * dilationW;
          if (in_frame >= 0 && in_row >= 0 && in_col >= 0 &&
              in_frame < iT && in_row < iH && in_col < iW) {
            scalar_t input_val = __ldg(input_ptr);
            Telement* pack_input = reinterpret_cast<Telement*>(&input_val);

            for (int tk = 0; tk < element_in_Tio; tk++){
              {% if dtype == "half" %}
                accscalar_t op1 = __half2float(pack_input[tk]);
                sum[tk] += op1 * __half2float(kernel_ptr[tk*kT*kH*kW]);
              {% elif dtype == "float" %}
                accscalar_t op1 = pack_input[tk];
                sum[tk] += op1 * kernel_ptr[tk*kT*kH*kW];
              {% endif %}
            }
          }
          kernel_ptr += 1;
          input_ptr += dilationW * iC;
        }
        input_ptr += iC * (iW * dilationH - kW * dilationW);
      }
      input_ptr += iC * iW * (iH * dilationT - kH * dilationH);
    }

    {% if has_bias %}
      const {{dtype}} *bias_ptr = bias + out_channel * element_in_Tio;
    {% endif %}


    scalar_t output_val;
    Telement* pack_output = reinterpret_cast<Telement*>(&output_val);
    for (int tk = 0; tk < element_in_Tio; tk++){
      {% if dtype == "half" %}
        {% if has_bias %}
          pack_output[tk] = __float2half(sum[tk]) + bias_ptr[tk];
        {% else %}
          pack_output[tk] = __float2half(sum[tk]);
        {% endif %}
      {% elif dtype == "float" %}
        {% if has_bias %}
          pack_output[tk] = sum[tk] + bias_ptr[tk];
        {% else %}
          pack_output[tk] = sum[tk];
        {% endif %}
      {% endif %}
    }
    output[out_offset] = output_val;
  }
}

#define NODEF_OR_EQUAL(x, y) ((y) < 0 || (x) == (y))
#define NODEF_OR_EQUAL_3(x, y1, y2, y3) \\
  (NODEF_OR_EQUAL(x, y1) && \\
   NODEF_OR_EQUAL(x, y2) && \\
   NODEF_OR_EQUAL(x, y3))


#define DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION(kernel_k, dil_d)                       \\
  if (NODEF_OR_EQUAL_3(kernel_k, (kernel_t), (kernel_h), (kernel_w)) &&                 \\
      NODEF_OR_EQUAL_3(dil_d, (dilation_t), (dilation_h), (dilation_w))) {              \\
    conv_depthwise3d_cuda_kernel                                        \\
    <scalar_t, accscalar_t, Telement, element_in_Tio, kernel_k, dil_d>  \\
    <<<grid, block, (smem), stream>>>(                                  \\
      (const scalar_t *)input,                                          \\
      weight,                                                           \\
      {% if has_bias %}                                                 \\
      bias,                                                             \\
      {% endif %}                                                       \\
      (scalar_t *)output,                                               \\
      kernel_t, kernel_h, kernel_w,                                     \\
      stride_t, stride_h, stride_w,                                     \\
      padding_t, padding_h, padding_w,                                  \\
      dilation_t, dilation_h, dilation_w,                               \\
      c, t, h, w,                                                       \\
      to, ho, wo,                                                       \\
      num_outputs);                                                     \\
  } else                                                                \\

#define DWCONV3D_FORWARD_DISPATCH_OTHERS                                \\
  {                                                                     \\
    conv_depthwise3d_cuda_kernel                                        \\
    <scalar_t, accscalar_t, Telement, element_in_Tio, -1, -1>           \\
    <<<grid, block, (smem), stream>>>(                                  \\
      (const scalar_t *)input,                                          \\
      weight,                                                           \\
      {% if has_bias %}                                                 \\
      bias,                                                             \\
      {% endif %}                                                       \\
      (scalar_t *)output,                                               \\
      kernel_t, kernel_h, kernel_w,                                     \\
      stride_t, stride_h, stride_w,                                     \\
      padding_t, padding_h, padding_w,                                  \\
      dilation_t, dilation_h, dilation_w,                               \\
      c, t, h, w,                                                       \\
      to, ho, wo,                                                       \\
      num_outputs);}                                                    \\


void conv_depthwise3d_launcher(
    const {{dtype}} * input,
    const {{dtype}} * weight,
    {% if has_bias %}
    const {{dtype}} * bias,
    {% endif %}
    {{dtype}} * output,
    int kernel_t,
    int kernel_h,
    int kernel_w,
    int stride_t,
    int stride_h,
    int stride_w,
    int padding_t,
    int padding_h,
    int padding_w,
    int dilation_t,
    int dilation_h,
    int dilation_w,
    int n,
    int c,
    int t,
    int h,
    int w,
    int to,
    int ho,
    int wo,
    cudaStream_t stream
    ) {

  assert(to > 0);
  assert(ho > 0);
  assert(wo > 0);

  int64_t num_outputs = n * to * ho * wo * c;
  int64_t block = 256;
  int64_t grid = std::min((num_outputs - 1) / block + 1, (int64_t)65536);

  int64_t num_inputs = n * t * h * w * c;
  int64_t num_weights = c * kernel_t * kernel_h * kernel_w;
  int64_t smem = 0;

  // Range check to avoid overflow in CUDA kernels.
  assert((num_inputs <= std::numeric_limits<int32_t>::max()) &&
              "Input tensor is too large.");
  assert((num_outputs <= std::numeric_limits<int32_t>::max()) &&
              "Output tensor is too large.");
  assert((num_weights <= 1024*8) &&
              "Weight tensor is too large.");

  assert((padding_t * 2 + t <= std::numeric_limits<int32_t>::max()) &&
                "Padded input tensor is too large.");
  assert((padding_h * 2 + h <= std::numeric_limits<int32_t>::max()) &&
                "Padded input tensor is too large.");
  assert((padding_w * 2 + w <= std::numeric_limits<int32_t>::max()) &&
                "Padded input tensor is too large.");


  using accscalar_t = float;
{% if dtype == "half" %}
  using Telement = half;
  {% if csize == 0 %}
    using scalar_t = float4;
    c = c/8;
    num_outputs = num_outputs/8;
    #define element_in_Tio 8
  {% elif csize == 2 %}
    using scalar_t = half2;
    c =c/2;
    num_outputs = num_outputs/2;
    #define element_in_Tio 2
  {% else %}
    using scalar_t = half;
    #define element_in_Tio 1
  {% endif %}
{% elif dtype == "float" %}
  using Telement = float;
  {% if csize == 2 %}
    using scalar_t = float2;
    c =c/2;
    num_outputs = num_outputs/2;
    #define element_in_Tio 2
  {% else %}
    using scalar_t = float;
    #define element_in_Tio 1
  {% endif %}
{% endif %}

  DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION(3, 1)
  DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION(-1, 1)
  DWCONV3D_FORWARD_DISPATCH_OTHERS
}

#undef DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION
#undef DWCONV3D_FORWARD_DISPATCH_OTHERS
#undef CUDA_KERNEL_LOOP
} // namespace

void {{function_name}} (
    void* in_ptr,
    void* weight_ptr,
{% if has_bias %}
    void* bias_ptr,
{% endif %}
    void* out_ptr,
    int64_t* p_kt,
    int64_t* p_kh,
    int64_t* p_kw,
    int stride_t,
    int stride_h,
    int stride_w,
    int padding_t,
    int padding_h,
    int padding_w,
    int dilation_t,
    int dilation_h,
    int dilation_w,
    int64_t* p_batch,
    int64_t* p_in_ch,
    int64_t* p_in_t,
    int64_t* p_in_h,
    int64_t* p_in_w,
    int64_t* p_out_ch,
    int64_t* p_out_t,
    int64_t* p_out_h,
    int64_t* p_out_w,
    cudaStream_t stream
) {
  int kt = *p_kt;
  int kh = *p_kh;
  int kw = *p_kw;
  int batch = *p_batch;
  int in_ch = *p_in_ch;
  int in_t = *p_in_t;
  int in_h = *p_in_h;
  int in_w = *p_in_w;
  int out_ch = *p_out_ch;
  int out_t = *p_out_t;
  int out_h = *p_out_h;
  int out_w = *p_out_w;

  conv_depthwise3d_launcher(
    (const {{dtype}}*)in_ptr,
    (const {{dtype}}*)weight_ptr,
    {% if has_bias %}
    (const {{dtype}}*)bias_ptr,
    {% endif %}
    ({{dtype}}*)out_ptr,
    kt,
    kh,
    kw,
    stride_t,
    stride_h,
    stride_w,
    padding_t,
    padding_h,
    padding_w,
    dilation_t,
    dilation_h,
    dilation_w,
    batch,
    in_ch,
    in_t,
    in_h,
    in_w,
    out_t,
    out_h,
    out_w,
    stream
  );

  return;
}
"""
)


@registry.reg("cuda.depthwise_conv3d_bias.gen_function")
def gen_function(func_attrs):
    func_name = func_attrs["name"]
    has_bias = func_attrs["bias"]
    csize = func_attrs["group"] % 8

    backend_spec = CUDASpec()
    dtype = backend_spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])

    return SRC_TEMPLATE.render(
        function_name=func_name,
        csize=csize,
        has_bias=has_bias,
        dtype=dtype,
    )


@registry.reg("cuda.depthwise_conv3d_bias.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return common_bias.gen_function_decl(func_name)


@registry.reg("cuda.depthwise_conv3d_bias.func_call")
def gen_function_call(func_attrs, indent="  "):
    return common_bias.gen_function_call(func_attrs, indent)
