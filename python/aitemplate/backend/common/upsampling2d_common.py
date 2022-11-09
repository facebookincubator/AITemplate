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
Backend-agnostic function templates for upsampling2d.
"""

import jinja2

# pylint: disable=C0103,C0415,W0613,C0301,W0612


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}bilinear_upsampling_launcher(
{{indent}}    static_cast<const {{dtype}}*>(in_ptr),
{% if bias_add %}
  {{indent}}    static_cast<const {{dtype}}*>(res_ptr),
{% endif %}
{{indent}}    static_cast<{{dtype}}*>(out_ptr),
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
{{header_files}}

namespace {
#define GPU_1D_KERNEL_LOOP(i, n) \
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

{% if mode == "bilinear"%}
__global__ void bilinear_upsampling_f16_nhwc_kernel(const half2* input,
                                                    {% if bias_add %}
                                                      const half2* input_res,
                                                    {% endif %}
                                                    half2* output,
                                                    const {{index_type}} batch,
                                                    const {{index_type}} in_height,
                                                    const {{index_type}} in_width,
                                                    const {{index_type}} channels,
                                                    const {{index_type}} out_height,
                                                    const {{index_type}} out_width) {

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);
    const int64_t num_threads = out_height * out_width * channels * batch;

GPU_1D_KERNEL_LOOP(out_idx, num_threads) {
    int64_t idx = out_idx;
    const int64_t c = idx % channels;
    idx /= channels;
    const int64_t x = idx % out_width;
    idx /= out_width;
    const int64_t y = idx % out_height;
    const int64_t b = idx / out_height;

    const float in_y = (static_cast<float>(y) + 0.5f) * height_scale - 0.5f;
    const int64_t top_y_index = in_y > 0.0 ? floorf(in_y) : 0;
    const int64_t bottom_y_index =
        (in_y < in_height - 1) ? ceilf(in_y) : in_height - 1;
    const float y_lerp = in_y - floorf(in_y);

    const float in_x = (static_cast<float>(x) + 0.5f) * width_scale - 0.5f;
    const int64_t left_x_index = in_x > 0.0 ? floorf(in_x) : 0;
    const int64_t right_x_index =
        (in_x < in_width - 1) ? ceilf(in_x) : in_width - 1;
    const float x_lerp = in_x - floorf(in_x);

    const half2 top_left = __ldg(
        input + ((b * in_height + top_y_index) * in_width + left_x_index) *
                   channels +
               c);

    const half2 top_right = __ldg(
        input + ((b * in_height + top_y_index) * in_width + right_x_index) *
                   channels +
               c);
    const half2 bottom_left = __ldg(
        input + ((b * in_height + bottom_y_index) * in_width + left_x_index) *
                   channels +
               c);
    const half2 bottom_right = __ldg(
        input + ((b * in_height + bottom_y_index) * in_width + right_x_index) *
                   channels +
               c);

    float top_x = __half2float(top_left{{half2_data_ref}}.x) + (__half2float(top_right{{half2_data_ref}}.x) - __half2float(top_left{{half2_data_ref}}.x)) * x_lerp;
    float top_y = __half2float(top_left{{half2_data_ref}}.y) + (__half2float(top_right{{half2_data_ref}}.y) - __half2float(top_left{{half2_data_ref}}.y)) * x_lerp;

    float bottom_x = __half2float(bottom_left{{half2_data_ref}}.x) + (__half2float(bottom_right{{half2_data_ref}}.x) - __half2float(bottom_left{{half2_data_ref}}.x)) * x_lerp;;
    float bottom_y = __half2float(bottom_left{{half2_data_ref}}.y) + (__half2float(bottom_right{{half2_data_ref}}.y) - __half2float(bottom_left{{half2_data_ref}}.y)) * x_lerp;;

    float2 out = {0.f, 0.f};
    out.x = top_x + (bottom_x - top_x) * y_lerp;
    out.y = top_y + (bottom_y - top_y) * y_lerp;

    {% if bias_add %}
      output[out_idx] = __hadd2(__float22half2_rn(out), __ldg(input_res + out_idx));
    {% else %}
      output[out_idx] = __float22half2_rn(out);
    {% endif %}
  }

}

{% else %}
template <typename T, typename Telement, int element_in_Tio>
__global__ void nearest_upsampling_f16_nhwc_kernel(const T* input,
                                                    {% if bias_add %}
                                                      const T* input_res,
                                                    {% endif %}
                                                    T* output,
                                                    const {{index_type}} batch,
                                                    const {{index_type}} in_height,
                                                    const {{index_type}} in_width,
                                                    const {{index_type}} channels,
                                                    const {{index_type}} out_height,
                                                    const {{index_type}} out_width) {

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);
    const int64_t nthreads = out_height * out_width * channels * batch;

GPU_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % out_width;
    n /= out_width;
    int out_y = n % out_height;
    n /= out_height;

    const T* bottom_data_n = input + n * channels * in_height * in_width;
    const int in_y =
        max(min(static_cast<int>(
                    floorf((static_cast<float>(out_y) + 0.5f) * height_scale)),
                static_cast<int>(in_height) - 1),
            0);
    const int in_x =
        max(min(static_cast<int>(
                    floorf((static_cast<float>(out_x) + 0.5f) * width_scale)),
                static_cast<int>(in_width) - 1),
            0);
    const int idx = (in_y * in_width + in_x) * channels + c;


  {% if bias_add %}
    T input_val = __ldg(bottom_data_n + idx);
    T input_res_val = __ldg(input_res + index);
    {% if tsize == 1 %}
    output[index] = input_val + input_res_val;

    {% elif tsize == 8 %}
    T output_val;
    Telement* pack_y = reinterpret_cast<Telement*>(&output_val);
    Telement* pack_x = reinterpret_cast<Telement*>(&input_val);
    Telement* pack_res = reinterpret_cast<Telement*>(&input_res_val);
    for (int k = 0 ; k < element_in_Tio ; k++)
      pack_y[k] = pack_x[k] + pack_res[k];
    output[index] =  output_val;

    {% else %}
    T output_val;
    output_val{{half2_data_ref}}.x = input_val{{half2_data_ref}}.x + input_res_val{{half2_data_ref}}.x;
    output_val{{half2_data_ref}}.y = input_val{{half2_data_ref}}.y + input_res_val{{half2_data_ref}}.y;
    output[index] = output_val;
    {% endif %}
  {% else %}
    output[index] = __ldg(bottom_data_n + idx);
  {% endif %}

  }
}

{% endif %}

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template<typename ELEM_T>
void bilinear_upsampling_launcher(const ELEM_T* input,
                    {% if bias_add %}
                      const ELEM_T* input_res,
                    {% endif %}
                      ELEM_T* output,
                      const {{index_type}} N,
                      const {{index_type}} H,
                      const {{index_type}} W,
                      const {{index_type}} C,
                      const {{index_type}} HO,
                      const {{index_type}} WO,
                      {{prefix}}Stream_t stream) {
    const int64_t output_size = N * (C) * HO * WO;
    dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
    dim3 block(512);

{% if mode == "bilinear" %}
    bilinear_upsampling_f16_nhwc_kernel<<<grid, block, 0, stream>>>(
      (const half2 *)input,
      {% if bias_add %}
        (const half2 *)input_res,
      {% endif %}
      (half2 *)output,
      N, H, W, C/2, HO, WO);
{% else %}
    {% if tsize == 1 %}
    nearest_upsampling_f16_nhwc_kernel<half, half, 1><<<grid, block, 0, stream>>>(
      (const half *)input,
      {% if bias_add %}
        (const half *)input_res,
      {% endif %}
      (half *)output,
      N, H, W, C, HO, WO);
    {% elif tsize == 8 %}
    nearest_upsampling_f16_nhwc_kernel<float4, half, 8><<<grid, block, 0, stream>>>(
      (const float4 *)input,
      {% if bias_add %}
        (const float4 *)input_res,
      {% endif %}
      (float4 *)output,
      N, H, W, C/8, HO, WO);
    {% else %}
    nearest_upsampling_f16_nhwc_kernel<half2, half, 2><<<grid, block, 0, stream>>>(
      (const half2 *)input,
      {% if bias_add %}
        (const half2 *)input_res,
      {% endif %}
      (half2 *)output,
      N, H, W, C/2, HO, WO);
    {% endif %}
{% endif %}
}
} // namespace

void {{function_name}} (
    const void* in_ptr,
    {% if bias_add %}
    const void* res_ptr,
    {% endif %}
    void* out_ptr,
    {{index_type}}* batch,
    {{index_type}}* in_h,
    {{index_type}}* in_w,
    {{index_type}}* in_ch,
    {{index_type}}* out_batch,
    {{index_type}}* out_h,
    {{index_type}}* out_w,
    {{prefix}}Stream_t stream
) {
  {{shape_function}}
  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this bilinear upsampling specialization."
  );
}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  const void*,
  {% if bias_add %}
  const void*,
  {% endif %}
  void*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{prefix}}Stream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{% if bias_add %}
{{indent}}    {{res_ptr}},
{% endif %}
{{indent}}    {{out_ptr}},
{{indent}}    {{p_batch}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    stream
{{indent}});
"""
)


def gen_function_decl(func_attrs, backend_spec, bias_add=False):
    """Function declaration generation

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        It describes the operation attributes
    backend_spec : custom class
        It specifies the corresponding backend dtypes of pytorch dtypes for many operations

    Returns
    -------
    str
        Rendered function declaration stmt
    """
    return FUNC_DECL_TEMPLATE.render(
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        func_name=func_attrs["name"],
        bias_add=bias_add,
    )


def gen_alignment(x):
    in_channel = x.shape()[-1].value()
    if in_channel % 8 == 0:
        tsize = 8
    elif in_channel % 4 == 0:
        tsize = 4
    elif in_channel % 2 == 0:
        tsize = 2
    else:
        tsize = 1
    return tsize


def gen_function_call(func_attrs, backend_spec, indent="  ", bias_add=False):
    """Function call generation

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        It describes the operation attributes
    indent : str, optional
        Indent for template, by default "  "

    Returns
    -------
    str
        Rendered function call
    """
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    if bias_add:
        r = func_attrs["inputs"][1]
        return FUNC_CALL_TEMPLATE.render(
            func_name=func_attrs["name"],
            index_type=backend_spec.index_type,
            in_ptr=x._attrs["name"],
            res_ptr=r._attrs["name"],
            out_ptr=y._attrs["name"],
            p_batch="&" + xshape[0]._attrs["name"],
            p_in_ch="&" + xshape[3]._attrs["name"],
            p_in_h="&" + xshape[1]._attrs["name"],
            p_in_w="&" + xshape[2]._attrs["name"],
            p_out_batch="&" + yshape[0]._attrs["name"],
            p_out_h="&" + yshape[1]._attrs["name"],
            p_out_w="&" + yshape[2]._attrs["name"],
            indent=indent,
            bias_add=bias_add,
        )
    else:
        return FUNC_CALL_TEMPLATE.render(
            func_name=func_attrs["name"],
            index_type=backend_spec.index_type,
            in_ptr=x._attrs["name"],
            out_ptr=y._attrs["name"],
            p_batch="&" + xshape[0]._attrs["name"],
            p_in_ch="&" + xshape[3]._attrs["name"],
            p_in_h="&" + xshape[1]._attrs["name"],
            p_in_w="&" + xshape[2]._attrs["name"],
            p_out_batch="&" + yshape[0]._attrs["name"],
            p_out_h="&" + yshape[1]._attrs["name"],
            p_out_w="&" + yshape[2]._attrs["name"],
            indent=indent,
            bias_add=bias_add,
        )
