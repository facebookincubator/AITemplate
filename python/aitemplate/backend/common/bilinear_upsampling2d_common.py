# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""

import jinja2

# pylint: disable=C0103,C0415,W0613,C0301,W0612


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}bilinear_upsampling_luncher(
{{indent}}    in_ptr,
{% if bias_add %}
  {{indent}}    res_ptr,
{% endif %}
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
{{header_files}}

namespace {
#define GPU_1D_KERNEL_LOOP(i, n) \
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

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

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

void bilinear_upsampling_luncher({{elem_input_type}}* input,
                    {% if bias_add %}
                      {{elem_input_type}}* input_res,
                    {% endif %}
                      {{elem_output_type}}* output,
                      const {{index_type}} N,
                      const {{index_type}} H,
                      const {{index_type}} W,
                      const {{index_type}} C,
                      const {{index_type}} HO,
                      const {{index_type}} WO,
                      {{prefix}}Stream_t stream) {
    const int64_t kThreadsPerBlock = 256;
    const int64_t output_size = N * (C) * HO * WO;
    dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
    dim3 block(512);
    bilinear_upsampling_f16_nhwc_kernel<<<grid, block, 0, stream>>>(
      (const half2 *)input,
      {% if bias_add %}
        (const half2 *)input_res,
      {% endif %}
      (half2 *)output,
      N, H, W, C/2, HO, WO);
}
} // namespace

void {{function_name}} (
    {{elem_input_type}}* in_ptr,
    {% if bias_add %}
      {{elem_input_type}}* res_ptr,
    {% endif %}
    {{elem_output_type}}* out_ptr,
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
  {{elem_input_type}}*,
  {% if bias_add %}
    {{elem_input_type}}*,
  {% endif %}
  {{elem_output_type}}*,
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
{{indent}}    static_cast<{{elem_input_type}}*>({{in_ptr}}),
{% if bias_add %}
  {{indent}}    static_cast<{{elem_input_type}}*>({{res_ptr}}),
{% endif %}
{{indent}}    static_cast<{{elem_output_type}}*>({{out_ptr}}),
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
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    input_type = backend_spec.dtype_to_lib_type(x._attrs["dtype"])
    output_type = backend_spec.dtype_to_lib_type(y._attrs["dtype"])
    return FUNC_DECL_TEMPLATE.render(
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        func_name=func_attrs["name"],
        elem_input_type=input_type,
        elem_output_type=output_type,
        bias_add=bias_add,
    )

    # return FUNC_DECL_TEMPLATE.render(func_name=func_name, data_type=data_type, stream_prefix=stream_prefix)


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
    input_type = backend_spec.dtype_to_lib_type(x._attrs["dtype"])
    output_type = backend_spec.dtype_to_lib_type(y._attrs["dtype"])
    if bias_add:
        r = func_attrs["inputs"][1]
        return FUNC_CALL_TEMPLATE.render(
            func_name=func_attrs["name"],
            elem_input_type=input_type,
            elem_output_type=output_type,
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
            elem_input_type=input_type,
            elem_output_type=output_type,
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
