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
roi align common functions for all backends.
"""

import jinja2

# pylint: disable=C0103,C0415,W0613,C0301,W0612


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}roi_align_launcher<float, {{num_rois}}, {{pooled_size}}>(
{{indent}}    in_ptr,
{{indent}}    rois_ptr,
{{indent}}    out_ptr,
{{indent}}    NI,
{{indent}}    HI,
{{indent}}    WI,
{{indent}}    CI,
{{indent}}    HO,
{{indent}}    WO,
{{indent}}    sampling_ratio,
{{indent}}    spatial_scale,
{{indent}}    position_sensitive,
{{indent}}    continuous_coordinate,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {
#define CUDA_KERNEL_LOOP(i, n) \
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename T>
__device__ float2 bilinear_interpolate(const half2* bottom_data,
                                  const int height,
                                  const int width,
                                  T y,
                                  T x,
                                  const int channels,
                                  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  float2 val = {0.f, 0.f};
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return val;
  }

  y = y <= 0 ? 0 : y;
  x = x <= 0 ? 0 : x;

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  y_high = y_low >= height - 1 ? height - 1 : y_low + 1;
  y_low = y_low >= height - 1 ? height - 1 : y_low;
  y = y_low >= height - 1 ? (T)y_low : y;

  x_high = x_low >= width - 1 ? width - 1 : x_low + 1;
  x_low = x_low >= width - 1 ? width - 1 : x_low;
  x = x_low >= width - 1 ? (T)x_low : x;


  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  const half2  v1 = __ldg(bottom_data + (y_low * width + x_low) * channels);
  const half2  v2 = __ldg(bottom_data + (y_low * width + x_high) * channels);
  const half2  v3 = __ldg(bottom_data + (y_high * width + x_low) * channels);
  const half2  v4 = __ldg(bottom_data + (y_high * width + x_high) * channels);

  T v1_x = __half2float(v1{{half2_data_ref}}.x);
  T v2_x = __half2float(v2{{half2_data_ref}}.x);
  T v3_x = __half2float(v3{{half2_data_ref}}.x);
  T v4_x = __half2float(v4{{half2_data_ref}}.x);

  T v1_y = __half2float(v1{{half2_data_ref}}.y);
  T v2_y = __half2float(v2{{half2_data_ref}}.y);
  T v3_y = __half2float(v3{{half2_data_ref}}.y);
  T v4_y = __half2float(v4{{half2_data_ref}}.y);

  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  val.x = (w1 * v1_x + w2 * v2_x + w3 * v3_x + w4 * v4_x);
  val.y = (w1 * v1_y + w2 * v2_y + w3 * v3_y + w4 * v4_y);

  return val;
}

template <typename T, int64_t num_rois, int pool_size>
__global__ void roi_align_f16_nhwc_kernel(const half2* bottom_data,
                                         const half* bottom_rois,
                                         half2* top_data,
                                         const int64_t N,
                                         const int64_t height,
                                         const int64_t width,
                                         const int64_t channels,
                                         const int64_t pooled_height,
                                         const int64_t pooled_width,
                                         const int sampling_ratio,
                                         const float spatial_scale,
                                         const bool position_sensitive,
                                         const bool continuous_coordinate) {

  const int64_t nthreads = num_rois * channels * pooled_width * pooled_height;

  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    // index = c + channels * (x + out_width * (y + out_height * b))
    int64_t idx = index;
    const int c = idx % channels;
    idx /= channels;
    const int pw = idx % pooled_width;
    idx /= pooled_width;
    const int ph = idx % pooled_height;
    const int n = idx / pooled_height;


    const half* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(__half2float(offset_bottom_rois[0]));

    float2 output_val = {0.f, 0.f};
    if (roi_batch_ind < 0) {
      top_data[index] = __float22half2_rn(output_val);
      continue;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_offset  = continuous_coordinate ? static_cast<T>(0.5) : static_cast<T>(0);
    T roi_start_w = __half2float(offset_bottom_rois[1]) * spatial_scale - roi_offset;
    T roi_start_h = __half2float(offset_bottom_rois[2]) * spatial_scale - roi_offset;
    T roi_end_w   = __half2float(offset_bottom_rois[3]) * spatial_scale - roi_offset;
    T roi_end_h   = __half2float(offset_bottom_rois[4]) * spatial_scale - roi_offset;

    T roi_width  = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!continuous_coordinate) {  // backward compatiblity
      // Force malformed ROIs to be 1x1
      roi_width  = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int c_unpooled        = c;
    int channels_unpooled = channels;
    if (position_sensitive) {
      c_unpooled        = c * pooled_height * pooled_width + ph * pooled_width + pw;
      channels_unpooled = channels * pooled_height * pooled_width;
    }

    const half2* offset_bottom_data =
           bottom_data + (roi_batch_ind * height * width * channels_unpooled + c_unpooled);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    // T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const T y =
          roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        float2 val = bilinear_interpolate(offset_bottom_data, height, width, y, x, channels, index);
        output_val.x += val.x;
        output_val.y += val.y;
      }
    }
    output_val.x /= count;
    output_val.y /= count;

    top_data[index] = __float22half2_rn(output_val);
  }

}


template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}


template <typename T, int64_t num_rois, int pool_size>
void roi_align_launcher({{elem_input_type}}* input,
                      {{elem_input_type}}* rois,
                      {{elem_output_type}}* output,
                      const {{index_type}} N,
                      const {{index_type}} H,
                      const {{index_type}} W,
                      const {{index_type}} C,
                      const {{index_type}} HO,
                      const {{index_type}} WO,
                      const int sampling_ratio,
                      const float spatial_scale,
                      const bool position_sensitive,
                      const bool continuous_coordinate,
                      {{prefix}}Stream_t stream) {

  const int64_t output_size = num_rois * C * HO * WO;

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  roi_align_f16_nhwc_kernel<T, num_rois, pool_size><<<grid, block, 0, stream>>>(
    (const half2*)input, (const half*)rois, (half2*)output, N, H, W, C / 2, HO, WO,
    sampling_ratio, spatial_scale, position_sensitive, continuous_coordinate);

}
} // namespace

void {{function_name}} (
    {{elem_input_type}}* in_ptr,
    {{elem_input_type}}* rois_ptr,
    {{elem_output_type}}* out_ptr,
    {{index_type}}* batch,
    {{index_type}}* in_h,
    {{index_type}}* in_w,
    {{index_type}}* in_ch,
    {{index_type}}* out_batch,
    {{index_type}}* out_h,
    {{index_type}}* out_w,
    int sampling_ratio,
    const float spatial_scale,
    const bool position_sensitive,
    const bool continuous_coordinate,
    {{prefix}}Stream_t stream
) {
  {{shape_function}}
  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this avg pool2d specialization."
  );
}

"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  {{elem_input_type}}*,
  {{elem_input_type}}*,
  {{elem_output_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  int,
  float,
  bool,
  bool,
  {{prefix}}Stream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    static_cast<{{elem_input_type}}*>({{in_ptr}}),
{{indent}}    static_cast<{{elem_input_type}}*>({{rois_ptr}}),
{{indent}}    static_cast<{{elem_output_type}}*>({{out_ptr}}),
{{indent}}    {{p_batch}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    {{sampling_ratio}},
{{indent}}    {{spatial_scale}},
{{indent}}    {{position_sensitive}},
{{indent}}    {{continuous_coordinate}},
{{indent}}    stream
{{indent}});
"""
)


def gen_function_decl(func_attrs, backend_spec):
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
    )


def gen_function_call(func_attrs, backend_spec, indent="  "):
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
    rois = func_attrs["inputs"][1]
    xshape = x._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]

    input_type = backend_spec.dtype_to_lib_type(x._attrs["dtype"])
    output_type = backend_spec.dtype_to_lib_type(y._attrs["dtype"])

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        rois_ptr=rois._attrs["name"],
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_in_ch="&" + xshape[3]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        sampling_ratio=func_attrs["sampling_ratio"],
        spatial_scale=func_attrs["spatial_scale"],
        position_sensitive="true" if func_attrs["position_sensitive"] else "false",
        continuous_coordinate="true"
        if func_attrs["continuous_coordinate"]
        else "false",
        backend_spec=backend_spec,
        elem_input_type=input_type,
        elem_output_type=output_type,
        indent=indent,
    )
