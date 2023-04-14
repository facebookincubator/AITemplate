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
multi-level roi align common functions for all backends.
"""

import jinja2

# pylint: disable=C0103,C0415,W0613,C0301,W0612


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}FPNRoiAlign<float, {{num_rois}}, {{pooled_size}}>(
{{indent}}    static_cast<{{elem_input_type}}*>(in_ptr_p2),
{{indent}}    static_cast<{{elem_input_type}}*>(in_ptr_p3),
{{indent}}    static_cast<{{elem_input_type}}*>(in_ptr_p4),
{{indent}}    static_cast<{{elem_input_type}}*>(in_ptr_p5),
{{indent}}    static_cast<{{elem_input_type}}*>(rois_ptr),
{{indent}}    static_cast<{{elem_output_type}}*>(out_ptr),
{{indent}}    batchSize,
{{indent}}    featureCount,
{{indent}}    imageSize,
{{indent}}    P2dims,
{{indent}}    P3dims,
{{indent}}    P4dims,
{{indent}}    P5dims,
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
// customized roi align kernel

struct xy_t {
  int64_t y;
  int64_t x;

  xy_t() : y(0), x(0) {}
  xy_t(int64_t y_, int64_t x_) : y(y_), x(x_) {}
};

template <typename T>
__device__ inline T interpolateBilinear(
    const T* src,
    xy_t srcDims,
    float y,
    float x,
    const int channels) {
  // deal with cases that inverse elements are out of feature map boundary
  int height = srcDims.y;
  int width = srcDims.x;
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = T(1.0) - ly, hx = T(1.0) - lx;
  // do bilinear interpolation
  T v1 = src[channels * (y_low * width + x_low)];
  T v2 = src[channels * (y_low * width + x_high)];
  T v3 = src[channels * (y_high * width + x_low)];
  T v4 = src[channels * (y_high * width + x_high)];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename Trois, typename Tfeat>
__global__ void roiAlign_kernel(
    xy_t imageSize,
    int featureCount,
    int roiCount,
    float threshold,
    int samplingRatio,
    const Trois* rois,
    const Tfeat* P2,
    const xy_t P2dims,
    const Tfeat* P3,
    const xy_t P3dims,
    const Tfeat* P4,
    const xy_t P4dims,
    const Tfeat* P5,
    const xy_t P5dims,
    Tfeat* pooled,
    const xy_t poolDims) {
  const int batch = blockIdx.x;
  const int feature = blockIdx.y;
  const int roiIdx = blockIdx.z;

  const Trois* roi = rois + 5 * (batch * roiCount + roiIdx);
  float hw;

{% if elem_input_type == "half" %}
  float x1 = __half2float(roi[1]);
  float y1 = __half2float(roi[2]);
  float x2 = __half2float(roi[3]);
  float y2 = __half2float(roi[4]);
{% elif elem_input_type == "float" %}
  float x1 = roi[1];
  float y1 = roi[2];
  float x2 = roi[3];
  float y2 = roi[4];
{% endif %}

  y1 = max(0.f, min((float)imageSize.y, y1)) / imageSize.y;
  x1 = max(0.f, min((float)imageSize.x, x1)) / imageSize.x;
  y2 = max(0.f, min((float)imageSize.y, y2)) / imageSize.y;
  x2 = max(0.f, min((float)imageSize.x, x2)) / imageSize.x;

  hw = (y2 - y1) * (x2 - x1);

  const Tfeat* src = P2;
  xy_t srcDims = P2dims;
  int iP = 2;

  if (hw > threshold) {
    src = P3;
    srcDims = P3dims;
    ++iP;
  }
  threshold *= 4;

  if (hw > threshold) {
    src = P4;
    srcDims = P4dims;
    ++iP;
  }
  threshold *= 4;

  if (hw > threshold) {
    src = P5;
    srcDims = P5dims;
    ++iP;
  }

  src += batch * srcDims.x * srcDims.y * featureCount + feature;
  // batch, roiCount, poolx, pooly, featureCount
  Tfeat* dst = pooled +
      poolDims.x * poolDims.y *
          (batch * roiCount * featureCount + roiIdx * featureCount) +
      feature;

  float samplingOffset = 0.5f;
  float inputOffset = 0.5f;

  float yStart = y1 * srcDims.y - inputOffset;
  float xStart = x1 * srcDims.x - inputOffset;

  float yEnd = y2 * srcDims.y - inputOffset;
  float xEnd = x2 * srcDims.x - inputOffset;

  float yDelta = (yEnd - yStart) / poolDims.y;
  float xDelta = (xEnd - xStart) / poolDims.x;

  const int samplingRatioX = samplingRatio > 0
      ? samplingRatio
      : max(1, (int)ceilf((xEnd - xStart) / poolDims.x));
  const int samplingRatioY = samplingRatio > 0
      ? samplingRatio
      : max(1, (int)ceilf((yEnd - yStart) / poolDims.y));
  const int samplingCount = samplingRatioX * samplingRatioY;

  for (int outIdx = threadIdx.x; outIdx < poolDims.x * poolDims.y;
       outIdx += blockDim.x) {
    int xx = outIdx % poolDims.x;
    int yy = outIdx / poolDims.x;
    Tfeat* out = dst + (poolDims.x * yy + xx) * featureCount;
    Tfeat result = 0;
    for (int iy = 0; iy < samplingRatioY; iy++) {
      float ySample = yStart + yDelta * yy;
      ySample += yDelta * (iy + samplingOffset) / samplingRatioY;
      ySample = min(max(ySample, 0.f), srcDims.y - 1.0f);

      for (int ix = 0; ix < samplingRatioX; ix++) {
        float xSample = xStart + xDelta * xx;
        xSample += xDelta * (ix + samplingOffset) / samplingRatioX;
        xSample = min(max(xSample, 0.f), srcDims.x - 1.0f);

        result +=
            interpolateBilinear(src, srcDims, ySample, xSample, featureCount);
      }
    }

{% if elem_output_type == "half" %}
    *out = __half(result) / __float2half_rn(samplingCount);
{% elif elem_output_type == "float" %}
    *out = result / samplingCount;
{% endif %}
  }
}

template <typename T, int roiCount, int pool_size>
void FPNRoiAlign(
    {{elem_input_type}}* P2,
    {{elem_input_type}}* P3,
    {{elem_input_type}}* P4,
    {{elem_input_type}}* P5,
    {{elem_input_type}}* rois,
    {{elem_output_type}}* output,
    const int batchSize,
    const int featureCount,
    const xy_t imageSize,
    const xy_t P2dims,
    const xy_t P3dims,
    const xy_t P4dims,
    const xy_t P5dims,
    const int samplingRatio,
    const float spatial_scale,
    const bool position_sensitive,
    const bool continuous_coordinate,
    {{prefix}}Stream_t stream) {
  float mFPNScale = 224;
  float normScale = sqrtf(mFPNScale * mFPNScale / (imageSize.x * imageSize.y));
  float firstThreshold = normScale * normScale / 4.f;

  const dim3 blocks(batchSize, featureCount, roiCount);
  const int threads(min(256, pool_size * pool_size));

  roiAlign_kernel<<<blocks, threads, 0, stream>>>(
      imageSize,
      featureCount,
      roiCount,
      firstThreshold,
      samplingRatio,
      reinterpret_cast<const {{elem_input_type}}*>(rois),
      reinterpret_cast<const {{elem_input_type}}*>(P2),
      P2dims,
      reinterpret_cast<const {{elem_input_type}}*>(P3),
      P3dims,
      reinterpret_cast<const {{elem_input_type}}*>(P4),
      P4dims,
      reinterpret_cast<const {{elem_input_type}}*>(P5),
      P5dims,
      output,
      {pool_size, pool_size});
}

} // namespace

void {{function_name}} (
    void* in_ptr_p2,
    void* in_ptr_p3,
    void* in_ptr_p4,
    void* in_ptr_p5,
    void* rois_ptr,
    void* out_ptr,
    {{index_type}}* batch, {{index_type}}* in_ch,
    {{index_type}}* p2_h, {{index_type}}* p2_w,
    {{index_type}}* p3_h, {{index_type}}* p3_w,
    {{index_type}}* p4_h, {{index_type}}* p4_w,
    {{index_type}}* p5_h, {{index_type}}* p5_w,
    const int im_h, const int im_w,
    int sampling_ratio,
    const float spatial_scale,
    const bool position_sensitive,
    const bool continuous_coordinate,
    {{prefix}}Stream_t stream
) {
  {{shape_function}}

  const xy_t imageSize = {im_h, im_w};
  const xy_t P2dims = {*p2_h, *p2_w};
  const xy_t P3dims = {*p3_h, *p3_w};
  const xy_t P4dims = {*p4_h, *p4_w};
  const xy_t P5dims = {*p5_h, *p5_w};
  const int featureCount = *in_ch;
  const int batchSize = *batch;

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
  void*,
  void*,
  void*,
  void*,
  void*,
  void*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  int,
  int,
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
{{indent}}    {{in_ptr_p2}},
{{indent}}    {{in_ptr_p3}},
{{indent}}    {{in_ptr_p4}},
{{indent}}    {{in_ptr_p5}},
{{indent}}    {{rois_ptr}},
{{indent}}    {{out_ptr}},
{{indent}}    {{p_batch}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p2_h}}, {{p2_w}},
{{indent}}    {{p3_h}}, {{p3_w}},
{{indent}}    {{p4_h}}, {{p4_w}},
{{indent}}    {{p5_h}}, {{p5_w}},
{{indent}}    {{im_h}}, {{im_w}},
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
    input_type = backend_spec.dtype_to_backend_type(x._attrs["dtype"])
    output_type = backend_spec.dtype_to_backend_type(y._attrs["dtype"])
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
    p2 = func_attrs["inputs"][0]
    p3 = func_attrs["inputs"][1]
    p4 = func_attrs["inputs"][2]
    p5 = func_attrs["inputs"][3]
    rois = func_attrs["inputs"][4]
    xshape = p2._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]

    input_type = backend_spec.dtype_to_backend_type(p2._attrs["dtype"])
    output_type = backend_spec.dtype_to_backend_type(y._attrs["dtype"])

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr_p2=p2._attrs["name"],
        in_ptr_p3=p3._attrs["name"],
        in_ptr_p4=p4._attrs["name"],
        in_ptr_p5=p5._attrs["name"],
        rois_ptr=rois._attrs["name"],
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_in_ch="&" + xshape[3]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        im_h=func_attrs["im_shape"][0],
        im_w=func_attrs["im_shape"][1],
        p2_h="&" + p2._attrs["shape"][1]._attrs["name"],
        p2_w="&" + p2._attrs["shape"][2]._attrs["name"],
        p3_h="&" + p3._attrs["shape"][1]._attrs["name"],
        p3_w="&" + p3._attrs["shape"][2]._attrs["name"],
        p4_h="&" + p4._attrs["shape"][1]._attrs["name"],
        p4_w="&" + p4._attrs["shape"][2]._attrs["name"],
        p5_h="&" + p5._attrs["shape"][1]._attrs["name"],
        p5_w="&" + p5._attrs["shape"][2]._attrs["name"],
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
