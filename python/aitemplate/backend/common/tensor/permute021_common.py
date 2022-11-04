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
Common implementations for all backends for permute021.

For three dimension input, shift the second and the third dimension.
i.e. Output[d0, d2, d1] = Input[d0, d1, d2]

"""
from typing import Any, Dict

import jinja2

# pylint: disable=C0301,W0613,W0612

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  const void* /*input*/,
  void* /* output */,
  int64_t* /* x_dim0 */,
  int64_t* /* x_dim1 */,
  int64_t* /* x_dim2 */,
  int64_t* /* y_dim0 */,
  int64_t* /* y_dim1 */,
  int64_t* /* y_dim2 */,
  {{prefix}}Stream_t /* stream */
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{out_ptr}},
{{indent}}    {{x_dim0}},
{{indent}}    {{x_dim1}},
{{indent}}    {{x_dim2}},
{{indent}}    {{y_dim0}},
{{indent}}    {{y_dim1}},
{{indent}}    {{y_dim2}},
{{indent}}    stream
{{indent}});
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}permute021_launcher(
{{indent}}    in_ptr,
{{indent}}    out_ptr,
{{indent}}    *x_dim0,
{{indent}}    *x_dim1,
{{indent}}    *x_dim2,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {
template <typename T>
__global__ void nhwc_to_nchw_kernel(T *output,
                                    const T *input,
                                    const int n,
                                    const int h,
                                    const int w,
                                    const int c) {

  const int hw = h*w;
  const int hwc = hw*c;
  __shared__ T shbuf[32 * (32 + 1)];
  const int32_t tid  = threadIdx.y*blockDim.x + threadIdx.x;
  const int32_t wid  = tid / 32;
  const int32_t lid  = tid % 32;
  const int32_t ni   = blockIdx.z;
  const int32_t hwi0  = blockIdx.y * 32;
  const int32_t ci0 = blockIdx.x * 32;

  const size_t input_idx = ni * hwc + (hwi0 + wid) * c + ci0;
  const T *A = input + input_idx;
  if (ci0 + lid < c) {
    const int lid_x_33 = lid * 33;
    if ((hwi0 + 32) <= hw) {
      int hwi = wid;  // between 0 and 7
      #pragma unroll
      for (int cLoopIdx = 0; cLoopIdx < 4; cLoopIdx++) {
        shbuf[lid_x_33 + hwi] = A[lid];
        A                     = &A[8 * c];
        hwi += 8;
      }
    } else {
      for (int hwi = wid; hwi < 32; hwi += 8) {
        if ((hwi + hwi0) < hw) {
          shbuf[lid_x_33 + hwi] = A[lid];
        }
        A = &A[8 * c];
      }
    }
  }
  __syncthreads();

  const int32_t hwiOut = hwi0 + lid;
  output = &output[ni * hwc + hwiOut];
  if (hwiOut < hw) {
    if (ci0 + 32 < c) {
      int cI = wid;
      #pragma unroll
      for (int hwLoopIdx = 0; hwLoopIdx < 4; ++hwLoopIdx) {
        output[(ci0 + cI) * hw] = shbuf[(cI)*33 + lid];
        cI += 8;
      }
    } else {
      for (int cI = wid; cI < 32; cI += 8) {
        if (ci0 + cI < c) {
          output[(ci0 + cI) * hw] = shbuf[(cI)*33 + lid];
        }
      }
    }
  }
}

void permute021_launcher(const void* in_ptr,
                         void* out_ptr,
                         int x_dim0,
                         int x_dim1,
                         int x_dim2,
                         {{prefix}}Stream_t stream) {
  const int n = x_dim0;
  const int h = 1;
  const int w = x_dim1;
  const int c = x_dim2;
  dim3 grid((c + 31)/32, (h*w + 31)/32, n);
  dim3 block(32, 8);
  nhwc_to_nchw_kernel<{{lib_dtype}}><<<grid, block, 0, stream>>>(
    static_cast<{{lib_dtype}}*>(out_ptr),
    static_cast<const {{lib_dtype}}*>(in_ptr),
    n,
    h,
    w,
    c
  );
}
} // namespace

void {{function_name}} (
    const void* in_ptr,
    void* out_ptr,
    int64_t* x_dim0,
    int64_t* x_dim1,
    int64_t* x_dim2,
    int64_t* y_dim0,
    int64_t* y_dim1,
    int64_t* y_dim2,
    {{prefix}}Stream_t stream
) {
  if (!in_ptr) {
    throw std::runtime_error("in_ptr is NULL!");
  }
  if (!out_ptr) {
    throw std::runtime_error("in_ptr is NULL!");
  }
  {{shape_function}}
  {{exec_paths}}
}

"""
)


def gen_function(
    func_attrs: Dict[str, Any],
    template_path: str,
    shape_eval_template,
    shape_save_template,
    header_files: str,
    backend_spec,
) -> str:
    """
    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Attributes from Operator
    template_path : str
        path to library used
    shape_eval_template : jinja template
    shape_save_template : jinja template
    header_files : str
        header files included in the function
    backend_spec : class
        specifies backend configs

    Returns
    -------
    str
        Source code for function generated.
    """

    func_name = func_attrs["name"]
    x = func_attrs["inputs"][0]
    xdtype = x._attrs["dtype"]
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*x_dim0",
        x_dim1="*x_dim1",
        x_dim2="*x_dim2",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*y_dim0",
        y_dim1="*y_dim1",
        y_dim2="*y_dim2",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = EXEC_TEMPLATE.render()
    return SRC_TEMPLATE.render(
        function_name=func_name,
        header_files=header_files,
        shape_function=shape_func,
        exec_paths=exec_paths,
        lib_dtype=backend_spec.dtype_to_lib_type(xdtype),
        prefix=backend_spec.prefix,
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    """
    Parameters
    ----------
    func_attrs : dict
        Attributes from Operator
    backend_spec : class
        specifies backend configs

    Returns
    -------
    str
        Function declaration
    """

    func_name = func_attrs["name"]
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        prefix=backend_spec.prefix,
    )


def gen_function_call(func_attrs: Dict[str, Any], backend_spec, indent="  ") -> str:
    """
    Parameters
    ----------
    func_attrs : dict
        Attributes from Operator
    backend_spec : class
        specifies backend configs
    indent : str, optional
        Indentation for function call template, by default "  "

    Returns
    -------
    str
        Driver code for invoking call
    """

    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        x_dim0="&" + xshape[0]._attrs["name"],
        x_dim1="&" + xshape[1]._attrs["name"],
        x_dim2="&" + xshape[2]._attrs["name"],
        y_dim0="&" + yshape[0]._attrs["name"],
        y_dim1="&" + yshape[1]._attrs["name"],
        y_dim2="&" + yshape[2]._attrs["name"],
        indent=indent,
    )
