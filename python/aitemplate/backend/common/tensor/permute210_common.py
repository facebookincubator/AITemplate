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
Common implementations for all backends for permute210.

For three dimension input, shift the first and the third dimension.
i.e. Output[d2, d1, d0] = Input[d0, d1, d2]

We invoke kernel with the following settings:
thread blocks of (TILE_SIZE x TILE_SIZE/4),
grid size of (ceil(d1/TILE_SIZE) x d2 x ceil(d3/TILE_SIZE))
For each, we have shared memory of size (TILE_SIZE, TILE_SIZE+1)

The 4 for thread blocks indicates each thread is responsible of 4 elements.
We use TILE_SIZE = 32 for the time being.
"""
from typing import Any, Dict

import jinja2

# pylint: disable=C0301,W0613,W0612

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  const void* /* input */,
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
{{indent}}permute210_launcher(
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

#define TILE_SIZE 32

namespace {
template <typename T>
__global__ void permute210_kernel(T *output,
                                  const T *input,
                                  const int n,
                                  const int c,
                                  const int w) {
  __shared__ T shbuf[TILE_SIZE][TILE_SIZE + 1];

  int32_t strides[2] = { c * w, w };
  int32_t offset = blockIdx.y * strides[1]; // We are slicing through static c.

  int32_t xBlock = blockIdx.x * TILE_SIZE;
  int32_t yBlock = blockIdx.z * TILE_SIZE;
  int32_t x = xBlock + threadIdx.x;
  int32_t y = yBlock + threadIdx.y;

  const int32_t inputIdx = y * strides[0] + offset + xBlock;
  const T *A = input + inputIdx;

  if (x < w) {
    if (y + 24 < n) { // This guards (y, y+8, y+16, y+24) are within boundary.
      int tid = threadIdx.y;
      #pragma unroll
      for (int loopIdx = 0; loopIdx < 4; loopIdx++) {
        shbuf[threadIdx.x][tid] = A[threadIdx.x];
        A                       = &A[8 * strides[0]];
        tid += 8;
      }
    } else {
      #pragma unroll
      for (int tid = threadIdx.y; tid < 32; tid += 8) {
        if (yBlock + tid < n) {
          shbuf[threadIdx.x][tid] = A[threadIdx.x];
        }
        A = &A[8 * strides[0]];
      }
    }
  }
  __syncthreads();

  // Now, we do the computation of transposes toward the new indices
  strides[0] = c * n;
  strides[1] = n;
  offset = blockIdx.y * strides[1];

  xBlock = blockIdx.z * TILE_SIZE;
  yBlock = blockIdx.x * TILE_SIZE;
  x = xBlock + threadIdx.x;
  y = yBlock + threadIdx.y;

  output = &output[y * strides[0] + offset + xBlock];
  if (x < n) {
    if (y + 24 < w) {
      int tid = threadIdx.y;
      #pragma unroll
      for (int loopIdx = 0; loopIdx < 4; loopIdx++) {
        output[threadIdx.x] = shbuf[tid][threadIdx.x];
        output              = &output[8 * strides[0]];
        tid += 8;
      }
    } else {
      #pragma unroll
      for (int tid = threadIdx.y; tid < 32; tid += 8) {
        if (yBlock + tid < w) {
          output[threadIdx.x] = shbuf[tid][threadIdx.x];
        }
        output = &output[8 * strides[0]];
      }
    }
  }
}

void permute210_launcher(const void* in_ptr,
                         void* out_ptr,
                         int x_dim0,
                         int x_dim1,
                         int x_dim2,
                         {{prefix}}Stream_t stream) {
  dim3 grid((x_dim2 + (TILE_SIZE-1))/TILE_SIZE, x_dim1, (x_dim0 + (TILE_SIZE-1))/TILE_SIZE);
  dim3 block(TILE_SIZE, TILE_SIZE/4);
  permute210_kernel<{{lib_dtype}}><<<grid, block, 0, stream>>>(
    static_cast<{{lib_dtype}}*>(out_ptr),
    static_cast<const {{lib_dtype}}*>(in_ptr),
    x_dim0,
    x_dim1,
    x_dim2
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
  {{exec_paths}}
}

"""
)


def gen_function(func_attrs: Dict[str, Any], header_files: str, backend_spec) -> str:
    """
    Parameters
    ----------
    func_attrs : dict
        Attributes from Operator
    header_files : str
        header files included in the function
    backend_spec : class
        specifies the backend configs

    Returns
    -------
    str
        Source code for function generated.
    """
    func_name = func_attrs["name"]
    x = func_attrs["inputs"][0]
    xdtype = x._attrs["dtype"]
    exec_paths = EXEC_TEMPLATE.render()
    return SRC_TEMPLATE.render(
        function_name=func_name,
        header_files=header_files,
        exec_paths=exec_paths,
        prefix=backend_spec.prefix,
        lib_dtype=backend_spec.dtype_to_lib_type(xdtype),
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    """
    Parameters
    ----------
    func_attrs : dict
        Attributes from Operator
    backend_spec : class
        specifies the backend configs

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
        specifies the backend configs
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
