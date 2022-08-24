# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"

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
      CUTLASS_PRAGMA_UNROLL
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
      CUTLASS_PRAGMA_UNROLL
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

void permute021_launcher(cutlass::half_t* in_ptr,
                         cutlass::half_t* out_ptr,
                         int x_dim0,
                         int x_dim1,
                         int x_dim2,
                         cudaStream_t stream) {
  const int n = x_dim0;
  const int h = 1;
  const int w = x_dim1;
  const int c = x_dim2;
  dim3 grid((c + 31)/32, (h*w + 31)/32, n);
  dim3 block(32, 8);
  nhwc_to_nchw_kernel<cutlass::half_t><<<grid, block, 0, stream>>>(
    out_ptr,
    (const cutlass::half_t*)in_ptr,
    n,
    h,
    w,
    c
  );
}
} // namespace

void {{function_name}} (
    cutlass::half_t* in_ptr,
    cutlass::half_t* out_ptr,
    int64_t* x_dim0,
    int64_t* x_dim1,
    int64_t* x_dim2,
    int64_t* y_dim0,
    int64_t* y_dim1,
    int64_t* y_dim2,
    cudaStream_t stream
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


@registry.reg("cuda.permute021.gen_function")
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
        function_name=func_name, shape_function=shape_func, exec_paths=exec_paths
    )


@registry.reg("cuda.permute021.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.permute021.func_call")
def gen_function_call(func_attrs, indent="  "):
    """[summary]

    Parameters
    ----------
    func_attrs : [type]
        [description]
    indent : str, optional
        [description], by default "  "

    Returns
    -------
    [type]
        [description]
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
