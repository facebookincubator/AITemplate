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
For higher-rank input, treat the first n-2 dims as a single flat dim.
i.e. Output[d0, ..., dn-3, dn-1, dn-2] = Input[d0, ..., dn-3, dn-2, dn-1]

"""
from typing import Any, Dict

import jinja2
from aitemplate.backend.common import tensor_accessor_codegen

# pylint: disable=C0301,W0613,W0612

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  const void* /* input */,
  void* /* output */,
  int64_t /* rank */,
  const int64_t* /* x_dims */,
  {{prefix}}Stream_t /* stream */
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  const int64_t x_dims[] = {{x_dims}};
{{indent}}  {{func_name}}(
{{indent}}      {{in_ptr}},
{{indent}}      {{out_ptr}},
{{indent}}      {{rank}},
{{indent}}      x_dims,
{{indent}}      stream
{{indent}}  );
{{indent}}}
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{input_accessor_def}}
{{indent}}permute021_launcher(
{{indent}}    in_ptr,
{{indent}}    out_ptr,
{{indent}}    rank,
{{indent}}    x_dims,
{{indent}}    input_accessor,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

#include <limits>

#define TILE_SIZE 32
#define CH_K 4

namespace {

{{tensor_accessor_libs}}

// blockIdx.x -> ni
// blockIdx.y -> hwi
// blockIdx.z -> ci
__device__ __forceinline__ void block_fn_nhc(int32_t& ni, int32_t& hwi, int32_t& ci) {
  ni = blockIdx.x;
  hwi = blockIdx.y;
  ci = blockIdx.z;
}

// blockIdx.x -> ni
// blockIdx.y -> ci
// blockIdx.z -> hwi
__device__ __forceinline__ void block_fn_nch(int32_t& ni, int32_t& hwi, int32_t& ci) {
  ni = blockIdx.x;
  ci = blockIdx.y;
  hwi = blockIdx.z;
}

// blockIdx.x -> ci
// blockIdx.y -> hwi
// blockIdx.z -> ni
__device__ __forceinline__ void block_fn_chn(int32_t& ni, int32_t& hwi, int32_t& ci) {
  ci = blockIdx.x;
  hwi = blockIdx.y;
  ni = blockIdx.z;
}

using BlockFunc = void (*)(int32_t&, int32_t&, int32_t&);

template <typename T, BlockFunc BLOCK_FN>
__global__ void permute021_kernel(T *output,
                                  const T *input,
                                  const int64_t n,
                                  const int32_t h,
                                  const int32_t w,
                                  const int32_t c,
                                  TensorAccessor input_accessor) {

  const int32_t hw = h * w;
  const int32_t hwc = hw * c;

  __shared__ T shbuf[TILE_SIZE * (TILE_SIZE + 1)];

  const int32_t tid  = threadIdx.y * blockDim.x + threadIdx.x;
  const int32_t wid  = tid / TILE_SIZE;
  const int32_t lid  = tid % TILE_SIZE;
  int32_t ni_tmp, hwi_tmp, ci_tmp;
  BLOCK_FN(ni_tmp, hwi_tmp, ci_tmp);
  const int32_t ni = ni_tmp;
  const int32_t hwi0 = hwi_tmp * TILE_SIZE;
  const int32_t ci0  = ci_tmp * TILE_SIZE;

  size_t input_idx = ni * hwc + (hwi0 + wid) * c + ci0;

  const T *A = input_accessor.get<const T, const T>(input, input_idx);

  if (ci0 + lid < c) {
    const int lid_x_33 = lid * (TILE_SIZE + 1);
    if ((hwi0 + TILE_SIZE) <= hw) {
      int hwi = wid;  // between 0 and 7
      #pragma unroll
      for (int cLoopIdx = 0; cLoopIdx < CH_K; cLoopIdx++) {
        shbuf[lid_x_33 + hwi] = *input_accessor.get<const T, const T>(input, input_idx + lid);
        input_idx += TILE_SIZE / CH_K * c;
        hwi += TILE_SIZE / CH_K;
      }
    } else {
      for (int hwi = wid; hwi < TILE_SIZE; hwi += TILE_SIZE / CH_K) {
        if (hwi + hwi0 < hw) {
          shbuf[lid_x_33 + hwi] = *input_accessor.get<const T, const T>(input, input_idx + lid);
        }
        input_idx += TILE_SIZE / CH_K * c;
      }
    }
  }
  __syncthreads();

  const int32_t hwiOut = hwi0 + lid;
  output = &output[ni * hwc + hwiOut];
  if (hwiOut < hw) {
    if (ci0 + TILE_SIZE < c) {
      int cI = wid;
      #pragma unroll
      for (int hwLoopIdx = 0; hwLoopIdx < CH_K; ++hwLoopIdx) {
        output[(ci0 + cI) * hw] = shbuf[cI * (TILE_SIZE + 1) + lid];
        cI += TILE_SIZE / CH_K;
      }
    } else {
      for (int cI = wid; cI < TILE_SIZE; cI += TILE_SIZE / CH_K) {
        if (ci0 + cI < c) {
          output[(ci0 + cI) * hw] = shbuf[cI * (TILE_SIZE + 1) + lid];
        }
      }
    }
  }
}

void permute021_launcher(const void* in_ptr,
                         void* out_ptr,
                         int64_t rank,
                         const int64_t* x_dims,
                         TensorAccessor input_accessor,
                         {{prefix}}Stream_t stream) {
  int64_t x_dim0 = 1;
  for (int i = 0; i < rank - 2; i++) {
    x_dim0 *= x_dims[i];
  }

  if (x_dims[rank-2] > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("The second last dim does not fit into int32_t.");
  }
  if (x_dims[rank-1] > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("The last dim does not fit into int32_t.");
  }

  // given the above checks, we know it's safe
  const int32_t x_dim1 = x_dims[rank-2];
  const int32_t x_dim2 = x_dims[rank-1];

#define THROW_INVALID_LAUNCH_CONFIG                   \
    throw std::runtime_error(                         \
        std::string("invalid cuda launch config: ") + \
        std::to_string(grid_c) + ", " +               \
        std::to_string(grid_hw) + ", " +              \
        std::to_string(grid_n));

  const int32_t n = static_cast<int32_t>(x_dim0);
  const int32_t h = 1;
  const int32_t w = x_dim1;
  const int32_t c = x_dim2;
  const int32_t grid_c = (c + TILE_SIZE - 1) / TILE_SIZE;
  const int32_t grid_hw = (h * w + TILE_SIZE - 1) / TILE_SIZE;
  const int32_t grid_n = n;
  constexpr int32_t max_grid_z = 65535;
  constexpr int32_t max_grid_x = 2147483647;
  if (grid_c > max_grid_x || grid_hw > max_grid_x || grid_n > max_grid_x) {
    THROW_INVALID_LAUNCH_CONFIG
  }
  if ((grid_c <= max_grid_z && grid_hw <= max_grid_z && grid_n <= max_grid_z) ||
      (grid_c > max_grid_z && grid_hw <= max_grid_z && grid_n <= max_grid_z)) {
    dim3 grid(grid_c, grid_hw, grid_n);
    dim3 block(TILE_SIZE, TILE_SIZE / CH_K);
    permute021_kernel<{{lib_dtype}}, block_fn_chn><<<grid, block, 0, stream>>>(
        static_cast<{{lib_dtype}}*>(out_ptr),
        static_cast<const {{lib_dtype}}*>(in_ptr),
        n, h, w, c, input_accessor
    );
  } else if (grid_n > max_grid_z && grid_hw <= max_grid_z && grid_c <= max_grid_z) {
    dim3 grid(grid_n, grid_c, grid_hw);
    dim3 block(TILE_SIZE, TILE_SIZE / CH_K);
    permute021_kernel<{{lib_dtype}}, block_fn_nch><<<grid, block, 0, stream>>>(
        static_cast<{{lib_dtype}}*>(out_ptr),
        static_cast<const {{lib_dtype}}*>(in_ptr),
        n, h, w, c, input_accessor
    );
  } else if (grid_n > max_grid_z && grid_hw <= max_grid_z && grid_c <= max_grid_z) {
    dim3 grid(grid_n, grid_hw, grid_c);
    dim3 block(TILE_SIZE, TILE_SIZE / CH_K);
    permute021_kernel<{{lib_dtype}}, block_fn_nhc><<<grid, block, 0, stream>>>(
        static_cast<{{lib_dtype}}*>(out_ptr),
        static_cast<const {{lib_dtype}}*>(in_ptr),
        n, h, w, c, input_accessor
    );
  } else {
    THROW_INVALID_LAUNCH_CONFIG
  }
  
}
} // namespace

void {{function_name}} (
    const void* in_ptr,
    void* out_ptr,
    int64_t rank,
    const int64_t* x_dims,
    {{prefix}}Stream_t stream
) {
  for (int i = 0; i < rank; i++) {
      if (x_dims[i] == 0) {
          // empty input: nothing to do
          return;
      }
  }
  if (!in_ptr) {
    throw std::runtime_error("in_ptr is NULL!");
  }
  if (!out_ptr) {
    throw std::runtime_error("out_ptr is NULL!");
  }
  {{exec_paths}}
}
"""
)


def gen_function(
    func_attrs: Dict[str, Any],
    template_path: str,
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
    tensor_accessor = func_attrs["input_accessors"][0]
    xdtype = x._attrs["dtype"]
    tensor_accessor_libs = tensor_accessor_codegen.get_libs()
    input_accessor_name = "input_accessor"
    input_accessor = tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
        name=input_accessor_name, tensor_accessor=tensor_accessor
    )
    exec_paths = EXEC_TEMPLATE.render(input_accessor_def=input_accessor)

    return SRC_TEMPLATE.render(
        function_name=func_name,
        exec_paths=exec_paths,
        header_files=header_files,
        lib_dtype=backend_spec.dtype_to_lib_type(xdtype),
        prefix=backend_spec.prefix,
        tensor_accessor_libs=tensor_accessor_libs,
    )


def gen_function_decl(
    func_attrs: Dict[str, Any],
    backend_spec,
) -> str:
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


def gen_function_call(
    func_attrs: Dict[str, Any],
    backend_spec,
    indent="  ",
) -> str:
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
    y = func_attrs["outputs"][0]

    input_accessor = func_attrs["input_accessors"][0]
    xshape = input_accessor.original_shapes
    x_dims = [dim._attrs["name"] for dim in xshape]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        x_dims=("{" + ", ".join(x_dims) + "}"),
        rank=len(xshape),
        indent=indent,
    )
