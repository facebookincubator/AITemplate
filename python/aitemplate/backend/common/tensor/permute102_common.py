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
Common implementations for all backends for permute012.

For three dimension input, shift the first and the second dimension.
i.e. Output[d1, d0, d2] = Input[d0, d1, d2]

After determining the largest movable vectorized type fitting into d2,
the implementation is based on two different kernels invoked depending
on the number of items of that type d2 consists of:

1. If the number is <= 16, the extension of the SMEM-tile approach (as
used in permute021) is used to maintain coalesced reads from and writes
to the global memory, with the SMEM layout for avoiding bank conflicts
on store and load. This approach assumes that the last dimension can be
fully covered with a single warp, hence can only work with the number
being <= 32.

2. If the number is > 16, the direct approach is used for copying
d2-sized blocks along the last dimension directly from the input to the
output global memory. This trivially corresponds to coalesced read and
write of the whole d2-sized block. The cutoff of > 16 is chosen, as
starting from 17 items, the approach #1 corresponds to the same data
movement, just through the SMEM and with more index computation.
"""
from typing import Any, Dict

import jinja2

# pylint: disable=C0301,W0613,W0612

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  const void* /* input */,
  void* /* output */,
  int64_t /* x_dim0 */,
  int64_t /* x_dim1 */,
  int64_t /* x_dim2 */,
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
{{indent}}    stream
{{indent}});
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{% if dtype == "half" %}
{{indent}}if (x_dim2 % 8 == 0) {
{{indent}}  permute102_launcher<float4>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2 / 8,
{{indent}}      stream
{{indent}}  );
{{indent}}} else if (x_dim2 % 4 == 0) {
{{indent}}  permute102_launcher<float2>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2 / 4,
{{indent}}      stream
{{indent}}  );
{{indent}}} else if (x_dim2 % 2 == 0) {
{{indent}}  permute102_launcher<float>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2 / 2,
{{indent}}      stream
{{indent}}  );
{{indent}}} else {
{{indent}}  permute102_launcher<half>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2,
{{indent}}      stream
{{indent}}  );
{{indent}}}
{% elif dtype == "float" %}
{{indent}}if (x_dim2 % 4 == 0) {
{{indent}}  permute102_launcher<float4>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2 / 4,
{{indent}}      stream
{{indent}}  );
{{indent}}} else if (x_dim2 % 2 == 0) {
{{indent}}  permute102_launcher<float2>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2 / 2,
{{indent}}      stream
{{indent}}  );
{{indent}}} else {
{{indent}}  permute102_launcher<float>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2,
{{indent}}      stream
{{indent}}  );
{{indent}}}
{% elif dtype == "bfloat16" %}
{{indent}}if (x_dim2 % 8 == 0) {
{{indent}}  permute102_launcher<float4>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2 / 8,
{{indent}}      stream
{{indent}}  );
{{indent}}} else if (x_dim2 % 4 == 0) {
{{indent}}  permute102_launcher<float2>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2 / 4,
{{indent}}      stream
{{indent}}  );
{{indent}}} else if (x_dim2 % 2 == 0) {
{{indent}}  permute102_launcher<float>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2 / 2,
{{indent}}      stream
{{indent}}  );
{{indent}}} else {
{{indent}}  permute102_launcher<bfloat16>(
{{indent}}      in_ptr,
{{indent}}      out_ptr,
{{indent}}      x_dim0,
{{indent}}      x_dim1,
{{indent}}      x_dim2,
{{indent}}      stream
{{indent}}  );
{{indent}}}
{% else %}
{{indent}} static_assert(std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16>, "Unsupported dtype");
{% endif %}
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

#define TILE_SIZE 32
#define ITEMS_PER_THREAD 4
#define DIRECT_BLOCK_Y 4
#define DIRECT_BLOCK_Z 2

namespace {
using bfloat16 = __nv_bfloat16;

template<typename T>
__global__ void permute102_tiled_kernel(T* output,
                                        const T *input,
                                        const int M,
                                        const int N,
                                        const int D,
                                        const int n) {
  __shared__ T shbuf[TILE_SIZE * TILE_SIZE];

  const int nD = n * D;
  const int ND = N * D;
  const int MD = M * D;
  const int bxn = blockIdx.x * n;
  const int DT = D * TILE_SIZE;
  int x, y, i, tid, threadIdxY;

  if (threadIdx.x < nD) {
    x = blockIdx.x * nD + threadIdx.x;
    if (x < ND) {
      threadIdxY = threadIdx.y;
      if ((blockIdx.y + 1) * TILE_SIZE <= M) {
        #pragma unroll
        for (i = 0; i < ITEMS_PER_THREAD; ++i) {
          y = blockIdx.y * TILE_SIZE + threadIdxY;
          shbuf[threadIdxY * TILE_SIZE + (D * threadIdxY + threadIdx.x) % TILE_SIZE] =
            input[y * ND + x];
          threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        }
      } else {
        #pragma unroll
        for (i = 0; i < ITEMS_PER_THREAD; ++i) {
          y = blockIdx.y * TILE_SIZE + threadIdxY;
          if (y >= M) break;
          shbuf[threadIdxY * TILE_SIZE + (D * threadIdxY + threadIdx.x) % TILE_SIZE] =
            input[y * ND + x];
          threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        }
      }
    }
  }

  __syncthreads();

  threadIdxY = threadIdx.y;
  if ((blockIdx.x + 1) * n <= N) {
    if ((blockIdx.y + 1) * TILE_SIZE * D <= MD) {
      #pragma unroll
      for (i = 0; i < ITEMS_PER_THREAD; i++) {
        tid = threadIdxY * TILE_SIZE + threadIdx.x;
        x = tid % DT;
        y = tid / DT;
        output[(bxn + y) * MD + blockIdx.y * DT + x] =
          shbuf[(x / D) * TILE_SIZE + (D * y + x) % TILE_SIZE];
        threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        if (threadIdxY >= nD) break;
      }
    } else {
      #pragma unroll
      for (i = 0; i < ITEMS_PER_THREAD; i++) {
        tid = threadIdxY * TILE_SIZE + threadIdx.x;
        x = tid % DT;
        y = tid / DT;
        if (blockIdx.y * DT + x < MD) {
          output[(bxn + y) * MD + blockIdx.y * DT + x] =
            shbuf[(x / D) * TILE_SIZE + (D * y + x) % TILE_SIZE];
        }
        threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        if (threadIdxY >= nD) break;
      }
    }
  } else {
    if ((blockIdx.y + 1) * TILE_SIZE * D <= MD) {
      #pragma unroll
      for (i = 0; i < ITEMS_PER_THREAD; i++) {
        tid = threadIdxY * TILE_SIZE + threadIdx.x;
        x = tid % DT;
        y = tid / DT;
        if (bxn + y < N) {
          output[(bxn + y) * MD + blockIdx.y * DT + x] =
            shbuf[(x / D) * TILE_SIZE + (D * y + x) % TILE_SIZE];
        }
        threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        if (threadIdxY >= nD) break;
      }
    } else {
      #pragma unroll
      for (i = 0; i < ITEMS_PER_THREAD; i++) {
        tid = threadIdxY * TILE_SIZE + threadIdx.x;
        x = tid % DT;
        y = tid / DT;
        if (bxn + y < N && blockIdx.y * DT + x < MD) {
          output[(bxn + y) * MD + blockIdx.y * DT + x] =
            shbuf[(x / D) * TILE_SIZE + (D * y + x) % TILE_SIZE];
        }
        threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        if (threadIdxY >= nD) break;
      }
    }
  }
}

template <typename T>
__global__ void permute102_direct_kernel(T* output,
                                         const T *input,
                                         const int M,
                                         const int N,
                                         const int D) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < D && y < N) {
    int bound = min(M, (blockIdx.z + 1) * TILE_SIZE);
    for (int z = blockIdx.z * TILE_SIZE + threadIdx.z; z < bound; z += DIRECT_BLOCK_Z) {
      output[y * M * D + z * D + x] = input[z * N * D + y * D + x];
    }
  }
}

template <typename T>
void permute102_launcher(const void* in_ptr,
                         void* out_ptr,
                         int x_dim0,
                         int x_dim1,
                         int x_dim2,
                         {{prefix}}Stream_t stream) {
  const int M = x_dim0;
  const int N = x_dim1;
  const int D = x_dim2;

  if (D <= 16) {
    // each warp reads n x d coalesced items of input
    const int d = min(TILE_SIZE, D);
    const int n = TILE_SIZE / d;

    dim3 grid((N + n - 1) / n, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE / ITEMS_PER_THREAD);

    permute102_tiled_kernel<T><<<grid, block, 0, stream>>>(
      static_cast<T*>(out_ptr),
      static_cast<const T*>(in_ptr),
      M,
      N,
      D,
      n
    );
  } else {
    dim3 grid((D + 31) / 32, (N + DIRECT_BLOCK_Y - 1) / DIRECT_BLOCK_Y, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(32, DIRECT_BLOCK_Y, DIRECT_BLOCK_Z);  // x = 32, the warp size

    permute102_direct_kernel<T><<<grid, block, 0, stream>>>(
      static_cast<T*>(out_ptr),
      static_cast<const T*>(in_ptr),
      M,
      N,
      D
    );
  }
}
} // namespace

void {{function_name}} (
    const void* in_ptr,
    void* out_ptr,
    int64_t x_dim0,
    int64_t x_dim1,
    int64_t x_dim2,
    {{prefix}}Stream_t stream
) {
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
    xdtype = x._attrs["dtype"]
    exec_paths = EXEC_TEMPLATE.render(
        indent="  ",
        dtype=backend_spec.dtype_to_backend_type(xdtype),
    )
    return SRC_TEMPLATE.render(
        function_name=func_name,
        exec_paths=exec_paths,
        header_files=header_files,
        prefix=backend_spec.prefix,
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
    xshape = x._attrs["shape"]
    y = func_attrs["outputs"][0]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        x_dim0=xshape[0]._attrs["name"],
        x_dim1=xshape[1]._attrs["name"],
        x_dim2=xshape[2]._attrs["name"],
        indent=indent,
    )
