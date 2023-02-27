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
GEMM Specialization for A[RowMajor], B[RowMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight

Special kernel for small K and N
K <= 8, N <= 8
A: [M, K] A can be ND with the first N - 1 dimensions as batch dimensions
B: [K, N]
C: [M, N]
"""

import jinja2

from ... import registry
from ...backend_spec import CUDASpec
from ...common import gemm_common
from ...target import Target
from ..gemm_universal import common

# pylint: disable=C0301,W0613,W0612


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
  {% for i in range(a_ndim) %}
  int64_t*,
  {% endfor %}
  {% for i in range(b_ndim) %}
  int64_t*,
  {% endfor %}
  {% for i in range(c_ndim) %}
  int64_t*,
  {% endfor %}
  bool,
  cudaStream_t
);
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{{indent}}    {{c_ptr}},
{% for adim in adims %}
{{indent}}    {{adim}},
{% endfor %}
{% for bdim in bdims %}
{{indent}}    {{bdim}},
{% endfor %}
{% for cdim in cdims %}
{{indent}}    {{cdim}},
{% endfor %}
{{indent}}    {{use_fp16_acc}},
{{indent}}    stream
{{indent}});
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}gemm_rrr_small_nk_launcher<{{elem_input_type}}, {{N}}, {{K}}>(
{{indent}}    ({{elem_input_type}}*)a_ptr,
{{indent}}    ({{elem_input_type}}*)b_ptr,
{{indent}}    ({{elem_input_type}}*)c_ptr,
{{indent}}    M,
{{indent}}    use_fp16_acc,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

namespace {

using bfloat16 = __nv_bfloat16;

__device__ float fma(float a, float b, float c) {
  return __fmaf_rn(a, b, c);
}

__device__ half fma(half a, half b, half c) {
  return __hfma(a, b, c);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ bfloat16 fma(bfloat16 a, bfloat16 b, bfloat16 c) {
  return __hfma(a, b, c);
}
#endif

// For each thread, read
// A tile: 8 x K
// B matrix: K x N
// C tile: 8 x N
template<typename TElem, int num_thread, int N, int K, bool USE_FP16_ACC>
__global__ void gemm_rrr_small_nk_kernel(
    const float4* a_ptr, const float4* b_ptr, float4* c_ptr, int M) {
  int idx = blockIdx.x * num_thread + threadIdx.x;
  constexpr int num_elems_in_float4 = sizeof(float4) / sizeof(TElem);

  if (idx >= (M + num_elems_in_float4 - 1) / num_elems_in_float4) {
    return;
  }

  int a_idx_base = idx * K;
  a_ptr += a_idx_base;

  // load b matrix
  TElem b[K][N];
  auto* b_e = reinterpret_cast<const TElem*>(b_ptr);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      b[i][j] = b_e[i * N + j];
    }
  }

  int c_idx_base = idx * N;
  c_ptr += c_idx_base;

  TElem c_tile[num_elems_in_float4][N];

  if (idx <= M / num_elems_in_float4 - 1) {
    // fast kernel
    // load a
    float4 a_tile_vec[K];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < K; i++) {
      a_tile_vec[i] = __ldg(a_ptr++);
    }
    auto* a_tile = reinterpret_cast<const TElem*>(&a_tile_vec);

    // compute
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < num_elems_in_float4; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < N; ++j) {
        if constexpr (USE_FP16_ACC) {
          TElem sum = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < K; ++k) {
            sum = fma(a_tile[i * K + k], b[k][j], sum);
          }
          c_tile[i][j] = sum;
        } else {
          float sum = 0;
          if constexpr (std::is_same_v<TElem, half>) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
              sum += __half2float(__hmul(a_tile[i * K + k], b[k][j]));
            }
            c_tile[i][j] = __float2half_rn(sum);
          } else {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
              sum += __fmul_rn(a_tile[i * K + k], b[k][j]);
            }
            c_tile[i][j] = sum;
          }
        }
      }
    }

    // write c
    float4* c_tile_vec = reinterpret_cast<float4*>(&c_tile);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; i++) {
      c_ptr[i] = c_tile_vec[i];
    }
  } else {
    // process tail
    // load a
    auto* a_e = reinterpret_cast<const TElem*>(a_ptr);
    int m = M - M / num_elems_in_float4 * num_elems_in_float4;
    TElem a_tile[num_elems_in_float4][K];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < m; i++) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < K; j++) {
        a_tile[i][j] = a_e[i * K + j];
      }
    }

    // compute
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < m; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < N; ++j) {
        if constexpr (USE_FP16_ACC) {
          TElem sum = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < K; ++k) {
            sum = fma(a_tile[i][k], b[k][j], sum);
          }
          c_tile[i][j] = sum;
        } else {
          float sum = 0;
          if constexpr (std::is_same_v<TElem, half>) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
              sum += __half2float(__hmul(a_tile[i][k], b[k][j]));
            }
            c_tile[i][j] = __float2half_rn(sum);
          }
          else {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
              sum += __fmul_rn(a_tile[i][k], b[k][j]);
            }
            c_tile[i][j] = sum;
          }
        }
      }
    }

    // write c
    auto* c_h = reinterpret_cast<TElem*>(c_ptr);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < m; i++) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < N; j++) {
        c_h[i * N + j] = c_tile[i][j];
      }
    }
  }
}

// N <= 8, K <= 8
template<typename ElemT, int N, int K,
         typename = std::enable_if_t<std::is_same_v<ElemT, float> || std::is_same_v<ElemT, half> || std::is_same_v<ElemT, bfloat16>, void>>
void gemm_rrr_small_nk_launcher(ElemT* a_ptr,
                         ElemT* b_ptr,
                         ElemT* c_ptr,
                         int M,
                         bool use_fp16_acc,
                         cudaStream_t stream) {
  constexpr int num_elems_in_float4 = sizeof(float4) / sizeof(ElemT);
  const int nthread = 256;
  dim3 thread_block(nthread);
  constexpr int n_element_per_t = nthread * num_elems_in_float4;
  dim3 grid((M + n_element_per_t - 1) / n_element_per_t);
  if (use_fp16_acc && (std::is_same_v<ElemT, half> || std::is_same_v<ElemT, bfloat16>)) {
    gemm_rrr_small_nk_kernel<ElemT, nthread, N, K, true><<<grid, thread_block, 0, stream>>>(
      reinterpret_cast<const float4*>(a_ptr),
      reinterpret_cast<const float4*>(b_ptr),
      reinterpret_cast<float4*>(c_ptr),
      M
    );
  } else {
    gemm_rrr_small_nk_kernel<ElemT, nthread, N, K, false><<<grid, thread_block, 0, stream>>>(
      reinterpret_cast<const float4*>(a_ptr),
      reinterpret_cast<const float4*>(b_ptr),
      reinterpret_cast<float4*>(c_ptr),
      M
    );
  }
}

} // namespace

void {{function_name}} (
    void* a_ptr,
    void* b_ptr,
    void* c_ptr,
    {% for i in range(a_ndim) %}
    int64_t *a_dim{{loop.index0}},
    {% endfor %}
    {% for i in range(b_ndim) %}
    int64_t *b_dim{{loop.index0}},
    {% endfor %}
    {% for i in range(c_ndim) %}
    int64_t *c_dim{{loop.index0}},
    {% endfor %}
    bool use_fp16_acc,
    cudaStream_t stream
) {
  {{shape_function}}
  {{input_output_checks}}
  {{exec_paths}}
}

"""
)


@registry.reg("cuda.gemm_rrr_small_nk.gen_function")
def gen_function(func_attrs, exec_cond_template, dim_info_dict):
    func_name = func_attrs["name"]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=1, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    b = func_attrs["inputs"][1]
    bshape = b._attrs["shape"]
    k = bshape[0]._attrs["values"][0]
    n = bshape[1]._attrs["values"][0]

    a_ndim = func_attrs["inputs"][0]._rank()
    b_ndim = func_attrs["inputs"][1]._rank()
    c_ndim = func_attrs["outputs"][0]._rank()

    input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=a_ndim,
        weight_ndims=2,
        output_ndims=c_ndim,
    )
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    if n == 0 or k == 0:
        # avoid "zero-sized variable not allowed in device code" error
        exec_paths = ""
    else:
        exec_paths = EXEC_TEMPLATE.render(
            indent="  ", elem_input_type=elem_input_type, N=n, K=k
        )
    return SRC_TEMPLATE.render(
        function_name=func_name,
        shape_function=shape_func,
        input_output_checks=input_output_checks,
        exec_paths=exec_paths,
        a_ndim=a_ndim,
        b_ndim=b_ndim,
        c_ndim=c_ndim,
    )


@registry.reg("cuda.gemm_rrr_small_nk.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    a_ndim = func_attrs["inputs"][0]._rank()
    b_ndim = func_attrs["inputs"][1]._rank()
    c_ndim = func_attrs["outputs"][0]._rank()
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name, a_ndim=a_ndim, b_ndim=b_ndim, c_ndim=c_ndim
    )


@registry.reg("cuda.gemm_rrr_small_nk.func_call")
def gen_function_call(func_attrs, indent="  "):
    a = func_attrs["inputs"][0]
    ashape = a._attrs["shape"]
    adims = ["&" + dim._attrs["name"] for dim in ashape]
    b = func_attrs["inputs"][1]
    bshape = b._attrs["shape"]
    bdims = ["&" + dim._attrs["name"] for dim in bshape]
    c = func_attrs["outputs"][0]
    cshape = c._attrs["shape"]
    cdims = ["&" + dim._attrs["name"] for dim in cshape]
    use_fp16_acc = False
    if "use_fp16_acc" in Target.current()._kwargs:
        use_fp16_acc = Target.current()._kwargs["use_fp16_acc"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        c_ptr=c._attrs["name"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        use_fp16_acc="true" if use_fp16_acc else "false",
        indent=indent,
    )


@registry.reg("cuda.gemm_rrr_small_nk.filter")
def function_filter(cfg, func_attrs, ab_alignment):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    ab_alignment:
        Input alignments.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common.function_filter(cfg, func_attrs, ab_alignment)
