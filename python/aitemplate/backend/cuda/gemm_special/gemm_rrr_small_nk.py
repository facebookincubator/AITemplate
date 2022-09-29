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
from ...common import gemm_common
from ...target import Target
from ..gemm_universal import common

# pylint: disable=C0301,W0613,W0612


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
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
{{indent}}gemm_rrr_small_nk_launcher<{{N}}, {{K}}>(
{{indent}}    a_ptr,
{{indent}}    b_ptr,
{{indent}}    c_ptr,
{{indent}}    M,
{{indent}}    use_fp16_acc,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

namespace {

// For each thread, read
// A tile: 8 x K
// B matrix: K x N
// C tile: 8 x N
template<int num_thread, int N, int K, bool USE_FP16_ACC>
__global__ void gemm_rrr_small_nk_kernel(float4* a_ptr,
                                         float4* b_ptr,
                                         float4* c_ptr,
                                         int M) {
  int idx = blockIdx.x * num_thread + threadIdx.x;

  if (idx >= (M + 7) / 8) {
    return;
  }

  int a_idx_base = idx * K;
  a_ptr += a_idx_base;

  // load b matrix
  half b[K][N];
  half* b_half = reinterpret_cast<half*>(b_ptr);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      b[i][j] = b_half[i * N + j];
    }
  }

  int c_idx_base = idx * N;
  c_ptr += c_idx_base;

  half c_tile[8][N];

  if (idx <= M / 8 - 1) {
    // fast kernel
    // load a
    float4 a_tile_vec[K];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < K; i++) {
      a_tile_vec[i] = __ldg(a_ptr++);
    }
    half* a_tile = reinterpret_cast<half*>(&a_tile_vec);

    // compute
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < N; ++j) {
        if (USE_FP16_ACC) {
          half sum = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < K; ++k) {
            sum = __hfma(a_tile[i * K + k], b[k][j], sum);
          }
          c_tile[i][j] = sum;
        } else {
          float sum = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < K; ++k) {
            sum += __half2float(__hmul(a_tile[i * K + k], b[k][j]));
          }
          c_tile[i][j] = __float2half_rn(sum);
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
    half* a_h = reinterpret_cast<half*>(a_ptr);
    int m = M - M / 8 * 8;
    half a_tile[8][K];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < m; i++) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < K; j++) {
        a_tile[i][j] = a_h[i * K + j];
      }
    }

    // compute
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < m; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < N; ++j) {
        if (USE_FP16_ACC) {
          half sum = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < K; ++k) {
            sum = __hfma(a_tile[i][k], b[k][j], sum);
          }
          c_tile[i][j] = sum;
        } else {
          float sum = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < K; ++k) {
            sum += __half2float(__hmul(a_tile[i][k], b[k][j]));
          }
          c_tile[i][j] = __float2half_rn(sum);
        }
      }
    }

    // write c
    half* c_h = reinterpret_cast<half*>(c_ptr);
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
template<int N, int K>
void gemm_rrr_small_nk_launcher(cutlass::half_t* a_ptr,
                         cutlass::half_t* b_ptr,
                         cutlass::half_t* c_ptr,
                         int M,
                         bool use_fp16_acc,
                         cudaStream_t stream) {
  const int nthread = 256;
  dim3 thread_block(nthread);
  const int n_element_per_t = nthread * 8;
  dim3 grid((M + n_element_per_t - 1) / n_element_per_t);
  if(use_fp16_acc) {
    gemm_rrr_small_nk_kernel<nthread, N, K, true><<<grid, thread_block, 0, stream>>>(
      (float4*)a_ptr,
      (float4*)b_ptr,
      (float4*)c_ptr,
      M
    );
  } else {
    gemm_rrr_small_nk_kernel<nthread, N, K, false><<<grid, thread_block, 0, stream>>>(
      (float4*)a_ptr,
      (float4*)b_ptr,
      (float4*)c_ptr,
      M
    );
  }
}

} // namespace

void {{function_name}} (
    cutlass::half_t* a_ptr,
    cutlass::half_t* b_ptr,
    cutlass::half_t* c_ptr,
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
    if n == 0 or k == 0:
        # avoid "zero-sized variable not allowed in device code" error
        exec_paths = ""
    else:
        exec_paths = EXEC_TEMPLATE.render(indent="  ", N=n, K=k)
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
