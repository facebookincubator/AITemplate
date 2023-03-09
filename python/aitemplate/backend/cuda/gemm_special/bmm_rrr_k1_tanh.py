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
Codegen for bmm_rrr_k1_tanh.

This kernel computes C = tanh(alpha * A @ B), where:
A[RowMajor]: [B, M, 1]
B[RowMajor]: [B, 1, N]
C[RowMajor]: [B, M, N]
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common import gemm_common
from aitemplate.backend.cuda.gemm_universal import common

# pylint: disable=C0301,W0613,W0612

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
  {% for i in range(3) %}
  int64_t*,
  {% endfor %}
  {% for i in range(3) %}
  int64_t*,
  {% endfor %}
  {% for i in range(3) %}
  int64_t*,
  {% endfor %}
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
{{indent}} {{adim}},
{% endfor %}
{% for bdim in bdims %}
{{indent}} {{bdim}},
{% endfor %}
{% for cdim in cdims %}
{{indent}} {{cdim}},
{% endfor %}
{{indent}}  stream
{{indent}});
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}bmm_rrr_k1_tanh_launcher<{{elem_input_type}}>(
{{indent}}    ({{elem_input_type}}*)a_ptr,
{{indent}}    ({{elem_input_type}}*)b_ptr,
{{indent}}    ({{elem_input_type}}*)c_ptr,
{{indent}}    B,
{{indent}}    M,
{{indent}}    N,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"
#include "cutlass/fast_math.h"

using bfloat16 = __nv_bfloat16;

#ifndef REINTERPRET_AS_U16
#define REINTERPRET_AS_U16(var) *(reinterpret_cast<unsigned short *>(&(var)))
#endif

namespace {

template <typename T>
__device__ __inline__ T fast_tanh(T x);

template <>
__device__ __inline__ half fast_tanh(half x) {
  #if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)

  asm volatile ( "tanh.approx.f16 %0, %1;" : "=h"(REINTERPRET_AS_U16(x)) : "h"(REINTERPRET_AS_U16(x)));
  return x;

  #else
  return half(cutlass::fast_tanh(float(x)));
  #endif
}

#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 800)

template <>
__device__ __inline__ bfloat16 fast_tanh(bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 12) && (__CUDA_ARCH__ >= 900)

  asm volatile ( "tanh.approx.bf16 %0, %1;" : "=h"(REINTERPRET_AS_U16(x)) : "h"(REINTERPRET_AS_U16(x)));
  return x;

#else
  return bfloat16(cutlass::fast_tanh(float(x)));
#endif
}

#endif // (__CUDA_ARCH__ >= 800)

template <>
__device__ __inline__ float fast_tanh(float x) {
  return cutlass::fast_tanh(x);
}

template<typename ElemT>
__device__ __inline__ ElemT intrinsic_mul(ElemT x, ElemT y);

template<>
__device__ __inline__ float intrinsic_mul(float x, float y) {
  return __fmul_rn(x, y);
}

template<>
__device__ __inline__ half intrinsic_mul(half x, half y) {
  return __hmul(x, y);
}

#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 800)

template<>
__device__ __inline__ bfloat16 intrinsic_mul(bfloat16 x, bfloat16 y) {
  return __hmul(x, y);
}

#endif

template<typename ElemT, int num_thread>
__global__ void bmm_rrr_k1_tanh_kernel(const float4* a_ptr,
                                  const float4* b_ptr,
                                  float4* c_ptr,
                                  const int B,
                                  const int M,
                                  const int N) {
  // TODO: check boundary
  constexpr int num_elems_in_float4 = sizeof(float4) / sizeof(ElemT);
  ElemT tmp[num_elems_in_float4 * num_elems_in_float4];
  int idx = blockIdx.x * num_thread + threadIdx.x;
  int m = idx % M;
  int b = idx / M;
  int a_idx_base = b * M + m;
  float4 a_vec = __ldg(a_ptr + a_idx_base);
  ElemT* a_vec_ptr = (ElemT*)(&a_vec);
  for (int n = 0; n < N; ++n) {
    int b_idx_base = b * N + n;
    float4 b_vec = __ldg(b_ptr + b_idx_base);
    ElemT* b_vec_ptr = (ElemT*)(&b_vec);
    for (int i = 0; i < num_elems_in_float4; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < num_elems_in_float4; ++j) {
        tmp[i * num_elems_in_float4 + j] = fast_tanh(intrinsic_mul(a_vec_ptr[i], b_vec_ptr[j]));
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < num_elems_in_float4; ++i) {
        int c_idx = (b * M * num_elems_in_float4  + m * num_elems_in_float4 + i) * N  + n;
        c_ptr[c_idx] = *((const float4*)(tmp + i * num_elems_in_float4));
    }
  }
}


template <typename ElemT>
void bmm_rrr_k1_tanh_launcher(ElemT* a_ptr,
                         ElemT* b_ptr,
                         ElemT* c_ptr,
                         int B,
                         int M,
                         int N,
                         cudaStream_t stream) {
  constexpr int num_elems_in_float4 = sizeof(float4) / sizeof(ElemT);
  if (M % num_elems_in_float4 != 0) {
     auto msg = std::string("Got error: ") + std::to_string(M) + "%" +
       std::to_string(num_elems_in_float4) + " != 0 " +
       " at " + __FILE__ + ": " + std::to_string(__LINE__);
     std::cerr << msg << std::endl;
     throw std::runtime_error(msg);
  }
  if (N % num_elems_in_float4 != 0) {
     auto msg = std::string("Got error: ") + std::to_string(N) + "%" +
       std::to_string(num_elems_in_float4) + " != 0 " +
       " at " + __FILE__ + ": " + std::to_string(__LINE__);
     std::cerr << msg << std::endl;
     throw std::runtime_error(msg);
  }
  const int nthread = 256;
  dim3 thread_block(nthread);
  dim3 grid(B * M / nthread / num_elems_in_float4);
  bmm_rrr_k1_tanh_kernel<ElemT, nthread><<<grid, thread_block, 0, stream>>>(
    (const float4*)a_ptr,
    (const float4*)b_ptr,
    (float4*) c_ptr,
    B,
    M / num_elems_in_float4,
    N / num_elems_in_float4
  );
}

} // namespace

void {{function_name}} (
    void* a_ptr,
    void* b_ptr,
    void* c_ptr,
    {% for i in range(3) %}
    int64_t *a_dim{{loop.index0}},
    {% endfor %}
    {% for i in range(3) %}
    int64_t *b_dim{{loop.index0}},
    {% endfor %}
    {% for i in range(3) %}
    int64_t *c_dim{{loop.index0}},
    {% endfor %}
    cudaStream_t stream
) {
  {{shape_function}}
  {{input_output_checks}}
  {{exec_paths}}
}

#undef REINTERPRET_AS_U16
"""
)


@registry.reg("cuda.bmm_rrr_k1_tanh.gen_function")
def gen_function(func_attrs, exec_cond_template, dim_info_dict):
    func_name = func_attrs["name"]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=1, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )
    input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=3,
        weight_ndims=3,
        output_ndims=3,
    )
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    exec_paths = EXEC_TEMPLATE.render(elem_input_type=elem_input_type)
    return SRC_TEMPLATE.render(
        function_name=func_name,
        shape_function=shape_func,
        input_output_checks=input_output_checks,
        exec_paths=exec_paths,
    )


@registry.reg("cuda.bmm_rrr_k1_tanh.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.bmm_rrr_k1_tanh.func_call")
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
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        c_ptr=c._attrs["name"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
    )


@registry.reg("cuda.bmm_rrr_k1_tanh.filter")
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
