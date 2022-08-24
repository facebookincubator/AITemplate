# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
import jinja2

from ... import registry
from ...common import gemm_common
from ..gemm_universal import common

# pylint: disable=C0301,W0613,W0612

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
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
{{indent}}bmm_rrr_k1_tanh_launcher(
{{indent}}    a_ptr,
{{indent}}    b_ptr,
{{indent}}    c_ptr,
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"
#include "cutlass/fast_math.h"

#ifndef __HALF_TO_US
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#endif

namespace {

__device__ half fast_tanh(half x) {
  #if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)

  asm volatile ( "tanh.approx.f16 %0, %1;" : "=h"(__HALF_TO_US(x)) : "h"(__HALF_TO_US(x)));
  return x;

  #else
  return half(cutlass::fast_tanh(float(x)));
  #endif
}

template<int num_thread>
__global__ void bmm_rrr_k1_tanh_kernel(const float4* a_ptr,
                                  const float4* b_ptr,
                                  float4* c_ptr,
                                  const int B,
                                  const int M,
                                  const int N) {
  // TODO: check boundary
  half tmp[64];
  int idx = blockIdx.x * num_thread + threadIdx.x;
  int m = idx % M;
  int b = idx / M;
  int a_idx_base = b * M + m;
  float4 a_vec = __ldg(a_ptr + a_idx_base);
  half* a_vec_ptr = (half*)(&a_vec);
  for (int n = 0; n < N; ++n) {
    int b_idx_base = b * N + n;
    float4 b_vec = __ldg(b_ptr + b_idx_base);
    half* b_vec_ptr = (half*)(&b_vec);
    for (int i = 0; i < 8; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j) {
        tmp[i * 8 + j] = fast_tanh(__hmul(a_vec_ptr[i], b_vec_ptr[j]));
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
        int c_idx = (b * M * 8  + m * 8 + i) * N  + n;
        c_ptr[c_idx] = *((const float4*)(tmp + i * 8));
    }
  }
}


void bmm_rrr_k1_tanh_launcher(cutlass::half_t* a_ptr,
                         cutlass::half_t* b_ptr,
                         cutlass::half_t* c_ptr,
                         int B,
                         int M,
                         int N,
                         cudaStream_t stream) {
  const int nthread = 256;
  dim3 thread_block(nthread);
  dim3 grid(B * M / nthread / 8);
  bmm_rrr_k1_tanh_kernel<nthread><<<grid, thread_block, 0, stream>>>(
    (const float4*)a_ptr,
    (const float4*)b_ptr,
    (float4*) c_ptr,
    B,
    M / 8,
    N / 8
  );
}

} // namespace

void {{function_name}} (
    cutlass::half_t* a_ptr,
    cutlass::half_t* b_ptr,
    cutlass::half_t* c_ptr,
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
    exec_paths = EXEC_TEMPLATE.render()
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
