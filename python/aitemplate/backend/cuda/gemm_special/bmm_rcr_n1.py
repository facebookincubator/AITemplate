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
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight

Special kernel for GEMV case:
A: [B, M, K]
B: [B, N, K]
C: [B, M, N]
where N = 1

This kernel computes C = alpha * A @ B
"""

import jinja2

from ....compiler.base import IntImm

from ... import registry
from ...backend_spec import CUDASpec
from ...common import gemm_common, tensor_accessor_codegen
from ...target import Target
from ..gemm_universal import common

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
  float,
  bool,
  cudaStream_t
);
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}{{local_dim_defs}}
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{{indent}}    {{c_ptr}},
{% for adim in adims %}
{{indent}}    {{adim}},
{% endfor %}
{% for bdim in bdims %}
{{indent}}     {{bdim}},
{% endfor %}
{% for cdim in cdims %}
{{indent}}    {{cdim}},
{% endfor %}
{{indent}}    {{alpha}},
{{indent}}    {{use_fp16_acc}},
{{indent}}    stream
{{indent}});
{{indent}}}
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}bmm_rcr_n1_launcher<{{elem_input_type}}, {{read_vec_type}}, {{K}}>(
{{indent}}    ({{elem_input_type}}*)a_ptr,
{{indent}}    ({{elem_input_type}}*)b_ptr,
{{indent}}    ({{elem_input_type}}*)c_ptr,
{{indent}}    B,
{{indent}}    M,
{{indent}}    alpha,
{{indent}}    use_fp16_acc,
{{indent}}    stream,
{{intent}}    input_a_accessor,
{{intent}}    input_b_accessor,
{{intent}}    output_accessor
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

{{tensor_accessor_libs}}

template<typename ElemT, typename ReadVecT, int64_t K>
__forceinline__ __device__ bool load_vec_data(
    ReadVecT* a_ptr,
    ReadVecT* b_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor,
    ReadVecT *a_vec,
    ReadVecT *b_vec) {

  int64_t batch_idx = blockIdx.y;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr int64_t N_READ_ELEMS_IN_V = sizeof(ReadVecT) / sizeof(ElemT);
  constexpr int64_t N_NUM_ELEMS_IN_V = K / N_READ_ELEMS_IN_V;

  int64_t b_idx_base = (batch_idx * K) / N_READ_ELEMS_IN_V;

  if (blockDim.x >= N_NUM_ELEMS_IN_V) {
    // We have enough threads in a thread block where each thread takes care
    // of loading one vector.
    if (threadIdx.x < N_NUM_ELEMS_IN_V) {
      b_vec[threadIdx.x] = *input_b_accessor.get<ElemT, ReadVecT>(b_ptr, b_idx_base + threadIdx.x);
    }
  } else {
    // We have more vectors than the available threads of a thread block, so each
    // thread may read multiple vectors.
    for (int64_t i = 0; i < N_NUM_ELEMS_IN_V / blockDim.x + 1; i++) {
      int64_t idx = i * blockDim.x + threadIdx.x;
      if (idx < N_NUM_ELEMS_IN_V) {
        b_vec[idx] = *input_b_accessor.get<ElemT, ReadVecT>(b_ptr, b_idx_base + idx);
      }
    }
  }

  __syncthreads();
  if (row_idx >= M) {
    return false;
  }

  int64_t a_batch_stride = M * K;
  int64_t a_idx_base = (batch_idx * a_batch_stride + row_idx * K) / N_READ_ELEMS_IN_V;

  CUTLASS_PRAGMA_UNROLL
  for (int64_t k = 0, i = 0; k < K; k += N_READ_ELEMS_IN_V, i++) {
    a_vec[i] = *input_a_accessor.get<ElemT, ReadVecT>(a_ptr, a_idx_base++);
  }

  return true;
}

// Each thread reads one row from "a" and one column from "b",
// computes dot_product(a_row, b_col), and writes the result to "c".
// This kernel assumes loading "a" and "b" can be fully vectorized,
// so it reads both "a" and "b" in ReadVecT.
template<typename ElemT, typename ReadVecT, int64_t K>
__global__ void bmm_rcr_n1_kernel_fp32_acc_vec(
    ReadVecT* a_ptr,
    ReadVecT* b_ptr,
    ElemT* c_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor) {

  static_assert(sizeof(ReadVecT) % sizeof(ElemT) == 0, "invalid vector type");
  constexpr int64_t N_READ_ELEMS_IN_V = sizeof(ReadVecT) / sizeof(ElemT);
  static_assert(N_READ_ELEMS_IN_V % 2 == 0, "invalid vector type for read");
  static_assert(K % N_READ_ELEMS_IN_V == 0, "cannot vectorize input");
  constexpr int64_t N_NUM_ELEMS_IN_V = K / N_READ_ELEMS_IN_V;

  __shared__ ReadVecT b_vec[N_NUM_ELEMS_IN_V];
  ReadVecT a_vec[N_NUM_ELEMS_IN_V];

  if (!load_vec_data<ElemT, ReadVecT, K>(
        a_ptr, b_ptr, M, alpha, input_a_accessor, input_b_accessor,
        output_accessor, a_vec, b_vec)) {
    return;
  }

  float result = 0.0;

  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < N_NUM_ELEMS_IN_V; i++) {
    const half2* a_vec_h2 = reinterpret_cast<const half2*>(&a_vec[i]);
    const half2* b_vec_h2 = reinterpret_cast<const half2*>(&b_vec[i]);
    CUTLASS_PRAGMA_UNROLL
    for (int64_t j = 0; j < N_READ_ELEMS_IN_V / 2; ++j) {
      half2 c_h2 = __hmul2(a_vec_h2[j], b_vec_h2[j]);
      result += float(__low2half(c_h2)) + float(__high2half(c_h2));
    }
  }

  int64_t batch_idx = blockIdx.y;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  *output_accessor.get<ElemT, ElemT>(c_ptr, batch_idx * M + row_idx) = alpha * result;
}

template<typename ElemT, int64_t K>
__forceinline__ __device__ bool load_data(
    ElemT* a_ptr,
    ElemT* b_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor,
    ElemT *a_data,
    ElemT *b_data) {

  int64_t batch_idx = blockIdx.y;
  int64_t b_idx_base = batch_idx * K;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (blockDim.x >= K) {
    // We have enough threads in a thread block where each thread takes care
    // of loading one element.
    if (threadIdx.x < K) {
      b_data[threadIdx.x] = *input_b_accessor.get<ElemT, ElemT>(b_ptr, b_idx_base + threadIdx.x);
    }
  } else {
    // We have more elements than the available threads of a thread block, so each
    // thread may load multiple elements.
    for (int64_t i = 0; i < K / blockDim.x + 1; i++) {
      int64_t idx = i * blockDim.x + threadIdx.x;
      if (idx < K) {
        b_data[idx] = *input_b_accessor.get<ElemT, ElemT>(b_ptr, b_idx_base + idx);
      }
    }
  }

  __syncthreads();

  if (row_idx >= M) {
    return false;
  }

  int64_t a_batch_stride = M * K;
  int64_t a_idx_base = batch_idx * a_batch_stride + row_idx * K;

  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < K; i++) {
    a_data[i] = *input_a_accessor.get<ElemT, ElemT>(a_ptr, a_idx_base++);
  }

  return true;
}

// Each thread reads one row from "a" and one column from "b",
// computes dot_product(a_row, b_col), and writes the result to "c".
// It reads both "a" and "b" one by one in ElemT.
template<typename ElemT, typename ReadVecT, int64_t K>
__global__ void bmm_rcr_n1_kernel_fp32_acc(
    ElemT* a_ptr,
    ElemT* b_ptr,
    ElemT* c_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor) {

  __shared__ ElemT b_data[K];
  ElemT a_data[K];

  if (!load_data<ElemT, K>(
        a_ptr, b_ptr, M, alpha, input_a_accessor, input_b_accessor,
        output_accessor, a_data, b_data)) {
    return;
  }

  float result = 0.0;

  const half2* a_data_h2 = reinterpret_cast<const half2*>(&a_data[0]);
  const half2* b_data_h2 = reinterpret_cast<const half2*>(&b_data[0]);
  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < K / 2; ++i) {
    half2 c_h2 = __hmul2(a_data_h2[i], b_data_h2[i]);
    result += float(__low2half(c_h2)) + float(__high2half(c_h2));
  }
  if (K % 2) {
    result += float(__hmul(reinterpret_cast<half&>(a_data[K-1]),
                           reinterpret_cast<half&>(b_data[K-1])));
  }

  int64_t batch_idx = blockIdx.y;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  *output_accessor.get<ElemT, ElemT>(c_ptr, batch_idx * M + row_idx) = alpha * result;
}

template<typename ElemT, typename ReadVecT, int64_t K>
__global__ void bmm_rcr_n1_kernel_fp16_acc_vec(
    ReadVecT* a_ptr,
    ReadVecT* b_ptr,
    ElemT* c_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor) {

  static_assert(sizeof(ReadVecT) % sizeof(ElemT) == 0, "invalid vector type");
  constexpr int64_t N_READ_ELEMS_IN_V = sizeof(ReadVecT) / sizeof(ElemT);
  static_assert(N_READ_ELEMS_IN_V % 2 == 0, "invalid vector type for read");
  static_assert(K % N_READ_ELEMS_IN_V == 0, "cannot vectorize input");
  constexpr int64_t N_NUM_ELEMS_IN_V = K / N_READ_ELEMS_IN_V;

  __shared__ ReadVecT b_vec[N_NUM_ELEMS_IN_V];
  ReadVecT a_vec[N_NUM_ELEMS_IN_V];

  if (!load_vec_data<ElemT, ReadVecT, K>(
        a_ptr, b_ptr, M, alpha, input_a_accessor, input_b_accessor,
        output_accessor, a_vec, b_vec)) {
    return;
  }

  half2 result_h2 = {0.0, 0.0};

  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < N_NUM_ELEMS_IN_V; i++) {
    const half2* a_vec_h2 = reinterpret_cast<const half2*>(&a_vec[i]);
    const half2* b_vec_h2 = reinterpret_cast<const half2*>(&b_vec[i]);
    CUTLASS_PRAGMA_UNROLL
    for (int64_t j = 0; j < N_READ_ELEMS_IN_V / 2; ++j) {
      result_h2 = __hfma2(a_vec_h2[j], b_vec_h2[j], result_h2);
    }
  }

  float result = __hadd(__low2half(result_h2), __high2half(result_h2));

  int64_t batch_idx = blockIdx.y;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  *output_accessor.get<ElemT, ElemT>(c_ptr, batch_idx * M + row_idx) = alpha * result;
}

template<typename ElemT, typename ReadVecT, int64_t K>
__global__ void bmm_rcr_n1_kernel_fp16_acc(
    ElemT* a_ptr,
    ElemT* b_ptr,
    ElemT* c_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor) {

  __shared__ ElemT b_data[K];
  ElemT a_data[K];

  if (!load_data<ElemT, K>(
        a_ptr, b_ptr, M, alpha, input_a_accessor, input_b_accessor,
        output_accessor, a_data, b_data)) {
    return;
  }

  half2 result_h2 = {0.0, 0.0};

  const half2* a_data_h2 = reinterpret_cast<const half2*>(&a_data[0]);
  const half2* b_data_h2 = reinterpret_cast<const half2*>(&b_data[0]);
  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < K / 2; ++i) {
    result_h2 = __hfma2(a_data_h2[i], b_data_h2[i], result_h2);
  }

  half result = __hadd(__low2half(result_h2), __high2half(result_h2));
  if (K % 2) {
    result = __hfma(reinterpret_cast<const half&>(a_data[K-1]),
                    reinterpret_cast<const half&>(b_data[K-1]),
                    result);
  }

  int64_t batch_idx = blockIdx.y;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  *output_accessor.get<ElemT, ElemT>(c_ptr, batch_idx * M + row_idx) =
      alpha * (float)result;
}

// N = 1, K is small
template<typename ElemT, typename ReadVecT, int64_t K>
void bmm_rcr_n1_launcher(ElemT* a_ptr,
                         ElemT* b_ptr,
                         ElemT* c_ptr,
                         int64_t B,
                         int64_t M,
                         float alpha,
                         bool use_fp16_acc,
                         cudaStream_t stream,
                         const TensorAccessor& input_a_accessor,
                         const TensorAccessor& input_b_accessor,
                         const TensorAccessor& output_accessor) {
  const int nthread = 256;
  dim3 thread_block(nthread);
  dim3 grid((M + nthread - 1) / nthread, B);

  if(use_fp16_acc) {
    {{bmm_rcr_n1_kernel_fp16}}<ElemT, ReadVecT, K>
    <<<grid, thread_block, 0, stream>>>(
      (ReadVecT*)a_ptr,
      (ReadVecT*)b_ptr,
      c_ptr,
      M,
      alpha,
      input_a_accessor,
      input_b_accessor,
      output_accessor
    );
  } else {
    {{bmm_rcr_n1_kernel_fp32}}<ElemT, ReadVecT, K>
    <<<grid, thread_block, 0, stream>>>(
      (ReadVecT*)a_ptr,
      (ReadVecT*)b_ptr,
      c_ptr,
      M,
      alpha,
      input_a_accessor,
      input_b_accessor,
      output_accessor
    );
  }
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
    float alpha,
    bool use_fp16_acc,
    cudaStream_t stream
) {
  {{shape_function}}
  {{input_output_checks}}
  {{input_accessors}}
  {{output_accessors}}
  {{exec_paths}}
}

"""
)


@registry.reg("cuda.bmm_rcr_n1.gen_function")
def gen_function(func_attrs, exec_cond_template, dim_info_dict):
    func_name = func_attrs["name"]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=1, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    def _get_original_dim_val(func_attrs, input_idx, dim):
        accessor = func_attrs["input_accessors"][input_idx]
        shape = accessor.original_shapes
        assert isinstance(
            shape[dim], IntImm
        ), f"input {input_idx}'s dim {dim} must be static. Instead it's dynamic"
        k = shape[dim]._attrs["values"][0]
        return k

    # Get original k value in case it's changed to a strided tensor after
    # fusing split op into bmm_rcr. Strided dim can only be the last dim.
    ak = _get_original_dim_val(func_attrs, 0, 2)
    bk = _get_original_dim_val(func_attrs, 1, 2)
    assert ak == bk, f"ak is not equal to bk. ak: {ak}, bk: {bk}"

    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    vec_lens = list(zip(*backend_spec.read_num_elements_to_backend_type))[0][:-1]
    alignment = tensor_accessor_codegen.find_max_alignment(
        ak, func_attrs["input_accessors"]
    )
    if alignment % 2:
        bmm_rcr_n1_kernel_fp32 = "bmm_rcr_n1_kernel_fp32_acc"
        bmm_rcr_n1_kernel_fp16 = "bmm_rcr_n1_kernel_fp16_acc"
        read_vec_type = elem_input_type
    else:
        for vec_idx, vec_len in enumerate(vec_lens):
            if ak % vec_len == 0:
                bmm_rcr_n1_kernel_fp32 = "bmm_rcr_n1_kernel_fp32_acc_vec"
                bmm_rcr_n1_kernel_fp16 = "bmm_rcr_n1_kernel_fp16_acc_vec"
                read_vec_type = backend_spec.read_num_elements_to_backend_type[vec_idx][
                    1
                ]
                break

    input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=3,
        weight_ndims=3,
        output_ndims=3,
    )
    if ak == 0:
        # avoid compilation failure (zero-sized variable not alowed in device code)
        # caused by instantiating the template with K=0
        exec_paths = ""
    else:
        exec_paths = EXEC_TEMPLATE.render(
            indent="  ",
            read_vec_type=read_vec_type,
            elem_input_type=elem_input_type,
            K=ak,
        )

    input_a_accessor = tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
        name="input_a_accessor", tensor_accessor=func_attrs["input_accessors"][0]
    )

    input_b_accessor = tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
        name="input_b_accessor", tensor_accessor=func_attrs["input_accessors"][1]
    )

    return SRC_TEMPLATE.render(
        function_name=func_name,
        elem_input_type=elem_input_type,
        bmm_rcr_n1_kernel_fp32=bmm_rcr_n1_kernel_fp32,
        bmm_rcr_n1_kernel_fp16=bmm_rcr_n1_kernel_fp16,
        shape_function=shape_func,
        input_output_checks=input_output_checks,
        exec_paths=exec_paths,
        tensor_accessor_libs=tensor_accessor_codegen.get_libs(),
        input_accessors=input_a_accessor + input_b_accessor,
        output_accessors=tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
            name="output_accessor", tensor_accessor=func_attrs["output_accessors"][0]
        ),
    )


@registry.reg("cuda.bmm_rcr_n1.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )


@registry.reg("cuda.bmm_rcr_n1.func_call")
def gen_function_call(func_attrs, indent="  "):
    a = func_attrs["inputs"][0]
    ashape = func_attrs["input_accessors"][0].original_shapes
    adims = ["&" + dim._attrs["name"] for dim in ashape]
    b = func_attrs["inputs"][1]
    bshape = func_attrs["input_accessors"][1].original_shapes
    bdims = ["&" + dim._attrs["name"] for dim in bshape]
    c = func_attrs["outputs"][0]
    cshape = func_attrs["output_accessors"][0].original_shapes
    cdims = ["&" + dim._attrs["name"] for dim in cshape]
    alpha = func_attrs["alpha"]
    use_fp16_acc = False
    if "use_fp16_acc" in Target.current()._kwargs:
        use_fp16_acc = Target.current()._kwargs["use_fp16_acc"]
    return FUNC_CALL_TEMPLATE.render(
        local_dim_defs=common.gen_local_dim_defs(func_attrs, indent=indent),
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        c_ptr=c._attrs["name"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        alpha=alpha,
        use_fp16_acc="true" if use_fp16_acc else "false",
        indent=indent,
    )


@registry.reg("cuda.bmm_rcr_n1.filter")
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
