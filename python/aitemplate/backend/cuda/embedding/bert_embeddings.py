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
bert_embeddings kernel codegen for CUDA.
"""

from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

#define FINAL_MASK 0xffffffff

namespace {

template <typename T>
__inline__ __device__ T warpReduceSum(T* val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val[0] += __shfl_xor_sync(FINAL_MASK, val[0], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T>
__inline__ __device__ T blockReduceSum(T* val) {
  __shared__ T shared[33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum<T>(val);

  if (lane == 0) {
#pragma unroll
    shared[wid] = val[0];
  }

  __syncthreads();

  // blockDim.x is round up to multiples of 32
  bool is_mask = threadIdx.x < (blockDim.x / 32);
#pragma unroll
  val[0] = is_mask ? shared[lane] : (T)(0.0f);

  warpReduceSum<T>(val);
  return (T)0.0f;
}

template <typename T>
__inline__ __device__ T normalize(T val, T mean, T variance, T gamma, T beta) {
  return (val - mean) * variance * gamma + beta;
}

// __inline__ __device__ float sigmoid(float val) {
//   return 1.0f / (1.0f + expf(-1.0f * val));
// }

// fast sigmoid
__inline__ __device__ float sigmoid(float val) {
  return (cutlass::fast_tanh(val * 0.5f) + 1.0f) * 0.5f;
}

template <typename ElemT, typename INDEX_T>
__global__ void bert_embeddings_kernel(
    uint4* output,
    INDEX_T* input_ids,
    INDEX_T* token_type_ids,
    INDEX_T* position_ids,
    uint4* word_embeddings,
    uint4* token_type_embeddings,
    uint4* position_embeddings,
    uint4* gamma,
    uint4* beta,
    const int64_t embedding_dim,
    const int64_t vocab_size,
    const int64_t type_vocab_size,
    const int64_t max_position_embeddings,
    const float eps) {
  constexpr int num_elems_in_uint4 = sizeof(uint4) / sizeof(ElemT);
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int embedding_dim_div_n = embedding_dim / num_elems_in_uint4;

  const int64_t input_id = input_ids[bid];
  const int64_t token_type_id = token_type_ids[bid];
  const int64_t position_id = position_ids[bid];

  // index bound check
  if (input_id < 0 || input_id >= vocab_size || token_type_id < 0 ||
      token_type_id >= type_vocab_size || position_id < 0 ||
      position_id >= max_position_embeddings) {
    return;
  }

  word_embeddings = word_embeddings + input_id * embedding_dim_div_n;
  token_type_embeddings =
      token_type_embeddings + token_type_id * embedding_dim_div_n;
  position_embeddings = position_embeddings + position_id * embedding_dim_div_n;

  uint4 word_embedding{0, 0, 0, 0};
  uint4 token_type_embedding{0, 0, 0, 0};
  uint4 position_embedding{0, 0, 0, 0};

  if (tid < embedding_dim_div_n) {
    word_embedding = word_embeddings[tid];
    token_type_embedding = token_type_embeddings[tid];
    position_embedding = position_embeddings[tid];
  }
  uint4 embedding{0, 0, 0, 0};

  ElemT* word_emb_vec = reinterpret_cast<ElemT*>(&word_embedding);
  ElemT* token_emb_vec = reinterpret_cast<ElemT*>(&token_type_embedding);
  ElemT* pos_emb_vec = reinterpret_cast<ElemT*>(&position_embedding);

  ElemT* emb_vec = reinterpret_cast<ElemT*>(&embedding);

  // layernorm
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};

#pragma unroll
  for (int i = 0; i < num_elems_in_uint4; i++) {
    float sum = word_emb_vec[i] + token_emb_vec[i] + pos_emb_vec[i];
    local_sums[0] += sum;
    emb_vec[i] = static_cast<ElemT>(sum);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float>(local_sums);
  } else {
    blockReduceSum<float>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / embedding_dim;
  }
  __syncthreads();

  local_sums[0] = 0.0f;

  if (tid < embedding_dim_div_n) {
#pragma unroll
    for (int i = 0; i < num_elems_in_uint4; i++) {
      float val = emb_vec[i];
      local_sums[0] += (val - s_mean) * (val - s_mean);
    }
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float>(local_sums);
  } else {
    blockReduceSum<float>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / embedding_dim + eps);
  }
  __syncthreads();

  if (tid < embedding_dim_div_n) {
    uint4 local_gamma = gamma[tid];
    ElemT* gamma_vec = reinterpret_cast<ElemT*>(&local_gamma);
    uint4 local_beta = beta[tid];
    ElemT* beta_vec = reinterpret_cast<ElemT*>(&local_beta);
#pragma unroll
    for (int i = 0; i < num_elems_in_uint4; i++) {
      emb_vec[i] = normalize(
          (float)emb_vec[i],
          s_mean,
          s_variance,
          (float)gamma_vec[i],
          (float)beta_vec[i]);
    }
  }

  // write to output
  if (tid < embedding_dim_div_n) {
    output = output + bid * embedding_dim_div_n;
    output[tid] = embedding;
  }
}

template <typename ElemT, typename INDEX_T>
void bert_embeddings_launcher(
    ElemT* output,
    INDEX_T* input_ids,
    INDEX_T* token_type_ids,
    INDEX_T* position_ids,
    ElemT* word_embeddings,
    ElemT* token_type_embeddings,
    ElemT* position_embeddings,
    ElemT* gamma,
    ElemT* beta,
    const int64_t indices_num,
    const int64_t embedding_dim,
    const int64_t vocab_size,
    const int64_t type_vocab_size,
    const int64_t max_position_embeddings,
    const float eps,
    cudaStream_t stream) {
  constexpr int num_elems_in_uint4 = sizeof(uint4) / sizeof(ElemT);
  if (embedding_dim % num_elems_in_uint4 != 0) {
    throw std::runtime_error(
        "embedding dim must be multiple of num_elems_in_uint4: " +
        std::to_string(num_elems_in_uint4)
    );
  }
  dim3 grid(indices_num);

  // round up to multiple of 32
  int64_t num_threads = embedding_dim / num_elems_in_uint4;
  num_threads = (num_threads + 31) / 32 * 32;
  dim3 block(num_threads);

  bert_embeddings_kernel<{{elem_input_type}}, INDEX_T><<<grid, block, 0, stream>>>(
      reinterpret_cast<uint4*>(output),
      input_ids,
      token_type_ids,
      position_ids,
      reinterpret_cast<uint4*>(word_embeddings),
      reinterpret_cast<uint4*>(token_type_embeddings),
      reinterpret_cast<uint4*>(position_embeddings),
      reinterpret_cast<uint4*>(gamma),
      reinterpret_cast<uint4*>(beta),
      embedding_dim,
      vocab_size,
      type_vocab_size,
      max_position_embeddings,
      eps);
}

} // namespace

{{func_signature}}
{
    bert_embeddings_launcher<{{elem_input_type}}, {{index_type}}>(
      static_cast<{{elem_input_type}}*>(output),
      input_ids,
      token_type_ids,
      position_ids,
      static_cast<{{elem_input_type}}*>(word_embeddings),
      static_cast<{{elem_input_type}}*>(token_type_embeddings),
      static_cast<{{elem_input_type}}*>(position_embeddings),
      static_cast<{{elem_input_type}}*>(gamma),
      static_cast<{{elem_input_type}}*>(beta),
      indices_num,
      embedding_dim,
      vocab_size,
      type_vocab_size,
      max_position_embeddings,
      eps,
      stream
    );
}

"""
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   {{index_type}}* input_ids,
                   {{index_type}}* token_type_ids,
                   {{index_type}}* position_ids,
                   void* word_embeddings,
                   void* token_type_embeddings,
                   void* position_embeddings,
                   void* gamma,
                   void* beta,
                   const int64_t indices_num,
                   const int64_t embedding_dim,
                   const int64_t vocab_size,
                   const int64_t type_vocab_size,
                   const int64_t max_position_embeddings,
                   const float eps,
                   cudaStream_t stream)
  """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  {{calculate_indices_num}}
{{indent}}  {{func_name}}(
{{indent}}            {{output}},
{{indent}}            {{input_ids}},
{{indent}}            {{token_type_ids}},
{{indent}}            {{position_ids}},
{{indent}}            {{word_embeddings}},
{{indent}}            {{token_type_embeddings}},
{{indent}}            {{position_embeddings}},
{{indent}}            {{gamma}},
{{indent}}            {{beta}},
{{indent}}            {{indices_num}},
{{indent}}            {{embedding_dim}},
{{indent}}            {{vocab_size}},
{{indent}}            {{type_vocab_size}},
{{indent}}            {{max_position_embeddings}},
{{indent}}            {{eps}},
{{indent}}            stream /* default stream */
{{indent}} );

{{indent}}}
    """
)

INDICES_NUM_TEMPLATE = jinja2.Template(
    """
  int64_t indices_num = 1;
  {% for dim_name in dim_names %}
  indices_num *= {{dim_name}};
  {% endfor %}
  """
)


def python_int_dtype_to_c_dtype(dtype):
    if dtype == "int64":
        return "int64_t"
    if dtype in ["int", "int32"]:
        return "int32_t"
    return dtype


@registry.reg("cuda.bert_embeddings.gen_function")
def bert_embeddings_gen_function(func_attrs: Dict[str, Any]) -> str:
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][3]._attrs["dtype"]
    )
    dtype = python_int_dtype_to_c_dtype(func_attrs["inputs"][0]._attrs["dtype"])
    return FUNC_TEMPLATE.render(
        index_type=dtype,
        elem_input_type=elem_input_type,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=dtype,
        ).strip(),
    )


@registry.reg("cuda.bert_embeddings.func_decl")
def bert_embeddings_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    dtype = python_int_dtype_to_c_dtype(func_attrs["inputs"][0]._attrs["dtype"])
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=dtype,
        ).strip()
    )


FUNC_CALL_INT64_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int64_t*>({{name}})")
FUNC_CALL_INT32_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int32_t*>({{name}})")


def get_int_param_template(tensor):
    name = tensor._attrs["name"]
    dtype = tensor._attrs["dtype"]
    if dtype == "int64":
        return FUNC_CALL_INT64_PARAM_TEMPLATE.render(name=name)
    elif dtype in ("int", "int32"):
        return FUNC_CALL_INT32_PARAM_TEMPLATE.render(name=name)
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype}")


@registry.reg("cuda.bert_embeddings.func_call")
def bert_embeddings_gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    (
        input_ids,
        token_type_ids,
        position_ids,
        word_embeddings,
        token_type_embeddings,
        position_embeddings,
        gamma,
        beta,
    ) = func_attrs["inputs"]

    indices_dims = [shape._attrs["name"] for shape in input_ids.shape()]
    indices_num_str = INDICES_NUM_TEMPLATE.render(
        dim_names=indices_dims,
    )
    embedding_dim = word_embeddings._size(-1).value()
    vocab_size = word_embeddings._size(0).value()
    type_vocab_size = token_type_embeddings._size(0).value()
    max_position_embeddings = position_embeddings._size(0).value()

    eps = func_attrs["eps"]
    output_str = func_attrs["outputs"][0]._attrs["name"]

    input_ids_str = get_int_param_template(input_ids)
    token_type_ids_str = get_int_param_template(token_type_ids)
    position_ids_str = get_int_param_template(position_ids)

    word_embeddings_str = word_embeddings._attrs["name"]
    token_type_embeddings_str = token_type_embeddings._attrs["name"]
    position_embeddings_str = position_embeddings._attrs["name"]

    gamma_str = gamma._attrs["name"]
    beta_str = beta._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        calculate_indices_num=indices_num_str,
        output=output_str,
        input_ids=input_ids_str,
        token_type_ids=token_type_ids_str,
        position_ids=position_ids_str,
        word_embeddings=word_embeddings_str,
        token_type_embeddings=token_type_embeddings_str,
        position_embeddings=position_embeddings_str,
        gamma=gamma_str,
        beta=beta_str,
        indices_num="indices_num",
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        eps=eps,
        indent=indent,
    )
