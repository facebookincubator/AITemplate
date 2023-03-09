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
CUDA gather function
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cuda import cuda_common

CAST_TO_CONST_INDEX_PTR_TEMPLATE = jinja2.Template(
    "reinterpret_cast<const {{index_type}}*>({{name}})"
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    void * /*output*/,
    const void * /*input*/,
    const {{index_type}} * /*indices*/,
    int64_t *[] /*output_shape*/,
    const int64_t * /*input_shape*/,
    const int64_t * /*index_shape*/,
    int /*gather_dim*/,
    int /*rank*/,
    cudaStream_t /*stream*/
    );
"""
)


KERNEL_SRC_TEMPLATE = jinja2.Template(
    """
#include <assert.h>
#include <cuda_fp16.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <string>

#ifndef CUDA_CHECK_ERROR_GATHER
#define CUDA_CHECK_ERROR_GATHER(expr)                        \\
  do {                                                       \\
    cudaError_t status = (expr);                             \\
    if (status != cudaSuccess) {                             \\
      auto msg = std::string("Got error: ") +                \\
       cudaGetErrorString(status) +                          \\
        " at " + __FILE__ + ": " + std::to_string(__LINE__); \\
      std::cerr << msg << std::endl;                         \\
      throw std::runtime_error(msg);                         \\
    }                                                        \\
  } while (0)
#endif // CUDA_CHECK_ERROR_GATHER

#ifndef CUDA_LAUNCH_CHECK_GATHER
#define CUDA_LAUNCH_CHECK_GATHER() CUDA_CHECK_ERROR_GATHER(cudaGetLastError())
#endif // CUDA_LAUNCH_CHECK_GATHER

namespace {

using INDEX_TYPE = {{index_type}};

template <int Rank>
struct InputMetaData {
  int64_t input_strides[Rank];
  int64_t index_shape[Rank];
};

__host__ __device__ __forceinline__
int64_t get_num_elems(const int64_t *shape, int rank) {
  int num = 1;
  for (int i = 0; i < rank; i++) {
    num *= shape[i];
  }
  return num;
}

template <int Rank>
__host__ __device__ int64_t compute_input_elem_offset(
    const int64_t *input_strides,
    const int64_t *index_shape,
    int64_t curr_gather_dim_size,
    int gather_dim,
    int64_t linear_index_idx) {
  int64_t input_offset = 0;
  for (int i = Rank - 1; i >= 0; --i) {
    int curr_index_idx = linear_index_idx % index_shape[i];
    int dim_size = i == gather_dim ? curr_gather_dim_size : curr_index_idx;
    input_offset += dim_size * input_strides[i];
    linear_index_idx /= index_shape[i];
  }
  assert(linear_index_idx == 0);
  return input_offset;
}

template <typename READ_T, typename READ_INDEX_T, typename ELEM_T,
          int Rank, int ElemsPerThread>
__global__ void
gather_kernel(
    ELEM_T *orig_output,
    const ELEM_T *orig_input,
    const INDEX_TYPE *orig_indices,
    InputMetaData<Rank> input_meta,
    const int gather_dim,
    const int64_t num_output_elems) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  READ_T* output = reinterpret_cast<READ_T*>(orig_output);
  const READ_INDEX_T* indices =
      reinterpret_cast<const READ_INDEX_T*>(orig_indices);

  constexpr unsigned read_t_sz = sizeof(READ_T);
  constexpr unsigned elem_t_sz = sizeof(ELEM_T);
  static_assert(read_t_sz >= elem_t_sz && (read_t_sz % elem_t_sz == 0));
  constexpr int n_of_elem_t = read_t_sz / elem_t_sz;
  static_assert(sizeof(READ_INDEX_T) % sizeof(INDEX_TYPE) == 0);
  static_assert(n_of_elem_t == (sizeof(READ_INDEX_T) / sizeof(INDEX_TYPE)));
  // number of READ_T elements per thread
  constexpr int reads_per_thread_in_read_t = ElemsPerThread / n_of_elem_t;
  const int num_elems_in_read_t = num_output_elems / n_of_elem_t;
  int read_idx = tid;

#pragma unroll
  for (int i = 0; i < reads_per_thread_in_read_t;
       i++, read_idx += blockDim.x * gridDim.x) {
    if (read_idx >= num_elems_in_read_t) {
      break;
    }
    READ_INDEX_T curr_gather_dim_sizes_vec = indices[read_idx];
    INDEX_TYPE *curr_gather_dim_sizes_ptr =
        reinterpret_cast<INDEX_TYPE*>(&curr_gather_dim_sizes_vec);
    READ_T input_values_vec;
    ELEM_T *input_values_ptr = reinterpret_cast<ELEM_T*>(&input_values_vec);
    #pragma unroll
    for (int j = 0; j < n_of_elem_t; j++) {
      int64_t input_elem_offset =
          compute_input_elem_offset<Rank>(input_meta.input_strides,
                                          input_meta.index_shape,
                                          curr_gather_dim_sizes_ptr[j],
                                          gather_dim,
                                          read_idx * n_of_elem_t + j);
      input_values_ptr[j] = orig_input[input_elem_offset];
    }
    output[read_idx] = input_values_vec;
  }
}

template <typename ELEM_T, int Rank, int ElemsPerThread, int ThreadsPerBlock>
void gather_kernel_launcher(
    ELEM_T *output,
    const ELEM_T *input,
    const INDEX_TYPE *indices,
    const int64_t *input_shape,
    const int64_t *index_shape,
    const int gather_dim,
    cudaStream_t stream) {

  InputMetaData<Rank> input_meta;
  input_meta.input_strides[Rank - 1] = 1;
  input_meta.index_shape[Rank - 1] = index_shape[Rank - 1];
  for (int i = Rank - 2; i >= 0; i--) {
    input_meta.input_strides[i] = input_meta.input_strides[i+1] * input_shape[i+1];
    input_meta.index_shape[i] = index_shape[i];
  }

  int64_t num_output_elems = get_num_elems(index_shape, Rank);
  int m = (num_output_elems % (ThreadsPerBlock * ElemsPerThread) != 0);
  int num_blocks_x =
      (num_output_elems / (ThreadsPerBlock * ElemsPerThread)) + m;
  int grid_config = num_blocks_x;

{% if elem_type == "half" %}
  if (num_output_elems % 2 == 0) {
    gather_kernel<float, int4, ELEM_T, Rank, ElemsPerThread>
    <<<grid_config, ThreadsPerBlock, 0, stream>>>(
        output,
        input,
        indices,
        input_meta,
        gather_dim,
        num_output_elems);
    CUDA_LAUNCH_CHECK_GATHER();
  } else{
    gather_kernel<half, INDEX_TYPE, ELEM_T, Rank, ElemsPerThread>
    <<<grid_config, ThreadsPerBlock, 0, stream>>>(
        output,
        input,
        indices,
        input_meta,
        gather_dim,
        num_output_elems);
    CUDA_LAUNCH_CHECK_GATHER();
  }
{% elif elem_type == "float" %}
  gather_kernel<float, INDEX_TYPE, ELEM_T, Rank, ElemsPerThread>
  <<<grid_config, ThreadsPerBlock, 0, stream>>>(
      output,
      input,
      indices,
      input_meta,
      gather_dim,
      num_output_elems);
  CUDA_LAUNCH_CHECK_GATHER();
{% endif %}
}

#undef CUDA_CHECK_ERROR_GATHER
#undef CUDA_LAUNCH_CHECK_GATHER

} // namespace
"""
)


EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if (rank == {{rank}}) {
{{indent}}  /* TODO: more profiling on ElemsPerThread and ThreadsPerBlock */
{{indent}}  gather_kernel_launcher<{{elem_type}},
                                   {{rank}}/*Rank*/,
                                   {{elems_per_thread}}/*ElemsPerThread*/,
{{indent}}                         {{threads_per_block}}/*THREADS_PER_BLOCK*/>(
{{indent}}    static_cast<{{elem_type}}*>(output), static_cast<const {{elem_type}}*>(input), indices, input_shape, index_shape, gather_dim, stream);
{{indent}}  return;
{{indent}}}
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
{{kernel_src}}

void {{func_name}}(
    void* output,
    const void* input,
    const INDEX_TYPE* indices,
    int64_t *output_shape[],
    const int64_t *input_shape,
    const int64_t *index_shape,
    int gather_dim,
    int rank,
    cudaStream_t stream
    ) {

  if (rank < 0) {
    throw std::runtime_error("rank must be larger than 0!");
  }
  if (rank == 0)
    return;
  if (gather_dim >= rank) {
    throw std::runtime_error("gather_dim must be smaller than rank!");
  }

  bool empty_tensor = false;
  for (int i = 0; i < rank; i++) {
    if (i != gather_dim && index_shape[i] > input_shape[i]) {
      throw std::runtime_error("index dimension must be <= input dimension");
    }
    *(output_shape[i]) = index_shape[i];
    if (index_shape[i] == 0)
      empty_tensor = true;
  }
  if (empty_tensor)
    return;

  // make sure input and output are valid
  if (!input) {
    throw std::runtime_error("input is NULL!");
  }
  if (!output) {
    throw std::runtime_error("output is NULL!");
  }

{{exec_paths}}

  throw std::runtime_error(
      "Unsupported cat kernel specialization!"
  );
}
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{

{{indent}}  int64_t *{{output_name}}_shape[] = {
{{indent}}      {{output_dims}}
{{indent}}  };

{{indent}}  const int64_t {{input_name}}_shape[] = {
{{indent}}    {{input_dims}}
{{indent}}  };

{{indent}}  const int64_t {{index_name}}_shape[] = {
{{indent}}    {{index_dims}}
{{indent}}  };

{{indent}}  {{func_name}}(
{{indent}}      {{output_ptr}},
{{indent}}      {{input_ptr}},
{{indent}}      {{index_ptr}},
{{indent}}      {{output_name}}_shape,
{{indent}}      {{input_name}}_shape,
{{indent}}      {{index_name}}_shape,
{{indent}}      {{gather_dim}}/*gather_dim*/,
{{indent}}      {{rank}}/*rank*/,
{{indent}}      stream
{{indent}}  );
{{indent}}}
"""
)


@registry.reg("cuda.gather.func_decl")
def gen_function_decl(func_attrs):
    inputs = func_attrs["inputs"]
    index = inputs[1]
    index_type = cuda_common.dtype_to_cuda_type(index._attrs["dtype"])
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=index_type,
    )


@registry.reg("cuda.gather.gen_function")
def gen_function(func_attrs):
    inputs = func_attrs["inputs"]
    x = inputs[0]
    index = inputs[1]
    y = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]

    input_type = cuda_common.dtype_to_cuda_type(x._attrs["dtype"])
    index_type = cuda_common.dtype_to_cuda_type(index._attrs["dtype"])
    output_type = cuda_common.dtype_to_cuda_type(y._attrs["dtype"])

    if input_type != output_type:
        raise TypeError("input type must equal to output type")

    # TODO: consider to add profiling paths for tuning
    # elems_per_thread and threads_per_block
    exec_paths = EXEC_COND_TEMPLATE.render(
        indent="  ",
        rank=len(x_shape),
        elem_type=input_type,
        elems_per_thread=2,
        threads_per_block=128,
    )
    kernel_src = KERNEL_SRC_TEMPLATE.render(
        index_type=index_type,
        elem_type=input_type,
    )
    return SRC_TEMPLATE.render(
        kernel_src=kernel_src,
        func_name=func_attrs["name"],
        exec_paths=exec_paths,
    )


@registry.reg("cuda.gather.func_call")
def gen_function_call(func_attrs, indent="  "):
    inputs = func_attrs["inputs"]
    x = inputs[0]
    index = inputs[1]
    y = func_attrs["outputs"][0]
    gather_dim = func_attrs["gather_dim"]

    def _dims(t, ref=""):
        return ", ".join([ref + dim._attrs["name"] for dim in t._attrs["shape"]])

    x_dims = _dims(x)
    index_dims = _dims(index)
    y_dims = _dims(y, ref="&")

    index_type = cuda_common.dtype_to_cuda_type(index._attrs["dtype"])
    casted_index_ptr = CAST_TO_CONST_INDEX_PTR_TEMPLATE.render(
        index_type=index_type, name=index._attrs["name"]
    )

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        output_name=y._attrs["name"],
        input_name=x._attrs["name"],
        index_name=index._attrs["name"],
        output_dims=y_dims,
        input_dims=x_dims,
        index_dims=index_dims,
        output_ptr=y._attrs["name"],
        input_ptr=x._attrs["name"],
        index_ptr=casted_index_ptr,
        gather_dim=gather_dim,
        rank=len(x._attrs["shape"]),
    )
