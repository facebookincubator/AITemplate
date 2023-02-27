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
Slice backend common implementation.
"""
import jinja2


SHAPE_UPDATE_FUNC = jinja2.Template(
    """
{{indent}}int64_t output_scatter_dim_value = 0;
{{indent}}for ({{index_type}} i = 0; i < num_inputs; i++) {
{{indent}}  output_scatter_dim_value +=
{{indent}}      slice_end_indices[i][scatter_dim] - slice_start_indices[i][scatter_dim];
{{indent}}}
{{indent}}
{{indent}}for ({{index_type}}  i = 0; i < rank; i++) {
{{indent}}  if (i == scatter_dim) {
{% if update_output_shape %}
{{indent}}    *output_shape[i] = output_scatter_dim_value;
{% else %}
{{indent}}    // skip updating output_shape[i]
{% endif %}
{{indent}}  } else {
{{indent}}    int64_t dim = slice_end_indices[0][i] - slice_start_indices[0][i];
{{indent}}    for ({{index_type}}  j = 1; j < num_inputs; j++) {
{{indent}}      if (slice_end_indices[j][i] - slice_start_indices[j][i] != dim) {
{{indent}}        throw std::runtime_error("invalid indices");
{{indent}}      }
{% if update_output_shape %}
{{indent}}      *output_shape[i] = dim;
{% else %}
{{indent}}    // skip updating output_shape[i]
{% endif %}
{{indent}}    }
{{indent}}  }
{{indent}}}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    void * /*output*/,
    int64_t *[] /*output_shape*/,
    const void *[] /*inputs*/,
    const int64_t *[] /*input_shapes*/,
    const int64_t *[] /*orig_slice_start_indices*/,
    const int64_t *[] /*orig_slice_end_indices*/,
    {{index_type}}  /*scatter_dim*/,
    {{index_type}}  /*rank*/,
    {{index_type}}  /*num_inputs*/,
    {{prefix}}Stream_t
    );
"""
)


KERNEL_SRC_TEMPLATE = jinja2.Template(
    """
{{header_src}}

#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

{% if element_func_def %}
//#include <cutlass/fast_math.h>
{% endif %}

namespace {
#ifndef CHECK_ERROR_SLICE
#define CHECK_ERROR_SLICE(expr)                              \\
  do {                                                       \\
    {{prefix}}Error_t status = (expr);                       \\
    if (status != {{prefix}}Success) {                       \\
      auto msg = std::string("Got error: ") +                \\
        {{prefix}}GetErrorString(status) +                   \\
        " at " + __FILE__ + ": " + std::to_string(__LINE__); \\
      std::cerr << msg << std::endl;                         \\
      throw std::runtime_error(msg);                         \\
    }                                                        \\
  } while (0)
#endif // CHECK_ERROR_SLICE

#ifndef LAUNCH_CHECK_SLICE
#define LAUNCH_CHECK_SLICE() CHECK_ERROR_SLICE({{prefix}}GetLastError())
#endif // LAUNCH_CHECK_SLICE

{% if element_func_def %}
{{element_func_def}}
{% endif %}

template <typename T, {{index_type}}  Rank, {{index_type}}  NumInputs>
struct SliceMetaData {
  const T *inputs[NumInputs];
  int64_t slice_start_indices[NumInputs][Rank];
  int64_t slice_end_indices[NumInputs][Rank];
  {{index_type}}  dim; // scatter dimension
  int64_t input_strides[NumInputs][Rank];
  int64_t num_elems[NumInputs];
  int64_t offsets[NumInputs];  // value of (dim_offset * output_dim_stride) at
                               // the dim axis in the output, where dim_offset
                               // is the offset of the scattered input at the
                               // dimension axis in the output
  int64_t dim_sizes[NumInputs];  // dimension size of the input to be scattered
                                 // at the dim axis
};

template <{{index_type}}  Rank, {{index_type}}  NumInputs>
struct ScatterMetaData {
  int64_t output_shape[Rank];
  int64_t output_strides[Rank];
};

__host__ __device__ __forceinline__
int64_t get_num_elems(const int64_t *shape, {{index_type}}  rank) {
  {{index_type}}  num = 1;
  for ({{index_type}}  i = 0; i < rank; i++) {
    num *= shape[i];
  }
  return num;
}

template <{{index_type}}  Rank>
__host__ __device__ int64_t compute_input_linear_index(
    const int64_t *input_strides,
    const int64_t *slice_start_indices,
    const int64_t *slice_end_indices,
    int64_t linear_idx) {
  int64_t input_offset = slice_start_indices[0] * input_strides[0];
  for ({{index_type}}  i = Rank - 1; i > 0; i--) {
    {{index_type}}  curr_output_dim_size = slice_end_indices[i] - slice_start_indices[i];
    int64_t curr_output_idx = linear_idx % curr_output_dim_size;
    int64_t curr_input_idx = curr_output_idx + slice_start_indices[i];
    input_offset += curr_input_idx * input_strides[i];
    linear_idx /= curr_output_dim_size;
  }
  return input_offset + linear_idx * input_strides[0];
}

template <{{index_type}}  Rank>
__host__ __device__ int64_t compute_output_elem_offset(
    const int64_t *output_shape,
    const int64_t *output_strides,
    int64_t scatter_dim_size,
    const {{index_type}}  scatter_dim,
    int64_t linear_idx) {
  int64_t offset = 0;
  for ({{index_type}}  i = Rank - 1; i >= 1; --i) {
    int64_t cur_dim_size = i == scatter_dim ?  scatter_dim_size : output_shape[i];
    int64_t next_dim_idx = linear_idx / cur_dim_size;
    int64_t cur_dim_idx = linear_idx - cur_dim_size * next_dim_idx;
    int64_t cur_dim_offset = cur_dim_idx * output_strides[i];
    offset += cur_dim_offset;
    linear_idx = next_dim_idx;
  }
  return offset + linear_idx * output_strides[0];
}

template <typename READ_T, typename ELEM_T, {{index_type}}  Rank,
          {{index_type}}  NumInputs, {{index_type}}  ElemsPerThread>
__global__ void
slice_scatter_kernel(
    ELEM_T *orig_output,
    SliceMetaData<ELEM_T, Rank, NumInputs> slice_meta_data,
    ScatterMetaData<Rank, NumInputs> scatter_meta_data) {
  const {{index_type}}  tid = blockIdx.x * blockDim.x + threadIdx.x;
  const {{index_type}}  block_y = blockIdx.y % NumInputs;

  READ_T* output = reinterpret_cast<READ_T*>(orig_output);
  const READ_T* input =
      reinterpret_cast<const READ_T*>(slice_meta_data.inputs[block_y]);
  int64_t num_elems = slice_meta_data.num_elems[block_y];
  const int64_t *input_strides = slice_meta_data.input_strides[block_y];
  const int64_t *slice_start_indices =
      slice_meta_data.slice_start_indices[block_y];
  const int64_t *slice_end_indices =
      slice_meta_data.slice_end_indices[block_y];

  {{index_type}}  scatter_dim = slice_meta_data.dim;
  int64_t scatter_dim_size = slice_meta_data.dim_sizes[block_y];
  int64_t scatter_offset = slice_meta_data.offsets[block_y];

  constexpr unsigned read_t_sz = sizeof(READ_T);
  constexpr unsigned elem_t_sz = sizeof(ELEM_T);
  static_assert(read_t_sz >= elem_t_sz && (read_t_sz % elem_t_sz == 0));
  {{index_type}}  n_of_elem_t = read_t_sz / elem_t_sz;
  // number of READ_T elements per thread
  {{index_type}}  reads_per_thread_in_read_t = ElemsPerThread / n_of_elem_t;
  const int64_t num_elems_in_read_t = num_elems / n_of_elem_t;
  {{index_type}}  read_idx = tid;

#pragma unroll
  for ({{index_type}}  i = 0; i < reads_per_thread_in_read_t;
       i++, read_idx += blockDim.x * gridDim.x) {
    if (read_idx >= num_elems_in_read_t) {
      break;
    }
    /* make sure to adjust read_idx, which refers to location at
       (read_idx * n_of_elem_t) actually */
    int64_t input_idx = compute_input_linear_index<Rank>(
        input_strides,
        slice_start_indices,
        slice_end_indices,
        read_idx * n_of_elem_t);
    int64_t output_elem_offset = compute_output_elem_offset<Rank>(
        scatter_meta_data.output_shape,
        scatter_meta_data.output_strides,
        scatter_dim_size,
        scatter_dim,
        read_idx * n_of_elem_t);

    READ_T tmp_v = input[input_idx / n_of_elem_t];
    int64_t output_idx = (scatter_offset + output_elem_offset) / n_of_elem_t;
    {% if element_func %}
    output[output_idx] = {{element_func}}(tmp_v);
    {% else %}
    output[output_idx] = tmp_v;
    {% endif %}
  }
}

enum class LoadVecType {
  VT_HALF = 0,
  VT_BFLOAT16 = 0,
  VT_FLOAT,
  VT_FLOAT2,
  VT_FLOAT4
};


template <typename ELEM_T>
static inline LoadVecType get_vec_type(int64_t dim_size) {
  {{index_type}}  size_elem_t = sizeof(ELEM_T);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)  \\
  if (sizeof(vec_type) % size_elem_t == 0) {          \\
    {{index_type}}  n_of_elem_t = sizeof(vec_type) / size_elem_t; \\
    if (dim_size % n_of_elem_t == 0) {                \\
      return load_vec_type;                           \\
    }                                                 \\
  }

  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT4, float4)
  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT2, float2)
  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT, float)
  if constexpr (std::is_same_v<ELEM_T, half>) {
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_HALF, half)
  } else if constexpr (std::is_same_v<ELEM_T, bfloat16>) {
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_BFLOAT16, bfloat16)
  }

#undef HANDLE_ONE_VEC_TYPE
  throw std::runtime_error(
      "Cannot resolve LoadVecType."
  );
}

template <typename ELEM_T, {{index_type}}  Rank>
static LoadVecType get_input_vec_type(
    const int64_t *output_strides,
    const int64_t *input_shape,
    const int64_t *input_strides,
    const int64_t *slice_start_indices,
    const int64_t *slice_end_indices,
    {{index_type}}  scatter_dim,
    {{index_type}}  scatter_offset,
    {{index_type}}  dim_size) {
  // get the outermost index where we continuous element accesses
  {{index_type}}  flatten_index = Rank - 1;
  for (; flatten_index >= 0; flatten_index--) {
    if (slice_end_indices[flatten_index] - slice_start_indices[flatten_index] !=
        input_shape[flatten_index]) {
      break;
    }
  }
  int64_t input_start_offset =
      compute_input_linear_index<Rank>(input_strides,
                                       slice_start_indices,
                                       slice_end_indices,
                                       /*linear_idx*/0);
  LoadVecType slice_vec_type1 =
      get_vec_type<ELEM_T>(input_start_offset);
  LoadVecType slice_vec_type2;
  if (Rank == 1) {
    int64_t continuous_read_size = slice_end_indices[0] - slice_start_indices[0];
    slice_vec_type2 = get_vec_type<ELEM_T>(continuous_read_size);
  } else {
    int64_t continuous_read_size =
      (slice_end_indices[flatten_index] - slice_start_indices[flatten_index]) *
      input_strides[flatten_index];
    LoadVecType vec_type1 = get_vec_type<ELEM_T>(continuous_read_size);
    continuous_read_size =
      (input_shape[flatten_index] - slice_end_indices[flatten_index]) *
      input_strides[flatten_index];
    LoadVecType vec_type2 = get_vec_type<ELEM_T>(continuous_read_size);
    // find the smaller alignment reqirement between the sliced piece
    // and the rest along the flattened dimensions
    slice_vec_type2 = vec_type1 < vec_type2 ?  vec_type1 : vec_type2;
  }
  LoadVecType slice_min_vec_type = slice_vec_type1 < slice_vec_type2 ?
                                   slice_vec_type1 : slice_vec_type2;

  LoadVecType scatter_vec_type1 = get_vec_type<ELEM_T>(dim_size);
  LoadVecType scatter_vec_type2 = get_vec_type<ELEM_T>(scatter_offset);
  LoadVecType scatter_min_vec_type = scatter_vec_type1 < scatter_vec_type2 ?
                                     scatter_vec_type1 : scatter_vec_type2;

  LoadVecType min_vec_type = slice_min_vec_type < scatter_min_vec_type ?
                             slice_min_vec_type : scatter_min_vec_type;
  return min_vec_type;
}

template <typename ELEM_T, {{index_type}}  Rank, {{index_type}}  NumInputs>
void prepare_one_meta_data(
    {{index_type}}  input_idx,
    SliceMetaData<ELEM_T, Rank, NumInputs> &slice_meta_data,
    ScatterMetaData<Rank, NumInputs> &scatter_meta_data,
    const ELEM_T *input,
    const int64_t *input_shape,
    const int64_t *slice_start_indices,
    const int64_t *slice_end_indices,
    {{index_type}}  scatter_dim,
    {{index_type}}  scatter_dim_offset) {
  slice_meta_data.inputs[input_idx] = input;
  slice_meta_data.input_strides[input_idx][Rank-1] = 1;
  for ({{index_type}}  i = Rank - 2; i >= 0; i--) {
    slice_meta_data.input_strides[input_idx][i] =
        slice_meta_data.input_strides[input_idx][i+1] * input_shape[i+1];
  }

  slice_meta_data.num_elems[input_idx] = 1;
  for ({{index_type}}  i = 0; i < Rank; i++) {
    assert(slice_start_indices[i] >= 0 &&
           slice_start_indices[i] <= input_shape[i]);
    assert(slice_end_indices[i] >= 0 && slice_end_indices[i] <= input_shape[i]);
    assert(slice_start_indices[i] <= slice_end_indices[i]);

    slice_meta_data.num_elems[input_idx] *=
        slice_end_indices[i] - slice_start_indices[i];
    slice_meta_data.slice_start_indices[input_idx][i] = slice_start_indices[i];
    slice_meta_data.slice_end_indices[input_idx][i] = slice_end_indices[i];
  }

  slice_meta_data.dim_sizes[input_idx] =
      slice_end_indices[scatter_dim] - slice_start_indices[scatter_dim];
  slice_meta_data.offsets[input_idx] =
      scatter_dim_offset * scatter_meta_data.output_strides[scatter_dim];
}

template <typename ELEM_T, {{index_type}}  Rank, {{index_type}}  NumInputs,
          {{index_type}}  ElemsPerThread, {{index_type}}  ThreadsPerBlock>
void slice_scatter_kernel_launcher(
    ELEM_T *output,
    {{index_type}} output_offset,
    const int64_t *output_shape,
    const ELEM_T *inputs[],
    const int64_t *input_shapes[],
    const std::vector<std::vector<int64_t>> &slice_start_indices,
    const std::vector<std::vector<int64_t>> &slice_end_indices,
    {{index_type}}  scatter_dim,
    {{prefix}}Stream_t stream
) {
  SliceMetaData<ELEM_T, Rank, NumInputs> slice_meta_data;
  ScatterMetaData<Rank, NumInputs> scatter_meta_data;

  // meta data for placing sliced output
  scatter_meta_data.output_strides[Rank-1] = 1;
  scatter_meta_data.output_shape[Rank-1] = output_shape[Rank-1];
  for ({{index_type}}  i = Rank - 2; i >= 0; i--) {
    scatter_meta_data.output_strides[i] =
        scatter_meta_data.output_strides[i+1] * output_shape[i+1];
    scatter_meta_data.output_shape[i] = output_shape[i];
  }

  {{index_type}}  scatter_dim_offset = 0;
  slice_meta_data.dim = scatter_dim;
  for ({{index_type}}  i = 0; i < NumInputs; i++) {
    prepare_one_meta_data(i, slice_meta_data, scatter_meta_data,
                          inputs[i], input_shapes[i],
                          slice_start_indices[i].data(),
                          slice_end_indices[i].data(),
                          scatter_dim, scatter_dim_offset);
    scatter_dim_offset += slice_meta_data.dim_sizes[i];
  }

  LoadVecType min_vec_type = get_vec_type<ELEM_T>(output_offset);
  for ({{index_type}}  i = 0; i < NumInputs; i++) {
    LoadVecType vec_type = get_input_vec_type<ELEM_T, Rank>(
        scatter_meta_data.output_strides,
        input_shapes[i],
        slice_meta_data.input_strides[i],
        slice_start_indices[i].data(),
        slice_end_indices[i].data(),
        scatter_dim,
        slice_meta_data.offsets[i],
        slice_meta_data.dim_sizes[i]);
    min_vec_type = vec_type < min_vec_type ? vec_type : min_vec_type;
  }

  // setup kernel configs
  int64_t max_num_elems = 0;
  for ({{index_type}}  i = 0; i < NumInputs; i++) {
    if (slice_meta_data.num_elems[i] > max_num_elems) {
      max_num_elems =  slice_meta_data.num_elems[i];
    }
  }

  {{index_type}}  m = max_num_elems % (ThreadsPerBlock * ElemsPerThread) != 0;
  {{index_type}}  num_blocks_x =
      (max_num_elems / (ThreadsPerBlock * ElemsPerThread)) + m;
  dim3 grid_config = dim3(num_blocks_x, NumInputs);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)                          \\
    if (min_vec_type == load_vec_type) {                                      \\
      if (ElemsPerThread * sizeof(ELEM_T) < sizeof(vec_type)) {               \\
         throw std::runtime_error(                                            \\
           std::string("No valid kernel available for ") + #vec_type);        \\
      }                                                                       \\
      slice_scatter_kernel<vec_type, ELEM_T, Rank, NumInputs, ElemsPerThread> \\
        <<<grid_config, ThreadsPerBlock, 0, stream>>>(                        \\
            output + output_offset,                                           \\
            slice_meta_data,                                                  \\
            scatter_meta_data);                                               \\
      LAUNCH_CHECK_SLICE();                                                   \\
      return;                                                                 \\
    }

    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT4, float4)
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT2, float2)
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT, float)
    if constexpr (std::is_same_v<ELEM_T, half>) {
      HANDLE_ONE_VEC_TYPE(LoadVecType::VT_HALF, half)
    } else if constexpr (std::is_same_v<ELEM_T, bfloat16>) {
      HANDLE_ONE_VEC_TYPE(LoadVecType::VT_BFLOAT16, bfloat16)
    }

  throw std::runtime_error("Invalid LoadVecType\\n");
#undef HANDLE_ONE_VEC_TYPE
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
normalize_slice_indices(
    const int64_t *input_shape,
    const int64_t *orig_slice_start_indices,
    const int64_t *orig_slice_end_indices,
    {{index_type}}  rank) {
  std::vector<int64_t> slice_start_indices(rank);
  std::vector<int64_t> slice_end_indices(rank);
  for ({{index_type}}  i = 0; i < rank; i++) {
    slice_start_indices[i] = orig_slice_start_indices[i] < 0 ?
                             input_shape[i] + orig_slice_start_indices[i]:
                             orig_slice_start_indices[i];
    // make it compatible with PyTorch
    slice_start_indices[i] = slice_start_indices[i] < 0 ?
                             0 : slice_start_indices[i];
    if (slice_start_indices[i] < 0) {
      slice_start_indices[i] = 0;
    }
    if (slice_start_indices[i] > input_shape[i]) {
      slice_start_indices[i] = input_shape[i];
    }

    slice_end_indices[i] =  orig_slice_end_indices[i] < 0 ?
                            input_shape[i] + orig_slice_end_indices[i]:
                            orig_slice_end_indices[i];
    // make it compatible with PyTorch
    slice_end_indices[i] = slice_end_indices[i] < 0 ?
                           0 : slice_end_indices[i];
    if (slice_end_indices[i] < 0) {
      slice_end_indices[i] = 0;
    }
    if (slice_end_indices[i] > input_shape[i]) {
      slice_end_indices[i] = input_shape[i];
    }

    // make it compatible with PyTorch
    if (slice_start_indices[i] > slice_end_indices[i]) {
      slice_start_indices[i] = slice_end_indices[i];
    }
  }

  return {slice_start_indices, slice_end_indices};
}
} // namespace

"""
)


EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if (rank == {{rank}} && num_inputs == {{num_inputs}}) {
{{indent}}  int64_t local_output_shape[{{rank}}];
{% for rank_idx in range(rank) %}
{{indent}}  local_output_shape[{{rank_idx}}] = *output_shape[{{rank_idx}}];
{% endfor %}
{{indent}}  slice_scatter_kernel_launcher<{{elem_type}},
{{indent}}                                {{rank}}/*Rank*/,
{{indent}}                                {{num_inputs}}/*NumInputs*/,
{{indent}}                                {{elems_per_thread}}/*ElemsPerThread*/,
{{indent}}                                {{threads_per_block}}/*ThreadsPerBlock*/>(
{{indent}}      static_cast<{{elem_type}}*>(output), {{output_offset}}, local_output_shape,
{{indent}}      reinterpret_cast<const {{elem_type}}**>(inputs), input_shapes,
{{indent}}      slice_start_indices, slice_end_indices, scatter_dim, stream);
{{indent}}  return;
{{indent}}}
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
{{kernel_src}}

void {{func_name}}(
    void *output,
    int64_t *output_shape[],
    const void *inputs[],
    const int64_t *input_shapes[],
    const int64_t *orig_slice_start_indices[],
    const int64_t *orig_slice_end_indices[],
    {{index_type}}  scatter_dim,
    {{index_type}}  rank,
    {{index_type}}  num_inputs,
    {{prefix}}Stream_t stream
    ) {

  if (rank <= 0) {
    throw std::runtime_error("rank must > 0!");
  }
  if (scatter_dim >= rank) {
    throw std::runtime_error("scatter_dim must < rank!");
  }

  // clip slip start and end indices
  std::vector<std::vector<int64_t>> slice_start_indices(num_inputs);
  std::vector<std::vector<int64_t>> slice_end_indices(num_inputs);
  std::vector<int64_t> output_dim_sizes;
  for ({{index_type}} i = 0; i < num_inputs; i++) {
    std::vector<int64_t> start_indices;
    std::vector<int64_t> end_indices;
    std::tie(start_indices, end_indices) =
        normalize_slice_indices(input_shapes[i],
                                orig_slice_start_indices[i],
                                orig_slice_end_indices[i],
                                rank);
    slice_start_indices[i] = start_indices;
    slice_end_indices[i] = end_indices;
  }

{{shape_function}}

  // If all input tensors are empty, we are done
  bool empty = true;
  for ({{index_type}} i = 0; i < num_inputs; i++) {
    if (get_num_elems(input_shapes[i], rank) != 0) {
      empty = false;
      // make sure input is valid for each non-zero-size tensor
      if (!inputs[i]) {
        throw std::runtime_error("NULL input is found at: " + std::to_string(i));
      }
    }
  }

  if (empty)
    return;

  // if we output has any zero dim size, we are done
  for ({{index_type}} i = 0; i < rank; i++) {
    if (*output_shape[i] == 0)
      return;
  }
  // make sure we have a valid output pointer
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


DEFAULT_OUTPUT_SHAPE_DEF_TEMPLATE = jinja2.Template(
    """
{{indent}}  int64_t *{{output_name}}_shape[] = {
{{indent}}    {{output_dim_refs}}
{{indent}}  };
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{output_shape_def}}

{{indent}}  const void *inputs[] = {
{{indent}}    {{inputs}}
{{indent}}  };

{{input_shape_defs}}

{{indent}}  const int64_t *input_shapes[] = {
{{indent}}    {{input_shapes}}
{{indent}}  };

{{start_indices_defs}}

{{indent}}  const int64_t *slice_start_indices[] = {
{{indent}}    {{slice_start_indices}}
{{indent}}  };

{{end_indices_defs}}

{{indent}}  const int64_t *slice_end_indices[] = {
{{indent}}    {{slice_end_indices}}
{{indent}}  };

{{indent}}  {{func_name}}(
{{indent}}    {{output_ptr}},
{{indent}}    {{output_name}}_shape,
{{indent}}    inputs,
{{indent}}    input_shapes,
{{indent}}    slice_start_indices,
{{indent}}    slice_end_indices,
{{indent}}    {{scatter_dim}}/*scatter_dim*/,
{{indent}}    {{rank}}/*rank*/,
{{indent}}    {{num_inputs}}/*num_inputs*/,
{{indent}}    stream
{{indent}}  );
{{indent}}}
"""
)


INPUT_SHAPE_DEF_TEMPLATE = jinja2.Template(
    """
{{indent}}int64_t {{input_shape_name}}[] = {
{{indent}}  {{input_dims}}
{{indent}}};
"""
)


INPUT_INDICES_DEF_TEMPLATE = jinja2.Template(
    """
{{indent}}int64_t {{input_indices_name}}[] = {
{{indent}}  {{input_indices}}
{{indent}}};
"""
)


def gen_function_decl(func_attrs, backend_spec):
    """Generate function declaration.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec: dataclass
        Backend specification.

    Returns
    -------
    str
        Rendered function declaration.
    """
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
    )


def gen_function(
    func_attrs,
    backend_spec,
    elems_per_thread=8,
    output_offset=0,
    update_output_shape=True,
    element_func=None,
    element_func_def=None,
    extra_header_template=None,
):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec: dataclass
        Backend specification.
    elems_per_thread: int
        Per thread elements.
    update_output_shape: bool
        Whether to update output shape, by default True.
    element_func: str
        Attributes for ease of tanh concatenate fusion, default is None.
    element_func_def: str
        Implmentation for fast_tanh, default is None.
    extra_header_template: str
        Header for fast_tanh, default is None.


    Returns
    -------
    str
        Rendered function body.
    """
    inputs = func_attrs["inputs"]
    x = inputs[0]
    y = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]

    input_type = backend_spec.dtype_to_backend_type(x._attrs["dtype"])
    output_type = backend_spec.dtype_to_backend_type(y._attrs["dtype"])

    if input_type != output_type:
        raise NotImplementedError("input type must equal to output type")

    # TODO: consider to add profiling paths for tuning
    # elems_per_thread and threads_per_block
    exec_paths = EXEC_COND_TEMPLATE.render(
        indent="  ",
        num_inputs=len(inputs),
        rank=len(x_shape),
        elem_type=input_type,
        elems_per_thread=elems_per_thread,
        threads_per_block=128,
        output_offset=output_offset,
    )

    shape_func = SHAPE_UPDATE_FUNC.render(
        indent="  ",
        update_output_shape=update_output_shape,
        index_type=backend_spec.index_type,
    )
    extra_header = (
        extra_header_template.render(element_func_def=element_func_def)
        if extra_header_template is not None
        else ""
    )
    header_src = backend_spec.header_src_template.render(extra_header=extra_header)
    kernel_src = KERNEL_SRC_TEMPLATE.render(
        element_func=element_func,
        element_func_def=element_func_def,
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        header_src=header_src,
    )
    return SRC_TEMPLATE.render(
        kernel_src=kernel_src,
        func_name=func_attrs["name"],
        shape_function=shape_func,
        exec_paths=exec_paths,
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        header_src=header_src,
    )


def gen_function_call(
    backend_spec,
    func_name,
    inputs,
    outputs,
    start_indices,
    end_indices,
    dim=0,
    indent="  ",
    output_shape_def=None,
):
    """Generates function call.

    Parameters
    ----------
    backend_spec: dataclass
        Backend specification.
    func_name : str
        Function neame
    inputs : List[Tensor]
        Input tensors.
    outputs : List[Tensor]
        Output tensors.
    start_indices : List[List[int]]
        each input has its own list of indices
    end_indices : List[List[int]]
        Each input has its own list of indices
    dim : int
        Specify the concat dim if we concat outputs of all inputs, by default 0.
    indent : str, optional
        Indent for template, by default "  ".
    output_shape_def: jinja2.Template
      output shape template, by default None.

    Returns
    -------
    str
        Rendered function call.
    """
    assert len(inputs) == len(start_indices) == len(end_indices)
    x = inputs[0]
    y = outputs[0]

    input_names = ",\n        ".join([i._attrs["name"] for i in inputs])

    input_shape_defs = []
    input_shape_names = []
    start_indices_defs = []
    start_indices_names = []
    end_indices_defs = []
    end_indices_names = []

    for idx, (i, s_indices, e_indices) in enumerate(
        zip(inputs, start_indices, end_indices)
    ):
        input_shape_name = "{}_shape".format(i._attrs["name"])
        s_indices_name = "{}_slice_start_indices_{}".format(i._attrs["name"], idx)
        e_indices_name = "{}_slice_end_indices_{}".format(i._attrs["name"], idx)
        if input_shape_name not in input_shape_names:
            dims = ", ".join([dim._attrs["name"] for dim in i._attrs["shape"]])
            one_shape_def = INPUT_SHAPE_DEF_TEMPLATE.render(
                indent="      ", input_shape_name=input_shape_name, input_dims=dims
            )
            input_shape_defs.append(one_shape_def)

        s_indices_str = ", ".join([str(i) for i in s_indices])
        one_s_indices_def = INPUT_INDICES_DEF_TEMPLATE.render(
            indent="      ",
            input_indices_name=s_indices_name,
            input_indices=s_indices_str,
        )
        start_indices_defs.append(one_s_indices_def)

        e_indices_str = ", ".join([str(i) for i in e_indices])
        one_e_indices_def = INPUT_INDICES_DEF_TEMPLATE.render(
            indent="      ",
            input_indices_name=e_indices_name,
            input_indices=e_indices_str,
        )
        end_indices_defs.append(one_e_indices_def)

        input_shape_names.append(input_shape_name)
        start_indices_names.append(s_indices_name)
        end_indices_names.append(e_indices_name)

    if output_shape_def is None:
        y_dim_refs = ", ".join(["&" + dim._attrs["name"] for dim in y._attrs["shape"]])
        output_shape_def = DEFAULT_OUTPUT_SHAPE_DEF_TEMPLATE.render(
            indent=indent, output_name=y._attrs["name"], output_dim_refs=y_dim_refs
        )

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_name,
        output_name=y._attrs["name"],
        output_ptr=y._attrs["name"],
        output_shape_def=output_shape_def,
        inputs=input_names,
        input_shape_defs="".join(input_shape_defs),
        input_shapes=", ".join(input_shape_names),
        start_indices_defs="".join(start_indices_defs),
        slice_start_indices=", ".join(start_indices_names),
        end_indices_defs="".join(end_indices_defs),
        slice_end_indices=", ".join(end_indices_names),
        scatter_dim=dim,
        rank=len(x._attrs["shape"]),
        num_inputs=len(inputs),
    )
