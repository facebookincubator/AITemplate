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
Backend-agnostic function templates for split.
"""
import jinja2

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    void *[] /*outputs*/,
    {{index_type}} **[] /*output_shapes*/,
    const bool [] /*output_masks*/,
    const void * /*input*/,
    const {{index_type}} * /*input_shape*/,
    {{index_type}} /*real_num_splits*/,
    {{index_type}} /*all_num_splits*/,
    {{index_type}} [] /*split_sizes*/,
    {{index_type}} /*split_dim*/,
    {{index_type}} /*rank*/,
    {{prefix}}Stream_t stream
);
"""
)


KERNEL_SRC_TEMPLATE = jinja2.Template(
    """
#include <vector>
#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <string>

{{header_src}}

#ifndef CHECK_ERROR_SPLIT
#define CHECK_ERROR_SPLIT(expr)                              \\
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
#endif // CHECK_ERROR_SPLIT

#ifndef LAUNCH_CHECK_SPLIT
#define LAUNCH_CHECK_SPLIT() CHECK_ERROR_SPLIT({{prefix}}GetLastError())
#endif // LAUNCH_CHECK_SPLIT

template <typename T, {{index_type}} NumSplits>
struct OutputMetaData {
  T* outputs[NumSplits]; /* pointer to each output */
  int64_t split_dim_offsets[NumSplits]; /* offset of each output along
                                           the split dimension */
  int64_t split_dim_sizes[NumSplits]; /* cat dimension size of each output */
  int64_t num_elems[NumSplits]; /* number of the elements of each output */
};

template <{{index_type}} Rank>
struct InputMetaData {
  {{index_type}} input_shape[Rank];
  int64_t input_strides[Rank];
};

__host__ __device__ __forceinline__
int64_t get_num_elems(const {{index_type}} *shape, {{index_type}} rank) {
  {{index_type}} num = 1;
  for ({{index_type}} i = 0; i < rank; i++) {
    num *= shape[i];
  }
  return num;
}

template <{{index_type}} Rank>
__host__ __device__ int64_t compute_input_elem_offset(
    const {{index_type}} *input_shape,
    int64_t *input_strides,
    int64_t split_dim_size,
    {{index_type}} split_dim,
    int64_t linear_idx) {
  int64_t offset = 0;
  for ({{index_type}} i = Rank - 1; i >= 1; --i) {
    int64_t cur_dim_size = i == split_dim ? split_dim_size : input_shape[i];
    int64_t next_dim_idx = linear_idx / cur_dim_size;
    int64_t cur_dim_idx = linear_idx - cur_dim_size * next_dim_idx;
    int64_t cur_dim_offset = cur_dim_idx * input_strides[i];
    offset += cur_dim_offset;
    linear_idx = next_dim_idx;
  }
  return offset + linear_idx * input_strides[0];
}

template <typename READ_T, typename ELEM_T, {{index_type}} Rank,
          {{index_type}} NumSplits, {{index_type}} ElemsPerThread>
__global__ void
split_kernel(
    const ELEM_T *orig_input,
    InputMetaData<Rank> input_meta,
    OutputMetaData<ELEM_T, NumSplits> output_meta,
    const {{index_type}} split_dim,
    const int64_t input_split_dim_stride) {
  // split is the inverse of concat, so we
  //   (1) use blockIdx.y to specify the blocks for each ouput; and
  //   (2) use tid to access each output;
  const {{index_type}} tid = blockIdx.x * blockDim.x + threadIdx.x;
  const READ_T* input = reinterpret_cast<const READ_T*>(orig_input);

  READ_T* output =
      reinterpret_cast<READ_T*>(output_meta.outputs[blockIdx.y]);
  int64_t output_offset = output_meta.split_dim_offsets[blockIdx.y];
  int64_t num_output_elems = output_meta.num_elems[blockIdx.y];
  int64_t split_dim_size = output_meta.split_dim_sizes[blockIdx.y];
  int64_t input_offset = output_offset * input_split_dim_stride;

  unsigned constexpr read_t_sz = sizeof(READ_T);
  unsigned constexpr elem_t_sz = sizeof(ELEM_T);
  static_assert(read_t_sz >= elem_t_sz && (read_t_sz % elem_t_sz == 0));
  {{index_type}} n_of_elem_t = read_t_sz / elem_t_sz;
  // number of READ_T elements per thread
  {{index_type}} reads_per_thread_in_read_t = ElemsPerThread / n_of_elem_t;
  const {{index_type}} num_elems_in_read_t = num_output_elems / n_of_elem_t;
  {{index_type}} read_idx = tid;

#pragma unroll
  for ({{index_type}} i = 0; i < reads_per_thread_in_read_t;
       i++, read_idx += blockDim.x * gridDim.x) {
    if (read_idx >= num_elems_in_read_t) {
      break;
    }
    /* make sure to adjust read_idx, which refers to location at
       (read_idx * n_of_elem_t) actually */
    int64_t input_elem_offset =
        compute_input_elem_offset<Rank>(input_meta.input_shape,
                                        input_meta.input_strides,
                                        split_dim_size,
                                        split_dim,
                                        read_idx * n_of_elem_t);

    READ_T tmp_v = input[(input_offset + input_elem_offset) / n_of_elem_t];
    output[read_idx] = tmp_v;
  }
}

enum class LoadVecType {
  VT_HALF = 0,
  VT_BFLOAT16,
  VT_FLOAT,
  VT_FLOAT2,
  VT_FLOAT4
};

template <typename ELEM_T>
static inline LoadVecType get_vec_type(
    const {{index_type}} *shape, {{index_type}} rank, {{index_type}} dim) {
  assert(rank > 0);
  assert(dim < rank && dim >= 0);
  int64_t running_stride = shape[rank - 1];
  for ({{index_type}} i = rank - 2; i >= dim; i--) {
    running_stride *= shape[i];
  }
  {{index_type}} size_elem_t = sizeof(ELEM_T);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)  \\
  if (sizeof(vec_type) % size_elem_t == 0) {          \\
    {{index_type}} n_of_elem_t = sizeof(vec_type) / size_elem_t; \\
    if (running_stride % n_of_elem_t == 0) {          \\
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

template <typename ELEM_T, {{index_type}} Rank, {{index_type}} NumSplits,
          {{index_type}} ElemsPerThread, {{index_type}} ThreadsPerBlock>
void split_kernel_launcher(
    void *outputs[],
    {{index_type}} *output_shapes[],
    const bool output_masks[],
    const void *input,
    const {{index_type}} *input_shape,
    const {{index_type}} split_dim,
    const {{index_type}} split_sizes[],
    {{prefix}}Stream_t stream
) {

  InputMetaData<Rank> input_meta;
  input_meta.input_strides[Rank - 1] = 1;
  input_meta.input_shape[Rank - 1] = input_shape[Rank - 1];
  for ({{index_type}} i = Rank - 2; i >= 0; i--) {
    input_meta.input_strides[i] =
        input_meta.input_strides[i+1] * input_shape[i+1];
    input_meta.input_shape[i] = input_shape[i];
  }

  OutputMetaData<ELEM_T, NumSplits> output_meta;
  {{index_type}} offset = 0;
  {{index_type}} split_sizes_idx = 0;
  LoadVecType min_vec_type = LoadVecType::VT_FLOAT4;
  for ({{index_type}} i = 0; i < NumSplits; i++) {
    while (!output_masks[split_sizes_idx]) {
      offset += split_sizes[split_sizes_idx];
      split_sizes_idx++;
    }
    output_meta.outputs[i] = static_cast<ELEM_T*>(outputs[i]);
    output_meta.split_dim_offsets[i] = offset;
    output_meta.split_dim_sizes[i] = output_shapes[i][split_dim];
    output_meta.num_elems[i] = get_num_elems(output_shapes[i], Rank);
    offset += output_meta.split_dim_sizes[i];
    split_sizes_idx++;
    LoadVecType vec_type =
        get_vec_type<ELEM_T>(output_shapes[i], Rank, split_dim);
    min_vec_type = vec_type < min_vec_type ? vec_type : min_vec_type;
  }

  int64_t max_num_output_elems = 0;
  for ({{index_type}} i = 0; i < NumSplits; i++) {
    {{index_type}} num_outputs = get_num_elems(output_shapes[i], Rank);
    max_num_output_elems = num_outputs > max_num_output_elems ?
                           num_outputs : max_num_output_elems;
  }
  {{index_type}} m = (max_num_output_elems % (ThreadsPerBlock * ElemsPerThread) != 0);
  {{index_type}} num_blocks_x =
      (max_num_output_elems / (ThreadsPerBlock * ElemsPerThread)) + m;
  dim3 grid_config = dim3(num_blocks_x, NumSplits);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)                   \\
    if (min_vec_type == load_vec_type) {                               \\
      if (ElemsPerThread * sizeof(ELEM_T) < sizeof(vec_type)) {        \\
         throw std::runtime_error(                                     \\
           std::string("No valid kernel available for ") + #vec_type); \\
      }                                                                \\
      split_kernel<vec_type, ELEM_T, Rank, NumSplits, ElemsPerThread>  \\
        <<<grid_config, ThreadsPerBlock, 0, stream>>>(                 \\
            static_cast<const ELEM_T*>(input),                         \\
            input_meta,                                                \\
            output_meta,                                               \\
            split_dim,                                                 \\
            input_meta.input_strides[split_dim]);                      \\
      LAUNCH_CHECK_SPLIT();                                            \\
      return;                                                          \\
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

#undef CHECK_ERROR_SPLIT
#undef LAUNCH_CHECK_SPLIT

"""
)


EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if (rank == {{rank}} && real_num_splits == {{real_num_splits}}) {
{% for split_idx in split_indices %}
{% set outer_loop = loop %}
{{indent}}  {{index_type}} local_shape{{outer_loop.index0}}[{{rank}}];
{% for rank_idx in range(rank) %}
{{indent}}  local_shape{{outer_loop.index0}}[{{rank_idx}}] = input_shape[{{rank_idx}}];
{% endfor %}
{{indent}}  local_shape{{outer_loop.index0}}[split_dim] = split_sizes[{{split_idx}}];

{% endfor %}

{{indent}}  {{index_type}}* local_output_shapes[{{real_num_splits}}] = {
{% for idx in range(real_num_splits - 1) %}
{{indent}}    local_shape{{idx}},
{% endfor %}
{{indent}}    local_shape{{real_num_splits - 1}}
{{indent}}  };
{{indent}}  /* TODO: more profiling on ElemsPerThread and ThreadsPerBlock */
{{indent}}  split_kernel_launcher<{{elem_type}},
{{indent}}                        {{rank}}/*Rank*/,
{{indent}}                        {{real_num_splits}}/*NumSplits*/,
{{indent}}                        {{elems_per_thread}}/*ElemsPerThread*/,
{{indent}}                        {{threads_per_block}}/*THREADS_PER_BLOCK*/>(
{{indent}}      outputs, local_output_shapes, output_masks, input, input_shape, split_dim, split_sizes, stream);
{{indent}}  return;
{{indent}}}
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
{{kernel_src}}
void {{func_name}}(
    void* outputs[],
    {{index_type}} **output_shapes[],
    const bool output_masks[],
    const void* input,
    const {{index_type}} *input_shape,
    {{index_type}} real_num_splits,
    {{index_type}} all_num_splits,
    {{index_type}} split_sizes[],
    {{index_type}} split_dim,
    {{index_type}} rank,
    {{prefix}}Stream_t stream
    ) {

  if (rank <= 0) {
    throw std::runtime_error("rank must be larger than 0!");
  }
  if (split_dim >= rank) {
    throw std::runtime_error("cat_dim must be smaller than rank!");
  }
  if (real_num_splits < 1) {
    throw std::runtime_error("the number of splits must be larger than 0!");
  }

  // now we update the shape for each output
  {{index_type}} real_idx = 0;
  for ({{index_type}} i = 0; i < all_num_splits; i++) {
    if (!output_masks[i]) {
      continue;
    }
    {{index_type}} **shape_ptr = output_shapes[real_idx];
    for ({{index_type}} dim_idx = 0; dim_idx < rank; dim_idx++) {
      *(shape_ptr[dim_idx]) = input_shape[dim_idx];
    }
    // update dim size for the split axis
    *(shape_ptr[split_dim]) = split_sizes[i];
    real_idx++;
  }

  {{index_type}} split_dim_size = input_shape[split_dim];
  {{index_type}} sum_of_split_sizes = 0;
  for ({{index_type}} i = 0; i < all_num_splits; i++) {
    sum_of_split_sizes += split_sizes[i];
  }
  if (split_dim_size != sum_of_split_sizes) {
      throw std::runtime_error("unmatched split dim size!");
  }

  // If split dim is zero, we are done
  if (split_dim_size == 0) {
    return;
  }
  // If the input tensor is empty, we are done
  if (get_num_elems(input_shape, rank) == 0) {
    return;
  }
  // make sure input and outputs are valid
  if (!input) {
    throw std::runtime_error("input is NULL!");
  }
  for (int i = 0; i < real_num_splits; i++) {
    if (!outputs[i]) {
      throw std::runtime_error("NULL output found at: " + std::to_string(i));
    }
  }

{{exec_paths}}

  throw std::runtime_error(
      "Unsupported split kernel specialization!"
  );
}
"""
)


OUTPUT_SHAPE_DEF_TEMPLATE = jinja2.Template(
    """
{{indent}}{{index_type}} *{{output_shape_name}}[] = {
{{indent}}  {{output_dim_refs}}
{{indent}}};
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{

{{indent}}  void *outputs[] = {
{{indent}}    {{outputs}}
{{indent}}  };

{{output_shape_defs}}

{{indent}}  {{index_type}} **output_shapes[] = {
{{indent}}    {{output_shapes}}
{{indent}}  };

{{indent}}  const {{index_type}} {{input_name}}_shape[] = {
{{indent}}    {{input_dims}}
{{indent}}  };

{{indent}}  {{index_type}} split_sizes[] = {
{{indent}}    {{split_sizes}}
{{indent}}  };

{{indent}}  bool output_masks[] = {
{{indent}}    {{output_masks}}
{{indent}}  };

{{indent}}  {{func_name}}(
{{indent}}      outputs,
{{indent}}      output_shapes,
{{indent}}      output_masks,
{{indent}}      {{input_ptr}},
{{indent}}      {{input_name}}_shape,
{{indent}}      {{real_num_splits}}/*real_num_splits*/,
{{indent}}      {{all_num_splits}}/*all_num_splits*/,
{{indent}}      split_sizes,
{{indent}}      {{split_dim}}/*split_dim*/,
{{indent}}      {{rank}}/*rank*/,
{{indent}}      stream
{{indent}}  );
{{indent}}}
"""
)


def gen_function_decl(func_attrs, backend_spec):
    """Generate function declaration.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec : BackendSpec
        Cuda/Rocm type definitions
    Returns
    -------
    str
        Rendered function declaration.
    """
    return FUNC_DECL_TEMPLATE.render(
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        func_name=func_attrs["name"],
    )


def gen_function(func_attrs, backend_spec):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.

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

    split_indices = [idx for idx, mask in enumerate(func_attrs["output_masks"]) if mask]

    # TODO: consider to add profiling paths for tuning
    # elems_per_thread and threads_per_block
    exec_paths = EXEC_COND_TEMPLATE.render(
        indent="  ",
        rank=len(x_shape),
        real_num_splits=len(func_attrs["outputs"]),
        split_indices=split_indices,
        elem_type=input_type,
        elems_per_thread=128,
        threads_per_block=128,
        index_type=backend_spec.index_type,
    )
    header_src = backend_spec.header_src_template.render()
    kernel_src = KERNEL_SRC_TEMPLATE.render(
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        header_src=header_src,
    )
    return SRC_TEMPLATE.render(
        kernel_src=kernel_src,
        func_name=func_attrs["name"],
        exec_paths=exec_paths,
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
    )


def gen_function_call(func_attrs, backend_spec, indent="  "):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    indent : str, optional
        Indent for template, by default "  ".

    Returns
    -------
    str
        Rendered function call.
    """
    x = func_attrs["inputs"][0]
    outputs = func_attrs["outputs"]
    split_dim = func_attrs["split_dim"]
    num_splits = len(func_attrs["outputs"])

    output_names = ",\n      ".join([i._attrs["name"] for i in outputs])

    output_shape_defs = []
    output_shape_names = []
    for i in outputs:
        output_shape_name = "{}_shape".format(i._attrs["name"])
        if output_shape_name not in output_shape_names:
            dim_refs = ", ".join(
                ["&" + dim._attrs["name"] for dim in i._attrs["shape"]]
            )
            one_shape_def = OUTPUT_SHAPE_DEF_TEMPLATE.render(
                indent="      ",
                output_shape_name=output_shape_name,
                output_dim_refs=dim_refs,
                index_type=backend_spec.index_type,
            )
            output_shape_defs.append(one_shape_def)
        output_shape_names.append(output_shape_name)

    x_shape = x._attrs["shape"]
    x_dims = ", ".join([dim._attrs["name"] for dim in x_shape])

    split_sizes = ", ".join([str(i) for i in func_attrs["split_sizes"]])

    output_masks_str = ", ".join(
        ["true" if mask is True else "false" for mask in func_attrs["output_masks"]]
    )

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        outputs=output_names,
        output_shape_defs="".join(output_shape_defs),
        output_shapes=", ".join(output_shape_names),
        output_masks=output_masks_str,
        input_dims=x_dims,
        func_name=func_attrs["name"],
        input_name=x._attrs["name"],
        input_ptr=x._attrs["name"],
        split_dim=split_dim,
        real_num_splits=len(func_attrs["outputs"]),
        all_num_splits=len(func_attrs["output_masks"]),
        rank=len(x._attrs["shape"]),
        num_splits=num_splits,
        split_sizes=split_sizes,
        index_type=backend_spec.index_type,
    )
