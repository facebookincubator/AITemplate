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
A special reduce kernel suitable for small reduction axes. The current upper
bound is set to 128.
"""

import math

import jinja2

from aitemplate.compiler.base import IntImm


EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if (input_shape[reduction_axis] <= {{reduction_dim_upperbound}}) {
{{indent}}  if (reduction_axis == rank - 1) {
{{indent}}    constexpr int64_t cst_n = {{reduction_dim_val}};
{{indent}}    if (rank == 3) {
{{indent}}      b = input_shape[0];
{{indent}}      m = input_shape[1];
{{indent}}      if (b > 1) {
{{indent}}        batch_stride_input = m * cst_n;
{{indent}}        batch_stride_output = m;
{{indent}}      }
{{indent}}    } else if (rank == 2) {
{{indent}}      m = input_shape[0];
{{indent}}    } else if (rank == 1) {
{{indent}}      // nothing to do
{{indent}}    } else {
{{indent}}      throw std::runtime_error("reduce_small_axis: invalid rank rank");
{{indent}}    }
{{indent}}    reduce_mean_launcher_small_axis<{{elem_output_type}},
{{indent}}                                    {{elem_input_type}},
{{indent}}                                    {{elem_compute_type}},
{{indent}}                                    cst_n>(
{{indent}}          static_cast<{{elem_output_type}}*>(output),
{{indent}}          static_cast<{{elem_input_type}}*>(input),
{{indent}}          b, m, batch_stride_input,
{{indent}}          batch_stride_output, stream);
{{indent}}    return;
{{indent}}  } else {
{{indent}}    // TODO: support more reduction axis
{{indent}}    // fall-through to the general reduction kernel for now
{{indent}}  }
{{indent}}}
{% if static_small_reduction_dim %}
#define SKIP_GENERAL_REDUCTION
{% endif %}
"""
)


KERNEL_SRC_TEMPLATE = jinja2.Template(
    """
constexpr const int ThreadsPerBlock = 128;

template <typename ElementInput,
          typename ElementOutput,
          typename ElementCompute,
          typename ReadVecT,
          typename WriteVecT,
          int64_t num_rows_per_thread,
          int64_t num_cols>
__global__ void reduce_small_in_v_out_v(
    ElementOutput *output,
    const ElementInput *input,
    int64_t num_rows,
    int64_t batch_stride_input,
    int64_t batch_stride_output) {
  int block_batch = blockIdx.y;
  // index within the batch
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t idx = tid * num_rows_per_thread;
  if (idx >= num_rows)
    return;
  // input within the batch
  int64_t input_offset = idx * num_cols;
  const ElementInput *this_input =
      input + block_batch * batch_stride_input + input_offset;
  size_t output_idx = block_batch * batch_stride_output + idx;
  ElementOutput *this_output = get_strided_address_at_idx<ElementOutput, ElementOutput>(output, output_idx);

  static_assert(sizeof(ReadVecT) % sizeof(ElementInput) == 0);
  constexpr int n_read_elems_in_v = sizeof(ReadVecT) / sizeof(ElementInput);
  // number of original elements
  constexpr int64_t num_elems_per_thread = num_rows_per_thread * num_cols;
  // number of vector elements
  static_assert(num_elems_per_thread % n_read_elems_in_v == 0);
  constexpr int64_t num_elems_per_thread_v =
      num_elems_per_thread / n_read_elems_in_v;

  ReadVecT read_elems_v[num_elems_per_thread_v];
  const ReadVecT *this_input_v = reinterpret_cast<const ReadVecT*>(this_input);
  // read
  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < num_elems_per_thread_v; i++) {
    cutlass::arch::global_load<ReadVecT, sizeof(ReadVecT)>(
        read_elems_v[i], this_input_v + i, true
    );
  }

  // compute
  using FragmentCompute = ElementCompute;
  ElementInput *read_elems = reinterpret_cast<ElementInput *>(read_elems_v);
  using ReduceScalarOp = {{reduce_op}}<ElementCompute>;
  ReduceScalarOp reduce_s_op;
  constexpr int num_reduced_elems = num_cols;

  auto prologue_fn = [&] (FragmentCompute fragment) {
    {{prologue_code}}
  };
  auto epilogue_scalar_fn = [&] (ElementCompute reduced_result) {
    {{epilogue_scalar_code}}
  };

  ElementOutput reduced_elems[num_rows_per_thread];
  static_assert(num_elems_per_thread % num_cols == 0);
  cutlass::NumericConverter<ElementCompute, ElementInput> convert_input;
  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < num_elems_per_thread / num_cols; i++) {
    static_assert(num_elems_per_thread % num_rows_per_thread == 0);
    FragmentCompute frag_compute = FragmentCompute(0);
    CUTLASS_PRAGMA_UNROLL
    for (int64_t j = 0; j < num_cols; j++) {
      int64_t read_idx = i * num_cols + j;
      FragmentCompute tmp = prologue_fn(convert_input(read_elems[read_idx]));
      frag_compute = reduce_s_op(frag_compute, tmp);
    }
    cutlass::NumericConverter<ElementOutput, ElementCompute> convert_output;
    ElementCompute tmp = epilogue_scalar_fn(frag_compute);
    reduced_elems[i] = convert_output(tmp);
  }

  WriteVecT *this_output_v = reinterpret_cast<WriteVecT*>(this_output);
  WriteVecT *reduced_elems_v = reinterpret_cast<WriteVecT*>(&reduced_elems[0]);
  constexpr int n_write_elems_in_v = sizeof(WriteVecT) / sizeof(ElementOutput);
  CUTLASS_PRAGMA_UNROLL
{% if output_accessor.is_contiguous %}
  for (int64_t i = 0; i < num_rows_per_thread / n_write_elems_in_v; i++) {
    WriteVecT tmp = reduced_elems_v[i];
    this_output_v[i] = tmp;
  }
{% else %}
  for (int64_t read_i = 0, write_i = 0;
       read_i < num_rows_per_thread / n_write_elems_in_v;
       read_i++,
       write_i += {{output_accessor.actual_total_elements_from_stride_dim}}
  ) {
    WriteVecT tmp = reduced_elems_v[read_i];
    this_output_v[write_i] = tmp;
  }
{% endif %}
}

template <typename ElemOutputType,
          typename ElemInputType,
          typename ElemComputeType,
          int64_t num_cols>
void reduce_mean_launcher_small_axis(
  ElemOutputType *output,
  ElemInputType *input,
  int64_t num_batches,
  int64_t num_rows,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {
  constexpr int64_t num_read_v =
      sizeof({{read_vec_type}}) / sizeof(ElemInputType);
  constexpr int64_t row_gcd = std::gcd(num_cols, num_read_v);
  constexpr int64_t num_rows_per_thread = num_read_v / row_gcd;
{% if output_accessor.is_contiguous %}
  constexpr int64_t num_write_bytes_v =
      num_rows_per_thread * sizeof(ElemOutputType);
{% else %}
  constexpr int64_t num_write_bytes_v =
      std::min(num_rows_per_thread, static_cast<int64_t>({{output_access_alignment}})) *
      sizeof(ElemOutputType);
{% endif %}

  assert(num_rows % num_rows_per_thread == 0);
  int64_t real_rows = num_rows / num_rows_per_thread;
  dim3 grid(static_cast<int>(real_rows + ThreadsPerBlock -1 ) / ThreadsPerBlock,
            static_cast<int>(num_batches));

  if (num_rows % num_rows_per_thread == 0) {

#define HANDLE_ONE_WRITE_VEC(write_bytes, write_vec_type) \\
    if (write_bytes == num_write_bytes_v) {               \\
      reduce_small_in_v_out_v<ElemInputType,              \\
                              ElemOutputType,             \\
                              ElemComputeType,            \\
                              {{read_vec_type}},          \\
                              write_vec_type,             \\
                              num_rows_per_thread,        \\
                              num_cols>                   \\
      <<<grid, ThreadsPerBlock, 0, stream>>>(             \\
          output,                                         \\
          input,                                          \\
          num_rows,                                       \\
          batch_stride_input,                             \\
          batch_stride_output);                           \\
      LAUNCH_CHECK_REDUCE();                              \\
      return;                                             \\
    }
    HANDLE_ONE_WRITE_VEC(16, uint4)
    HANDLE_ONE_WRITE_VEC(8, uint2)
    HANDLE_ONE_WRITE_VEC(4, unsigned)
    if constexpr (std::is_same_v<ElemOutputType, cutlass::half_t>) {
      HANDLE_ONE_WRITE_VEC(2, cutlass::half_t)
    }
    else if constexpr (std::is_same_v<ElemOutputType, cutlass::bfloat16_t>) {
      HANDLE_ONE_WRITE_VEC(2, cutlass::bfloat16_t)
    }
    throw std::runtime_error("unsupported vector size for write");
  } else {
    throw std::runtime_error("unsupported num_row_per_threads");
  }
}

template <typename ElemOutputType, typename ElemInputType>
void reduce_mean_launcher_small_axis_column_major(
  ElemOutputType *output,
  ElemInputType *input,
  int64_t num_batches,
  int64_t num_rows,
  int64_t num_columns,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {
}

"""
)


def _get_read_vector_type(input_shape, input_type, force_min_vec_type=False) -> str:
    """return vector_type for reading input along reduction axis -1 (for row-major).
    In a long run, we should consider to add profiling support to reduce kernels
    and then tune vector_type. Currently, we use the following heuristics based on
    manual experiments:
    (1) for 1-d input, it's straightforward - we pickup the max vector type
        whose size is modulo of dim_val
    (2) for n-d input, if the n-2 dimension value is less than 128, we utilize
        thread-level parallelism, i.e. we don't try to read inputs across multiple
        dimensions
    (3) we ensure each thread won't read too many elements (maximum is 256 at
        the moment)

    Parameters
    ----------
    input_shape: List[IntImm]

    Returns
    -------
    str
        returns the vector type (uint4/uint2/unsigned/cutlass::half_t)
        for reading input
    """
    type_to_size_in_bit = {
        "half": 16,
        "cutlass::half_t": 16,
        "bfloat16": 16,
        "cutlass::bfloat16_t": 16,
        "float": 32,
    }

    # FIXME
    # (1) note that we don't support int8, so we don't have vector_type for one byte
    # (2) the input type is inherited from reduce_3d, so we still
    #     use cutlass::half_t for fp16. We will replace it to half once we
    #     unify our half representation
    vector_types = {
        "cutlass::half_t": [
            ("uint4", 16),
            ("uint2", 8),
            ("unsigned", 4),
            ("cutlass::half_t", 2),
        ],
        "cutlass::bfloat16_t": [
            ("uint4", 16),
            ("uint2", 8),
            ("unsigned", 4),
            ("cutlass::bfloat16_t", 2),
        ],
        "float": [
            ("uint4", 16),
            ("uint2", 8),
            ("unsigned", 4),
        ],
    }

    def _size_to_vector_type(sz_in_byte) -> str:
        """return vector_type for the given size"""
        for vec_type, sz in vector_types[input_type]:
            if sz_in_byte % sz == 0:
                return vec_type
        raise NotImplementedError("Unsupported vector size: {}".format(sz_in_byte))

    reduction_axis = -1

    if not isinstance(input_shape[reduction_axis], IntImm):
        # the last dimension is IntVar, so the best we can do in
        # terms of the read vector type is the input_type iteself
        return input_type

    rank = len(input_shape)
    reduction_dim_val = input_shape[reduction_axis]._attrs["values"][0]
    input_type_sz_in_bit = type_to_size_in_bit.get(input_type)
    assert input_type_sz_in_bit is not None and input_type_sz_in_bit % 8 == 0
    input_type_sz_in_byte = input_type_sz_in_bit / 8

    # return the minimal vector type based on the input_type
    if force_min_vec_type:
        return _size_to_vector_type(input_type_sz_in_byte)

    # 1-d tensor
    if rank == 1:
        return _size_to_vector_type(reduction_dim_val * input_type_sz_in_byte)

    # no matter input_shape[-2] is IntImm or IntVar, we get the minimal dim value
    dim_2_val = input_shape[-2]._attrs["values"][0]
    # If input_shape[-2] is too small, we utilize thread-level parallelism
    tlp_lower_bound = 128
    if dim_2_val <= tlp_lower_bound:
        return _size_to_vector_type(reduction_dim_val * input_type_sz_in_byte)
    # When dim_2_val % 2 == 1, we cannot read inputs across dimension
    if dim_2_val % 2 == 1:
        return _size_to_vector_type(reduction_dim_val * input_type_sz_in_byte)

    # Let's make sure that each thread won't read too many.
    # Currently set it to be less than 256 elements
    max_num_elems_per_thread = 256

    def _valid_vector_type(vec_type, sz_in_byte):
        if sz_in_byte % input_type_sz_in_byte != 0:
            return False
        num_elems_in_a_vec = int(sz_in_byte / input_type_sz_in_byte)
        gcd = math.gcd(reduction_dim_val, num_elems_in_a_vec)
        num_reduction_dim = int(num_elems_in_a_vec / gcd)
        if num_reduction_dim >= dim_2_val or dim_2_val % num_reduction_dim != 0:
            return False
        if num_reduction_dim == 1:
            return True
        if num_reduction_dim * reduction_dim_val > max_num_elems_per_thread:
            return False
        return True

    for vec_type, sz in vector_types[input_type]:
        if _valid_vector_type(vec_type, sz):
            return vec_type

    raise RuntimeError("Cannot find valid vector type!")


def get_exec_cond_and_kernel(
    func_attrs,
    reduce_op,
    reduction_axis,
    prologue_code,
    epilogue_scalar_code,
    input_type,
    output_type,
    acc_type,
    output_accessors,
    output_alignment,
) -> str:
    """return a pair that contains the execution condition for this special
       reduction kernel and the source code of this reduction kernel

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce op
    reduce_op : str
        the reduce op's string representation such as cutlass::plus
    reduction_axis : int
        the axis along which the reduction is performed
    prologue_code : str
        prologue code to process input before reduction
    epilogue_scalar_code : str
        epilogue code to process each final reduced element
    input_type : str
        specifies the input data type
    output_type : str
        specifies the output data type
    acc_type : str,
        specifies the data type for accumulation
    output_accessors : List[TensorAccessor]
        output TensorAccessor(s)
    output_alignment : int
        max alignment value that meets the requirement for accessing strided output

    Returns
    -------
    str
        returns the rendered code for the complete implementation of the reduce op
    """
    x = func_attrs["inputs"][0]
    x_shape = x._attrs["shape"]
    reduction_dim = x_shape[reduction_axis]
    # TODO: support dynamic reduction axis
    if not isinstance(reduction_dim, IntImm):
        return ("", "")

    reduction_dim_upperbound = 128
    if reduction_dim._attrs["values"][0] > reduction_dim_upperbound:
        return ("", "")

    rank = len(x_shape)
    # TODO: support reduction_axis = rank - 2
    valid_static_small_reduction_dim = reduction_axis == rank - 1
    reduction_dim_val = reduction_dim._attrs["values"][0]

    exec_cond = EXEC_COND_TEMPLATE.render(
        indent="  ",
        func_name=func_attrs["name"],
        elem_output_type=output_type,
        elem_input_type=input_type,
        elem_compute_type=acc_type,
        reduction_dim_upperbound=reduction_dim_upperbound,
        reduction_dim_val=reduction_dim_val,
        static_small_reduction_dim=valid_static_small_reduction_dim,
    )

    if output_accessors[0].is_contiguous:
        read_vec_type = _get_read_vector_type(x_shape, input_type)
    else:
        # For strided accesses, we force to take minimal vector type to
        # reduce uncoalesced device memory accesses. The perf penalty
        # of uncoalesced accesses (both read and write) seems to outweigh the
        # benefit from vector load.
        read_vec_type = _get_read_vector_type(
            x_shape, input_type, force_min_vec_type=True
        )
    kernel_src = KERNEL_SRC_TEMPLATE.render(
        reduce_op=reduce_op,
        prologue_code=prologue_code,
        epilogue_scalar_code=epilogue_scalar_code,
        read_vec_type=read_vec_type,
        output_accessor=output_accessors[0],
        output_access_alignment=output_alignment,
    )
    return (exec_cond, kernel_src)
