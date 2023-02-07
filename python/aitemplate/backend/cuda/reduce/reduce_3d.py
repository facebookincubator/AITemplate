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
A common reduce kernel that renders 3d tensors.
In particular, the kernel template accepts a prologue and an epilogue like
lambda functions that can be applied to the input and final reduced result,
respectively. So, this common kernel can be used to implement a family of
reduction ops such as reduce_mean and norm where we need to apply a scalar-op
to each final element.
"""
import bisect

import jinja2

from ...backend_spec import CUDASpec
from ...common import tensor_accessor_codegen

from . import reduce_small_axis


DEFAULT_PROLOGUE_TEMPLATE = jinja2.Template(
    """
{{indent}}return fragment;
"""
)


DEFAULT_EPILOGUE_SCALAR_TEMPLATE = jinja2.Template(
    """
{{indent}}return reduced_result;
"""
)


REDUCE_KERNEL_INSTANCE = jinja2.Template(
    """
using ReductionKernel{{layout}}_{{align}} = ReductionKernel3D<
    {{elem_output_type}}, /* ElementOutput */
    {{elem_input_type}}, /* ElementInput */
    {{elem_compute_type}}, /*ElementCompute */
    {{align}},
    cutlass::layout::{{layout}}, /*Layout*/
    cutlass::MatrixShape<1, {{shared_col_size}}>
>;
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void * /*output*/,
  void * /*input*/,
  int /*reduction_axis*/,
  int64_t *[] /*output_shape*/,
  const int64_t * /*input_shape*/,
  int /*rank*/,
  bool /*keep_dim*/,
  cudaStream_t /*stream*/
);
"""
)


EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{special_exec_cond}}

// If we can statically determine we always fall into special exec cond above,
// it's safe to skip the general exec path below
#ifndef SKIP_GENERAL_REDUCTION
{{indent}}if (reduction_axis == rank - 1) {
{{indent}}  if (rank == 3) {
{{indent}}    b = input_shape[0];
{{indent}}    m = input_shape[1];
{{indent}}    n = input_shape[2];
{{indent}}    if (b > 1) {
{{indent}}      batch_stride_input = m * n;
{{indent}}      batch_stride_output = m;
{{indent}}    }
{{indent}}  } else if (rank == 2) {
{{indent}}    m = input_shape[0];
{{indent}}    n = input_shape[1];
{{indent}}  } else if (rank == 1) {
{{indent}}    n = input_shape[0];
{{indent}}  } else {
{{indent}}    throw std::runtime_error("unreachable: invalid rank");
{{indent}}  }
{% for align in alignments %}
{{indent}}  if (input_shape[reduction_axis] % {{align}} == 0) {
{{indent}}    reduce_mean_launcher_RowMajor_{{align}}(
{{indent}}      static_cast<{{elem_output_type}}*>(output),
{{indent}}      static_cast<{{elem_input_type}}*>(input),
{{indent}}      b, m, n, batch_stride_input, batch_stride_output, stream);
{{indent}}    return;
{{indent}}  }
{% endfor %}
{{indent}}  throw std::runtime_error("unreachable: invalid alignment");
{{indent}}} else if (reduction_axis == rank - 2) {
{{indent}}  if (rank == 3) {
{{indent}}    b = input_shape[0];
{{indent}}    m = input_shape[2];
{{indent}}    n = input_shape[1];
{{indent}}    if (b > 1) {
{{indent}}      batch_stride_input = m * n;
{{indent}}      batch_stride_output = m;
{{indent}}    }
{{indent}}  } else if (rank == 2) {
{{indent}}    m = input_shape[1];
{{indent}}    n = input_shape[0];
{{indent}}  } else {
{{indent}}    throw std::runtime_error("unreachable: invalid rank");
{{indent}}  }
{{indent}}  reduce_mean_launcher_ColumnMajor_1(
{{indent}}    static_cast<{{elem_output_type}}*>(output),
{{indent}}    static_cast<{{elem_input_type}}*>(input),
{{indent}}    b, m, n, batch_stride_input, batch_stride_output, stream);
{{indent}}  return;
{{indent}}}
#else
#undef SKIP_GENERAL_REDUCTION
#endif // !SKIP_GENERAL_REDUCTION
"""
)


KERNEL_SRC_TEMPLATE = jinja2.Template(
    """
// Modified from cutlass/examples/35_gemm_softmax/gemm_with_softmax.h

/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**

*/

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include <numeric>

#include "cutlass/cutlass.h"

#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/array.h"
#include "cutlass/device_kernel.h"
#include "cutlass/functional.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/fast_math.h"

#ifndef CHECK_ERROR_REDUCE
#define CHECK_ERROR_REDUCE(expr)                             \\
  do {                                                       \\
    cudaError_t status = (expr);                             \\
    if (status != cudaSuccess) {                             \\
      auto msg = std::string("Got error: ") +                \\
        cudaGetErrorString(status) +                         \\
        " at " + __FILE__ + ": " + std::to_string(__LINE__); \\
      std::cerr << msg << std::endl;                         \\
      throw std::runtime_error(msg);                         \\
    }                                                        \\
  } while (0)
#endif // CHECK_ERROR_REDUCE

#ifndef LAUNCH_CHECK_REDUCE
#define LAUNCH_CHECK_REDUCE() CHECK_ERROR_REDUCE(cudaGetLastError())
#endif // LAUNCH_CHECK_REDUCE

{{extra_code}}

namespace {

template <
  typename ElementOutput,
  typename ElementInput,
  typename ElementCompute,
  int Alignment,
  typename Layout_ = cutlass::layout::RowMajor,
  typename Shape_ = cutlass::MatrixShape<4, 16>
>
struct ReductionKernel3D {

  static int const kAlignment = Alignment;

  using Layout = Layout_;
  using Shape = Shape_;

  using TensorOutput = cutlass::TensorRef<ElementOutput, Layout>;
  using TensorInput = cutlass::TensorRef<ElementInput, Layout>;
  using TensorCompute = cutlass::TensorRef<ElementCompute, Layout>;

  struct Arguments {

    TensorOutput ref_output;     ///< Output tensor
    TensorInput ref_input;       ///< Input tensor
    cutlass::MatrixCoord extent; ///< Extent of input and output tensors
    int64_t input_row_stride;    ///< stride for accessing next element in
                                 ///< the same row. It's 1 for RowMajor and
                                 ///< extent.row() for ColMajor
    int64_t batch_count;         ///< Batch count
    int64_t batch_stride_output; ///< Batch stride for Output tensor
    int64_t batch_stride_input;  ///< Batch stride for Input tensor

    Arguments(
      TensorOutput    ref_output_,        ///< Output tensor
      TensorInput     ref_input_,         ///< Input tensor
      cutlass::MatrixCoord extent_,       ///< Extent of input and output tensors
      int64_t         input_row_stride_,  ///< stride for accessing input rows
      int64_t         batch_count_,       ///< Batch count
      int64_t         batch_stride_output_ = 0,
      int64_t         batch_stride_input_ = 0
    ):
      ref_output(ref_output_),
      ref_input(ref_input_),
      extent(extent_),
      input_row_stride(input_row_stride_),
      batch_count(batch_count_),
      batch_stride_output(batch_stride_output_),
      batch_stride_input(batch_stride_input_)
    { }
  };

  struct Params {
    Arguments args;
    Params() { }
    Params(Arguments const &args_): args(args_) { }
  };

  struct SharedStorage {
    cutlass::AlignedArray<ElementCompute, Shape::kCount, Shape::kCount * alignof(ElementCompute)> exchange;
  };

  CUTLASS_DEVICE
  ReductionKernel3D() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    reduce_partial(params.args, shared_storage);

    __syncthreads();

    reduce_final(params.args, shared_storage);

    __syncthreads();
  }

  /// Partial reduction
  CUTLASS_DEVICE
  void reduce_partial(Arguments const &args, SharedStorage &shared_storage) {

    using AccessTypeInput = cutlass::AlignedArray<ElementInput, kAlignment>;

    int block_batch = blockIdx.z;
    int block_m = blockIdx.x * Shape::kRow;
    int block_n = 0;

    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x * kAlignment;

    int idx_m = block_m + thread_m;
    int idx_n = block_n + thread_n;

    AccessTypeInput *access_input = reinterpret_cast<AccessTypeInput *>(
      args.ref_input.data() +
      args.batch_stride_input * block_batch +
      args.ref_input.layout()({idx_m, idx_n}));

    using ConvertS = cutlass::NumericArrayConverter<ElementCompute, ElementInput, kAlignment>;
    ConvertS convert_s;

    using FragmentCompute = cutlass::Array<ElementCompute, kAlignment>;
    using ReduceVectorOp = {{reduce_op}}<FragmentCompute>;
    using ReduceScalarOp = {{reduce_op}}<ElementCompute>;
    ReduceVectorOp reduce_v_op;
    ReduceScalarOp reduce_s_op;

    FragmentCompute frag_compute;
    frag_compute.clear();

    if (idx_m < args.extent.row()) {

      CUTLASS_PRAGMA_UNROLL
      for (
        int idx = 0;
        idx < args.extent.column();
        idx += Shape::kColumn * kAlignment) {

        if (idx_n < args.extent.column()) {

          AccessTypeInput fetch;
          cutlass::arch::global_load<AccessTypeInput, sizeof(AccessTypeInput)>(
              fetch, access_input, true);
          auto prologue_fn = [&] (FragmentCompute fragment) {

{{prologue_code}}

          };
          FragmentCompute tmp = prologue_fn(convert_s(fetch));
          frag_compute = reduce_v_op(frag_compute, tmp);
        }

        access_input += Shape::kColumn * args.input_row_stride;
        idx_n += Shape::kColumn * kAlignment;
      }

      // Reduce the elements owned by one thread
      ElementCompute result = frag_compute[0];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kAlignment; ++i) {
        result = reduce_s_op(result, frag_compute[i]);
      }

      shared_storage.exchange.data()[threadIdx.x + threadIdx.y * Shape::kColumn] = result;
    }
  }

  /// Compute the final summation from data in SMEM
  CUTLASS_DEVICE
  void reduce_final(Arguments const &args, SharedStorage &shared_storage) {

    //
    // SMEM has shape `Shape::Row`-by-`Shape::Column`
    //
    // This computes a reduction across the `Column` dimension yielding a `Row-by-1` vector.
    //

    //
    // Tuning parameters tradeoff ILP with latency
    //
    // During each step of the reduction, each thread performs `kAccesses` of
    // vector size `kReduceVector`

    // Tune the number of accesses per reduction
    int const kAccesses = 2;

    // Tune the memory access size
    int const kReduceVector = 4;

    //
    // Static asserts to ensure integrity
    //

    static_assert(kAccesses * kReduceVector,
      "Zero-size steps would infinitely loop.");

    static_assert(
      cutlass::is_pow2<Shape::kColumn>::value &&
      cutlass::is_pow2<kAccesses>::value &&
      cutlass::is_pow2<kReduceVector>::value,
      "Powers of two only.");

    static_assert(!(Shape::kColumn % (kAccesses * kReduceVector)),
      "Divisibility not satisfied");

    //
    // Reduction operators
    //

    using FragmentCompute = cutlass::Array<ElementCompute, kReduceVector>;
    using ReduceVectorOp = {{reduce_op}}<FragmentCompute>;
    using ReduceScalarOp = {{reduce_op}}<ElementCompute>;
    ReduceVectorOp reduce_v_op;
    ReduceScalarOp reduce_s_op;

    // Tree reduction
    ElementCompute *smem_ptr = shared_storage.exchange.data() + threadIdx.y * Shape::kColumn;

    ElementCompute result = ElementCompute();

    CUTLASS_PRAGMA_UNROLL
    for (
      int tidx_limit = Shape::kColumn / (kAccesses * kReduceVector);
      tidx_limit > 0;
      tidx_limit /= (kAccesses * kReduceVector)) {

      if (threadIdx.x < tidx_limit) {
        FragmentCompute fetch;

        cutlass::arch::shared_load<sizeof(FragmentCompute)>(
            &fetch,
            cutlass::arch::cutlass_get_smem_pointer(smem_ptr + threadIdx.x * kReduceVector));

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kAccesses; ++i) {
          FragmentCompute extra;

          cutlass::arch::shared_load<sizeof(FragmentCompute)>(
              &extra,
              cutlass::arch::cutlass_get_smem_pointer(
                  smem_ptr + threadIdx.x * kReduceVector + tidx_limit * kReduceVector * i));

          fetch = reduce_v_op(fetch, extra);
        }

        // Reduce to one element
        result = fetch[0];

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kReduceVector; ++i) {
          result = reduce_s_op(result, fetch[i]);
        }
      }
      __syncthreads();

      if (threadIdx.x < tidx_limit) {
        smem_ptr[threadIdx.x] = result;
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {

      int const kLgResidual =
        (cutlass::log2_down<Shape::kColumn>::value %
         cutlass::log2_down<kAccesses * kReduceVector>::value);

      // Certain shape combinations require an additional reduction step
      if (kLgResidual) {
        result = ElementCompute();

        int const kResidualVector = (1 << kLgResidual);
        cutlass::Array<ElementCompute, kResidualVector> fetch;

        cutlass::arch::shared_load<sizeof(FragmentCompute)>(
            &fetch,
            cutlass::arch::cutlass_get_smem_pointer(smem_ptr));

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kResidualVector; ++i) {
          result = reduce_s_op(result, fetch[i]);
        }
      }

      int block_batch = blockIdx.z;
      int block_m = blockIdx.x * Shape::kRow;
      int thread_m = threadIdx.y;
      int idx_m = block_m + thread_m;
      if (idx_m >= args.extent.row()) {
        return;
      }

      int64_t output_idx = args.batch_stride_output * block_batch +
                           args.ref_output.layout()({idx_m, 0});
      ElementOutput *access_output =
          get_strided_address_at_idx<ElementOutput, ElementOutput>(
              reinterpret_cast<ElementOutput*>(args.ref_output.data()), output_idx);

      cutlass::NumericConverter<ElementOutput, ElementCompute> convert_output;

      auto epilogue_scalar_fn = [&] (ElementCompute reduced_result,
                                     int num_reduced_elems) {

{{epilogue_scalar_code}}

      };
      ElementCompute tmp = epilogue_scalar_fn(result, args.extent.column());
      *access_output = convert_output(tmp);
    }
  }
};

{{reduce_kernel_instance}}

{% for align in alignments %}
template<typename ElementOutput, typename ElementInput>
void reduce_mean_launcher_RowMajor_{{align}}(
  ElementOutput *output,
  ElementInput *input,
  int64_t batch_count,
  int64_t rows,
  int64_t columns,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {

  using ReductionKernel = ReductionKernelRowMajor_{{align}};

  dim3 apply_block(ReductionKernel::Shape::kColumn,
                   ReductionKernel::Shape::kRow);

  int cta_rows = ReductionKernel::Shape::kRow;
  int cta_columns = ReductionKernel::Shape::kColumn * ReductionKernel::kAlignment;

  dim3 apply_grid(static_cast<int>((rows + cta_rows - 1) / cta_rows),
                  static_cast<int>((columns + cta_columns - 1) / cta_columns),
                  static_cast<int>(batch_count));

  // row major
  int64_t lda_output = 1;
  int64_t lda_input = columns;
  ReductionKernel::Layout output_layout(lda_output);
  ReductionKernel::Layout input_layout(lda_input);

  ReductionKernel::TensorOutput output_tensor(output, output_layout);
  ReductionKernel::TensorInput input_tensor(input, input_layout);
  ReductionKernel::Arguments kernel_args(
      output_tensor,
      input_tensor,
      cutlass::MatrixCoord(static_cast<int>(rows), static_cast<int>(columns)),
      1 /*input_row_stride*/,
      static_cast<int>(batch_count),
      batch_stride_output,
      batch_stride_input
  );

  cutlass::Kernel<ReductionKernel><<<
      apply_grid,
      apply_block,
      sizeof(typename ReductionKernel::SharedStorage),
      stream
  >>>(kernel_args);

  LAUNCH_CHECK_REDUCE();
}
{% endfor %}

template<typename ElementOutput, typename ElementInput>
void reduce_mean_launcher_ColumnMajor_1(
  ElementOutput *output,
  ElementInput *input,
  int64_t batch_count,
  int64_t rows,
  int64_t columns,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {

  using ReductionKernel = ReductionKernelColumnMajor_1;

  dim3 apply_block(ReductionKernel::Shape::kColumn,
                   ReductionKernel::Shape::kRow);

  int cta_rows = ReductionKernel::Shape::kRow;
  int cta_columns = ReductionKernel::Shape::kColumn * ReductionKernel::kAlignment;

  dim3 apply_grid(static_cast<int>((rows + cta_rows - 1) / cta_rows),
                  static_cast<int>((columns + cta_columns - 1) / cta_columns),
                  static_cast<int>(batch_count));

  // column major
  int64_t lda_output = 1;
  int64_t lda_input = rows;
  ReductionKernel::Layout output_layout(lda_output);
  ReductionKernel::Layout input_layout(lda_input);

  ReductionKernel::TensorOutput output_tensor(output, output_layout);
  ReductionKernel::TensorInput input_tensor(input, input_layout);
  ReductionKernel::Arguments kernel_args(
      output_tensor,
      input_tensor,
      cutlass::MatrixCoord(static_cast<int>(rows), static_cast<int>(columns)),
      rows /*input_row_stride*/,
      static_cast<int>(batch_count),
      batch_stride_output,
      batch_stride_input
  );

  cutlass::Kernel<ReductionKernel><<<
      apply_grid,
      apply_block,
      sizeof(typename ReductionKernel::SharedStorage),
      stream
  >>>(kernel_args);

  LAUNCH_CHECK_REDUCE();
}

{{special_kernel}}

} // namespace

"""
)


SRC_TEMPLATE = jinja2.Template(
    """
{{kernel_source}}

static int normalize_axis(int axis, int rank) {
  if (axis >= 0) return axis;
  return rank + axis;
}

static int64_t get_size(const int64_t *input_shape, int from, int to) {
  int64_t sz = 1;
  for (int i = from; i < to; i++) {
    sz *= input_shape[i];
  }
  return sz;
}

static void normalize_input_shape(
  int64_t *new_input_shape,
  int *reduction_axis,
  const int64_t *input_shape,
  int *rank
) {
  if (*reduction_axis == 0 && *rank > 1) {
      new_input_shape[0] = 1;
      new_input_shape[1] = input_shape[0];
      new_input_shape[2] = get_size(input_shape, 1, *rank);
      *reduction_axis = 1;
      *rank = 3;
      return;
  }

  if (*rank <= 3) {
    for (int i = 0; i < *rank; i++) {
      new_input_shape[i] = input_shape[i];
    }
    return;
  }

  if (*reduction_axis == *rank - 1) {
    new_input_shape[0] = input_shape[0];
    new_input_shape[1] = get_size(input_shape, 1, *reduction_axis);
    new_input_shape[2] = input_shape[*reduction_axis];
    *reduction_axis = 2;
    *rank = 3;
    return;
  }

  new_input_shape[0] = get_size(input_shape, 0, *reduction_axis);
  new_input_shape[1] = input_shape[*reduction_axis];
  new_input_shape[2] = get_size(input_shape, *reduction_axis + 1, *rank);
  *reduction_axis = 1;
  *rank = 3;
}

void {{func_name}}(
  void *output,
  void *input,
  int reduction_axis,
  int64_t *output_shape[],
  const int64_t *orig_input_shape,
  int rank,
  bool keep_dim,
  cudaStream_t stream
) {

  reduction_axis = normalize_axis(reduction_axis, rank);
  if (reduction_axis >= rank) {
    throw std::runtime_error("reduction_axis must < rank");
  }
  if (reduction_axis < 0) {
    throw std::runtime_error("reduction_axis must >= 0");
  }
  if (rank == 0) {
    return;
  }

{% if not output_accessor.is_from_strided_tensor %}
  for (int i = 0, j = 0; i < rank; i++, j++) {
    if (i == reduction_axis) {
      if (keep_dim) {
        *(output_shape[j]) = orig_input_shape[j] == 0 ? 0 : 1;
      } else {
        j--;
      }
    } else {
      if (orig_input_shape[i] != *(output_shape[j])) {
        throw std::runtime_error("input/output dim values do not match");
      }
    }
  }
{% endif %}

  int64_t input_shape[3] = {1, 1, 1};
  normalize_input_shape(
      input_shape, &reduction_axis, orig_input_shape, &rank
  );

  for (int i = 0; i < rank; i++) {
    if (input_shape[i] == 0)
      return;
  }
  // make sure input and output are valid
  if (!output) {
    throw std::runtime_error("output is NULL!");
  }
  if (!input) {
    throw std::runtime_error("input is NULL!");
  }

  int64_t b = 1;
  int64_t m = 1;
  int64_t n = 1;
  int64_t batch_stride_input = 0;
  int64_t batch_stride_output = 0;

  {{exec_paths}}

  throw std::runtime_error(
    "unsupported reduction_axis value for {{func_name}}"
  );
}
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  const int64_t {{input_name}}_shape[] = {
{{indent}}    {{input_dims}}
{{indent}}  };

{{indent}}  int64_t *{{output_name}}_shape[] = {
{{indent}}    {{output_dim_refs}}
{{indent}}  };

{{indent}}  {{func_name}}(
{{indent}}      {{output_name}},
{{indent}}      {{input_name}},
{{indent}}      {{reduction_axis}}, /*reduction_axis*/
{{indent}}      {{output_name}}_shape,
{{indent}}      {{input_name}}_shape,
{{indent}}      {{rank}}, /*rank*/
{{indent}}      {{keep_dim}}, /*keep_dim*/
{{indent}}      stream
{{indent}}  );
{{indent}}}
"""
)


def gen_function_decl(func_attrs) -> str:
    """a common function for generating the function declaration of a
    reduce-family kernel

    Parameters
    ----------
    func_attrs : Dit[str, Any]
        holds attributes of this reduce op

    Returns
    -------
    str
        returns the rendered function declaration with appropriate replacements
    """
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


def gen_function(
    func_attrs,
    reduce_op,
    prologue_template=DEFAULT_PROLOGUE_TEMPLATE,
    epilogue_scalar_template=DEFAULT_EPILOGUE_SCALAR_TEMPLATE,
    extra_code_str="",
    accumulation_type=None,
) -> str:
    """a common function for generating a reduce-family kernel

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce op
    reduce_op : str
        the reduce op's string representation such as cutlass::plus
    prologue_template : str, optional
        a Template that will be rendered to process input before reduction
    epilogue_scalar_template : str, optional
        a Template that will be rendered to process each final reduced element
    epilogue_scalar_template : str, optional
        a Template that will be rendered to hold extra code for reduction
    accmulation_type : str, optional
        specifies the data type for accumulation
        (default is None so that we will use output's type for accumulation)

    Returns
    -------
    str
        returns the rendered code for the complete implementation of the reduce op
    """
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    backend_spec = CUDASpec()
    input_type = backend_spec.dtype_to_lib_type(x._attrs["dtype"])
    output_type = backend_spec.dtype_to_lib_type(y._attrs["dtype"])
    if accumulation_type is None:
        # follow pytorch's semantics
        acc_type = output_type
    else:
        acc_type = accumulation_type

    # FIXME: Our kernel launch configs are determined by the amount of shared
    # memory assigned to each block. Currently, we set up an approximate size for
    # shared memroy based on the reduction dim value. We should consider to
    # determine this value by profiling later.
    axes = func_attrs["reduction_axes"]
    if not len(axes) == 1:
        raise NotImplementedError("Multiple reduction axes are not supported yet")
    reduction_axis = axes[0]
    x_shape = x._attrs["shape"]
    reduction_dim_val = max(x_shape[reduction_axis]._attrs["values"])
    col_size_buckets = [8, 16, 32, 64, 128]
    # If reduction axis is not the last dim, let's increase the shared memory a bit.
    # Note that this heuristic is observed with some test inputs, and we need a
    # more systematic approach to determining the size of shared memory.
    if reduction_axis < len(x_shape) - 1:
        col_size_buckets.extend([256, 512])
    col_pos = bisect.bisect_right(col_size_buckets, reduction_dim_val - 1)
    col_pos = min(col_pos, len(col_size_buckets) - 1)
    shared_col_size = col_size_buckets[col_pos]

    row_layout = "RowMajor"
    col_layout = "ColumnMajor"

    # FIXME: these alignments values are only for half_t type.
    # make it adjustable to other types such as float.
    alignments = [8, 4, 2, 1]
    if x._attrs["dtype"] in ("float16", "bfloat16"):
        alignments.append(16)
    # This is ugly. Ideally, we should have templated code like below:
    # template <typename Alignment>
    # reduce_launcher(...) {
    #   using ReductionKernel = ReductionKernel3D<..., 4, ...>
    #   ...
    #   typename ReductionKernel::Layout output_layout(lda_output);
    #   ...
    # }
    #
    # However, this dependent template pattern caused cicc (at least 11.4) to
    # segfault. To workaround this cicc issue, we manually "instantiate" template.
    reduce_instances = [
        REDUCE_KERNEL_INSTANCE.render(
            indent="  ",
            elem_output_type=output_type,
            elem_input_type=input_type,
            elem_compute_type=acc_type,
            align=align,
            shared_col_size=shared_col_size,
            layout=row_layout,
        )
        for align in alignments
    ]
    reduce_instances.append(
        REDUCE_KERNEL_INSTANCE.render(
            indent="  ",
            elem_output_type=output_type,
            elem_input_type=input_type,
            elem_compute_type=acc_type,
            align=1,
            shared_col_size=shared_col_size,
            layout=col_layout,
        )
    )
    reduce_instance = "\n".join(reduce_instances)

    prologue_code = prologue_template.render(indent=" " * 8)
    epilogue_scalar_code = epilogue_scalar_template.render(indent=" " * 12)

    output_accessors = func_attrs["output_accessors"]
    assert (
        len(output_accessors) == 1
    ), f"expected the length of output_accessors to be one but got {len(output_accessors)}"
    dtype = func_attrs["inputs"][0].dtype()
    output_alignment = tensor_accessor_codegen.find_max_alignment_for_accessors(
        dtype, output_accessors
    )
    special_exec_path, special_kernel = reduce_small_axis.get_exec_cond_and_kernel(
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
    )
    exec_paths = EXEC_COND_TEMPLATE.render(
        indent="  ",
        func_name=func_attrs["name"],
        elem_output_type=output_type,
        elem_input_type=input_type,
        elem_compute_type=acc_type,
        alignments=alignments,
        special_exec_cond=special_exec_path,
    )

    strided_address_func_str = (
        tensor_accessor_codegen.STRIDED_ADDRESS_AT_IDX_FUNC_TEMPLATE.render(
            output_accessor=output_accessors[0],
        )
    )
    tensor_accessor_libs = tensor_accessor_codegen.get_libs()
    extra_code_str += "\n\n" + tensor_accessor_libs
    extra_code_str += "\n\nnamespace {\n" + strided_address_func_str + "\n}\n\n"

    kernel_src = KERNEL_SRC_TEMPLATE.render(
        extra_code=extra_code_str,
        reduce_op=reduce_op,
        reduce_kernel_instance=reduce_instance,
        alignments=alignments,
        prologue_code=prologue_code,
        epilogue_scalar_code=epilogue_scalar_code,
        special_kernel=special_kernel,
    )

    return SRC_TEMPLATE.render(
        func_name=func_attrs["name"],
        kernel_source=kernel_src,
        exec_paths=exec_paths,
        output_accessor=output_accessors[0],
    )


def gen_function_call(func_attrs, indent="  ") -> str:
    """a common function for generating a call to a reduce-family function

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce op
    indent : str, optional
        indent for each line of the function call code (default "  ")

    Returns
    -------
    str
        returns rendered code for invoking the reduce op
    """
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    axes = func_attrs["reduction_axes"]
    if not len(axes) == 1:
        raise NotImplementedError("Multiple reduction axes are not supported yet")

    x_shape = x._attrs["shape"]
    x_dims = ", ".join([dim._attrs["name"] for dim in x_shape])
    y_shape = func_attrs["output_accessors"][0].original_shapes
    y_dim_refs = ", ".join(["&" + dim._attrs["name"] for dim in y_shape])
    keep_dim = "true" if func_attrs["keepdim"] else "false"

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        output_name=y._attrs["name"],
        input_name=x._attrs["name"],
        input_dims=x_dims,
        output_dim_refs=y_dim_refs,
        reduction_axis=axes[0],
        rank=str(len(x_shape)),
        keep_dim=keep_dim,
    )
