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
A simply reduction implementation for "slim" tensors. Currently only reduction
along dim=1 for 3D tensors is supported.
"""


from typing import List

import jinja2

from aitemplate.compiler.base import IntVar
from aitemplate.utils.shape_utils import is_static_dimension

# These are supported by cutlass::NumericConverter.
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

bytesize = {
    "half": 2,
    "cutlass::half_t": 2,
    "bfloat16": 2,
    "cutlass::bfloat16_t": 2,
    "float": 4,
}


EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if (shape[rank - 1] % {{vlen}} == 0) {
{{indent}}  {{func_name}}_launcher<{{elem_output_type}}, {{elem_input_type}}, {{elem_compute_type}}, {{vector_type}}>
{{indent}}    (dst_ptr, src_ptr, reduction_axis, shape, rank, stream);
{{indent}}  return;
{{indent}}}
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include "cutlass/arch/memory.h"

#define SKIP_GENERAL_REDUCTION

template <typename ElemOutputType, typename ElemInputType, typename ElementCompute, typename VecType>
__global__ void {{func_name}}_kernel(
    ElemOutputType* output,
    ElemInputType* input,
    const int nbatches,
    const int nrows,
    const int ncols,
    ElementCompute reductionIdentity) {
  constexpr int32_t elemsPerThread = sizeof(VecType) / sizeof(ElemInputType);
  int32_t batch = blockIdx.x;
  int32_t batchOffset = batch * nrows * ncols;
  int32_t colOffset = threadIdx.x * elemsPerThread;

  VecType vecIn[1];
  ElementCompute fragReduce[elemsPerThread];
  cutlass::NumericConverter<ElementCompute, ElemInputType> toComputeType;
  using ReductionOp = {{reduction_op}}<ElementCompute>;
  ReductionOp reduce;
#pragma unroll
  for (int32_t r = 0; r < nrows; r++) {
    // Vectorized load and reduce.
    int32_t rowOffset = r * ncols;
    int readIdx = batchOffset + rowOffset + colOffset;

    cutlass::arch::global_load<VecType, sizeof(VecType)>(*vecIn, input + readIdx, true);
    ElemInputType* in = reinterpret_cast<ElemInputType*>(vecIn);

    #pragma unroll
    for (int32_t i = 0; i < elemsPerThread; i++) {
        if (r == 0) fragReduce[i] = reductionIdentity;
        fragReduce[i] = reduce(fragReduce[i], toComputeType(in[i]));
    }
  }

  // Finished reduction now convert back to output type.
  alignas(sizeof(VecType)) ElemOutputType reduced[elemsPerThread];
  cutlass::NumericConverter<ElemOutputType, ElementCompute> toOutputType;
  for (int32_t i = 0; i < elemsPerThread; i++) {
    reduced[i] = toOutputType(fragReduce[i]);
  }

  // Vectorized stores.
  int writeIdx = (batch * ncols) + colOffset;
  VecType* vecOut = reinterpret_cast<VecType*>(&output[writeIdx]);
  *vecOut = *reinterpret_cast<VecType*>(reduced);  // vectorized store
}

template <typename ElemOutputType, typename ElemInputType, typename ElementCompute, typename VecType>
void {{func_name}}_launcher(
    void* dst_ptr,
    void* src_ptr,
    int reduction_axis,
    const int64_t* shape,
    const int rank,
    cudaStream_t stream) {
    static_assert(sizeof(ElemOutputType) == sizeof(ElemInputType));
    int nbatches = shape[rank - 3];
    int nrows = shape[rank - 2];
    int ncols = shape[rank - 1];
    int elemsPerThread = sizeof(VecType) / sizeof(ElemInputType);
    int nthreads = ncols / elemsPerThread;

    {{func_name}}_kernel<ElemOutputType, ElemInputType, ElementCompute, VecType>
        <<<nbatches, nthreads, 0, stream>>>(
            static_cast<ElemOutputType*>(dst_ptr),
            static_cast<ElemInputType*>(src_ptr),
            nbatches,
            nrows,
            ncols,
            {{reduction_identity}});
}
"""
)


def meets_special_kernel_conditions(func_attrs, input_type, output_type) -> bool:
    """
    The reduction op must meet all conditions to use the special kernel:
    1. The input and output types are the same.
    2. The input and output memory are contiguous.
    3. The tensor is rank-3 and the reduction_dim=1.
    4. The reduction dimension length is less than 10 and the # of threads needed
    to do vectorize loads/stores is less than 1024.
    """
    if input_type != output_type:
        return False

    input_accessors = func_attrs["output_accessors"]
    output_accessors = func_attrs["output_accessors"]
    if not (
        output_accessors
        and input_accessors
        and output_accessors[0].is_contiguous
        and input_accessors[0].is_contiguous
    ):
        return False

    input_shape: List[IntVar] = func_attrs["inputs"][0].shape()
    reduction_axes = func_attrs["reduction_axes"]
    reduction_axis = reduction_axes[0]
    if not (
        len(input_shape) == 3  # This kernel only supports rank 3 tensors.
        and len(reduction_axes) == 1
        and reduction_axis == 1
        and is_static_dimension(input_shape, reduction_axis)
        and is_static_dimension(input_shape, reduction_axis + 1)
    ):
        return False

    def _get_largest_aligned_vector(input_type, nelems) -> (str, int):
        for vtype, vbytesize in vector_types[input_type]:
            vcapacity = int(vbytesize / bytesize[input_type])
            if nelems % vcapacity == 0:
                return vtype, vcapacity

    MAX_BLOCK_DIM = 1024
    reduction_axis_len = input_shape[reduction_axis].value()
    cross_axis_len = input_shape[reduction_axis + 1].value()
    _, vector_capacity = _get_largest_aligned_vector(input_type, nelems=cross_axis_len)
    if not (
        reduction_axis_len <= 10 and cross_axis_len / vector_capacity <= MAX_BLOCK_DIM
    ):
        return False

    return True


def get_special_exec_cond_and_kernel(
    func_attrs,
    input_type,
    output_type,
    acc_type,
    output_accessors,
    reduction_op,
    reduction_identity,
) -> (str, str):
    """
    In the current implementation, each thread performs one vector load per row,
    runs the reduction, then with the results does one vector store.

    Returns
    ---
    exec_cond : str
        This should replace the exec conditions for original kernel.

    special_reduction_code : str
        Includes the kernel code and the launcher for the host.
    """
    exec_conds = []

    for vector_type, vec_bytesize in vector_types[input_type]:
        vlen = int(vec_bytesize / bytesize[input_type])
        exec_cond = EXEC_COND_TEMPLATE.render(
            indent="  ",
            func_name=func_attrs["name"],
            elem_input_type=input_type,
            elem_output_type=output_type,
            elem_compute_type=acc_type,
            vector_type=vector_type,
            vlen=vlen,
        )
        exec_conds.append(exec_cond)

    special_reduction_code = SRC_TEMPLATE.render(
        func_name=func_attrs["name"],
        reduction_op=reduction_op,
        reduction_identity=reduction_identity,
    )
    return "".join(exec_conds), special_reduction_code
