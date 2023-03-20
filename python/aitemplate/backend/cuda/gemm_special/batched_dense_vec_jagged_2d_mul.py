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
Define batched_dense_vec_jagged_2d_mul codegen and CUDA kernel
"""
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common.elementwise_common import gen_offsets_str
from aitemplate.backend.target import Target


CONSTANT_TEMPLATE = jinja2.Template(
    """
#define WARP_SIZE 32
#define MAX_THREADS 1024
    """
)

KERNEL_TEMPLATE = jinja2.Template(
    """
__global__ void {{func_name}}(
    {{data_t}}* output,
    const {{data_t}}* vectors,
    const {{data_t}}* matrices,
    {{offsets}}
    {{index_type}} b, {{index_type}} h,
    {{index_type}} n, {{index_type}} d
) {
  const int b_h_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int b_h_step = gridDim.x * blockDim.y;
  for (int b_h = b_h_begin; b_h < b * h; b_h += b_h_step) {
    const int b_idx = b_h / h;
    const int h_idx = b_h % h;

    const {{index_type}} row_start = offsets.data[0][b_idx];
    const {{index_type}} row_end = offsets.data[0][b_idx + 1];
    const {{index_type}} length = min(row_end - row_start, n);
    if (length == 0) {
      for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
        output[b_h * d + d_idx] = 0;
      }
    } else {
      for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
        {{acc_t}} acc =
            {{acc_t}}(vectors[b_h * n] * matrices[row_start * h * d + h_idx * d + d_idx]);
        for (int l = 1; l < length; ++l) {
          acc += {{acc_t}}(vectors[b_h * n + l] * matrices[(row_start + l) * h * d + h_idx * d + d_idx]);
        }
        output[b_h * d + d_idx] = {{data_t}}(acc);
      }
    }
  }
}
    """
)

FUNC_TEMPLATE = jinja2.Template(
    """
{{head}}

#include "jagged.h"

namespace {

{{constant}}

{{kernel_function}}

}  // namespace

void invoke_{{func_name}}(void* output, const void* vectors, const void* matrices, {{index_type}} b, {{index_type}} h, {{index_type}} n, {{index_type}} d, {{offsets_decl}} {{prefix}}Stream_t stream) {
    if (b == 0 || d == 0) {
      return;
    }
    int block_dim_x = std::min(static_cast<int>(std::ceil(static_cast<double>(d) / WARP_SIZE) * WARP_SIZE), MAX_THREADS);
    int block_dim_y = MAX_THREADS / block_dim_x;
    int block_size = static_cast<int>(std::ceil(static_cast<double>(b * h) / block_dim_y));
    {{func_name}}<<<block_size, dim3(block_dim_x, block_dim_y), 0, stream>>>(
        reinterpret_cast<{{data_t}}*>(output),
        reinterpret_cast<const {{data_t}}*>(vectors),
        reinterpret_cast<const {{data_t}}*>(matrices),
        {{offsets_call}}
        b,
        h,
        n,
        d
    );
}
    """
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void invoke_{{func_name}}(void* output, const void* vectors, const void* matrices, {{index_type}} b, {{index_type}} h, {{index_type}} n, {{index_type}} d, {{offsets}} {{prefix}}Stream_t stream);
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
    {{indent}}invoke_{{func_name}}({{output}}, {{vectors}}, {{matrices}}, {{b}}, {{h}}, {{n}}, {{d}}, {{offsets}} {{stream}});
{{indent}}}
    """
)


def _gen_kernel_function(
    func_attrs: Dict[str, Any],
    index_type: str,
    data_type: str,
) -> str:
    matrices = func_attrs["inputs"][1]

    acc_t = "float"
    if (
        data_type in ["half", "bfloat16"]
        and "use_fp16_acc" in Target.current()._kwargs
        and Target.current()._kwargs["use_fp16_acc"]
    ):
        acc_t = data_type

    kernel_func = KERNEL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=index_type,
        data_t=data_type,
        offsets=gen_offsets_str(
            matrices._attrs["shape"][0],
            has_type=True,
            # the offsets are passed
            # by value to the kernel
            const_ref=False,
            name="offsets",
        ),
        acc_t=acc_t,
    )
    return kernel_func


@registry.reg("cuda.batched_dense_vec_jagged_2d_mul.gen_function")
def jagged_to_dense_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generates jagged_to_dense function definition."""

    vectors = func_attrs["inputs"][0]
    matrices = func_attrs["inputs"][1]
    backend_spec = CUDASpec()

    dtype = vectors.dtype()
    data_type = backend_spec.dtype_to_backend_type(dtype)

    kernel_function = _gen_kernel_function(
        func_attrs,
        backend_spec.index_type,
        data_type,
    )

    constant = CONSTANT_TEMPLATE.render()

    function = FUNC_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        head=backend_spec.header_src_template.render(),
        constant=constant,
        kernel_function=kernel_function,
        func_name=func_attrs["name"],
        offsets_decl=gen_offsets_str(
            matrices._attrs["shape"][0],
            has_type=True,
            # the offsets are passed
            # by const reference to the function
            const_ref=True,
            name="offsets",
        ),
        offsets_call=gen_offsets_str(
            matrices._attrs["shape"][0],
            has_type=False,
            const_ref=False,
            name="offsets",
        ),
        data_t=data_type,
    )
    return function


@registry.reg("cuda.batched_dense_vec_jagged_2d_mul.func_decl")
def jagged_to_dense_gen_function_decl(func_attrs) -> str:
    """Generate jagged_to_dense function declaration."""

    matrices = func_attrs["inputs"][1]
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()

    return FUNC_DECL_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        func_name=func_name,
        offsets=gen_offsets_str(
            matrices._attrs["shape"][0],
            has_type=True,
            const_ref=True,
            name="offsets",
        ),
    )


@registry.reg("cuda.batched_dense_vec_jagged_2d_mul.func_call")
def jagged_to_dense_gen_function_call(
    func_attrs,
    indent: str,
) -> str:
    """Generate jagged_to_dense function call."""

    vectors = func_attrs["inputs"][0]
    vshape = vectors._attrs["shape"]
    matrices = func_attrs["inputs"][1]
    jshape = matrices._attrs["shape"]
    output = func_attrs["outputs"][0]
    backend_spec = CUDASpec()

    return FUNC_CALL_TEMPLATE.render(
        stream=backend_spec.stream,
        func_name=func_attrs["name"],
        matrices=matrices._attrs["name"],
        vectors=vectors._attrs["name"],
        b=vshape[0]._attrs["name"],
        h=vshape[1]._attrs["name"],
        n=vshape[2]._attrs["name"],
        d=jshape[2]._attrs["name"],
        output=output._attrs["name"],
        offsets=gen_offsets_str(
            matrices._attrs["shape"][0],
            has_type=False,
            const_ref=False,
        ),
        indent=indent,
    )
