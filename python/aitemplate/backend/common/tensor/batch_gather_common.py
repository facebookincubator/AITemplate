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
batch_gather kernel codegen.
"""

from typing import Any, Dict

import jinja2

# pylint: disable=C0301

FUNC_CALL_INT64_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int64_t*>({{name}})")

FUNC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

{{kernel}}

}  // namespace

{{func_signature}}
{
    const int64_t gather_size = (gather_dim != 0) ? (*batch_size * batch_num) : batch_num;
    batch_gather_launcher<{{dtype}}, int64_t>(stream, gather_size, indices_num, instance_size, gather_dim_size, static_cast<const {{dtype}}*>(input), indices, workspace, static_cast<{{dtype}}*>(output));
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   const void* input,
                   const int64_t* indices,
                   const {{index_type}}* batch_size,
                   const {{index_type}} batch_num,
                   const {{index_type}} indices_num,
                   const {{index_type}} instance_size,
                   const {{index_type}} gather_dim,
                   const {{index_type}} gather_dim_size,
                   uint8_t* workspace,
                   {{prefix}}Stream_t stream)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{output}}, {{input}}, {{indices}},
{{indent}}    {{batch_size}},
{{indent}}    {{batch_num}},
{{indent}}    {{indices_num}},
{{indent}}    {{instance_size}},
{{indent}}    {{gather_dim}},
{{indent}}    {{gather_dim_size}},
{{indent}}    global_workspace_, stream /* default stream */
{{indent}});
    """
)

KERNEL_TEMPLATE = jinja2.Template(
    """
const int64_t kThreadsNumPerBlock = 256;
const int64_t kMaxBlocksNum = 8192;

#define GPU_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename K>
__device__ int64_t GetInOffset(
    const int64_t out_offset,
    const K* indices,
    const int64_t indices_num,
    const int64_t instance_size,
    const int64_t gather_dim_size) {
  const int64_t batch_idx = out_offset / (indices_num * instance_size);
  const int64_t indices_idx =
      out_offset % (indices_num * instance_size) / instance_size;
  const int64_t inner_idx = out_offset % instance_size;
  const int64_t idx = indices[batch_idx * indices_num + indices_idx];
  assert(idx >= 0 && idx < gather_dim_size);
  return batch_idx * gather_dim_size * instance_size + idx * instance_size +
      inner_idx;
}

template <typename T, typename K>
__global__ void BatchGatherGpu(
    const int64_t elem_cnt,
    const T* in,
    const K* indices,
    const int64_t indices_num,
    const int64_t instance_size,
    const int64_t gather_dim_size,
    T* out) {
  GPU_KERNEL_LOOP(i, elem_cnt) {
    out[i] = in[GetInOffset<K>(
        i, indices, indices_num, instance_size, gather_dim_size)];
  }
}

inline int64_t BlocksNum4ThreadsNum(const int64_t n) {
  return std::min(
      (n + kThreadsNumPerBlock - 1) / kThreadsNumPerBlock,
      kMaxBlocksNum);
}
template <typename T, typename K>
void batch_gather_launcher(
    {{prefix}}Stream_t stream,
    const {{index_type}} batch_num,
    const {{index_type}} indices_num,
    const {{index_type}} instance_size,
    const {{index_type}} gather_dim_size,
    const T* input,
    const K* indices,
    void* workspace,
    T* output) {
  const int64_t elem_cnt = batch_num * indices_num * instance_size;
  BatchGatherGpu<T, K>
      <<<BlocksNum4ThreadsNum(elem_cnt), kThreadsNumPerBlock, 0, stream>>>(
          elem_cnt,
          input,
          indices,
          indices_num,
          instance_size,
          gather_dim_size,
          output);
}
    """
)


def gen_function_call(func_attrs: Dict[str, Any], indent="  ", is_cuda=False) -> str:
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 2

    output_name = func_attrs["outputs"][0]._attrs["name"]

    input_name = func_attrs["inputs"][0]._attrs["name"]

    indices_name = FUNC_CALL_INT64_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][1]._attrs["name"]
    )

    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    indices = func_attrs["inputs"][1]
    ind_shape = indices._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]

    axis = len(ind_shape) - 1
    batch_num = 1
    for i in range(1, axis):
        batch_num *= yshape[i]._attrs["values"][0]

    indices_num = yshape[axis]._attrs["values"][0]

    instance_size = 1
    for i in range(axis + 1, len(yshape)):
        instance_size *= yshape[i]._attrs["values"][0]

    gather_dim_size = xshape[axis]._attrs["values"][0]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        input=input_name,
        indices=indices_name,
        batch_size="&" + xshape[0]._attrs["name"],
        batch_num=batch_num,
        indices_num=indices_num,
        instance_size=instance_size,
        gather_dim=axis,
        gather_dim_size=gather_dim_size,
        indent=indent,
    )


def gen_function(func_attrs: Dict[str, Any], header_files: str, backend_spec) -> str:
    index_type = backend_spec.index_type
    prefix = backend_spec.prefix
    return FUNC_TEMPLATE.render(
        header_files=header_files,
        kernel=KERNEL_TEMPLATE.render(index_type=index_type, prefix=prefix),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], index_type=index_type, prefix=prefix
        ),
        dtype=backend_spec.dtype_to_backend_dtype[
            func_attrs["inputs"][0]._attrs["dtype"]
        ],
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=backend_spec.index_type,
            prefix=backend_spec.prefix,
        ).strip()
    )
