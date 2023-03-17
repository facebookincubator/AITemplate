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

import os

import jinja2

from aitemplate.backend.backend_spec import CUDASpec

from aitemplate.backend.common import tensor_accessor_codegen
from aitemplate.backend.target import Target

from aitemplate.compiler.ops.tensor import concatenate


KERNEL_SRC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "logging.h"

{{header_src}}

{% if element_func_def %}
{{element_func_def}}
{% endif %}

namespace {

{{tensor_accessor_libs}}

// TODO: support strided tensor with TensorAccessor
// For strided tensor, the index can be much larger than original if the stride is large
bool can_use_32bit_index_math(const int64_t elements, int64_t max_elem=std::numeric_limits<int32_t>::max()) {
  if (elements >= max_elem) {
    return false;
  }
  if (elements == 0) {
    return max_elem > 0;
  }

  return true;
}

__host__ __device__ __forceinline__
int64_t get_num_elems(const {{index_type}} *shape, {{index_type}} rank) {
  int64_t num = 1;
  for ({{index_type}} i = 0; i < rank; i++) {
    num *= shape[i];
  }
  return num;
}

{{custom_libs}}

}  // namespace

"""
)


EXEC_COND_TEMPLATE = jinja2.Template(
    """

{{input_accessor_defs}}

{{indent}}{{index_type}} local_output_shape[] = {
{% for idx in range(rank - 1) %}
{{indent}}  *(output_shape[{{idx}}]),
{% endfor %}
{{indent}}  *(output_shape[{{rank - 1}}])
{{indent}}};
{{indent}}
{{indent}}{% if element_func == "fast_tanh" %}
{{indent}}using transform_type = TanhTransform<{{elem_type}}>;
{{indent}}{% else %}
{{indent}}using transform_type = NoopTransform<{{elem_type}}>;
{{indent}}{% endif %}
{{indent}}
{{indent}}invoke_concatenate_fast<{{elem_type}}, {{elem_type}}, {{num_all_inputs}}, {{rank}}, transform_type>(
{{indent}}    real_input_shapes,
{{indent}}    inputs,
{{indent}}    input_accessors,
{{indent}}    local_output_shape,
{{indent}}    concat_dim_offsets.data(),
{{indent}}    output,
{{indent}}    concat_dim,
{{indent}}    "{{func_name}}",
{{indent}}    stream);
{{indent}}return;
"""
)

INPUT_ACCESSOR_DEFS_TEMPLATE = jinja2.Template(
    """
{{input_accessors}}

{{indent}}const TensorAccessor *input_accessors[{{num_real_inputs}}] = {

{{indent}}  {{input_accessor_refs}}

{{indent}}};
"""
)


def gen_function(
    func_attrs,
    src_template,
    element_func=None,
    element_func_def=None,
):
    backend_spec = CUDASpec()

    inputs = func_attrs["inputs"]
    original_inputs = func_attrs["original_inputs"]
    concatenate.check_rank(original_inputs, func_attrs["concat_dim"])
    orig_x = original_inputs[0]
    y = func_attrs["outputs"][0]
    x_shape = orig_x._attrs["shape"]

    input_type = backend_spec.dtype_to_backend_type(orig_x._attrs["dtype"])
    output_type = backend_spec.dtype_to_backend_type(y._attrs["dtype"])

    # TODO: support type cast
    if input_type != output_type:
        raise NotImplementedError("input type must equal to output type")

    concat_dim = func_attrs["concat_dim"]
    assert concat_dim < len(x_shape)

    input_accessors = []
    input_accessor_refs = []
    for i in range(len(inputs)):
        accessor_name = f"input_accessor{i}"
        input_accessor_refs.append(f"&{accessor_name}")
        input_accessors.append(
            tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
                name=accessor_name, tensor_accessor=func_attrs["input_accessors"][i]
            )
        )
    input_accessor_defs = INPUT_ACCESSOR_DEFS_TEMPLATE.render(
        indent="    ",
        input_accessors="".join(input_accessors),
        num_real_inputs=len(inputs),
        input_accessor_refs=", ".join(input_accessor_refs),
    )

    # load the file from the drive
    custom_libs = Target.current().get_custom_libs(
        os.path.dirname(__file__), "concatenate_fast.cuh"
    )

    header_src = backend_spec.header_src_template.render()
    tensor_accessor_libs = tensor_accessor_codegen.get_libs()
    kernel_src = KERNEL_SRC_TEMPLATE.render(
        custom_libs=custom_libs,
        element_func=element_func,
        element_func_def=element_func_def,
        header_src=header_src,
        index_type=backend_spec.index_type,
        tensor_accessor_libs=tensor_accessor_libs,
    )
    exec_paths = EXEC_COND_TEMPLATE.render(
        indent="  ",
        rank=len(x_shape),
        num_all_inputs=len(inputs),
        elem_type=input_type,
        element_func=element_func,
        element_func_def=element_func_def,
        index_type=backend_spec.index_type,
        input_accessor_defs=input_accessor_defs,
        func_name=func_attrs["name"],
    )

    return src_template.render(
        kernel_src=kernel_src,
        func_name=func_attrs["name"],
        exec_paths=exec_paths,
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
    )
