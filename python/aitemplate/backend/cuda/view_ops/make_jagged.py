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
Codegen functions for the make_jagged op.

The main responsibilities of the make_jagged backend are:

  1. Associate the offsets structure members (lengths and data)
  with the corresponding rank-1 offsets Tensors' first dimension
  and data pointer, respectively.

  2. Check the validity of the offset content (non-strict
  monotonicity, first and last values in each array). Offset
  contents are on the device, hence are checked by a simple
  CUDA kernel doing an assertion for each constraint. Some
  of the constraints can be checked on the device, in which
  case an std::runtime_error is thrown on violation.
"""
from typing import Set

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.compiler.base import IntImm, IntVar, JaggedIntVar


SRC_TEMPLATE = jinja2.Template(
    """
#include <assert.h>
#include <stdexcept>

#include "jagged.h"


#define THREADS_PER_BLOCK 128


namespace {

struct OffsetBounds {
  {{offsets_type}} min_values[{{num_offsets}}]{0};
  {{offsets_type}} max_values[{{num_offsets}}]{0};
  {{offsets_type}} last_values[{{num_offsets}}]{0};
};

__global__ void check_offsets(
  {{offsets_struct_type}} offsets,
  OffsetBounds bounds
) {
  {{index_type}} dim_id = blockIdx.y;
  {{index_type}} offset_id = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  {{index_type}} length = offsets.lengths[dim_id];
  const {{offsets_type}}* data = offsets.data[dim_id];

  if (offset_id >= length - 1) {
    // out of bounds of the offset array
    return;
  }

{% if check_sequence_lengths %}
  {{offsets_type}} group_size = data[offset_id + 1] - data[offset_id];
  if (group_size < bounds.min_values[dim_id] || group_size > bounds.max_values[dim_id]) {
    printf(
      "\\n[func name: {{func_name}}, block: [%d, %d, %d], thread: [%d, %d, %d]]: "
      "Error: the offset difference %d is out of bounds of the jagged dimension %d (min: %d, max: %d).",
      (int32_t)blockIdx.x,
      (int32_t)blockIdx.y,
      (int32_t)blockIdx.z,
      (int32_t)threadIdx.x,
      (int32_t)threadIdx.y,
      (int32_t)threadIdx.z,
      (int32_t)group_size,
      (int32_t)dim_id,
      (int32_t)bounds.min_values[dim_id],
      (int32_t)bounds.max_values[dim_id]
    );
    __trap();
  }
{% endif %}

  if (offset_id == 0) {
    {{offsets_type}} first_offset = data[0];
    if (first_offset != 0)
    {
      printf(
      "\\n[func name: {{func_name}}, block: [%d, %d, %d], thread: [%d, %d, %d]]: "
        "Error: the first offset of the jagged dimension %d is non-zero: %d.",
        (int32_t)blockIdx.x,
        (int32_t)blockIdx.y,
        (int32_t)blockIdx.z,
        (int32_t)threadIdx.x,
        (int32_t)threadIdx.y,
        (int32_t)threadIdx.z,
        (int32_t)dim_id,
        (int32_t)first_offset
      );
      __trap();
    }
  }

  if (offset_id == length - 2) {
    {{offsets_type}} last_offset = data[length - 1];
    if (last_offset != bounds.last_values[dim_id])
    {
      printf(
      "\\n[func name: {{func_name}}, block: [%d, %d, %d], thread: [%d, %d, %d]]: "
        "Error: the last offset of the jagged dimension %d is incorrect: %d (must be %d).",
        (int32_t)blockIdx.x,
        (int32_t)blockIdx.y,
        (int32_t)blockIdx.z,
        (int32_t)threadIdx.x,
        (int32_t)threadIdx.y,
        (int32_t)threadIdx.z,
        (int32_t)dim_id,
        (int32_t)last_offset,
        (int32_t)bounds.last_values[dim_id]
      );
      __trap();
    }
  }
}

} // namespace


void {{func_name}}(
{% for idx in range(num_offsets) %}
  {{index_type}} offsets_length_{{idx}},
  const void* offsets_data_{{idx}},
{% endfor %}
{% for name in jagged_dynamic_bound_names %}
  {{index_type}} {{name}},
{% endfor %}
  {{offsets_struct_type}}& offsets,
  {{index_type}}* batch_dim,
  {{index_type}} total_length,
  cudaStream_t stream
) {
{% for idx in range(num_offsets) %}
    offsets.lengths[{{idx}}] = offsets_length_{{idx}};
    offsets.data[{{idx}}] = reinterpret_cast<const {{offsets_type}}*>(offsets_data_{{idx}});
{% endfor %}

{% if isolated_batch_dim %}
    // batch_dim is not present in any input shape
    // we should set it here from the offsets length
    *batch_dim = offsets.lengths[0] - 1;
{% else %}
    if (*batch_dim != offsets.lengths[0] - 1) {
        // batch_dim must have been set before this code
        throw std::runtime_error("batch_dim != len(offsets[0]) - 1");
    }
{% endif %}

    {{index_type}} max_offset_length = 0;
    for (int i = 0; i < {{num_offsets}}; ++i) {
        if (offsets.lengths[i] <= 1) {
            throw std::runtime_error("offset array's length must be at least 2");
        }
        if (offsets.lengths[i] > max_offset_length) {
            max_offset_length = offsets.lengths[i];
        }
    }

    OffsetBounds bounds;
{% for idx in range(num_offsets) %}
    bounds.min_values[{{idx}}] = {{jagged_dim_min_values[idx]}};
    bounds.max_values[{{idx}}] = {{jagged_dim_max_values[idx]}};
    bounds.last_values[{{idx}}] = {{ "offsets.lengths[" + ((idx + 1) | string) + "] - 1" if idx < num_offsets - 1 else "total_length" }};
{% endfor %}

    dim3 grid_size((max_offset_length - 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, {{num_offsets}});
    check_offsets<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(offsets, bounds);
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
{% for idx in range(num_offsets) %}
  {{index_type}},
  const void*,
{% endfor %}
{% for _ in range(num_jagged_dynamic_bound_dims) %}
  {{index_type}},
{% endfor %}
  {{offsets_struct_type}}&,
  {{index_type}}*,
  {{index_type}},
  cudaStream_t
);
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{% for idx in range(num_offsets) %}
{{indent}}  {{offsets_first_dim_names[idx]}},
{{indent}}  {{offsets_data_names[idx]}},
{% endfor %}
{% for name in jagged_dynamic_bound_names %}
{{indent}}  {{name}},
{% endfor %}
{{indent}}  {{offsets_var_name}},
{{indent}}  &{{batch_dim_name}},
{{indent}}  {{total_length_name}},
{{indent}}  stream
{{indent}});
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


def _get_jagged_dynamic_bound_dims(jagged_int_var: JaggedIntVar) -> Set[IntVar]:
    """Get the set of dynamic dims in JaggedIntVar's JaggedDims' min / max values."""
    return set(
        [
            dim.min_value()
            for dim in jagged_int_var.jagged_dims()
            if type(dim.min_value()) == IntVar
        ]
        + [
            dim.max_value()
            for dim in jagged_int_var.jagged_dims()
            if type(dim.max_value()) == IntVar
        ]
    )


@registry.reg("cuda.make_jagged.gen_function")
def make_jagged_gen_function(func_attrs):
    func_name = func_attrs["name"]
    num_sources = func_attrs["num_sources"]
    offsets_list = func_attrs["inputs"][num_sources:]
    backend_spec = CUDASpec()

    output = func_attrs["outputs"][0]
    jagged_int_var = output._attrs["shape"][0]
    offsets_struct_type = jagged_int_var.offsets_struct_type()

    jagged_dim_min_values = [
        dim.min_value().value()
        if isinstance(dim.min_value(), IntImm)
        else dim.min_value()._attrs["name"]
        for dim in jagged_int_var.jagged_dims()
    ]
    jagged_dim_max_values = [
        dim.max_value().value()
        if isinstance(dim.max_value(), IntImm)
        else dim.max_value()._attrs["name"]
        for dim in jagged_int_var.jagged_dims()
    ]

    jagged_dynamic_bound_dims = _get_jagged_dynamic_bound_dims(jagged_int_var)
    jagged_dynamic_bound_names = [
        dim._attrs["name"] for dim in jagged_dynamic_bound_dims
    ]

    for dim in jagged_dynamic_bound_dims:
        if dim._attrs.get("isolated", False):
            raise ValueError(
                "Dynamic dimension (IntVar) in the min / max value "
                "of a JaggedDim in the JaggedIntVar is isolated "
                f"(not present in any input shape): {jagged_int_var}."
            )

    batch_dim = jagged_int_var.batch_dim()
    isolated_batch_dim = batch_dim._attrs.get("isolated", False)
    check_sequence_lengths = func_attrs["check_sequence_lengths"]

    return SRC_TEMPLATE.render(
        func_name=func_name,
        num_offsets=len(offsets_list),
        offsets_struct_type=offsets_struct_type,
        jagged_dim_min_values=jagged_dim_min_values,
        jagged_dim_max_values=jagged_dim_max_values,
        offsets_type=jagged_int_var.offsets_type(),
        isolated_batch_dim=isolated_batch_dim,
        jagged_dynamic_bound_names=jagged_dynamic_bound_names,
        index_type=backend_spec.index_type,
        check_sequence_lengths=check_sequence_lengths,
    )


@registry.reg("cuda.make_jagged.func_decl")
def make_jagged_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    num_sources = func_attrs["num_sources"]
    offsets_list = func_attrs["inputs"][num_sources:]
    backend_spec = CUDASpec()

    output = func_attrs["outputs"][0]
    jagged_int_var = output._attrs["shape"][0]
    offsets_struct_type = jagged_int_var.offsets_struct_type()
    jagged_dynamic_bound_dims = _get_jagged_dynamic_bound_dims(jagged_int_var)

    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        num_offsets=len(offsets_list),
        offsets_struct_type=offsets_struct_type,
        num_jagged_dynamic_bound_dims=len(jagged_dynamic_bound_dims),
        index_type=backend_spec.index_type,
    )


@registry.reg("cuda.make_jagged.func_call")
def make_jagged_gen_function_call(func_attrs, indent="  "):
    func_name = func_attrs["name"]
    num_sources = func_attrs["num_sources"]
    total_length = func_attrs["inputs"][0]._attrs["shape"][0]
    offsets_list = func_attrs["inputs"][num_sources:]
    output = func_attrs["outputs"][0]
    jagged_int_var = output._attrs["shape"][0]

    offsets_first_dim_names = [
        offsets._attrs["shape"][0]._attrs["name"] for offsets in offsets_list
    ]
    offsets_data_names = [offsets._attrs["name"] for offsets in offsets_list]
    batch_dim_name = jagged_int_var.batch_dim()._attrs["name"]
    total_length_name = total_length._attrs["name"]

    jagged_dynamic_bound_names = [
        dim._attrs["name"] for dim in _get_jagged_dynamic_bound_dims(jagged_int_var)
    ]

    return FUNC_CALL_TEMPLATE.render(
        indent="      ",
        func_name=func_name,
        num_offsets=len(offsets_list),
        offsets_var_name=jagged_int_var.offsets_var_name(),
        offsets_first_dim_names=offsets_first_dim_names,
        offsets_data_names=offsets_data_names,
        batch_dim_name=batch_dim_name,
        total_length_name=total_length_name,
        jagged_dynamic_bound_names=jagged_dynamic_bound_names,
    )
