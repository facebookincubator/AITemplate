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
import jinja2

from ....backend import registry


SRC_TEMPLATE = jinja2.Template(
    """
#include <assert.h>
#include <stdexcept>

#include "jagged.h"


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
  int64_t length = offsets.lengths[blockIdx.x];
  const {{offsets_type}}* data = offsets.data[blockIdx.x];

  if (threadIdx.x >= length - 1) {
    // out of bounds of the offset array
    return;
  }

  {{offsets_type}} group_size = data[threadIdx.x + 1] - data[threadIdx.x];
  if (group_size < bounds.min_values[blockIdx.x] || group_size > bounds.max_values[blockIdx.x]) {
    printf(
      "\\n[func name: {{func_name}}, blockIdx.x: %d, threadIdx.x: %d]: "
      "Error: the offset difference %d is out of bounds of the jagged dimension %d (min: %d, max: %d).",
      (int32_t)blockIdx.x,
      (int32_t)threadIdx.x,
      (int32_t)group_size,
      (int32_t)blockIdx.x,
      (int32_t)bounds.min_values[blockIdx.x],
      (int32_t)bounds.max_values[blockIdx.x]
    );
    __trap();
  }

  if (threadIdx.x == 0) {
    {{offsets_type}} first_offset = data[0];
    if (first_offset != 0)
    {
      printf(
        "\\n[func name: {{func_name}}, blockIdx.x: %d, threadIdx.x: %d]: "
        "Error: the first offset of the jagged dimension %d is non-zero: %d.",
        (int32_t)blockIdx.x,
        (int32_t)threadIdx.x,
        (int32_t)blockIdx.x,
        (int32_t)first_offset
      );
      __trap();
    }
  }

  if (threadIdx.x == length - 2) {
    {{offsets_type}} last_offset = data[length - 1];
    if (last_offset != bounds.last_values[blockIdx.x])
    {
      printf(
        "\\n[func name: {{func_name}}, blockIdx.x: %d, threadIdx.x: %d]: "
        "Error: the last offset of the jagged dimension %d is incorrect: %d (must be %d).",
        (int32_t)blockIdx.x,
        (int32_t)threadIdx.x,
        (int32_t)blockIdx.x,
        (int32_t)last_offset,
        (int32_t)bounds.last_values[blockIdx.x]
      );
      __trap();
    }
  }
}

} // namespace


void {{func_name}}(
{% for idx in range(num_offsets) %}
  int64_t offsets_length_{{idx}},
  const void* offsets_data_{{idx}},
{% endfor %}
  {{offsets_struct_type}}& offsets,
  int64_t* batch_dim,
  int64_t total_length
) {
{% for idx in range(num_offsets) %}
    offsets.lengths[{{idx}}] = offsets_length_{{idx}};
    offsets.data[{{idx}}] = reinterpret_cast<const {{offsets_type}}*>(offsets_data_{{idx}});
{% endfor %}

{% if set_batch_dim %}
    // batch_dim must be set by this code
    *batch_dim = offsets.lengths[0] - 1;
{% else %}
    // batch_dim must have been set before this code
    if (*batch_dim != offsets.lengths[0] - 1) {
      throw std::runtime_error("batch_dim != len(offsets[0]) - 1");
    }
{% endif %}

    int64_t max_offset_length = 0;
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

    check_offsets<<<{{num_offsets}}, max_offset_length - 1, 0, 0>>>(offsets, bounds);
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
{% for idx in range(num_offsets) %}
  int64_t,
  const void*,
{% endfor %}
  {{offsets_struct_type}}&,
  int64_t*,
  int64_t
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
{{indent}}  {{offsets_var_name}},
{{indent}}  &{{batch_dim_name}},
{{indent}}  {{source_first_dim_name}}
{{indent}});
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


@registry.reg("cuda.make_jagged.gen_function")
def make_jagged_gen_function(func_attrs):
    func_name = func_attrs["name"]
    offsets_list = func_attrs["inputs"][1:]

    output = func_attrs["outputs"][0]
    jagged_int_var = output._attrs["shape"][0]
    set_batch_dim = jagged_int_var.batch_dim()._attrs.get("isolated", False)
    offsets_struct_type = jagged_int_var.offsets_struct_type()
    jagged_dim_min_values = [dim.min_value() for dim in jagged_int_var.jagged_dims()]
    jagged_dim_max_values = [dim.max_value() for dim in jagged_int_var.jagged_dims()]

    return SRC_TEMPLATE.render(
        func_name=func_name,
        num_offsets=len(offsets_list),
        set_batch_dim=set_batch_dim,
        offsets_struct_type=offsets_struct_type,
        jagged_dim_min_values=jagged_dim_min_values,
        jagged_dim_max_values=jagged_dim_max_values,
        offsets_type=jagged_int_var.offsets_type(),
    )


@registry.reg("cuda.make_jagged.func_decl")
def make_jagged_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    offsets_list = func_attrs["inputs"][1:]

    output = func_attrs["outputs"][0]
    jagged_int_var = output._attrs["shape"][0]
    offsets_struct_type = jagged_int_var.offsets_struct_type()

    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        num_offsets=len(offsets_list),
        offsets_struct_type=offsets_struct_type,
    )


@registry.reg("cuda.make_jagged.func_call")
def make_jagged_gen_function_call(func_attrs, indent="  "):
    func_name = func_attrs["name"]
    source = func_attrs["inputs"][0]
    offsets_list = func_attrs["inputs"][1:]
    output = func_attrs["outputs"][0]
    jagged_int_var = output._attrs["shape"][0]

    offsets_first_dim_names = [
        offsets._attrs["shape"][0]._attrs["name"] for offsets in offsets_list
    ]
    offsets_data_names = [offsets._attrs["name"] for offsets in offsets_list]
    batch_dim_name = jagged_int_var.batch_dim()._attrs["name"]
    source_first_dim_name = source._attrs["shape"][0]._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        indent="      ",
        func_name=func_name,
        num_offsets=len(offsets_list),
        offsets_var_name=jagged_int_var.offsets_var_name(),
        offsets_first_dim_names=offsets_first_dim_names,
        offsets_data_names=offsets_data_names,
        batch_dim_name=batch_dim_name,
        source_first_dim_name=source_first_dim_name,
    )
