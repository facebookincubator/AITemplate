# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] common template for conv2d
"""

import jinja2

# pylint: disable=C0301,C0415,R1705

INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
using {{name}} = {{config_name}};
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

{{extra_code}}

#define CUTLASS_CHECK(status)                                                                    \\
  {                                                                                              \\
    cutlass::Status error = status;                                                              \\
    if (error != cutlass::Status::kSuccess) {                                                    \\
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __FILE__ \\
                << ":" << __LINE__ << std::endl;                                                 \\
      exit(EXIT_FAILURE);                                                                        \\
    }                                                                                            \\
  }

{{instances}}

void {{function_name}} (
    cutlass::half_t* a_ptr,
    cutlass::half_t* b_ptr,
    cutlass::half_t* bias_ptr,
    cutlass::half_t* c_ptr,
    uint8_t* workspace,
{% if support_split_k %}
    int split_k,
{% endif %}
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(input_ndims) %}
    int64_t* c_dim{{idx}},
{% endfor %}
  cudaStream_t stream
  ) {
  {{shape_eval}}
  {{input_addr_calculator}}
  {{output_addr_calculator}}
  {{extra_shape}}
  {{input_output_checks}}

  if (!bias_ptr) {
    throw std::runtime_error("bias_ptr is null!");
  }

  {{exec_paths}}
  {% for idx in range(input_ndims) %}
      std::cout << "input_ndims{{idx}}: " << *a_dim{{idx}} << std::endl;
  {% endfor %}
  {% for idx in range(weight_ndims) %}
      std::cout << "weight_ndims{{idx}}: " << *b_dim{{idx}} << std::endl;
  {% endfor %}
  {% for idx in range(input_ndims) %}
      std::cout << "output_ndims{{idx}}: " << *c_dim{{idx}} << std::endl;
  {% endfor %}
  throw std::runtime_error(
      "Unsupported workload for this {{function_name}} specialization."
  );
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  uint8_t*,
{% if support_split_k %}
    int,
{% endif %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(weight_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
  cudaStream_t
);
"""
)
