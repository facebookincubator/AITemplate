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
Common functions and templates for group-gemm-family kernels
"""
import re
from hashlib import sha1
from typing import Any, Dict, List

import jinja2

from ...backend_spec import CUDASpec
from ...common import tensor_accessor_codegen
from . import common

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703

DIM_DEFS_TEMPLATE = jinja2.Template(
    """
{% for dim_name in dim_names %}
{% set dim_value = dim_values[loop.index - 1] %}
{{indent}}int64_t {{dim_name}} = {{dim_value}};
{% endfor %}
"""
)


GROUP_OUTPUT_ADDR_CALCULATOR = jinja2.Template(
    """
  {% if output_accessor.is_contiguous %}
  int64_t output_stride_{{group_id}} = GROUP_{{group_id}}_{{output_stride_dim}};
  int64_t output_offset_{{group_id}} = 0;
  {% else %}
  int64_t output_stride_{{group_id}} = {{output_accessor.actual_total_elements_from_stride_dim}};
  int64_t output_offset_{{group_id}} = {{output_accessor.offset}};
  {% endif %}
"""
)


GROUP_INPUT_A_ADDR_CALCULATOR = jinja2.Template(
    """
  {% if input_a_accessor.is_contiguous %}
  int64_t input_a_stride_{{group_id}} = GROUP_{{group_id}}_{{input_a_stride_dim}};
  int64_t input_a_offset_{{group_id}} = 0;
  {% else %}
  int64_t input_a_stride_{{group_id}} = {{input_a_accessor.actual_total_elements_from_stride_dim}};
  int64_t input_a_offset_{{group_id}} = {{input_a_accessor.offset}};
  {% endif %}
"""
)


INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
using {{name}} = cutlass::gemm::device::GemmGrouped<{{config_name}}>;
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
{{indent}}void {{func_name}}(
{{indent}} int,
{{indent}} int,
{{indent}} int64_t*,
{{indent}} int,
{{indent}} void*,
{% for i in range(groups) %}
{{indent}} void*,
{{indent}} void*,
{{indent}} void*,
{% if has_bias %}
{{indent}} void*,
{% endif %}
{% endfor %}
{{indent}} uint8_t*,
{% for i in range(groups) %}
{{indent}} int64_t*,
{{indent}} int64_t*,
{{indent}} int64_t*,
{{indent}} int64_t*,
{{indent}} int64_t*,
{{indent}} int64_t*,
{% endfor %}
{{indent}} cudaStream_t
{{indent}});
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{% if is_profiler %}
{{indent}}    gemm_op,
{% endif %}
{{indent}} device_properties_.sharedMemPerMultiprocessor,
{{indent}} device_properties_.multiProcessorCount,
{{indent}} &{{func_name}}_state,
{{indent}} {{problem_count}},
{{indent}} {{device_args}},
{% for operand in group_operands %}
{{indent}} {{operand[0]}},
{{indent}} {{operand[1]}},
{{indent}} {{operand[2]}},
{% if has_bias %}
{{indent}} {{operand[3]}},
{% endif %}
{% endfor %}
{{indent}} global_workspace_,
{% for operand_dim in group_operand_dims %}
{{indent}} {{operand_dim[0]}},
{{indent}} {{operand_dim[1]}},
{{indent}} {{operand_dim[2]}},
{{indent}} {{operand_dim[3]}},
{{indent}} {{operand_dim[4]}},
{{indent}} {{operand_dim[5]}},
{% endfor %}
{{indent}} stream
{{indent}});
"""
)


ADAPTOR_FUNCTION_TEMPLATE = jinja2.Template(
    """
{% if is_profiler %}
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"

using bfloat16 = nv_bfloat16;

#define CUTLASS_CHECK(status)                                                         \\
  {                                                                                   \\
    cutlass::Status error = status;                                                   \\
    if (error != cutlass::Status::kSuccess) {                                         \\
      auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +              \\
          cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \\
      std::cerr << msg << std::endl;                                                  \\
      throw std::runtime_error(msg);                                                  \\
    }                                                                                 \\
  }

{{instances}}

{% endif %}

{{indent}}template<typename GemmInstance>
{{indent}}void {{func_name}}_adapter(
{%if is_profiler %}
    GemmInstance& gemm_op,
{% endif %}
    int sharedMemPerMultiprocessor,
    int multiProcessorCount,
    uint8_t* workspace,
    int problem_count,
    cutlass::gemm::GemmCoord* problem_sizes_device,
    void **ptr_A,
    void **ptr_B,
    void **ptr_C,
{% if has_bias %}
    void **ptr_bias,
{% endif %}
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc,
{% if has_bias %}
    int64_t* ldd,
{% endif %}
    int occupancy,
    cudaStream_t stream) {
  {{exec_program}}
  throw std::runtime_error(
      "Unsupported workload for this gemm specialization."
  );
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


ADAPTER_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}_adapter<{{instance}}>(
{% if is_profiler %}
    gemm_op,
{% endif %}
    {{sharedMemPerMultiprocessor}},
    {{multiProcessorCount}},
    {{workspace}},
    {{problem_count}},
    {{problem_sizes_device}},
    {{ptr_A}},
    {{ptr_B}},
    {{ptr_C}},
{% if has_bias %}
    {{ptr_bias}},
{% endif %}
    {{lda}},
    {{ldb}},
    {{ldc}},
{% if has_bias %}
    {{ldd}},
{% endif %}
    {{instance}}::maximum_active_blocks(),
    stream
    );
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


BENCHMARK_INSTANCE_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}
{{indent}}{{instance_name}} {{gemm_op}};
{{indent}}const char *gemm_op_name = "{{gemm_op_name}}";
{{indent}}int ret = {{func_name}}_adapter(
{{indent}}    {{gemm_op}},
{{indent}}    gemm_op_name,
{{indent}}    {{sharedMemPerMultiprocessor}},
{{indent}}    {{multiProcessorCount}},
{{indent}}    {{workspace}},
{{indent}}    {{problem_count}},
{{indent}}    {{problem_sizes_device}},
{{indent}}    (void**)({{ptr_A}}),
{{indent}}    (void**)({{ptr_B}}),
{{indent}}    (void**)({{ptr_C}}),
{% if has_bias %}
{{indent}}    (void**)({{ptr_bias}}),
{% endif %}
{{indent}}    {{lda}},
{{indent}}    {{ldb}},
{{indent}}    {{ldc}},
{% if has_bias %}
{{indent}}    {{ldd}},
{% endif %}
{{indent}}    {{instance_name}}::maximum_active_blocks(),
{{indent}}    stream
{{indent}}    );
{{indent}}if (ret != 0)
{{indent}}  return ret;
{{indent}}
{{indent}}}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


EXEC_TEMPLATE = jinja2.Template(
    """
//  TODO: cast to right dtype
{{indent}}using ElementComputeEpilogue = typename {{instance}}::ElementAccumulator;
{{indent}}// int smem_size = int(sizeof(typename {{instance}}::GemmKernel::SharedStorage));
{{indent}}// int occupancy = std::min(2, int(sharedMemPerMultiprocessor / smem_size));
{{indent}}int threadblock_count = multiProcessorCount * occupancy;
{{indent}}// Early exit
{{indent}}if (!threadblock_count) {
{{indent}}  throw std::runtime_error(
{{indent}}    "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel."
{{indent}}  );
{{indent}}}


{{indent}}typename {{instance}}::Arguments arguments{

{{problem_args}}

{{indent}}};
{% if is_profiler %}
{{indent}}// Debug BGM: https://www.youtube.com/watch?v=rRwxfYlgG-M
{{indent}}size_t workspace_size = gemm_op.get_workspace_size(arguments);
{{indent}}cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);
{{indent}}workspace = local_workspace.get();
{{indent}}GLOBAL_WORKSPACE_SIZE = workspace_size;
{% else %}
{{indent}}{{instance}} gemm_op;
{% endif %}
{{indent}}// TODO: cutlass bug here
{{indent}}// auto status = gemm_op.can_implement(arguments);
{{indent}}// CUTLASS_CHECK(status);
{{indent}}auto status = gemm_op.initialize(arguments, workspace, stream);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = gemm_op(stream);
{{indent}}CUTLASS_CHECK(status);
{{indent}}return;
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#define CUTLASS_CHECK(status)                                                         \\
  {                                                                                   \\
    cutlass::Status error = status;                                                   \\
    if (error != cutlass::Status::kSuccess) {                                         \\
      auto msg = std::string("Got cutlass error: ") + cutlassGetStatusString(error) + \\
          " at: " + std::to_string(__LINE__);                                         \\
      std::cerr << msg << std::endl;                                                  \\
      throw std::runtime_error(msg);                                                  \\
    }                                                                                 \\
  }

namespace {
template <typename T>
void copy(T* dst, T const* src, size_t count, cudaMemcpyKind kind) {
  size_t bytes = count * cutlass::sizeof_bits<T>::value / 8;
  if (bytes == 0 && count > 0)
    bytes = 1;
  cudaError_t cuda_error = (cudaMemcpy(dst, src, bytes, kind));
  if (cuda_error != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy() failed");
  }
}
} // namespace

{{instances}}

{{func_adapter}}

{% if is_profiler %}
template <typename GemmInstance>
void {{function_name}} (
    GemmInstance& gemm_op,
{% else %}
void {{function_name}} (
{% endif %}
    int sharedMemPerMultiprocessor,
    int multiProcessorCount,
    int64_t* func_state,
    int problem_count,
    void* device_args,
    {% for operand in group_operands %}
    void* {{operand[0]}},
    void* {{operand[1]}},
    void* {{operand[2]}},
    {% if has_bias %}
    void* {{operand[3]}},
    {% endif %}
    {% endfor %}
    uint8_t* global_workspace_,
{% for operand_dim in group_operand_dims %}
    int64_t* {{operand_dim[0]}},
    int64_t* {{operand_dim[1]}},
    int64_t* {{operand_dim[2]}},
    int64_t* {{operand_dim[3]}},
    int64_t* {{operand_dim[4]}},
    int64_t* {{operand_dim[5]}},
{% endfor %}
    cudaStream_t stream) {

    {{shape_function}}

    if (!device_args) {
      throw std::runtime_error("device_args is NULL!");
    }
    // It's a bit tricky to check individual gemms in group_gemm cases,
    // so let's rule out them all if any of the input/output tensors is zero-sized.
    // We can re-visit this part if we hit any use case, e.g. one input of the
    // gemm is zero-sized, but all others are non-zero-sized.
{% for operand_dim in group_operand_dims %}
    if (*{{operand_dim[0]}} == 0 || *{{operand_dim[1]}} == 0 ||
        *{{operand_dim[2]}} == 0 || *{{operand_dim[3]}} == 0 ||
        *{{operand_dim[4]}} == 0 || *{{operand_dim[5]}} == 0) {
      throw std::runtime_error("Zero-sized tensors are not supported yet");
    }
{% endfor %}
{% for operand in group_operands %}
    if (!{{operand[0]}}) {
      throw std::runtime_error("{{operand[0]}} is NULL!");
    }
    if (!{{operand[1]}}) {
      throw std::runtime_error("{{operand[1]}} is NULL!");
    }
    if (!{{operand[2]}}) {
      throw std::runtime_error("{{operand[2]}} is NULL!");
    }
{% if has_bias %}
    if (!{{operand[3]}}) {
      throw std::runtime_error("{{operand[3]}} is NULL!");
    }
{% endif %}

{% endfor %}

    void* arg_ptr = device_args;
    // problem_sizes_device: N * GemmCoord -> N * 3 * sizeof(int64_t) -> 32 * N
    // ptrA/B/C/D: N * 8 for each
    // lda/b/c/d: N * 8 for each
    // total: N * 8 * 4 + N * 8 * 4 + N * 8 * 4
    // total: 3 * 32 * N
    int offset = 0;
    auto problem_sizes_device =
        (cutlass::gemm::GemmCoord*)(arg_ptr + offset);
    offset += 32 * problem_count;

    auto ptr_A = (void**)(arg_ptr + offset);
    offset += 8 * problem_count;
    auto ptr_B = (void**)(arg_ptr + offset);
    offset += 8 * problem_count;
    auto ptr_C = (void**)(arg_ptr + offset);
    offset += 8 * problem_count;
    {% if has_bias %}
    auto ptr_bias = (void**)(arg_ptr + offset);
    offset += 8 * problem_count;
    {% endif %}

    auto lda = (int64_t*)(arg_ptr + offset);
    offset += 8 * problem_count;
    auto ldb = (int64_t*)(arg_ptr + offset);
    offset += 8 * problem_count;
    auto ldc = (int64_t*)(arg_ptr + offset);
    {% if has_bias %}
    offset += 8 * problem_count;
    auto ldd = (int64_t*)(arg_ptr + offset);
    {% endif %}
    // offset += 8 * problem_count;

    if (*func_state != GROUP_0_AM) {
        // need update
        std::vector<cutlass::gemm::GemmCoord> problem_sizes;
        std::vector<void*> ptr_A_host;
        std::vector<void*> ptr_B_host;
        std::vector<void*> ptr_C_host;
        {% if has_bias %}
        std::vector<void*> ptr_bias_host;
        {% endif %}
        std::vector<int64_t> lda_host;
        std::vector<int64_t> ldb_host;
        std::vector<int64_t> ldc_host;
        {% if has_bias %}
        std::vector<int64_t> ldd_host;
        {% endif %}

        {% for operand in group_operands %}
        ptr_A_host.push_back(({{elem_input_type}}*)({{operand[0]}}) + input_a_offset_{{loop.index0}});
        ptr_B_host.push_back(({{elem_input_type}}*)({{operand[1]}}));
        ptr_C_host.push_back(({{elem_output_type}}*)({{operand[2]}}) + output_offset_{{loop.index0}});
        {% if has_bias %}
        ptr_bias_host.push_back(({{elem_input_type}}*)({{operand[3]}}));
        {% endif %}
        {% endfor %}

        // AM: 0
        // AK: 1
        // BN: 2
        {% for operand_dim in group_operand_dims %}
        cutlass::gemm::GemmCoord problem_{{loop.index0}}(
            GROUP_{{loop.index0}}_M,
            GROUP_{{loop.index0}}_N,
            GROUP_{{loop.index0}}_K);
        problem_sizes.emplace_back(problem_{{loop.index0}});
        lda_host.push_back(input_a_stride_{{loop.index0}});
        ldb_host.push_back(GROUP_{{loop.index0}}_K);
        {% if has_bias %}
        ldc_host.push_back(0);
        ldd_host.push_back(output_stride_{{loop.index0}});
        {% else %}
        ldc_host.push_back(output_stride_{{loop.index0}});
        {% endif %}
        {% endfor %}

        copy(problem_sizes_device,
             problem_sizes.data(),
             problem_count, cudaMemcpyHostToDevice);

        copy(ptr_A,
             ptr_A_host.data(),
             problem_count, cudaMemcpyHostToDevice);

        copy(ptr_B,
             ptr_B_host.data(),
             problem_count, cudaMemcpyHostToDevice);

        copy(ptr_C,
             ptr_C_host.data(),
             problem_count, cudaMemcpyHostToDevice);

        {% if has_bias %}
        copy(ptr_bias,
             ptr_bias_host.data(),
             problem_count, cudaMemcpyHostToDevice);
        {% endif %}

        copy(lda,
             lda_host.data(),
             problem_count, cudaMemcpyHostToDevice);

        copy(ldb,
             ldb_host.data(),
             problem_count, cudaMemcpyHostToDevice);

        copy(ldc,
             ldc_host.data(),
             problem_count, cudaMemcpyHostToDevice);

        {% if has_bias %}
        copy(ldd,
             ldd_host.data(),
             problem_count, cudaMemcpyHostToDevice);
        {% endif %}

        *func_state = GROUP_0_AM;
    }
    {{exec_paths}}
}


"""
)


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int problem_count = std::atoi(argv[1]);
  int64_t idx = 2;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes;
  while (idx < argc) {
    int64_t M = std::atoi(argv[idx++]);
    int64_t N = std::atoi(argv[idx++]);
    int64_t K = std::atoi(argv[idx++]);
    cutlass::gemm::GemmCoord problem(M, N, K);
    problem_sizes.push_back(problem);
  }
"""
)


TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  using ElementOutput = {{elem_output_type}};
  using ElementInputA = {{elem_input_type}};
  using ElementInputB = {{elem_input_type}};
  cutlass::DeviceAllocation<ElementInputA> blob_A;
  cutlass::DeviceAllocation<ElementInputB> blob_B;
  cutlass::DeviceAllocation<ElementOutput> blob_C;
{% if has_bias %}
  cutlass::DeviceAllocation<ElementOutput> blob_Bias;
{% endif %}
  int64_t total_size_A = 0;
  int64_t total_size_B = 0;
  int64_t total_size_C = 0;
{% if has_bias %}
  int64_t total_size_Bias = 0;
{% endif %}

  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;

  std::vector<int64_t> lda_host;
  std::vector<int64_t> ldb_host;
  std::vector<int64_t> ldc_host;
{% if has_bias %}
  std::vector<int64_t> ldd_host;
{% endif %}


  cutlass::DeviceAllocation<int64_t> lda;
  cutlass::DeviceAllocation<int64_t> ldb;
  cutlass::DeviceAllocation<int64_t> ldc;
{% if has_bias %}
  cutlass::DeviceAllocation<int64_t> ldd;
{% endif %}

  std::vector<ElementInputA *> ptr_A_host;
  std::vector<ElementInputB *> ptr_B_host;
  std::vector<ElementOutput *> ptr_C_host;
{% if has_bias %}
  std::vector<ElementOutput *> ptr_bias_host;
{% endif %}


  cutlass::DeviceAllocation<ElementInputA *> ptr_A;
  cutlass::DeviceAllocation<ElementInputB *> ptr_B;
  cutlass::DeviceAllocation<ElementOutput *> ptr_C;
{% if has_bias %}
  cutlass::DeviceAllocation<ElementOutput *> ptr_bias;
{% endif %}


  for (auto & mnk : problem_sizes) {
    int64_t M = mnk.m();
    int64_t N = mnk.n();
    int64_t K = mnk.k();
    lda_host.push_back(K);
    ldb_host.push_back(K);
{% if has_bias %}
    ldc_host.push_back(0);
    ldd_host.push_back(N);
{% else %}
    ldc_host.push_back(N);
{% endif %}

    total_size_A += M * K;
    total_size_B += N * K;
    total_size_C += M * N;
{% if has_bias %}
    total_size_Bias += N;
{% endif %}
  }

  blob_A.reset(total_size_A);
  blob_B.reset(total_size_B);
  blob_C.reset(total_size_C);
{% if has_bias %}
  blob_Bias.reset(total_size_Bias);
{% endif %}

  int64_t offset_A = 0;
  int64_t offset_B = 0;
  int64_t offset_C = 0;
{% if has_bias %}
  int64_t offset_Bias = 0;
{% endif %}

  for (int i = 0; i < problem_sizes.size(); ++i) {
    auto & mnk = problem_sizes.at(i);
    int64_t M = mnk.m();
    int64_t N = mnk.n();
    int64_t K = mnk.k();

    ptr_A_host.push_back(blob_A.get() + offset_A);
    ptr_B_host.push_back(blob_B.get() + offset_B);
    ptr_C_host.push_back(blob_C.get() + offset_C);
{% if has_bias %}
    ptr_bias_host.push_back(blob_Bias.get() + offset_Bias);
{% endif %}
    offset_A += M * K;
    offset_B += N * K;
    offset_C += M * N;
{% if has_bias %}
    offset_Bias += N;
{% endif %}
  }


  lda.reset(problem_count);
  ldb.reset(problem_count);
  ldc.reset(problem_count);
{% if has_bias %}
  ldd.reset(problem_count);
{% endif %}
  lda.copy_from_host(lda_host.data());
  ldb.copy_from_host(ldb_host.data());
  ldc.copy_from_host(ldc_host.data());
{% if has_bias %}
  ldd.copy_from_host(ldd_host.data());
{% endif %}

  ptr_A.reset(problem_count);
  ptr_B.reset(problem_count);
  ptr_C.reset(problem_count);
{% if has_bias %}
  ptr_bias.reset(problem_count);
{% endif %}
  ptr_A.copy_from_host(ptr_A_host.data());
  ptr_B.copy_from_host(ptr_B_host.data());
  ptr_C.copy_from_host(ptr_C_host.data());
{% if has_bias %}
  ptr_bias.copy_from_host(ptr_bias_host.data());
{% endif %}

  problem_sizes_device.reset(problem_count);
  problem_sizes_device.copy_from_host(problem_sizes.data());

"""
)


def get_group_gemm_instance_template_params(op_def: str) -> List[str]:
    """
    For a given op_def string generated by cutlass's group_gemm emiter, parse and
    return the group_gemm instance's template parameters.
    """
    params = re.findall(
        r"cutlass::gemm::kernel::DefaultGemmUniversal<([\s\S]+)>::GemmKernel;", op_def
    )
    assert len(params) == 1
    param = params[0]
    gemm_universal_params = param.strip().split("\n")
    gemm_universal_params = [param.strip(",") for param in gemm_universal_params]
    assert len(gemm_universal_params) == 20, (
        "expected len(gemm_universal_params) to be 20, but got "
        "{len(gemm_universal_params)}, {gemm_universal_params=}"
    )
    return gemm_universal_params


def update_alignments_in_group_gemm_instance(
    op_def: str, func_attrs: Dict[str, Any], for_profiler: bool
) -> str:
    """
    update kAlignmentA, kAlignmentB, and epilogue_alignment in op_def,
    which is a group_gemm instance emitted by the gemm instance emitter of cutlass.
    """
    if for_profiler:
        return op_def

    # TODO: adjust a_alignment, b_alignment based on input_accessors

    gemm_params = get_group_gemm_instance_template_params(op_def)
    epilogue_align_idx = 12
    epilogue_curr_align = gemm_params[epilogue_align_idx].strip()

    output_accessors = func_attrs["output_accessors"]
    epilogue_alignment = int(epilogue_curr_align)
    for output_accessor in output_accessors:
        epilogue_alignment = min(
            epilogue_alignment,
            tensor_accessor_codegen.find_max_alignment_for_accessor(output_accessor),
        )

    instance_lines = op_def.split("\n")
    # a_align_idx + 4 in the full instance string
    idx_offset = 4

    epilogue_curr_align_line = instance_lines[epilogue_align_idx + idx_offset]
    assert epilogue_curr_align == epilogue_curr_align_line.strip(
        " ,"
    ), f"expected {epilogue_curr_align=} equal to {epilogue_curr_align_line=}"
    instance_lines[epilogue_align_idx + idx_offset] = epilogue_curr_align_line.replace(
        epilogue_curr_align, str(epilogue_alignment)
    )
    return "\n".join(instance_lines)


def group_gemm_instance(op_def: str, func_attrs: Dict[str, Any], for_profiler: bool):
    # TODO: This is a dirty thing need to add an extra emitter to clean this up
    op_def = update_alignments_in_group_gemm_instance(op_def, func_attrs, for_profiler)
    tmp = op_def.replace("DefaultGemmUniversal", "DefaultGemmGrouped")
    tmp = tmp.replace("false,", "")
    # force output to be row major
    # cutlass lib can't generate row major output kernels
    tmp = re.sub(
        r"cutlass::layout::ColumnMajor,\n", "cutlass::layout::RowMajor,\n", tmp
    )
    tmp = re.sub(
        r"GemmIdentityThreadblockSwizzle<\d>",
        "GemmBatchedIdentityThreadblockSwizzle",
        tmp,
    )
    tmp = re.sub(
        r"cutlass::arch::OpMultiplyAdd",
        "cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,\n"
        + "cutlass::arch::OpMultiplyAdd",
        tmp,
    )
    return tmp


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
    problem_args_template,
    has_bias=False,
    output_addr_calculator="",
):
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    elem_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )

    instance_name_base = "GemmInstance"
    exec_program = EXEC_TEMPLATE.render(
        indent="  ",
        instance=instance_name_base,
        is_profiler=True,
        problem_args=problem_args_template.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
        ),
    )

    instances = []
    benchmark_instances = []
    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        config = common.emit_instance(
            op,
            for_profiler=True,
            f_instance_convertor=group_gemm_instance,
            emit_kernel=True,
        )
        config_name = common.extract_config_name(config)
        instance_name = f"{instance_name_base}_{instance_idx}"
        gemm_op = f"gemm_op_{instance_idx}"
        instance = INSTANCE_TEMPLATE.render(
            config_name=config_name, name=instance_name, config=config
        )
        benchmark_instance = BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            instance_name=instance_name,
            gemm_op=gemm_op,
            gemm_op_name=op_name,
            func_name=f"benchmark_{instance_name_base}",
            sharedMemPerMultiprocessor="device_properties.sharedMemPerMultiprocessor",
            multiProcessorCount="device_properties.multiProcessorCount",
            workspace="global_workspace_",
            problem_count="problem_count",
            problem_sizes_device="problem_sizes_device.get()",
            ptr_A="ptr_A.get()",
            ptr_B="ptr_B.get()",
            ptr_C="ptr_C.get()",
            has_bias=has_bias,
            ptr_bias="ptr_bias.get()",
            lda="lda.get()",
            ldb="ldb.get()",
            ldc="ldc.get()",
            ldd="ldd.get()",
        )
        instances.append(instance)
        benchmark_instances.append(benchmark_instance)
    op_func = ADAPTOR_FUNCTION_TEMPLATE.render(
        instances="\n".join(instances),
        is_profiler=True,
        func_name=instance_name_base,
        indent=" ",
        exec_program=exec_program,
        has_bias=has_bias,
    )
    func_call = ADAPTER_CALL_TEMPLATE.render(
        is_profiler=True,
        func_name=instance_name_base,
        instance=instance_name_base,
        sharedMemPerMultiprocessor="sharedMemPerMultiprocessor",
        multiProcessorCount="multiProcessorCount",
        workspace="global_workspace_",
        problem_count="problem_count",
        problem_sizes_device="problem_sizes_device",
        ptr_A="ptr_A",
        ptr_B="ptr_B",
        ptr_C="ptr_C",
        has_bias=has_bias,
        ptr_bias="ptr_bias",
        lda="lda",
        ldb="ldb",
        ldc="ldc",
        ldd="ldd",
    )
    tensor_decl = TENSOR_DECL_TEMPLATE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        has_bias=has_bias,
    )
    code = common.PROFILER_TEMPLATE.render(
        is_group_gemm=True,
        op_func=op_func,
        has_bias=has_bias,
        args_parse=ARGS_PARSER_TEMPLATE.render(),
        function_name=f"{instance_name_base}_adapter",
        func_call=func_call,
        name=instance_name_base,
        tensor_decl=tensor_decl,
        benchmark_instances="\n".join(benchmark_instances),
        elem_type=elem_type,
    )
    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    common.add_profiler(file_pairs, workdir, op_type, profiler_filename, code)
    # build
    return common.build_profiler(file_pairs)


def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    problem_args_template,
    has_bias=False,
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    problem_args = problem_args_template.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]
    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    emit_kernel = True
    for key, value in exec_path.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        algo = value.algo
        if algo not in inst_def_flag:
            config = common.emit_instance(
                op_instance[algo],
                for_profiler=False,
                f_instance_convertor=group_gemm_instance,
                emit_kernel=emit_kernel,
                func_attrs=func_attrs,
            )
            inst_def_flag.add(algo)
        else:
            raise ValueError(f"Algo {algo} already in inst_def_flags")

        inst = INSTANCE_TEMPLATE.render(
            config=config,
            name=fname,
            config_name=common.extract_config_name(config),
        )
        instances[key] = inst
        instance_decl += inst
    kwargs = {}
    kwargs["indent"] = "  "
    kwargs["dtype"] = "int64_t "
    group_operand_dims = []
    output_addr_cals = []
    input_a_addr_cals = []
    num_inputs_per_group = 3 if has_bias else 2

    for i in range(func_attrs["groups"]):
        dim_names = []
        for j in range(6):
            dim_names.append("*dim_{group}_{dim}".format(group=i, dim=j))
        group_operand_dims.append(dim_names)
        output_addr_cal = GROUP_OUTPUT_ADDR_CALCULATOR.render(
            group_id=i,
            output_stride_dim="CN",
            output_accessor=func_attrs["output_accessors"][i],
        )
        output_addr_cals.append(output_addr_cal)
        input_a_addr_cal = GROUP_INPUT_A_ADDR_CALCULATOR.render(
            group_id=i,
            input_a_stride_dim="AK",
            input_a_accessor=func_attrs["input_accessors"][i * num_inputs_per_group],
        )
        input_a_addr_cals.append(input_a_addr_cal)
    kwargs["group_operand_dims"] = group_operand_dims
    kwargs["output_addr_cals"] = output_addr_cals
    kwargs["input_a_addr_cals"] = input_a_addr_cals
    shape_func = shape_eval_template.render(**kwargs)
    exec_paths = ""
    #
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        program = ADAPTER_CALL_TEMPLATE.render(
            indent="    ",
            func_name=func_name,
            instance=fname,
            sharedMemPerMultiprocessor="sharedMemPerMultiprocessor",
            multiProcessorCount="multiProcessorCount",
            workspace="global_workspace_",
            problem_count=func_attrs["groups"],
            problem_sizes_device="problem_sizes_device",
            ptr_A="ptr_A",
            ptr_B="ptr_B",
            ptr_C="ptr_C",
            has_bias=has_bias,
            ptr_bias="ptr_bias",
            lda="lda",
            ldb="ldb",
            ldc="ldc",
            ldd="ldd",
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst

    exec_program = EXEC_TEMPLATE.render(
        indent="  ",
        instance="GemmInstance",
        is_profiler=False,
        problem_args=problem_args,
    )
    adapter_func = ADAPTOR_FUNCTION_TEMPLATE.render(
        func_name=func_name, exec_program=exec_program, has_bias=has_bias
    )
    group_operands = []
    group_operand_dims = []
    for i in range(func_attrs["groups"]):
        operand = []
        operand.append("ptr_{group}_a".format(group=i))
        operand.append("ptr_{group}_b".format(group=i))
        operand.append("ptr_{group}_c".format(group=i))
        if has_bias:
            operand.append("ptr_{group}_bias".format(group=i))
        dims = []
        for j in range(6):
            dims.append("dim_{group}_{dim}".format(group=i, dim=j))
        group_operands.append(operand)
        group_operand_dims.append(dims)

    return SRC_TEMPLATE.render(
        instances=instance_decl,
        func_adapter=adapter_func,
        function_name=func_name,
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        shape_function=shape_func,
        group_operands=group_operands,
        group_operand_dims=group_operand_dims,
        exec_paths=exec_paths,
        has_bias=has_bias,
    )


def gen_function_call(func_attrs, ndims, has_bias=False, indent="  "):
    group_operands = []
    group_operand_dims = []
    output_accessors = [a.is_contiguous for a in func_attrs["output_accessors"]]
    with_single_strided_output = False
    if "output_stride_dim" in func_attrs:
        output_accessors = list(set(output_accessors))
        # we only support two cases: either all outputs are contiguous or none
        # of them are
        assert len(output_accessors) == 1
        with_single_strided_output = not output_accessors[0]
    for i in range(func_attrs["groups"]):
        a = func_attrs["inputs"][i * ndims]
        b = func_attrs["inputs"][i * ndims + 1]
        if has_bias:
            bias = func_attrs["inputs"][i * ndims + 2]
        c_idx = 0 if with_single_strided_output else i
        c = func_attrs["outputs"][c_idx]
        input_a_accessor = func_attrs["input_accessors"][i * ndims]
        input_b_accessor = func_attrs["input_accessors"][i * ndims + 1]
        output_accessor = func_attrs["output_accessors"][i]

        ashape = input_a_accessor.original_shapes
        bshape = input_b_accessor.original_shapes
        cshape = output_accessor.original_shapes
        operands = []
        operand_dims = []
        operands.append(a._attrs["name"])
        operands.append(b._attrs["name"])
        operands.append(c._attrs["name"])
        if has_bias:
            operands.append(bias._attrs["name"])
        operand_dims.append("&" + ashape[0]._attrs["name"])
        operand_dims.append("&" + ashape[1]._attrs["name"])
        operand_dims.append("&" + bshape[0]._attrs["name"])
        operand_dims.append("&" + bshape[1]._attrs["name"])
        operand_dims.append("&" + cshape[0]._attrs["name"])
        operand_dims.append("&" + cshape[1]._attrs["name"])
        group_operands.append(operands)
        group_operand_dims.append(operand_dims)
    device_args = f'unique_workspace_ + {func_attrs["unique_workspace_offset"]}'
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        problem_count=func_attrs["groups"],
        device_args=device_args,
        group_operands=group_operands,
        group_operand_dims=group_operand_dims,
        indent=indent,
        has_bias=has_bias,
    )
