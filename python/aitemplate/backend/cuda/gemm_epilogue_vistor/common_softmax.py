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
Common template for softmax.
"""
import os
import re
from hashlib import sha1

import jinja2

from ...common import gemm_common
from ...target import Target
from ..gemm_universal import common

# pylint: disable=C0301,C0415,R1705


SRC_TEMPLATE = jinja2.Template(
    """
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
#include "cutlass/util/device_memory.h"

#include "gemm_with_softmax.h"

{{custom_libs}}

{{extra_code}}

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

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;


void {{function_name}} (
    cutlass::half_t* a_ptr,
    cutlass::half_t* b_ptr,
{% if has_d %}
    cutlass::half_t* d_ptr,
{% endif %}
    cutlass::half_t* c_ptr,
    cutlass::half_t* d_ptr,
    float* n_ptr,
    cutlass::half_t* soft_ptr,
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
  {{output_addr_calculator}}
  {{extra_shape}}

  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this gemm specialization."
  );
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}typename {{instance}}::Arguments arguments{

{{problem_args}}

{{indent}}};
{{indent}}{{instance}} gemm_op;
{% if is_profiler %}
{{indent}}size_t workspace_size = 0; //gemm_op.get_workspace_size(arguments);
{{indent}}cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);
{{indent}}workspace = local_workspace.get();
{{indent}}GLOBAL_WORKSPACE_SIZE = workspace_size;
{% endif %}

{{indent}}auto status = gemm_op.initialize(arguments);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = gemm_op(stream);
{{indent}}CUTLASS_CHECK(status);
{{indent}}return;

"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  float*,
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


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{% if has_bias %}
{{indent}}    {{bias_ptr}},
{% endif %}
{{indent}}    {{c_ptr}},
{{indent}}    {{d_ptr}},
{{indent}}    {{n_ptr}},
{{indent}}    {{soft_ptr}},
{{indent}}    global_workspace_,
{{indent}}    {{split_k}},
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    stream
{{indent}});
"""
)


TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  // cast to int64_t to avoid overflow
  int64_t a_ptr_sz = static_cast<int64_t>(a_dim0) * static_cast<int64_t>(a_dim1);
  int64_t b_ptr_sz = static_cast<int64_t>(b_dim0) * static_cast<int64_t>(b_dim1);
  int64_t c_ptr_sz = static_cast<int64_t>(c_dim0) * static_cast<int64_t>(c_dim1);
  int64_t ptr_max_sz = std::max({a_ptr_sz, b_ptr_sz, c_ptr_sz});
  // TODO: special pool size for A100 L2 cache 40M
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 25) / ptr_max_sz)));

  memory_pool->AllocateHalfTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateHalfTensor(b_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // c_ptr: index 2
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // d_ptr: index 3
  memory_pool->AllocateFloatTensor(c_dim0,  mem_pool_sz);  // n_ptr: index 4
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // soft_ptr: index 5
"""
)


DEFAULT_EXTRA_SHAPE_TEMPLATE = jinja2.Template(
    """
{{indent}}const int M = AM;
{{indent}}const int N = BN;
{{indent}}const int K = AK;
"""
)


# TODO Merge all alignment into single profiler
PROFILER_TEMPLATE = jinja2.Template(
    """
size_t GLOBAL_WORKSPACE_SIZE = 0;

{{op_func}}

struct ProfilerMemoryPool {
  ProfilerMemoryPool() {
    std::random_device rd;
    gen = std::mt19937(rd());
    uniform_dist = std::uniform_int_distribution<int64_t>(1, 48964896);
    offsets.reserve(512);
    strides.reserve(512);
    copies.reserve(512);
    ptrs.reserve(512);
    blobs.reserve(512);
  }
  ~ProfilerMemoryPool() {}

  template <typename DType>
  DType* AllocateGaussianTensor(int64_t size) {
    size_t length = size * sizeof(DType);
    blobs.emplace_back(length);
    DType* ptr = reinterpret_cast<DType*>(blobs.back().get());

    uint64_t seed = uniform_dist(gen);
    double mean = 0.f;
    double std = 1.f;

    cutlass::reference::device::BlockFillRandomGaussian(ptr, size, seed, mean,
                                                        std);

    return ptr;
  }


  cutlass::half_t* AllocateHalfGaussianTensor(int64_t size) {
    return reinterpret_cast<cutlass::half_t*>(
        AllocateGaussianTensor<__half>(size));
  }

  int AllocateHalfTensor(int64_t size, int64_t copy) {
    offsets.push_back(0);
    strides.push_back(size);
    copies.push_back(copy);
    auto ptr = AllocateHalfGaussianTensor(size * copy);
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  float* AllocateFloatGaussianTensor(int64_t size) {
    return reinterpret_cast<float*>(
        AllocateGaussianTensor<float>(size));
  }

  int AllocateFloatTensor(int64_t size, int64_t copy) {
    offsets.push_back(0);
    strides.push_back(size);
    copies.push_back(copy);
    auto ptr = AllocateFloatGaussianTensor(size * copy);
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  template <typename T>
  T* RequestTensorByIdx(int idx) {
    auto copy = copies.at(idx);
    auto offset = offsets.at(idx);
    auto stride = strides.at(idx);
    T* ptr = reinterpret_cast<T*>(ptrs.at(idx));
    ptr += offset;
    offset += stride;
    if (offset == copy * stride) {
        offset = 0;
    }
    offsets[idx] = offset;
    return ptr;
  }

  std::vector<int64_t> offsets;
  std::vector<int64_t> strides;
  std::vector<int64_t> copies;
  std::vector<void*> ptrs;
  std::vector<cutlass::DeviceAllocation<uint8_t> > blobs;
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> uniform_dist;
};

int main(int argc, char** argv) {
  int device_idx;
  cudaDeviceProp device_properties;
  cudaError_t result = cudaGetDevice(&device_idx);
  auto memory_pool = std::make_unique<ProfilerMemoryPool>();
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDevice() API call failed.");
  }

  result = cudaGetDeviceProperties(&device_properties, device_idx);

  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDeviceProperties() failed");
  }



  {{args_parse}}

  using ElementOutput = typename {{name}}::ElementC;
  using ElementInputA = typename {{name}}::ElementA;
  using ElementInputB = typename {{name}}::ElementB;
  using ElementInputN = typename {{name}}::ElementN;
  uint8_t* global_workspace = nullptr;
  cudaStream_t stream = nullptr;

  {{tensor_decl}}

  // warmup
  {{func_call}}
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0], stream);
  for (int i = 0; i < 5; ++i) {
    {{func_call}}
  }
  cudaEventRecord(events[1], stream);
  cudaEventSynchronize(events[1]);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  // TODO: output workspace
  if (runtime_ms < 0.00001) {
      throw std::runtime_error(
      "OOB in cutlass."
    );
  }
  std::cout << "TIME:" << runtime_ms << std::endl;
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
}
"""
)


def gen_custom_libs():
    custom_libs = Target.current().get_custom_libs(
        os.path.dirname(__file__), "include/gemm_with_softmax.h"
    )
    return custom_libs


def _gemm_softmax_instance(op_def):
    tmp = op_def.replace("GemmSoftmax", "GemmSoftmaxUniversal")
    tmp = re.sub(
        r"GemmIdentityThreadblockSwizzle<\d>",
        "GemmBatchedIdentityThreadblockSwizzle",
        tmp,
    )
    return tmp


def emit_instance(op, f_instance_convertor=_gemm_softmax_instance, emit_kernel=False):
    import cutlass_lib

    emiter = cutlass_lib.gemm_operation.EmitGemmInstance()
    if emit_kernel:
        emiter = cutlass_lib.gemm_operation.EmitGemmSoftmaxInstance()

    op_def = emiter.emit(op)
    op_def = f_instance_convertor(op_def)
    return op_def


def gen_function(
    func_attrs,
    src_template,
    exec_cond_template,
    problem_args,
    input_ndims,
    weight_ndims,
    dim_info_dict,
    f_instance_convertor=_gemm_softmax_instance,
    emit_kernel=False,
    support_split_k=False,
    output_addr_calculator="",
    extra_code="",
):
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]
    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for exec_item in exec_path.values():
        fname = "f" + sha1(exec_item.exec_cond.encode()).hexdigest()
        algo = exec_item.algo
        if algo not in inst_def_flag:
            config = emit_instance(op_instance[algo], f_instance_convertor, emit_kernel)
            inst_def_flag.add(algo)
        else:
            config = ""
        inst = common.INSTANCE_TEMPLATE.render(
            config=config, name=fname, config_name=common.extract_config_name(config)
        )
        instances[exec_item.exec_cond] = inst
        instance_decl += inst
    shape_eval_func = gemm_common.gen_shape_eval_code(
        indent=1, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        program = EXEC_TEMPLATE.render(
            indent="    ",
            instance=fname,
            problem_args=problem_args,
            support_split_k=support_split_k,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return src_template.render(
        custom_libs=gen_custom_libs(),
        instances=instance_decl,
        function_name=func_name,
        dtype="cutlass::half_t",
        shape_eval=shape_eval_func,
        output_addr_calculator=output_addr_calculator,
        exec_paths=exec_paths,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=support_split_k,
        has_d=common.has_d(func_attrs),
        has_d1=common.has_d1(func_attrs),
        extra_code=extra_code,
    )


def gen_profiler(
    func_attrs,
    workdir,
    dim_info_dict,
    src_template,
    problem_args_template,
    args_parser_template,
    emit_kernel=False,
    support_split_k=False,
    output_addr_calculator="",
    bias_ptr_arg=None,
    extra_code="",
):
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]

    ndims = 2
    adims = ["&a_dim" + str(i) for i in range(ndims)]
    bdims = ["&b_dim" + str(i) for i in range(ndims)]
    cdims = ["&c_dim" + str(i) for i in range(ndims)]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    file_pairs = []
    has_bias = bias_ptr_arg is not None
    for op_name, op in op_instance.items():
        config = emit_instance(op, emit_kernel=emit_kernel)
        config_name = common.extract_config_name(config)
        name = "GemmInstance"
        instance = common.INSTANCE_TEMPLATE.render(
            config_name=config_name, name=name, config=config
        )
        exec_program = EXEC_TEMPLATE.render(
            indent="  ",
            instance=name,
            is_profiler=True,
            support_split_k=support_split_k,
            problem_args=problem_args_template.render(),
        )
        op_func = src_template.render(
            custom_libs=gen_custom_libs(),
            instances=instance,
            function_name="gemm",
            input_ndims=2,
            weight_ndims=2,
            shape_eval=shape_func,
            exec_paths=exec_program,
            output_addr_calculator=output_addr_calculator,
            support_split_k=support_split_k,
            extra_code=extra_code,
        )
        func_call = FUNC_CALL_TEMPLATE.render(
            func_name="gemm",
            a_ptr="memory_pool->RequestTensorByIdx<cutlass::half_t>(0)",
            b_ptr="memory_pool->RequestTensorByIdx<cutlass::half_t>(1)",
            has_bias=has_bias,
            bias_ptr=bias_ptr_arg,
            c_ptr="memory_pool->RequestTensorByIdx<cutlass::half_t>(2)",
            d_ptr="memory_pool->RequestTensorByIdx<cutlass::half_t>(3)",
            n_ptr="memory_pool->RequestTensorByIdx<float>(4)",
            soft_ptr="memory_pool->RequestTensorByIdx<cutlass::half_t>(5)",
            split_k="split_k",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
        )
        code = PROFILER_TEMPLATE.render(
            op_func=op_func,
            args_parse=args_parser_template.render(),
            func_call=func_call,
            name=name,
            tensor_decl=TENSOR_DECL_TEMPLATE.render(name=name, has_bias=has_bias),
        )
        common.add_profiler(file_pairs, workdir, op_type, op_name, code)
    # build
    return common.build_profiler(file_pairs)
