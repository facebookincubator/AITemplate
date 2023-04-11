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

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common import gemm_common
from aitemplate.backend.cuda.gemm_universal import common
from aitemplate.backend.target import Target

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

{% if is_profiler %}
template <typename {{instance_name_base}}>
void {{func_name}} (
    {{instance_name_base}}& gemm_op,
{% else %}
void {{func_name}} (
{% endif %}
    void* a_ptr,
    void* b_ptr,
{% if has_bias %}
    void* bias_ptr,
{% endif %}
    void* soft_ptr,
    uint8_t* workspace,
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
{{indent}}int block_num = (N + {{instance}}::ThreadblockShape::kN - 1) / {{instance}}::ThreadblockShape::kN;

{% if is_profiler %}
{{indent}}size_t workspace_size = 2 * M * N * sizeof({{elem_output_type}}) + 2 * block_num * M * sizeof(float);
{{indent}}cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);
{{indent}}workspace = local_workspace.get();
{{indent}}GLOBAL_WORKSPACE_SIZE = workspace_size;
{% else %}
{{indent}}{{instance}} gemm_op;
{% endif %}

{{indent}}using coord_t = cutlass::gemm::GemmCoord::Index;
{{indent}}typename {{instance}}::Arguments arguments{
{{problem_args}}
{{indent}}};

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
  void*,
  void*,
{% if has_bias %}
  void*,
{% endif %}
  void*,
  uint8_t*,
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
{% if is_profiler %}
{{indent}}    gemm_op,
{% endif %}
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{% if has_bias %}
{{indent}}    {{bias_ptr}},
{% endif %}
{{indent}}    {{soft_ptr}},
{{indent}}    global_workspace_,
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


BENCHMARK_INSTANCE_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}
{{indent}}{{instance_name}} {{gemm_op}};
{{indent}}const char *gemm_op_name = "{{gemm_op_name}}";
{{indent}}int ret = 0;
{{indent}}try {
{{indent}}ret = {{func_name}}(
{{indent}}    {{gemm_op}},
{{indent}}    gemm_op_name,
{{indent}}    memory_pool.get(),
{{indent}}    global_workspace_,
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
{{indent}}} catch (...) {}
{{indent}}if (ret != 0)
{{indent}}  return ret;
{{indent}}
{{indent}}}
"""
)


TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = a_dim0 * a_dim1;
  int64_t b_ptr_sz = b_dim0 * b_dim1;
  int64_t c_ptr_sz = c_dim0 * c_dim1;

  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b_ptr_sz, c_ptr_sz});

  size_t one_copy_sz = a_ptr_sz + b_ptr_sz + c_ptr_sz;
{% if has_bias %}
  one_copy_sz += c_dim1;
{%endif%}

  int64_t mem_pool_sz = memory_pool->ComputeMemPoolSize(one_copy_sz, ptr_max_sz);

  memory_pool->AllocateTensor(a_ptr_sz, mem_pool_sz);                      // a_ptr: index 0
  memory_pool->AllocateTensor(b_ptr_sz, mem_pool_sz);                      // b_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz, /*is_output*/ true);  // soft_ptr: index 2

{% if has_bias %}
  memory_pool->AllocateTensor(c_dim1, mem_pool_sz);                        // bias_ptr: index 3
{% endif %}
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

#include <sstream>

{{op_func}}

template <typename DType>
struct ProfilerMemoryPool;

template <typename GemmInstance>
int benchmark_{{func_name}} (
    GemmInstance &gemm_op,
    const char *gemm_op_name,
    ProfilerMemoryPool<{{elem_type}}>* memory_pool,
    uint8_t* global_workspace_,
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
  // warmup
  for (int i = 0; i < 5; ++i) {
    {{func_call}}
  }
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0], stream);
  for (int i = 0; i < 10; ++i) {
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
  std::cout << "OP:" << gemm_op_name << ",";
  std::cout << "TIME:" << runtime_ms << ",";
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
  return 0;
}

template <typename DType>
struct ProfilerMemoryPool {
  ProfilerMemoryPool() : shared_input_tensor(false) {
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

  int64_t ComputeMemPoolSize(size_t one_copy_sz, size_t ptr_max_sz) {
    // TODO: special pool size for A100 L2 cache 40M
    // need to tune it for other devices
    int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 25) / ptr_max_sz)));
    size_t free_global_mem = 0;
    size_t total_global_mem = 0;
    cudaError_t cuda_error = cudaMemGetInfo(&free_global_mem, &total_global_mem);
    if (cuda_error != cudaSuccess) {
      auto error_msg = std::string("Failed to invoke cudaMemGetInfo: ") +
          cudaGetErrorName(cuda_error) + ", at " + __FILE__;
      throw std::runtime_error(error_msg);
    }
    size_t single_copy_nbytes = one_copy_sz * sizeof(DType);
    while (mem_pool_sz > 0) {
      size_t nbytes = single_copy_nbytes * mem_pool_sz;
      if (nbytes < free_global_mem) {
        break;
      }
      mem_pool_sz--;
    }

    if (mem_pool_sz <= 1) {
      size_t minimal_required_nbytes = ptr_max_sz * sizeof(DType);
      if (minimal_required_nbytes > free_global_mem) {
        // We absolutely run out of memory
        auto error_msg = std::string("no enough GPU memory: requested ") +
            std::to_string(minimal_required_nbytes) + ", available: " +
            std::to_string(free_global_mem) + ", ptr_max_sz: " +
            std::to_string(ptr_max_sz) + ", at " + __FILE__;
        throw std::runtime_error(error_msg);
      } else {
        // Let's try to allocate a single blob that is large enough to hold
        // all input tensors. Note that this is still an approximation, because
        // we may still hit cudaErrorMemoryAllocation error while allocating
        // memory for the output. We will rely on cudaMalloc to throw out
        // an exception in such a case.
        shared_input_tensor = true;
        AllocateGaussianTensor(ptr_max_sz);
      }
      return 1;
    }
    return mem_pool_sz;
  }

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

  int AllocateTensor(int64_t size, int64_t copy, bool is_output = false) {
    offsets.push_back(0);
    strides.push_back(size);
    copies.push_back(copy);
    DType *ptr;
    if (!is_output && shared_input_tensor) {
      ptr = reinterpret_cast<DType*>(blobs.back().get());
    } else {
      ptr = AllocateGaussianTensor(size * copy);
    }
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  DType* RequestTensorByIdx(int idx) {
    auto copy = copies.at(idx);
    auto offset = offsets.at(idx);
    auto stride = strides.at(idx);
    DType* ptr = reinterpret_cast<DType*>(ptrs.at(idx));
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
  // make a shared blob to hold all inputs in cases we do not have
  // enough GPU memory
  bool shared_input_tensor;
};

int main(int argc, char** argv) {
  int device_idx;
  cudaDeviceProp device_properties;
  cudaError_t result = cudaGetDevice(&device_idx);
  auto memory_pool = std::make_unique<ProfilerMemoryPool<{{elem_type}}>>();
  if (result != cudaSuccess) {
    std::ostringstream errorStream;
    errorStream << "cudaGetDevice() call failed! "
                << "Error code: " << cudaGetErrorName(result)
                << " Error message: " << cudaGetErrorString(result);
    throw std::runtime_error(errorStream.str());
  }

  result = cudaGetDeviceProperties(&device_properties, device_idx);

  if (result != cudaSuccess) {
    std::ostringstream errorStream;
    errorStream << "cudaGetDeviceProperties() call failed! "
                << "Error code: " << cudaGetErrorName(result)
                << " Error message: " << cudaGetErrorString(result);
    throw std::runtime_error(errorStream.str());
  }

  {{args_parse}}

  uint8_t* global_workspace_ = nullptr;
  cudaStream_t stream = nullptr;

  {{tensor_decl}}

  {{benchmark_instances}}

  return 0;
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
    has_bias=False,
    f_instance_convertor=_gemm_softmax_instance,
    emit_kernel=False,
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
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return src_template.render(
        custom_libs=gen_custom_libs(),
        instances=instance_decl,
        func_name=func_name,
        shape_eval=shape_eval_func,
        output_addr_calculator=output_addr_calculator,
        exec_paths=exec_paths,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        extra_code=extra_code,
        has_bias=has_bias,
    )


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    src_template,
    problem_args_template,
    args_parser_template,
    emit_kernel=False,
    output_addr_calculator="",
    bias_ptr_arg=None,
    extra_code="",
    ndims=2,
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

    adims = ["&a_dim" + str(i) for i in range(ndims)]
    bdims = ["&b_dim" + str(i) for i in range(ndims)]
    cdims = ["&c_dim" + str(i) for i in range(ndims)]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    has_bias = bias_ptr_arg is not None
    instance_name_base = "GemmSoftmaxInstance"
    exec_program = EXEC_TEMPLATE.render(
        indent="  ",
        instance=instance_name_base,
        is_profiler=True,
        problem_args=problem_args_template.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
        ),
        elem_output_type=elem_output_type,
    )

    instances = []
    benchmark_instances = []
    func_name = "gemm_softmax"
    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        config = emit_instance(op, emit_kernel=emit_kernel)
        config_name = common.extract_config_name(config)
        instance_name = f"{instance_name_base}_{instance_idx}"
        gemm_op = f"gemm_softmax_op_{instance_idx}"
        instance = common.INSTANCE_TEMPLATE.render(
            config_name=config_name, name=instance_name, config=config
        )
        benchmark_instance = BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            instance_name=instance_name,
            gemm_op=gemm_op,
            gemm_op_name=op_name,
            func_name=f"benchmark_{func_name}",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
        )
        instances.append(instance)
        benchmark_instances.append(benchmark_instance)

    op_func = src_template.render(
        is_profiler=True,
        instances="\n".join(instances),
        func_name=func_name,
        instance_name_base=instance_name_base,
        custom_libs=gen_custom_libs(),
        has_bias=has_bias,
        input_ndims=ndims,
        weight_ndims=ndims,
        shape_eval=shape_func,
        exec_paths=exec_program,
        output_addr_calculator=output_addr_calculator,
        extra_code=extra_code,
    )
    benchmark_adims = ["a_dim" + str(i) for i in range(ndims)]
    benchmark_bdims = ["b_dim" + str(i) for i in range(ndims)]
    benchmark_cdims = ["c_dim" + str(i) for i in range(ndims)]
    func_call = FUNC_CALL_TEMPLATE.render(
        is_profiler=True,
        func_name=func_name,
        a_ptr="memory_pool->RequestTensorByIdx(0)",
        b_ptr="memory_pool->RequestTensorByIdx(1)",
        has_bias=has_bias,
        bias_ptr=bias_ptr_arg,
        soft_ptr="memory_pool->RequestTensorByIdx(2)",
        adims=benchmark_adims,
        bdims=benchmark_bdims,
        cdims=benchmark_cdims,
    )
    code = PROFILER_TEMPLATE.render(
        op_func=op_func,
        has_bias=has_bias,
        args_parse=args_parser_template.render(),
        func_name=func_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        func_call=func_call,
        name=instance_name_base,
        tensor_decl=TENSOR_DECL_TEMPLATE.render(
            has_bias=has_bias,
        ),
        benchmark_instances="\n".join(benchmark_instances),
        elem_output_type=elem_output_type,
        elem_type=elem_type,
    )

    file_pairs = []
    common.add_profiler(file_pairs, workdir, op_type, profiler_filename, code)
    return common.build_profiler(file_pairs)
