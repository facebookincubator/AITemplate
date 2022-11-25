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
Common template for gemm
"""
import os
import re
from collections import OrderedDict
from hashlib import sha1

import jinja2

from ...common import gemm_common
from ...target import Target

INPUT_ADDR_CALCULATOR = jinja2.Template(
    """
    {% if accessor_a.is_from_strided_tensor %}
      stride_a = {{accessor_a.stride(0)}};
      offset_a = {{accessor_a.offset}}; // default to 0
    {% endif %}
    {% if accessor_b.is_from_strided_tensor %}
      stride_b = {{accessor_b.stride(0)}};
      offset_b = {{accessor_b.offset}}; // default to 0
    {% endif %}
    """
)

# pylint: disable=C0103,C0415,W0611,C0301
OUTPUT_ADDR_CALCULATOR = jinja2.Template(
    """
  {% if output_accessor.is_from_strided_tensor %}
    output_stride = {{output_accessor.actual_total_elements_from_stride_dim}};
    output_offset = {{output_accessor.offset}};
  {% endif %}
    """
)

EXTRA_SHAPE_TEMPLATE = jinja2.Template(
    """
{{indent}}int64_t stride_a = *a_dim1;
{{indent}}int64_t stride_b = *b_dim1;
{{indent}}const int64_t stride_c = *c_dim1;
{{indent}}int64_t output_stride = stride_c;
"""
)


INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
using {{name}} = {{ config_name }};
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}auto op =  {{instance}}{};
{{indent}}auto invoker  = op.MakeInvoker();
{{indent}}auto argument = op.MakeArgument(
{{problem_args}}
{{indent}});
{{indent}}if(!op.IsSupportedArgument(argument)) {
{{indent}}  LOG(FATAL) << "wrong! " << op.GetTypeString() << " with the specified compilation parameters does not support this Gemm problem.";
{{indent}}}
{% if is_profiler %}
{{indent}}auto workspace_size = op.GetWorkSpaceSize(&argument);
{{indent}}GLOBAL_WORKSPACE_SIZE = workspace_size;
{% endif %}
{{indent}}invoker.Run(argument, StreamConfig{stream, false});
{{indent}}return;
"""
)


EXTRA_HEADER_TEMPLATE = jinja2.Template(
    """
{% if gemm_flag == "" %}
#include "include/ck/tensor_operation/gpu/device/device_gemm_xdl_cshuffle.hpp"
{% elif gemm_flag == "permute_m2n3" %}
#include "ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d_xdl_cshuffle.hpp"
{% elif "bias" in gemm_flag or has_d0 %}
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_xdl_cshuffle.hpp"
    {% if gemm_flag == "bias_permute" %}
#include "ck/tensor_operation/gpu/device/device_gemm_bias_e_permute_xdl.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
    {% elif gemm_flag in ["bias_permute_m2n3", "bias_permute_m3n2"]  %}
#include "ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d_xdl_cshuffle.hpp"
    {% endif %}
{% endif %}
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
// #include <half.hpp>
#include <random>
#include <rocrand/rocrand.h>
#include "logging.h"
#include "include/ck/utility/print.hpp"
#include "library/include/ck/library/utility/device_memory.hpp"
#include "library/include/ck/library/utility/host_tensor.hpp"
#include "library/include/ck/library/utility/host_tensor_generator.hpp"
#include "include/ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/tensor_operation/gpu/element/element_wise_operation.hpp"
{{extra_header}}


{{extra_code}}

{{instances}}


void {{function_name}}(
    void * in_ptr,
    void * weight_ptr,
    void * out_ptr,
{% if "bias" in gemm_flag %}
    void * bias_ptr,
{% endif %}
{% if has_d0 %}
    void * d0_ptr,
{% endif %}
{% if has_d1 %}
    void * d1_ptr,
{% endif %}
{% for idx in range(ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(ndims) %}
    int64_t* c_dim{{idx}},
{% endfor %}
{% for idx in range(pdims) %}
    const int p_dim{{idx}},
{% endfor %}
    hipStream_t stream
    ) {
  {{shape_func}}
  int64_t offset_a = 0;
  int64_t offset_b = 0;
  int64_t output_offset = 0;
  {{extra_shape}}
  {{input_addr_calculator}}
  {{output_addr_calculator}}
  {{exec_paths}}

  throw std::runtime_error(
      "Unsupported workload for this gemm specialization."
  );
}
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{weight_ptr}},
{{indent}}    {{out_ptr}},
{% if "bias" in gemm_flag %}
{{indent}}    {{bias_ptr}},
{% endif %}
{% if d0_ptr != "" %}
{{indent}}    {{d0_ptr}},
{% endif %}
{% if d1_ptr != "" %}
{{indent}}    {{d1_ptr}},
{% endif %}
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in pdims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    stream
{{indent}});
"""
)


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t *>(in_ptr) + offset_a,
{{indent}}                                static_cast<ck::half_t *>(weight_ptr) + offset_b,

{% if gemm_flag == "bias_permute" %}
{{indent}}                                static_cast<ck::half_t *>(bias_ptr),
{% elif gemm_flag == "bias_permute_m2n3" %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% elif gemm_flag == "permute_m2n3" %}
{{indent}}                                {},
{% else %}
{% if "bias" in gemm_flag and not has_d0 %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% elif has_d0 and not has_d1 %}
{{indent}}                                std::array<const void*, 2>{static_cast<ck::half_t *>(bias_ptr),
                                                                    static_cast<ck::half_t *>(d0_ptr)},
{% elif has_d1 %}
{{indent}}                                std::array<const void*, 3>{static_cast<ck::half_t *>(bias_ptr),
                                                                    static_cast<ck::half_t *>(d0_ptr),
                                                                    static_cast<ck::half_t *>(d1_ptr)},
{% endif %}
{% endif %}
{{indent}}                                static_cast<ck::half_t *>(out_ptr) + output_offset,
{% if gemm_flag not in ["permute_m2n3", "bias_permute_m2n3", "bias_permute_m3n2"]  %}
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{% endif %}
{% if gemm_flag == "bias_permute" %}
{{indent}}                                {M0, M1, M2, N0, N1, stride_D_M0, stride_D_M1, stride_D_M2, stride_D_N0, stride_D_N1},
{{indent}}                                {M0, M1, M2, N0, N1, stride_E_M0, stride_E_M1, stride_E_M2, stride_E_N0, stride_E_N1},
{% elif gemm_flag in ["permute_m2n3", "bias_permute_m2n3", "bias_permute_m3n2"]  %}
{{indent}}                                a_ms_ks_lengths,
{{indent}}                                a_ms_ks_strides,
{{indent}}                                b_ns_ks_lengths,
{{indent}}                                b_ns_ks_strides,
    {% if gemm_flag == "permute_m2n3"  %}
{{indent}}                                {},
{{indent}}                                {},
    {% else %}
{{indent}}                                std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
{{indent}}                                std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
    {% endif %}
{{indent}}                                e_ms_ns_lengths,
{{indent}}                                e_ms_ns_strides,
{% else %}
{% if "bias" in gemm_flag and not has_d0 %}
{{indent}}                                std::array<ck::index_t, 1>{0},
{% elif has_d0 and not has_d1 %}
{{indent}}                                std::array<ck::index_t, 2>{0, static_cast<int>(stride_c)},
{% elif has_d1 %}
{{indent}}                                std::array<ck::index_t, 3>{0, static_cast<int>(stride_c), static_cast<int>(stride_c)},
{% endif %}
{{indent}}                                output_stride,
{% endif %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{% if gemm_flag == "" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag == "permute_m2n3" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag == "bias" or "bias_permute" in gemm_flag %}
{{indent}}                                ck::tensor_operation::element_wise::Add{}
{% elif gemm_flag == "bias_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddRelu{}
{% elif gemm_flag == "bias_fast_gelu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddFastGelu{}
{% elif gemm_flag == "bias_swish" %}
{{indent}}                                ck::tensor_operation::element_wise::AddHardswish{}
{% elif gemm_flag == "bias_tanh" %}
{{indent}}                                ck::tensor_operation::element_wise::AddTanh{}
{% elif gemm_flag == "bias_sigmoid" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoid{}
{% elif gemm_flag == "bias_add" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAdd{}
{% elif gemm_flag == "bias_mul" %}
{{indent}}                                ck::tensor_operation::element_wise::AddMul{}
{% elif gemm_flag == "bias_mul_tanh" %}
{{indent}}                                ck::tensor_operation::element_wise::AddMulTanh{}
{% elif gemm_flag == "bias_add_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddRelu{}
{% elif gemm_flag == "bias_add_fast_gelu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddFastGelu{}
{% elif gemm_flag == "bias_sigmoid_mul" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoidMul{}
{% elif gemm_flag == "bias_sigmoid_mul_tanh" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoidMulTanh{}
{% elif gemm_flag == "bias_mul_add" %}
{{indent}}                                ck::tensor_operation::element_wise::AddMulAdd{}
{% elif gemm_flag == "bias_add_add" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddAdd{}
{% elif gemm_flag == "bias_add_add_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddAddRelu{}
{% endif %}
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = M*K;
  int64_t b_ptr_sz = N*K;
  int64_t c_ptr_sz = M*N;
  int64_t ptr_max_sz = std::max({a_ptr_sz, b_ptr_sz, c_ptr_sz});
  // TODO: special pool size for 8M L2 cache
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_max_sz)));

  memory_pool->AllocateHalfTensor(a_ptr_sz, mem_pool_sz);  // x: index 0
  memory_pool->AllocateHalfTensor(b_ptr_sz, mem_pool_sz);  // w: index 1
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // y: index 2
{% if "bias" in gemm_flag %}
  memory_pool->AllocateHalfTensor(N, mem_pool_sz);  // b: index 3
{% endif %}
{% if has_d0 %}
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // d0 ptr: index 4
{% endif %}
{% if has_d1 %}
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // d1 ptr: index 5
{% endif %}
"""
)

STRUCTS_DEF_TEMPLATE = jinja2.Template(
    """

struct ProfilerMemoryPool {
  ProfilerMemoryPool() {
    std::random_device rd;
    gen = std::mt19937(rd());
    uniform_dist = std::uniform_int_distribution<int64_t>(1, 48964896);
    offsets.reserve(512);
    strides.reserve(512);
    copies.reserve(512);
    ptrs.reserve(512);
  }
  ~ProfilerMemoryPool() {
    for(int i = 0; i < ptrs.size(); i++){
      hipFree(ptrs[i]);
    }
  }

  template <typename DType>
  DType* AllocateGaussianTensor(int64_t size) {
    size_t length = size * sizeof(DType);
    DType *d_x;
    hipMalloc(&d_x, length);

    float mean = 0.0f;
    float stddev = 1.0f;
    uint64_t seed = uniform_dist(gen);
    rocrand_set_seed(generator, seed);
    rocrand_generate_normal(generator, reinterpret_cast<float*>(d_x), size, mean, stddev);
    return d_x;
  }

  ck::half_t* AllocateHalfGaussianTensor(int64_t size) {
    return reinterpret_cast<ck::half_t*>(
        AllocateGaussianTensor<ck::half_t>(size));
  }

  int AllocateHalfTensor(int64_t size, int64_t copy) {
    offsets.push_back(0);
    strides.push_back(size);
    copies.push_back(copy);
    auto ptr = AllocateHalfGaussianTensor(size * copy);
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  ck::half_t* RequestHalfTensorByIdx(int idx) {
    auto copy = copies.at(idx);
    auto offset = offsets.at(idx);
    auto stride = strides.at(idx);
    ck::half_t* ptr = reinterpret_cast<ck::half_t*>(ptrs.at(idx));
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
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> uniform_dist;
  rocrand_generator generator;
};

// hack for DeviceMem linking error
// TODO fix this by making CK a header-only lib
// <<< hack begin
DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
  hipGetErrorString(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}
void* DeviceMem::GetDeviceBuffer() const { return mpDeviceBuf; }
void DeviceMem::ToDevice(const void* p) const
{
  hipGetErrorString(
        hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}
void DeviceMem::FromDevice(void* p) const
{
  hipGetErrorString(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}
DeviceMem::~DeviceMem() { hipGetErrorString(hipFree(mpDeviceBuf)); }
struct KernelTimerImpl
{
  KernelTimerImpl() {
    hipGetErrorString(hipEventCreate(&mStart));
    hipGetErrorString(hipEventCreate(&mEnd));
  }
  ~KernelTimerImpl() {
    hipGetErrorString(hipEventDestroy(mStart));
    hipGetErrorString(hipEventDestroy(mEnd));
  }
  void Start() {
    hipGetErrorString(hipDeviceSynchronize());
    hipGetErrorString(hipEventRecord(mStart, nullptr));
  }
  void End() {
    hipGetErrorString(hipEventRecord(mEnd, nullptr));
    hipGetErrorString(hipEventSynchronize(mEnd));
  }
  float GetElapsedTime() const {
    float time;
    hipGetErrorString(hipEventElapsedTime(&time, mStart, mEnd));
    return time;
  }
  hipEvent_t mStart, mEnd;
};
// >>> hack end

"""
)

PROFILER_TEMPLATE = jinja2.Template(
    """
size_t GLOBAL_WORKSPACE_SIZE = 0;
{{op_func}}

{{structs_def}}

int main(int argc, char** argv) {
  if (argc < 4) {
    throw std::runtime_error("wrong params");
  }
  {{args_parse}}
  auto memory_pool = std::make_unique<ProfilerMemoryPool>();
  hipStream_t stream = nullptr;
  {{tensor_decl}}
  // TODO: random init
  // warmup
  for(int i = 0; i < 3; ++i) {
    {{func_call}}
  }
  // run
  auto timer = new KernelTimerImpl();
  timer->Start();
  for(int i = 0; i < 5; ++i) {
    {{func_call}}
  }
  timer->End();
  std::cout << "OP:" << "{{op_name}}" << ",";
  std::cout << "TIME:" << timer->GetElapsedTime() << ",";
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
  delete(timer);
}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void *,
  void *,
  void *,
{% if "bias" in gemm_flag %}
  void *,
{% endif %}
{% if has_d0 %}
  void *,
{% endif %}
{% if has_d1 %}
  void *,
{% endif %}
{% for idx in range(ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(pdims) %}
  const int,
{% endfor %}
  hipStream_t
);
"""
)


def has_d0(func_attrs):
    return func_attrs.get("num_sources", 0) >= 1


def has_d1(func_attrs):
    return func_attrs.get("num_sources", 0) >= 2


def emit_instance(op):
    """Emit instance."""
    import ck_lib  # noqa: F401

    op_def = op.emit()
    return op_def


def extract_config(op_kind, extra_kind, f_proc_op):
    """Extract (operation name, operation instance) pair
    from all operation candidates.

    Parameters
    ----------
    op_kind : ck_lib.library.OperationKind
        Operation kind.
    extra_kind : ck_lib.library.[AnyKind]
        Used to as extra flag to distinguish kernels.
        E.g. bias_add_relu vs. add_relu_bias
    f_prop_op: function
        Used to filter operation.

    Returns
    -------
    Dict
        Extracted (operation name, operation instance) pair.
    """
    gemm_ops = OrderedDict()
    extract_ops = list(Target.current()._operators[op_kind][extra_kind].items())

    for key, value in extract_ops:
        op = value[0]
        ret = f_proc_op(op)
        if len(ret) > 0:
            for op_inst in ret:
                gemm_ops[key] = op_inst
    return gemm_ops


def extract_config_name(config):
    """Exract name from the statement, e.g. 'model' for 'using model = xxx'.

    Parameters
    ----------
    config : str
        Configuration as a string in the format of 'using model = xxx'.

    Returns
    -------
    str
        Extracted name from the statement.

    Raises
    ------
    RuntimeError
        Invalid config.
    """
    pattern = re.compile(r"\s*using\s(.*?)\s=")
    decl = config.split("\n")[1]
    match = pattern.match(decl)
    if match is None:
        raise RuntimeError("Invalid config: \n" + config)
    return match.groups()[0]


def gen_profiler(
    func_attrs,
    workdir,
    dim_info_dict,
    args_parse,
    gemm_flag,
    extra_code="",
    ndims=2,
    extra_shape_template=EXTRA_SHAPE_TEMPLATE,
    problem_args_template=PROBLEM_ARGS_TEMPLATE,
    extra_header_template=EXTRA_HEADER_TEMPLATE,
    tensor_decl_template=TENSOR_DECL_TEMPLATE,
):
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    args_parse: str
        Profiler input argument parser.
    gemm_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu','bias_sigmoid','bias_add_relu'.
    extra_code : str
        Extra code for self-defined operators.
    ndims : int
        Number of dims for each parameter, 2 for gemm, 3 for bmm
    extra_shape_template: jinja2.Template
        Shape evaluation template.
    problem_args_template: jinja2.Template
        Problem args template for profiler.
    extra_header_template: jinja2.Template
        Extra header template as we have different headers for gemm and bmm.
    tensor_decl_template: jinja2.Template
        Tensor declaration template.
    """
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    # shape function
    op_func_shape = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    adims = ["&a_dim" + str(i) for i in range(ndims)]
    bdims = ["&b_dim" + str(i) for i in range(ndims)]
    cdims = ["&c_dim" + str(i) for i in range(ndims)]
    pdims = []
    if func_attrs.get("shape") is not None:
        pdims = ["p_dim" + str(i) for i in range(len(func_attrs["shape"]))]
    extra_shape_func = extra_shape_template.render(indent="  ")
    file_pairs = []
    has_d0_flag = has_d0(func_attrs)
    has_d1_flag = has_d1(func_attrs)
    for op_name, op in op_instance.items():
        config = emit_instance(op)
        config_name = extract_config_name(config)
        instance = INSTANCE_TEMPLATE.render(
            name="DeviceGemmInstance", config_name=config_name, config=config
        )
        problem_args = problem_args_template.render(
            indent="  ",
            gemm_flag=gemm_flag,
            has_d0=has_d0_flag,
            has_d1=has_d1_flag,
        )
        exec_program = EXEC_TEMPLATE.render(
            indent="  ",
            instance="DeviceGemmInstance",
            problem_args=problem_args,
            is_profiler=True,
        )
        extra_header = extra_header_template.render(
            gemm_flag=gemm_flag, has_d0=has_d0_flag
        )
        op_func = SRC_TEMPLATE.render(
            instances=instance,
            function_name="gemm",
            ndims=ndims,
            pdims=len(pdims),
            has_d0=has_d0_flag,
            has_d1=has_d1_flag,
            shape_func=op_func_shape,
            extra_shape=extra_shape_func,
            exec_paths=exec_program,
            extra_code=extra_code,
            gemm_flag=gemm_flag,
            extra_header=extra_header,
        )
        structs_def = STRUCTS_DEF_TEMPLATE.render()
        tensor_decl = tensor_decl_template.render(
            gemm_flag=gemm_flag, has_d0=has_d0_flag, has_d1=has_d1_flag
        )
        func_call = FUNC_CALL_TEMPLATE.render(
            func_name="gemm",
            in_ptr="(void *) memory_pool->RequestHalfTensorByIdx(0)",
            weight_ptr="(void *) memory_pool->RequestHalfTensorByIdx(1)",
            out_ptr="(void *) memory_pool->RequestHalfTensorByIdx(2)",
            bias_ptr="(void *) memory_pool->RequestHalfTensorByIdx(3)",
            d0_ptr="(void *) memory_pool->RequestHalfTensorByIdx(4)"
            if has_d0_flag
            else "",
            d1_ptr="(void *) memory_pool->RequestHalfTensorByIdx(5)"
            if has_d1_flag
            else "",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
            pdims=pdims,
            gemm_flag=gemm_flag,
        )

        code = PROFILER_TEMPLATE.render(
            structs_def=structs_def,
            op_func=op_func,
            args_parse=args_parse,
            tensor_decl=tensor_decl,
            func_call=func_call,
            op_name=op_name,
        )
        prefix = os.path.join(workdir, "profiler", op_type)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        src_path = os.path.join(prefix, op_name + ".cpp")
        obj_path = os.path.join(prefix, op_name)
        if os.path.exists(obj_path):
            continue
        with open(src_path, "w") as fo:
            fo.write(code)
        file_pairs.append((src_path, obj_path))
    return file_pairs


def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    gemm_flag,
    extra_code="",
    ndims=2,
    extra_shape_template=EXTRA_SHAPE_TEMPLATE,
    problem_args_template=PROBLEM_ARGS_TEMPLATE,
    extra_header_template=EXTRA_HEADER_TEMPLATE,
    input_addr_calculator="",
    output_addr_calculator="",
):
    """Generate function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    gemm_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu','bias_add_relu'.
    extra_code : str
        Extra code for self-defined operators.
    ndims : int
        Number of dims for each parameter, 2 for gemm, 3 for bmm.
    extra_shape_template: jinja2.Template
        Shape evaluation template.
    extra_header_template: jinja2.Template
        Extra header template as we have different headers for gemm and bmm.
    input_addr_calculator : str
        Used to adjust input address based on input tensor accessors if accessors exist
    output_addr_calculator : str
        Used to adjust output address based on output tensor accessors if accessors exist


    Returns
    -------
    str
        The rendered template of generated function body.
    """
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]

    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    has_d0_flag = has_d0(func_attrs)
    has_d1_flag = has_d1(func_attrs)
    for key, value in exec_path.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        algo = value.algo
        if algo not in inst_def_flag:
            config = emit_instance(op_instance[algo])
            inst_def_flag.add(algo)
        else:
            config = ""
        inst = INSTANCE_TEMPLATE.render(
            config=config, name=fname, config_name=extract_config_name(config)
        )
        instances[key] = inst
        instance_decl += inst

    extra_shape_func = extra_shape_template.render(indent="  ")
    shape_eval_func = gemm_common.gen_shape_eval_code(
        indent=1, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        problem_args = problem_args_template.render(
            indent="    ",
            gemm_flag=gemm_flag,
            has_d0=has_d0_flag,
            has_d1=has_d1_flag,
        )
        program = EXEC_TEMPLATE.render(
            indent="    ",
            instance=fname,
            problem_args=problem_args,
            is_profiler=False,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    extra_header = extra_header_template.render(
        gemm_flag=gemm_flag, has_d0=has_d0(func_attrs)
    )
    pdims = len(func_attrs["shape"]) if func_attrs.get("shape") is not None else 0
    return SRC_TEMPLATE.render(
        instances=instance_decl,
        function_name=func_name,
        shape_func=shape_eval_func,
        extra_shape=extra_shape_func,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=output_addr_calculator,
        exec_paths=exec_paths,
        extra_code=extra_code,
        extra_header=extra_header,
        gemm_flag=gemm_flag,
        ndims=ndims,
        pdims=pdims,
        has_d0=has_d0_flag,
        has_d1=has_d1_flag,
    )


def gen_function_decl(func_name, gemm_flag, ndims=2, pdims=0, has_d0="", has_d1=""):
    """Generates function declarations.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    gemm_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu'.
    ndims : int
        Number of dims for each parameter, 2 for gemm, 3 for bmm.

    Returns
    -------
    str
        The rentered template of function declaration.
    """
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        gemm_flag=gemm_flag,
        ndims=ndims,
        pdims=pdims,
        has_d0=has_d0,
        has_d1=has_d1,
    )


def gen_function_call(func_attrs, indent="  ", gemm_flag=""):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict
        Stores the operation attributes.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".
    gemm_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu'.

    Returns
    -------
    str
        The rendered template of generated function call.
    """
    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    c = func_attrs["outputs"][0]
    bias_ptr = ""
    if "bias" in gemm_flag:
        bias = func_attrs["inputs"][2]
        bias_ptr = bias._attrs["name"]
    d0_ptr = ""
    if has_d0(func_attrs):
        d0 = func_attrs["inputs"][3]
        d0_ptr = d0._attrs["name"]
    d1_ptr = ""
    if has_d1(func_attrs):
        d1 = func_attrs["inputs"][4]
        d1_ptr = d1._attrs["name"]
    adims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["input_accessors"][0].original_shapes
    ]
    bdims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["input_accessors"][1].original_shapes
    ]
    cdims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["output_accessors"][0].original_shapes
    ]
    pdims = []
    if func_attrs.get("shape") is not None:
        pdims = list(func_attrs["shape"])

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=a._attrs["name"],
        weight_ptr=b._attrs["name"],
        out_ptr=c._attrs["name"],
        bias_ptr=bias_ptr,
        d0_ptr=d0_ptr,
        d1_ptr=d1_ptr,
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        pdims=pdims,
        indent=indent,
        gemm_flag=gemm_flag,
    )


def default_fproc_f16(*, op, a_layout, b_layout, c_layout):
    """Filter the input operation by layouts.

    Parameters
    ----------
    op: operation
        aitemplate operation
    a_layout: ck_lib.library.LayoutType
        a layout type.
    b_layout: ck_lib.library.LayoutType
        b layout type.
    c_layout: ck_lib.library.LayoutType
        c layout type.
    Returns
    -------
    List
        List of filtered op (can be empty).
    """
    import copy

    import ck_lib

    ret = []
    data_type = ck_lib.library.DataType.f16
    acc_type = ck_lib.library.DataType.f32
    if (
        op.A.element == data_type
        and op.B.element == data_type
        and op.C.element == data_type
        and op.accumulator_type() == acc_type
        and op.A.layout == a_layout
        and op.B.layout == b_layout
        and op.C.layout == c_layout
    ):
        ret += [copy.deepcopy(op)]
    return ret


def make_fproc_f16(func_attrs, layout, op_kind, extra_kind):
    """This function sets a callback for processing the epilogue of the kernel
    associated with func_attrs.

    Parameters
    ----------
    func_attrs: Dictionary
        kernel attributes dictionary
    layout: layout object
        kernel layout
    op_kind : ck_lib.library.OperationKind
        Operation kind.
    extra_kind : ck_lib.library.[AnyKind]
        Used to as extra flag to distinguish kernels.
        E.g. bias_add_relu vs. add_relu_bias
    """

    def fproc_f16(op):
        a_layout, b_layout, c_layout = layout.ck_lib_layouts()
        return default_fproc_f16(
            op=op,
            a_layout=a_layout,
            b_layout=b_layout,
            c_layout=c_layout,
        )

    func_attrs["op_instance"] = extract_config(op_kind, extra_kind, fproc_f16)
