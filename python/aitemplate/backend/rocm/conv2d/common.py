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
Common ROCM template for conv2d.
"""
import os
import re
from collections import OrderedDict
from hashlib import sha1

import jinja2

from aitemplate.backend.target import Target

# pylint: disable=C0103,C0415,W0611,C0301


INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
using {{name}} = {{ config_name }};
"""
)
PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t *>(in_ptr),
{{indent}}                                static_cast<ck::half_t *>(weight_ptr),

{% if conv2d_flag == "" %}
{{indent}}                                {},
{% elif conv2d_flag in ["bias", "bias_relu", "bias_sigmoid"] %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% elif conv2d_flag in ["bias_add_relu", "bias_add_identity"] %}
{{indent}}                                std::array<const void*, 2>{static_cast<ck::half_t *>(bias_ptr), static_cast<ck::half_t *>(res_ptr)},
{% endif %}
{{indent}}                                static_cast<ck::half_t *>(out_ptr),
{{indent}}                                a_g_n_c_wis_lengths,
{{indent}}                                a_g_n_c_wis_strides,
{{indent}}                                b_g_k_c_xs_lengths,
{{indent}}                                b_g_k_c_xs_strides,
{% if conv2d_flag == "" %}
{{indent}}                                {}, {},
{% elif conv2d_flag in ["bias", "bias_relu", "bias_sigmoid"] %}
{{indent}}                                std::array<std::array<ck::index_t, NDimSpatial + 3>, 1>{ {d_g_n_k_wos_lengths} },
{{indent}}                                std::array<std::array<ck::index_t, NDimSpatial + 3>, 1>{ {d_g_n_k_wos_strides} },
{% elif conv2d_flag in ["bias_add_relu", "bias_add_identity"] %}
{{indent}}                                std::array<std::array<ck::index_t, NDimSpatial + 3>, 2>{ {d_g_n_k_wos_lengths, e_g_n_k_wos_lengths} },
{{indent}}                                std::array<std::array<ck::index_t, NDimSpatial + 3>, 2>{ {d_g_n_k_wos_strides, e_g_n_k_wos_strides} },
{% endif %}
{{indent}}                                e_g_n_k_wos_lengths,
{{indent}}                                e_g_n_k_wos_strides,
{{indent}}                                conv_filter_strides,
{{indent}}                                conv_filter_dilations,
{{indent}}                                input_left_pads,
{{indent}}                                input_right_pads,
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{% if conv2d_flag == "" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif conv2d_flag == "bias" %}
{{indent}}                                ck::tensor_operation::element_wise::Add{}
{% elif conv2d_flag == "bias_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddRelu{}
{% elif conv2d_flag == "bias_sigmoid" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoid{}
{% elif conv2d_flag == "bias_add_identity" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAdd{}
{% elif conv2d_flag == "bias_add_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddRelu{}
{% endif %}
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
{{indent}}  LOG(FATAL) << "wrong! " << op.GetTypeString() << " with the specified compilation parameters does not support this Conv problem.";
{{indent}}}
{% if is_profiler %}
{{indent}}auto workspace_size = op.GetWorkSpaceSize(&argument);
{{indent}}GLOBAL_WORKSPACE_SIZE = workspace_size;
{% endif %}
{{indent}}invoker.Run(argument, StreamConfig{stream, false});
{{indent}}return;
"""
)

HEADER_CODE = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_xdl_cshuffle.hpp"
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


{{extra_code}}

{{instances}}


void {{function_name}}(
    void * in_ptr,
    void * weight_ptr,
    void * out_ptr,
{% if "bias" in conv2d_flag %}
    void * bias_ptr,
{% endif %}
{% if conv2d_flag in ["bias_add_relu", "bias_add_identity"] %}
    void * res_ptr,
{% endif %}
    int64_t* batch,
    int64_t* out_ch,
    int64_t* in_ch,
    int64_t* kernel_h,
    int64_t* kernel_w,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    int64_t stride,
    int64_t dilation,
    int64_t pad,
    int64_t group,
    hipStream_t stream
    ) {
  {{shape_function}}

  int C_ = (*in_ch)  / group;
  int K_ = (*out_ch) / group;
  int N_ = (*batch);
  const int NDimSpatial = 2;

  // input
  std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{
    static_cast<ck::index_t>(group),
    static_cast<ck::index_t>(N_),
    static_cast<ck::index_t>(C_),
    static_cast<ck::index_t>(*in_h),
    static_cast<ck::index_t>(*in_w)
  };

  std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{
    static_cast<ck::index_t>(C_ ),
    static_cast<ck::index_t>((*in_h) * (*in_w) * group * C_),
    static_cast<ck::index_t>(1),
    static_cast<ck::index_t>((*in_w) * group * C_),
    static_cast<ck::index_t>(group * C_)
  };

  // weight
  std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{
    static_cast<ck::index_t>(group),
    static_cast<ck::index_t>(K_), // K
    static_cast<ck::index_t>(C_),
    static_cast<ck::index_t>(*kernel_h),
    static_cast<ck::index_t>(*kernel_w)
  };
  std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{
    static_cast<ck::index_t>((*kernel_h) * (*kernel_w)  * C_ * K_),
    static_cast<ck::index_t>((*kernel_h) * (*kernel_w)  * C_),
    static_cast<ck::index_t>(1),
    static_cast<ck::index_t>((*kernel_w)  * C_),
    static_cast<ck::index_t>(C_)
  };

  // bias
  std::array<ck::index_t, NDimSpatial + 3> d_g_n_k_wos_lengths{
    static_cast<ck::index_t>(group),
    static_cast<ck::index_t>(N_), // N
    static_cast<ck::index_t>(K_), // K
    static_cast<ck::index_t>(*out_h),
    static_cast<ck::index_t>(*out_w)
  };
  std::array<ck::index_t, NDimSpatial + 3> d_g_n_k_wos_strides{
    static_cast<ck::index_t>(K_), 0, 1, 0, 0};

  // output
  std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{
    static_cast<ck::index_t>(group),
    static_cast<ck::index_t>(N_),
    static_cast<ck::index_t>(K_),
    static_cast<ck::index_t>(*out_h),
    static_cast<ck::index_t>(*out_w)
  };

  std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{
    static_cast<ck::index_t>(K_),
    static_cast<ck::index_t>((*out_h) * (*out_w) * group * K_),
    static_cast<ck::index_t>(1),
    static_cast<ck::index_t>((*out_w) * group * K_),
    static_cast<ck::index_t>(group * K_)
  };

  std::array<ck::index_t, NDimSpatial> conv_filter_strides{static_cast<ck::index_t>(stride), static_cast<ck::index_t>(stride)};
  std::array<ck::index_t, NDimSpatial> conv_filter_dilations{static_cast<ck::index_t>(dilation), static_cast<ck::index_t>(dilation)};
  std::array<ck::index_t, NDimSpatial> input_left_pads{static_cast<ck::index_t>(pad), static_cast<ck::index_t>(pad)};
  std::array<ck::index_t, NDimSpatial> input_right_pads{static_cast<ck::index_t>(pad), static_cast<ck::index_t>(pad)};


  {{exec_paths}}

  throw std::runtime_error(
      "Unsupported workload for this conv2d specialization."
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
{% if "bias" in conv2d_flag %}
{{indent}}    {{bias_ptr}},
{% endif %}
{% if conv2d_flag in ["bias_add_relu", "bias_add_identity"] %}
{{indent}}    {{res_ptr}},
{% endif %}
{{indent}}    {{p_batch}},
{{indent}}    {{p_out_ch}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_kernel_h}},
{{indent}}    {{p_kernel_w}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    {{stride}},
{{indent}}    {{dilation}},
{{indent}}    {{pad}},
{{indent}}    {{group}},
{{indent}}    stream
{{indent}});
"""
)

ARGS_PARSE_TEMPLATE = jinja2.Template(
    """
  const int64_t batch = std::stoi(argv[1]);
  const int64_t in_h = std::stoi(argv[2]);
  const int64_t in_w = std::stoi(argv[3]);
  const int64_t in_ch = std::stoi(argv[4]);
  const int64_t kernel_h = std::stoi(argv[5]);
  const int64_t kernel_w = std::stoi(argv[6]);
  const int64_t out_ch = std::stoi(argv[7]);
  const int64_t stride = std::stoi(argv[8]);
  const int64_t pad = std::stoi(argv[9]);
  const int64_t dilation = std::stoi(argv[10]);
  const int64_t group = std::stoi(argv[11]);
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = NI * HI * WI * CI;
  int64_t b_ptr_sz = CO * KH * KW * CI  / group;
  int64_t c_ptr_sz = NO * HO * WO * CO;
  int64_t ptr_max_sz = std::max({a_ptr_sz, b_ptr_sz, c_ptr_sz});
  // TODO: special pool size for 8M L2 cache
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_max_sz)));

  memory_pool->AllocateHalfTensor(a_ptr_sz, mem_pool_sz);  // x: index 0
  memory_pool->AllocateHalfTensor(b_ptr_sz, mem_pool_sz);  // w: index 1
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // y: index 2
{% if "bias" in conv2d_flag %}
  memory_pool->AllocateHalfTensor(CO, 8);  // b: index 3
{% endif %}
{% if conv2d_flag in ["bias_add_relu", "bias_add_identity"] %}
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // r: index 4
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
      if (hipFree(&ptrs[i]) != hipSuccess) {
      // ...
      }
    }
  }

  template <typename DType>
  DType* AllocateGaussianTensor(int64_t size) {
    size_t length = size * sizeof(DType);
    DType *d_x;
    //hipMalloc(&d_x, length);
    if (hipMalloc(&d_x, length) != hipSuccess) {
      throw std::runtime_error(
          " ROCMWorkspace: hipMalloc( " + std::to_string(length) +
          " ) failed.");
    }

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
  if (argc < 10) {
    throw std::runtime_error("wrong params");
  }
  {{args_parse}}
  {{shape_func}}
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
{% if "bias" in conv2d_flag %}
  void *,
{% endif %}
{% if conv2d_flag in ["bias_add_relu", "bias_add_identity"] %}
  void *,
{% endif %}
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  hipStream_t stream
);
"""
)


def emit_instance(op):
    """Emits instance."""
    import ck_lib  # noqa: F401

    op_def = op.emit()
    return op_def


def extract_config(op_kind, extra_kind):
    """
    Parameters
    ----------
    op_kind : ck_lib.library.OperationKind
        Operation kind.
    extra_kind : ck_lib.library.[AnyKind]
        Used to as extra flag to distinguish kernels.
        E.g. bias_add_relu vs. add_relu_bias

    Returns
    -------
    Dict
        Extracted (operation name, operation instance) pair
        from all operation candidates.
    """
    import ck_lib  # noqa: F401

    conv2d_ops = OrderedDict()
    extract_ops = list(Target.current()._operators[op_kind][extra_kind].items())
    for key, value in extract_ops:
        conv2d_ops[key] = value[0]
    return conv2d_ops


def extract_config_name(config):
    """
    Parameters
    ----------
    config : str
        Configuration as a string in the format of 'using model = xxx'.

    Returns
    -------
    str
        Extracted name from the statement, e.g. 'model' for 'using model = xxx'.

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
    shape_template,
    conv2d_flag,
    extra_code="",
    src_template=SRC_TEMPLATE,
    prob_args_template=PROBLEM_ARGS_TEMPLATE,
):
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    shape_template : jinja2.Template
        Generates shape calculation.
        The template is passed from compiler/ops/pool.
    conv2d_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu','bias_add_relu','bias_add_identity'.
    extra_code : str
        Extra code for self-defined operators.
    """
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    # shape function
    shape_func = shape_template.render(
        indent="  ",
        dtype="int64_t ",
        div="/",
        x_dim0="batch",
        x_dim1="in_h",
        x_dim2="in_w",
        x_dim3="in_ch",
        w_dim0="out_ch",
        w_dim1="kernel_h",
        w_dim2="kernel_w",
        strideh="stride",
        dilateh="dilation",
        padh="pad",
        stridew="stride",
        dilatew="dilation",
        padw="pad",
    )
    file_pairs = []
    for op_name, op in op_instance.items():
        config = emit_instance(op)
        config_name = extract_config_name(config)
        instance = INSTANCE_TEMPLATE.render(
            name="DeviceConvFwdInstance", config_name=config_name, config=config
        )
        problem_args = prob_args_template.render(
            indent="  ",
            conv2d_flag=conv2d_flag,
        )
        exec_program = EXEC_TEMPLATE.render(
            indent="  ",
            instance="DeviceConvFwdInstance",
            problem_args=problem_args,
            is_profiler=True,
        )
        op_func = src_template.render(
            instances=instance,
            function_name="conv",
            shape_func="",
            exec_paths=exec_program,
            extra_code=extra_code,
            conv2d_flag=conv2d_flag,
        )
        structs_def = STRUCTS_DEF_TEMPLATE.render()
        args_parse = ARGS_PARSE_TEMPLATE.render()
        tensor_decl = TENSOR_DECL_TEMPLATE.render(conv2d_flag=conv2d_flag)
        func_call = FUNC_CALL_TEMPLATE.render(
            func_name="conv",
            in_ptr="(void *) memory_pool->RequestHalfTensorByIdx(0)",
            weight_ptr="(void *) memory_pool->RequestHalfTensorByIdx(1)",
            out_ptr="(void *) memory_pool->RequestHalfTensorByIdx(2)",
            bias_ptr="(void *) memory_pool->RequestHalfTensorByIdx(3)",
            res_ptr="(void *) memory_pool->RequestHalfTensorByIdx(4)",
            p_batch="&NI",
            p_out_ch="&CO",
            p_in_ch="&CI",
            p_kernel_h="&KH",
            p_kernel_w="&KW",
            p_in_h="&HI",
            p_in_w="&WI",
            p_out_batch="&NO",
            p_out_h="&HO",
            p_out_w="&WO",
            stride="SH",
            dilation="DH",
            pad="PH",
            group="group",
            conv2d_flag=conv2d_flag,
        )

        code = PROFILER_TEMPLATE.render(
            structs_def=structs_def,
            op_func=op_func,
            shape_func=shape_func,
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
    shape_eval_template,
    shape_save_template,
    conv2d_flag,
    extra_code="",
    src_template=SRC_TEMPLATE,
    prob_args_template=PROBLEM_ARGS_TEMPLATE,
):
    """
    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    shape_eval_template : jinja2.Template
        Generates shape calculation.
        The template is passed from compiler/ops/pool.
    shape_save_template : jinja2.Template
        Generates output dimensions.
        The template is passed from compiler/ops/pool.
    conv2d_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu','bias_add_relu','bias_add_identity'.
    extra_code : str
        Extra code for self-defined operators.


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
    for key, value in exec_path.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        if value not in inst_def_flag:
            config = emit_instance(op_instance[value])
            inst_def_flag.add(value)
        else:
            config = ""
        inst = INSTANCE_TEMPLATE.render(
            config=config, name=fname, config_name=extract_config_name(config)
        )
        instances[key] = inst
        instance_decl += inst

    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
        x_dim3="*in_ch",
        w_dim0="*out_ch",
        w_dim1="*kernel_h",
        w_dim2="*kernel_w",
        strideh="stride",
        dilateh="dilation",
        padh="pad",
        stridew="stride",
        dilatew="dilation",
        padw="pad",
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
        y_dim3="*out_ch",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        problem_args = prob_args_template.render(
            indent="    ",
            conv2d_flag=conv2d_flag,
        )
        program = EXEC_TEMPLATE.render(
            indent="    ",
            instance=fname,
            problem_args=problem_args,
            is_profiler=False,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return src_template.render(
        instances=instance_decl,
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
        extra_code=extra_code,
        conv2d_flag=conv2d_flag,
    )


def gen_function_decl(func_name, conv2d_flag):
    """Generates function declarations.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    conv2d_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu','bias_add_relu','bias_add_identity'.

    Returns
    -------
    str
        The rentered template of function declaration.
    """
    return FUNC_DECL_TEMPLATE.render(func_name=func_name, conv2d_flag=conv2d_flag)


def gen_function_call(func_attrs, indent="  ", conv2d_flag=""):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict
        Stores the operation attributes.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".

    Returns
    -------
    str
        The rendered template of generated function call.
    """
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    w = func_attrs["inputs"][1]
    wshape = w._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    bias_ptr = ""
    res_ptr = ""
    if "bias" in conv2d_flag:
        b = func_attrs["inputs"][2]
        bias_ptr = b._attrs["name"]
    if conv2d_flag in ["bias_add_relu", "bias_add_identity"]:
        r = func_attrs["inputs"][3]
        res_ptr = r._attrs["name"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        weight_ptr=w._attrs["name"],
        out_ptr=y._attrs["name"],
        bias_ptr=bias_ptr,
        res_ptr=res_ptr,
        p_batch="&" + xshape[0]._attrs["name"],
        p_out_ch="&" + wshape[0]._attrs["name"],
        p_in_ch="&" + xshape[3]._attrs["name"],
        p_kernel_h="&" + wshape[1]._attrs["name"],
        p_kernel_w="&" + wshape[2]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        stride=func_attrs["stride"],
        dilation=func_attrs["dilate"],
        pad=func_attrs["pad"],
        group=func_attrs["group"],
        indent=indent,
        conv2d_flag=conv2d_flag,
    )


def function_filter(cfg, func_attrs, offset):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    offset: Int
        Offset of split(cfg,"_") to get conv2d specialization

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    from ck_lib.conv2d_operation import Conv2DSpecialization

    kh = func_attrs["KH"]
    kw = func_attrs["KW"]
    pad = func_attrs["pad"]
    stride = func_attrs["stride"]
    cond0 = kh == 1 and kw == 1 and pad == 0
    cond1 = stride == 1
    spec = int(cfg.split("_")[offset])
    if cond0 and cond1:  # 1x1, stride 1, pad 0
        return spec == Conv2DSpecialization.ConvFwd1x1S1P0.value
    if cond0 and not cond1:  # 1x1, pad 0
        return spec == Conv2DSpecialization.ConvFwd1x1P0.value
    return True
