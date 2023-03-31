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
Normalization common codegen for ROCM.
"""

import os
import re
from collections import OrderedDict
from hashlib import sha1
from typing import Any, Dict

import jinja2

from aitemplate.backend.target import Target

FUNC_CALL_PARAM_TEMPLATE = jinja2.Template("(void *)({{name}})")

INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
using {{name}} = {{ config_name }};
"""
)

ARGS_PARSE_TEMPLATE = jinja2.Template(
    """
{% for idx in range(rank) %}
  const int64_t in_{{idx}} = std::stoi(argv[{{ idx + 1 }}]);
{% endfor %}
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
  {{args_parse}}
  auto memory_pool = std::make_unique<ProfilerMemoryPool>();
  hipStream_t stream = nullptr;
  {{tensor_decl}}
  // warmup
  for(int i = 0; i < 3; ++i) {
    {{func_call}}
  }
  // run
  KernelTimerImpl timer;
  timer.Start();
  for(int i = 0; i < 5; ++i) {
    {{func_call}}
  }
  timer.End();
  std::cout << "OP:" << "{{op_name}}" << ",";
  std::cout << "TIME:" << timer.GetElapsedTime() << ",";
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
}
"""
)

FUNC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <random>
#include <rocrand/rocrand.h>
#include "logging.h"
#include "include/ck/utility/print.hpp"
#include "library/include/ck/library/utility/device_memory.hpp"
#include "library/include/ck/library/utility/host_tensor.hpp"
#include "library/include/ck/library/utility/host_tensor_generator.hpp"
#include "include/ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/utility/reduction_operator.hpp"
{{extra_headers}}

{{extra_code}}

{{instances_decl}}

{{func_signature}}
{
{{shape_eval}}
{{exec_paths}}
}
    """
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{input}},
{{indent}}   {{output}},
{% for name in input_dim_names %}
{{indent}}    const_cast<int64_t *>(&{{name}}),
{% endfor %}
{{indent}}   stream
{{indent}});
    """
)


def extract_config(func_attrs):
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
    import ck_lib

    op_kind = ck_lib.library.OperationKind.Softmax
    extra_kind = len(func_attrs["inputs"][0]._attrs["shape"])
    extract_ops = list(Target.current()._operators[op_kind][extra_kind].items())
    softmax_ops = OrderedDict()
    for key, value in extract_ops:
        softmax_ops[key] = value[0]
    func_attrs["op_instance"] = softmax_ops


def emit_instance(op):
    """Emit instance"""
    import ck_lib  # noqa: F401

    op_def = op.emit()
    return op_def


def extract_config_name(config):
    """Extract configuration names.

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
    func_attrs: Dict[str, Any],
    workdir: str,
    rank: int,
    shape_eval_template: jinja2.Template,
    exec_template: jinja2.Template,
    tensor_decl_template: jinja2.Template,
    extra_header_template: jinja2.Template,
    get_func_signature: Any,
    extra_code: str = "",
    func_call_template: jinja2.Template = FUNC_CALL_TEMPLATE,
    indent: str = "  ",
) -> str:
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    rank: int
        Rank of the input tensor. If using [M, N] in exec_key, the rank here
        must be 2 because if implies that the inputs are reshaped for profiling.
        For code gen, the real shapes are used.
    exec_template : jinja2.Template
        Execution block template.
    tensor_decl_template: jinja2.Template
        Tensor declaration template.
    extra_header_template : jinja2.Template
        Extra header template.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".
    """
    op_type = func_attrs["op"]
    shape_eval = shape_eval_template.render(rank=rank) if shape_eval_template else ""
    eps = func_attrs.get("eps", "1e-5")

    op_instance = func_attrs["op_instance"]
    file_pairs = []
    for op_name, op in op_instance.items():
        config = emit_instance(op)
        config_name = extract_config_name(config)
        instances = INSTANCE_TEMPLATE.render(
            name="DeviceInstance", config_name=config_name, config=config
        )
        exe_path = exec_template.render(
            instance="DeviceInstance",
            dtype="void",
            reduce_dims=rank - 1,
            rank=rank,
            eps=eps,
        )

        op_func = FUNC_TEMPLATE.render(
            instances_decl=instances,
            func_signature=get_func_signature(func_attrs),
            shape_eval=shape_eval,
            exec_paths=exe_path,
            extra_headers=extra_header_template.render(),
            extra_code=extra_code,
        )
        structs_def = STRUCTS_DEF_TEMPLATE.render()
        args_parse = ARGS_PARSE_TEMPLATE.render(rank=rank)
        tensor_decl = tensor_decl_template.render(rank=rank)

        input_dim_names = [f"in_{i}" for i in range(rank)]
        func_call = func_call_template.render(
            func_name=func_attrs["name"],
            input="(void *) memory_pool->RequestHalfTensorByIdx(0)",
            gamma="(void *) memory_pool->RequestHalfTensorByIdx(2)",
            beta="(void *) memory_pool->RequestHalfTensorByIdx(3)",
            output="(void *) memory_pool->RequestHalfTensorByIdx(1)",
            input_dim_names=input_dim_names,
            indent=indent,
        )
        code = PROFILER_TEMPLATE.render(
            op_func=op_func,
            structs_def=structs_def,
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


# no longer used by layernorm
def gen_function(
    func_attrs: Dict[str, Any],
    exec_template: jinja2.Template,
    extra_header_template: jinja2.Template,
    get_func_signature: Any,
) -> str:
    """Generate function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_template : jinja2.Template
        Execution block template.
    extra_header_template : jinja2.Template
        Extra header template.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    shapes = func_attrs["inputs"][0]._attrs["shape"]
    rank = len(shapes)

    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]

    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for exec_item in exec_path.values():
        fname = "f" + sha1(exec_item.exec_cond.encode()).hexdigest()
        algo = exec_item.algo
        if algo not in inst_def_flag:
            config = emit_instance(op_instance[algo])
            inst_def_flag.add(algo)
        else:
            config = ""
        inst = INSTANCE_TEMPLATE.render(
            config=config, name=fname, config_name=extract_config_name(config)
        )
        instances[exec_item.exec_cond] = inst
        instance_decl += inst

    exec_cond_template = func_attrs["exec_cond_template"]
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        program = exec_template.render(
            instance=fname, dtype="void", reduce_dims=rank - 1, rank=rank
        )
        cond_vars = re.findall(r"\S+(?= >=)", key)
        cond_vars += re.findall(r"\S+(?= ==)", key)
        cond = key
        for i, var in enumerate(cond_vars):
            cond = cond.replace(var + " ", "*in_" + str(i))
        exec_inst = exec_cond_template.render(indent="  ", cond=cond, program=program)
        exec_paths += exec_inst

    return FUNC_TEMPLATE.render(
        instances_decl=instance_decl,
        func_signature=get_func_signature(func_attrs),
        exec_paths=exec_paths,
        extra_headers=extra_header_template.render(),
    )


def gen_function_call(func_attrs, indent="  "):
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
    input_name = FUNC_CALL_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][0]._attrs["name"]
    )
    output_name = FUNC_CALL_PARAM_TEMPLATE.render(
        name=func_attrs["outputs"][0]._attrs["name"]
    )

    shapes = func_attrs["inputs"][0]._attrs["shape"]
    input_dim_names = [shape._attrs["name"] for shape in shapes]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=input_name,
        output=output_name,
        input_dim_names=input_dim_names,
        indent=indent,
    )
