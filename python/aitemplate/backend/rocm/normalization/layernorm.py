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
Layernorm codegen for ROCM.
"""
from collections import OrderedDict
from hashlib import sha1
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.rocm.normalization import norm_common
from aitemplate.backend.target import Target

from aitemplate.compiler.base import IntImm

EXTRA_HEADERS = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/impl/device_normalization_impl.hpp"
"""
)

FUNC_CALL_FP16_PARAM_TEMPLATE = jinja2.Template(
    "reinterpret_cast<ck::half_t*>(({{name}}))"
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t ptr_sz = in_{{ range(rank)|join(' * in_') }};

  int64_t norm_dim = in_{{rank - 1}};

  // TODO: special pool size for 8M L2 cache
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_sz)));

  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // in: index 0
  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // out: index 1
  memory_pool->AllocateHalfTensor(norm_dim, mem_pool_sz);  // gamma: index 2
  memory_pool->AllocateHalfTensor(norm_dim, mem_pool_sz);  // beta: index 3

"""
)

SHAPE_EVAL_TEMPLATE = jinja2.Template(
    """
    int M = *in_{{ range(rank - 1)|join(' * *in_') }};
    int N = *in_{{rank - 1}};
    """
)

EXEC_TEMPLATE = jinja2.Template(
    """
    std::vector<ck::index_t> i_inStrides;
    std::vector<ck::index_t> i_outStrides;
    {% if input_strides is defined %}
    i_inStrides.push_back({{input_strides[-2]}});
    i_inStrides.push_back({{input_strides[-1]}});
    {% else %}
    i_inStrides.push_back(N);
    i_inStrides.push_back(1);
    {% endif %}

    {% if output_strides is defined %}
    i_outStrides.push_back({{output_strides[-2]}});
    i_outStrides.push_back({{output_strides[-1]}});
    {% else %}
    i_outStrides.push_back(N);
    i_outStrides.push_back(1);
    {% endif %}

    auto device_instance = {{instance}}{};
    auto argument_ptr = device_instance.MakeArgumentPointer(
        {M, N},
        i_inStrides,
        std::vector<ck::index_t>{0, 1},
        std::vector<ck::index_t>{0, 1},
        i_outStrides,
        {1},
        {{eps}},
        static_cast<ck::half_t *>(input) + {{ input_offset if input_offset is defined else 0 }},
        static_cast<ck::half_t *>(gamma),
        static_cast<ck::half_t *>(beta),
        static_cast<ck::half_t *>(output) + {{ output_offset if output_offset is defined else 0 }},
        nullptr,
        nullptr,
        ck::tensor_operation::element_wise::PassThrough{}
    );

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        LOG(FATAL) << "wrong! " << device_instance.GetTypeString() << " with the specified compilation parameters does not support this Layernorm problem.";
    };
    auto invoker_ptr = device_instance.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, false});
    return;
"""
)
FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}({{dtype}}* input,
                   {{dtype}}* gamma,
                   {{dtype}}* beta,
                   {{dtype}}* output,
{% for idx in range(input_ndim) %}
                   int64_t* in_{{idx}},
{% endfor %}
                   hipStream_t stream)
    """
)


FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{input}},
{{indent}}   {{gamma}},
{{indent}}   {{beta}},
{{indent}}   {{output}},
{% for name in input_dim_names %}
{{indent}}    const_cast<int64_t *>(&{{name}}),
{% endfor %}
{{indent}}   stream
{{indent}});
    """
)


@registry.reg("rocm.layernorm.config")
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

    op_kind = ck_lib.library.OperationKind.LayerNorm
    extra_kind = 2
    extract_ops = list(Target.current()._operators[op_kind][extra_kind].items())
    layernorm_ops = OrderedDict()
    for key, value in extract_ops:
        layernorm_ops[key] = value[0]
    func_attrs["op_instance"] = layernorm_ops


def get_func_signature_profiler(func_attrs: Dict[str, Any]) -> str:
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        dtype="void",
        input_ndim=2,
    ).strip()


@registry.reg("rocm.layernorm.gen_profiler")
def layernorm_gen_profiler(
    func_attrs: Dict[str, Any], workdir: str, indent: str = "  "
) -> str:
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".
    """
    dim = -1
    shapes = func_attrs["inputs"][0]._attrs["shape"]

    assert isinstance(
        shapes[dim], IntImm
    ), "layernorm requires reduction dim to be static"

    return norm_common.gen_profiler(
        func_attrs,
        workdir,
        2,  # rank
        SHAPE_EVAL_TEMPLATE,
        EXEC_TEMPLATE,
        TENSOR_DECL_TEMPLATE,
        EXTRA_HEADERS,
        get_func_signature_profiler,
        "",  # extra_code
        FUNC_CALL_TEMPLATE,
        indent,
    )


# This function has diverged from norm_common.gen_function
# due to the change to the profiler exec_key
# TODO: merge with norm_common.gen_function after fixing softmax
def gen_function(
    func_attrs: Dict[str, Any],
    shape_eval_template: jinja2.Template,
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
    rank = func_attrs["inputs"][0]._rank()
    eps = func_attrs.get("eps", "1e-5")
    input_accessor = func_attrs["input_accessors"][0]
    output_accessor = func_attrs["output_accessors"][0]
    input_strides = []
    output_strides = []
    for i, _ in enumerate(input_accessor.original_shapes):
        input_strides.append(input_accessor.stride(i))
        output_strides.append(output_accessor.stride(i))

    input_offset = input_accessor.offset
    output_offset = output_accessor.offset

    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]

    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for exec_item in exec_path.values():
        fname = "f" + sha1(exec_item.exec_cond.encode()).hexdigest()
        algo = exec_item.algo
        if algo not in inst_def_flag:
            config = norm_common.emit_instance(op_instance[algo])
            inst_def_flag.add(algo)
        else:
            config = ""
        inst = norm_common.INSTANCE_TEMPLATE.render(
            config=config,
            name=fname,
            config_name=norm_common.extract_config_name(config),
        )
        instances[exec_item.exec_cond] = inst
        instance_decl += inst

    shape_eval = shape_eval_template.render(rank=rank) if shape_eval_template else ""
    exec_cond_template = func_attrs["exec_cond_template"]
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        program = exec_template.render(
            instance=fname,
            dtype="void",
            reduce_dims=rank - 1,
            eps=eps,
            input_strides=input_strides,
            output_strides=output_strides,
            input_offset=input_offset,
            output_offset=output_offset,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst

    return norm_common.FUNC_TEMPLATE.render(
        instances_decl=instance_decl,
        func_signature=get_func_signature(func_attrs),
        shape_eval=shape_eval,
        exec_paths=exec_paths,
        extra_headers=extra_header_template.render(),
    )


@registry.reg("rocm.layernorm.gen_function")
def layernorm_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generate function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    dim = -1
    shapes = func_attrs["inputs"][0]._attrs["shape"]

    assert isinstance(
        shapes[dim], IntImm
    ), "layernorm requires reduction dim to be static"

    return gen_function(
        func_attrs,
        SHAPE_EVAL_TEMPLATE,
        EXEC_TEMPLATE,
        EXTRA_HEADERS,
        get_func_signature,
    )


def get_func_signature(func_attrs: Dict[str, Any]) -> str:
    input_ndim = func_attrs["inputs"][0]._rank()
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        dtype="void",
        input_ndim=input_ndim,
    ).strip()


@registry.reg("rocm.layernorm.func_decl")
def layernorm_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(func_signature=get_func_signature(func_attrs))


@registry.reg("rocm.layernorm.func_call")
def layernorm_gen_function_call(func_attrs, indent="  "):
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 3

    input_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][0]._attrs["name"]
    )
    gamma_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][1]._attrs["name"]
    )
    beta_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][2]._attrs["name"]
    )
    output_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["outputs"][0]._attrs["name"]
    )

    shapes = func_attrs["inputs"][0]._attrs["shape"]
    assert (
        len(shapes) >= 2
    ), f"LayerNorm only supports input with rank >= 2, current rank: {len(shapes)}"

    input_dim_names = [shape._attrs["name"] for shape in shapes]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=input_name,
        gamma=gamma_name,
        beta=beta_name,
        output=output_name,
        input_dim_names=input_dim_names,
        indent=indent,
    )
