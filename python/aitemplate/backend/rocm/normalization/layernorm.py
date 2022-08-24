# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Layernorm codegen for ROCM.
"""

from typing import Any, Dict, OrderedDict

import jinja2

from ....compiler.base import IntImm

from ... import registry
from ...target import Target
from . import norm_common

EXTRA_HEADERS = jinja2.Template(
    """
#include "include/ck/tensor_operation/gpu/device/device_layernorm_impl.hpp"
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

EXEC_TEMPLATE = jinja2.Template(
    """

    int M = *in_{{ range(rank - 1)|join(' * *in_') }};
    int N = *in_{{rank - 1}};

    std::vector<ck::index_t> i_inStrides;

    i_inStrides.push_back(N);
    i_inStrides.push_back(1);


    auto device_instance = {{instance}}{};
    auto argument_ptr = device_instance.MakeArgumentPointer(
        {M, N},
        i_inStrides,
        std::vector<ck::index_t>{1},
        std::vector<ck::index_t>{1},
        i_inStrides,
        {1},
        1e-5,
        static_cast<ck::half_t *>(input),
        static_cast<ck::half_t *>(gamma),
        static_cast<ck::half_t *>(beta),
        static_cast<ck::half_t *>(output),
        ck::tensor_operation::element_wise::PassThrough{}
    );

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        throw std::runtime_error(
            "wrong! device_layernorm with the specified compilation parameters does "
            "not support this Softmax problem");
    };
    std::string instance_name = device_instance.GetTypeString();
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
    ), "softmax requires reduction dim to be static"

    norm_common.gen_profiler(
        func_attrs,
        workdir,
        EXEC_TEMPLATE,
        TENSOR_DECL_TEMPLATE,
        EXTRA_HEADERS,
        get_func_signature,
        FUNC_CALL_TEMPLATE,
        indent,
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
    ), "softmax requires reduction dim to be static"
    return norm_common.gen_function(
        func_attrs, EXEC_TEMPLATE, EXTRA_HEADERS, get_func_signature
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
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]

    elem_cnt = 1
    for shape in xshape:
        elem_cnt *= shape._attrs["values"][0]
    instance_size = xshape[-1]._attrs["values"][0]
    instance_num = elem_cnt // instance_size

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=input_name,
        gamma=gamma_name,
        beta=beta_name,
        output=output_name,
        M=instance_num,
        N=instance_size,
        input_dim_names=input_dim_names,
        indent=indent,
    )
