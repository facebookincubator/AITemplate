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
Softmax codegen for ROCM.
"""

from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.rocm.normalization import norm_common

from aitemplate.compiler.base import IntImm

EXTRA_HEADERS = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "include/ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp"
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t ptr_sz = in_{{ range(rank)|join(' * in_') }};
  // TODO: special pool size for 8M L2 cache
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_sz)));

  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // in: index 0
  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // out: index 1
"""
)

EXEC_TEMPLATE = jinja2.Template(
    """
    float alpha = 1.0f;
    float beta  = 0.0f;
    const std::vector<int> reduceDims{ {{reduce_dims}} };
    std::vector<ck::index_t> i_inLengths;
    std::vector<ck::index_t> i_inStrides;
{% for idx in range(rank) %}
    i_inLengths.push_back(*in_{{idx}});
{% endfor %}
{% for start in range(1, rank) %}
    i_inStrides.push_back( (*in_{{ range(start, rank)|join(') * (*in_') }}) );
{% endfor %}
    i_inStrides.push_back(1);
    auto device_instance = {{instance}}{};
    auto argument_ptr = device_instance.MakeArgumentPointer(i_inLengths,
                                                            i_inStrides,
                                                            reduceDims,
                                                            alpha,
                                                            beta,
                                                            static_cast<ck::half_t *>(input),
                                                            static_cast<ck::half_t *>(output),
                                                            ck::tensor_operation::element_wise::PassThrough{},
                                                            ck::tensor_operation::element_wise::PassThrough{}
                                                            );
    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        LOG(FATAL) << "wrong! " << device_instance.GetTypeString() << " with the specified compilation parameters does not support this Softmax problem.";
    };
    auto invoker_ptr = device_instance.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, false});
    return;
"""
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}({{dtype}}* input,
                   {{dtype}}* output,
{% for idx in range(input_ndim) %}
                   int64_t* in_{{idx}},
{% endfor %}
                   hipStream_t stream)
    """
)


def get_func_signature(func_attrs: Dict[str, Any]) -> str:
    input_ndim = func_attrs["inputs"][0]._rank()
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        dtype="void",
        input_ndim=input_ndim,
    ).strip()


@registry.reg("rocm.softmax.config")
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
    norm_common.extract_config(func_attrs)


@registry.reg("rocm.softmax.gen_profiler")
def softmax_gen_profiler(
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
    dim = func_attrs["dim"]
    shapes = func_attrs["inputs"][0]._attrs["shape"]
    rank = len(shapes)

    assert (
        dim == rank - 1
    ), f"rocm softmax only supports dim == rank - 1, dim={dim}, rank={rank}"

    assert isinstance(
        shapes[dim], IntImm
    ), "softmax requires reduction dim to be static"

    return norm_common.gen_profiler(
        func_attrs,
        workdir,
        rank,
        "",
        EXEC_TEMPLATE,
        TENSOR_DECL_TEMPLATE,
        EXTRA_HEADERS,
        get_func_signature=get_func_signature,
        indent=indent,
    )


@registry.reg("rocm.softmax.gen_function")
def softmax_gen_function(func_attrs: Dict[str, Any]) -> str:
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
    dim = func_attrs["dim"]
    shapes = func_attrs["inputs"][0]._attrs["shape"]
    rank = len(shapes)

    assert (
        dim == rank - 1
    ), f"rocm softmax only supports dim == rank - 1, dim={dim}, rank={rank}"

    assert isinstance(
        shapes[dim], IntImm
    ), "softmax requires reduction dim to be static"
    return norm_common.gen_function(
        func_attrs, EXEC_TEMPLATE, EXTRA_HEADERS, get_func_signature
    )


@registry.reg("rocm.softmax.func_decl")
def softmax_gen_function_decl(func_attrs: Dict[str, Any]):
    """Generates function declarations.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.

    Returns
    -------
    str
        The rentered template of function declaration.
    """
    return get_func_signature(func_attrs) + ";"


@registry.reg("rocm.softmax.func_call")
def softmax_gen_function_call(func_attrs, indent="  "):
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
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 1
    shapes = func_attrs["inputs"][0]._attrs["shape"]
    assert (
        len(shapes) >= 2
    ), f"Softmax only supports input with rank >= 2, current rank: {len(shapes)}"

    return norm_common.gen_function_call(func_attrs, indent)
