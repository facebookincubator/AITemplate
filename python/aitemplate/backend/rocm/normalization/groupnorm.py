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
Groupnorm codegen for ROCM.
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

EXTRA_CODE_TEMPLATE = jinja2.Template(
    """
{%if use_swish %}
struct YElementOp
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(ck::is_same<T, float>::value || ck::is_same<T, double>::value ||
                          ck::is_same<T, ck::half_t>::value,
                      "Data type is not supported by this operation!");

        T a;

        ck::tensor_operation::element_wise::Sigmoid{}(a, x);

        y = x * a;
    };
};

{% else %}

using YElementOp   = ck::tensor_operation::element_wise::PassThrough;

{% endif %}
"""
)

FUNC_CALL_FP16_PARAM_TEMPLATE = jinja2.Template(
    "reinterpret_cast<ck::half_t*>(({{name}}))"
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  // N, H, W, G, C
  const int64_t N = in_0;
  const int64_t H = in_1;
  const int64_t W = in_2;
  const int64_t G = in_3;
  const int64_t C = in_4 ;
  int64_t ptr_sz = N * H * W * G * C;

  // TODO: special pool size for 8M L2 cache
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_sz)));

  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // in: index 0
  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // out: index 1
  memory_pool->AllocateHalfTensor(G * C, mem_pool_sz);  // gamma: index 2
  memory_pool->AllocateHalfTensor(G * C, mem_pool_sz);  // beta: index 3

"""
)

SHAPE_EVAL_TEMPLATE = jinja2.Template(
    """
    """
)

EXEC_TEMPLATE = jinja2.Template(
    """
    C = C / G;
    std::vector<ck::index_t> i_inStrides;

    i_inStrides.push_back(H * W * G * C);
    i_inStrides.push_back(W * G * C);
    i_inStrides.push_back(G * C);
    i_inStrides.push_back(C);
    i_inStrides.push_back(1);

    std::vector<ck::index_t> gamma_beta_Strides;
    gamma_beta_Strides.push_back(0);
    gamma_beta_Strides.push_back(0);
    gamma_beta_Strides.push_back(0);
    gamma_beta_Strides.push_back(C);
    gamma_beta_Strides.push_back(1);

    auto device_instance = {{instance}}{};
    auto argument_ptr = device_instance.MakeArgumentPointer(
        {static_cast<ck::index_t>(N),
         static_cast<ck::index_t>(H),
         static_cast<ck::index_t>(W),
         static_cast<ck::index_t>(G),
         static_cast<ck::index_t>(C)},
        i_inStrides, // x stride
        gamma_beta_Strides,
        gamma_beta_Strides,
        i_inStrides, // y stride
        {1, 2, 4}, // reduction dimension: [H, W, C]
        1e-5,
        static_cast<ck::half_t *>(input),
        static_cast<ck::half_t *>(gamma),
        static_cast<ck::half_t *>(beta),
        static_cast<ck::half_t *>(output),
        nullptr,
        nullptr,
        YElementOp{}
    );

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        LOG(FATAL) << "wrong! " << device_instance.GetTypeString() << " with the specified compilation parameters does not support this Groupnorm problem.";
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
                   int64_t N,
                   int64_t H,
                   int64_t W,
                   int64_t G,
                   int64_t C,
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
{{indent}}   {{N}},
{{indent}}   {{H}},
{{indent}}   {{W}},
{{indent}}   {{G}},
{{indent}}   {{C}},
{{indent}}   stream
{{indent}});
    """
)


PROFILER_FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{input}},
{{indent}}   {{gamma}},
{{indent}}   {{beta}},
{{indent}}   {{output}},
{{indent}}   N,
{{indent}}   H,
{{indent}}   W,
{{indent}}   G,
{{indent}}   C,
{{indent}}   stream
{{indent}});
    """
)


@registry.reg("rocm.groupnorm.config")
def groupnorm_extract_config(func_attrs):
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

    op_kind = ck_lib.library.OperationKind.GroupNorm
    extra_kind = 5
    extract_ops = list(Target.current()._operators[op_kind][extra_kind].items())
    groupnorm_ops = OrderedDict()
    for key, value in extract_ops:
        groupnorm_ops[key] = value[0]
    func_attrs["op_instance"] = groupnorm_ops


def get_func_signature_profiler(func_attrs: Dict[str, Any]) -> str:
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        dtype="void",
        input_ndim=5,
    ).strip()


@registry.reg("rocm.groupnorm.gen_profiler")
def groupnorm_gen_profiler(
    func_attrs: Dict[str, Any],
    workdir: str,
    indent: str = "  ",
    use_swish: bool = False,
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
    use_swish : bool, optional
        Use swish if True
    """
    # N, H, W, C
    shapes = func_attrs["inputs"][0]._attrs["shape"]

    for dim_idx in (1, 2, 3):
        assert isinstance(
            shapes[dim_idx], IntImm
        ), f"groupnorm requires reduction dim {dim_idx=} to be static"

    return norm_common.gen_profiler(
        func_attrs,
        workdir,
        5,  # rank
        SHAPE_EVAL_TEMPLATE,
        EXEC_TEMPLATE,
        TENSOR_DECL_TEMPLATE,
        EXTRA_HEADERS,
        get_func_signature_profiler,
        EXTRA_CODE_TEMPLATE.render(use_swish=use_swish),
        PROFILER_FUNC_CALL_TEMPLATE,
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
    extra_code_template: jinja2.Template,
    get_func_signature: Any,
    use_swish: bool = False,
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
    extra_code_template : jinja2.Template
        Extra code template.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
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

    exec_cond_template = func_attrs["exec_cond_template"]
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        program = exec_template.render(instance=fname, dtype="void")
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst

    return norm_common.FUNC_TEMPLATE.render(
        instances_decl=instance_decl,
        func_signature=get_func_signature(func_attrs),
        shape_eval="",
        exec_paths=exec_paths,
        extra_headers=extra_header_template.render(),
        extra_code=extra_code_template.render(use_swish=use_swish),
    )


@registry.reg("rocm.groupnorm.gen_function")
def groupnorm_gen_function(func_attrs: Dict[str, Any], use_swish: bool = False) -> str:
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
    # N, H, W, C
    shapes = func_attrs["inputs"][0]._attrs["shape"]

    for dim_idx in (1, 2, 3):
        assert isinstance(
            shapes[dim_idx], IntImm
        ), f"groupnorm requires reduction dim {dim_idx=} to be static"

    return gen_function(
        func_attrs,
        SHAPE_EVAL_TEMPLATE,
        EXEC_TEMPLATE,
        EXTRA_HEADERS,
        EXTRA_CODE_TEMPLATE,
        get_func_signature,
        use_swish,
    )


def get_func_signature(func_attrs: Dict[str, Any]) -> str:
    input_ndim = func_attrs["inputs"][0]._rank()
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        dtype="void",
        input_ndim=input_ndim,
    ).strip()


@registry.reg("rocm.groupnorm.func_decl")
def groupnorm_gen_func_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(func_signature=get_func_signature(func_attrs))


@registry.reg("rocm.groupnorm.func_call")
def groupnorm_gen_func_call(func_attrs, indent="  "):
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
        len(shapes) == 4
    ), f"GroupNorm only supports input with rank == 4, current rank: {len(shapes)}"

    N = shapes[0]._attrs["name"]
    H = shapes[1]._attrs["name"]
    W = shapes[2]._attrs["name"]
    G = func_attrs["num_groups"]
    C = shapes[3]._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=input_name,
        gamma=gamma_name,
        beta=beta_name,
        output=output_name,
        N=N,
        H=H,
        W=W,
        G=G,
        C=C,
        indent=indent,
    )
