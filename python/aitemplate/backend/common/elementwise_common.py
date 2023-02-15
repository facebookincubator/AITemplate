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
Backend-agnostic functions for elementwise codegen.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jinja2
from aitemplate.backend.backend_spec import BackendSpec

from ...compiler.base import IntImm, IntVar, Operator, Tensor
from ...compiler.tensor_accessor import TensorAccessor
from ...utils import shape_utils
from . import tensor_accessor_codegen

CONSTANT_TEMPLATE = jinja2.Template(
    """
#define FUSED_ELE_THREAD_SIZE 256

const int N_ELEMENTS_PER_THREAD = sizeof({{read_t}}) / sizeof({{data_t}});
const int N_ELEMENTS_PER_READ = sizeof({{read_t}}) / sizeof({{data_t}});
const int N_OPS_PER_THREAD = sizeof({{read_t}}) / sizeof({{op_t}});
    """
)

KERNEL_DECL_INPUT_PARAM_TEMPLATE = jinja2.Template("const {{read_t}}* input{{idx}}")
KERNEL_DECL_OUTPUT_PARAM_TEMPLATE = jinja2.Template("{{read_t}}* output{{idx}}")

KERNEL_TMP_INPUT_TEMPLATE = jinja2.Template("p_tmp_i{{idx}}[i]")
KERNEL_TMP_OUTPUT_TEMPLATE = jinja2.Template("p_tmp_o{{idx}}[i]")


GET_STRIDED_ADDRESS_TEMPLATE = jinja2.Template(
    """
  {% if tensor_accessor.is_contiguous %}
  {{data_ptr}} = get_strided_address</*data_t*/ {{data_t}},
                                     /*read_t*/ {{read_t}},
                                     /*is_contiguous*/ true>(
      {{data_ptr}}, {{data_idx}}, {{tensor_accessor.offset}}, 0, 0);
  {% else %}
  {{data_ptr}} = get_strided_address</*data_t*/ {{data_t}},
                                     /*read_t*/ {{read_t}},
                                     /*is_contiguous*/ false>(
      {{data_ptr}}, {{data_idx}},
      {{tensor_accessor.offset}},
      {{tensor_accessor.original_total_elements_from_stride_dim}},
      {{tensor_accessor.actual_total_elements_from_stride_dim}});
  {% endif %}
    """
)


KERNEL_READ_INPUT_TEMPLATE = jinja2.Template(
    """
  {{read_t}} *{{input_name}} = const_cast<{{read_t}}*>(input{{input_idx}});
  {{get_strided_address}}
  {{read_t}} tmp_i{{input_idx}} = *{{input_name}};
  const {{op_t}}* p_tmp_i{{input_idx}} = reinterpret_cast<const {{op_t}}*>(&tmp_i{{input_idx}});

    """
)


KERNEL_DEFINE_OUTPUTS_TEMPLATE = jinja2.Template(
    """
  {% for idx in indexes %}
  {{read_t}} tmp_o{{idx}};
  {{op_t}}* p_tmp_o{{idx}} = reinterpret_cast<{{op_t}}*>(&tmp_o{{idx}});
  {% endfor %}
    """
)


KERNEL_WRITE_OUTPUT_TEMPLATE = jinja2.Template(
    """
  {{get_strided_address}}
  *{{output_name}} = tmp_o{{output_idx}};
    """
)


KERNEL_TEMPLATE = jinja2.Template(
    """
__global__ void
{{func_name}}({{output_params}}, {{input_params}}, {{dynamic_dims}} {{index_type}} n_elements) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const {{index_type}} idx = bid * FUSED_ELE_THREAD_SIZE + tid;
  const {{index_type}} idx_elem = idx * N_ELEMENTS_PER_THREAD;
  if (idx_elem >= n_elements) {
    return;
  }
  {{read_inputs}}
  {{define_outputs}}
#pragma unroll
  for (int i = 0; i < N_OPS_PER_THREAD; ++i) {
    {{fused_funcs}}
  }
  {{write_outputs}}
}
    """
)

FUNC_DECL_INPUT_PARAM_TEMPLATE = jinja2.Template("const void* input{{idx}}")
FUNC_DECL_OUTPUT_PARAM_TEMPLATE = jinja2.Template("void* output{{idx}}")
KERNEL_CALL_INPUT_PARAM_TEMPLATE = jinja2.Template(
    "reinterpret_cast<const {{read_t}}*>(input{{idx}})"
)
KERNEL_CALL_OUTPUT_PARAM_TEMPLATE = jinja2.Template(
    "reinterpret_cast<{{read_t}}*>(output{{idx}})"
)

FUNC_TEMPLATE = jinja2.Template(
    """
{{head}}

namespace {

{{constant}}

{{custom_libs}}

{{tensor_accessor_lib}}

{{kernel_function}}

}  // namespace

void invoke_{{func_name}}({{output_params}}, {{input_params}}, {{dynamic_dims_decl}} {{index_type}} n_elements, {{prefix}}Stream_t stream) {
    if (n_elements == 0) {
      return;
    }
    int block_size = static_cast<int>(std::ceil(static_cast<double>(n_elements) / N_ELEMENTS_PER_THREAD / FUSED_ELE_THREAD_SIZE));
    {{func_name}}<<<block_size, FUSED_ELE_THREAD_SIZE, 0, stream>>>(
        {{kernel_call_output_params}},
        {{kernel_call_input_params}},
        {{dynamic_dims_call}}
        n_elements
    );
}
    """
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void invoke_{{func_name}}({{output_params}}, {{input_params}}, {{dynamic_dims}} {{index_type}} n_elements, {{prefix}}Stream_t stream);
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
    {{indent}}{{index_type}} {{func_name}}_n_elements = {{calculate_n}};
    {{indent}}invoke_{{func_name}}({{output_params}}, {{input_params}}, {{dynamic_dims}} {{func_name}}_n_elements, {{stream}});
{{indent}}}
    """
)


@dataclass
class ElementwiseMetaData:
    func_name: str
    op_t: str
    args: List[Tensor]
    outputs: List[Tensor]


@dataclass
class FusedElementwiseMetaData:
    # Input / output Tensors and TensorAccessors.
    inputs: List[Tensor]
    outputs: List[Tensor]
    input_accessors: List[TensorAccessor]
    output_accessors: List[TensorAccessor]

    # Original input / output Tensors before graph transformation.
    # Kept here for elementwise -> fused elementwise Tensor mapping.
    original_inputs: List[Tensor]
    original_outputs: List[Tensor]

    read_t: str
    op_t: str
    data_t: str
    input_broadcast_sizes: List[List[IntVar]]
    dynamic_dims: List[IntVar]
    sub_funcs: List[ElementwiseMetaData]


def gen_function_single_thread(
    fused_func_metadata,
    input_names,
    output_names,
    type_converter,
) -> str:
    """Per thread elementwise function codegen."""
    tensor_to_expr: Dict[Tensor, str] = {}
    body = ""

    for tensor, name in zip(fused_func_metadata.original_inputs, input_names):
        tensor_to_expr[tensor] = name

    tmp_output_idx: int = 0
    for func_metadata in fused_func_metadata.sub_funcs:
        params: List[str] = []
        func_op_t = func_metadata.op_t
        input_converter = None
        output_converter = None
        if func_op_t != fused_func_metadata.op_t:
            input_converter = type_converter.get(fused_func_metadata.op_t).get(
                func_op_t
            )
            output_converter = type_converter.get(func_op_t).get(
                fused_func_metadata.op_t
            )
            assert (
                input_converter is not None
            ), "Unsupported convertion from {} to {}".format(
                fused_func_metadata.op_t, func_op_t
            )
            assert (
                output_converter is not None
            ), "Unsupported convertion from {} to {}".format(
                func_op_t, fused_func_metadata.op_t
            )

        for arg in func_metadata.args:
            if arg in tensor_to_expr:
                param = tensor_to_expr[arg]
                params.append(
                    "{}({})".format(input_converter, param)
                    if input_converter is not None
                    else param
                )
            elif arg.is_a_const_num():
                arg_str = ""
                if math.isinf(arg._attrs["value"]):
                    arg_str = "CUDART_INF_F"
                else:
                    arg_str = str(arg._attrs["value"])
                if func_op_t[-1] == "2":
                    params.append(
                        "{}({},{})".format(
                            func_op_t,
                            arg_str,
                            arg_str,
                        )
                    )
                else:
                    params.append("{}({})".format(func_op_t, arg_str))
            else:
                raise RuntimeError(
                    "Cannot generate expression for node {}, ops: {}".format(
                        arg, func_metadata
                    )
                )
        assert (
            len(func_metadata.outputs) == 1
        ), "Operator has more than 1 output! Operator: {}".format(func_metadata)

        output = func_metadata.outputs[0]
        func_def = "{}({})".format(func_metadata.func_name, ",".join(params))
        func_def = (
            "{}({})".format(output_converter, func_def)
            if output_converter is not None
            else func_def
        )
        if len(output._attrs["dst_ops"]) > 1:
            name = "tmp_" + (str)(tmp_output_idx)
            tmp_output_idx += 1
            body += "{} {} = {};\n".format(fused_func_metadata.op_t, name, func_def)
            tensor_to_expr[output] = name
        else:
            tensor_to_expr[output] = func_def

    for tensor, name in zip(fused_func_metadata.original_outputs, output_names):
        if tensor not in tensor_to_expr:
            raise RuntimeError(
                "Cannot generate expression for node {}, outputs: {}".format(
                    tensor, fused_func_metadata.original_outputs
                )
            )
        expr = tensor_to_expr[tensor]
        body += "{} = {};\n".format(name, expr)

    return body


def _get_sub_func_metadata(
    ops: List[Operator], data_t: str, op_t: str, backend_spec: BackendSpec
) -> Tuple[List[ElementwiseMetaData], str]:
    candidate_op_types = backend_spec.get_candidate_op_types(op_t)
    func_enums = []
    for op in ops:
        func_enum = op._attrs["func"]
        func_enums.append(func_enum)
        funcs = backend_spec.func_enum_to_func_name.get(func_enum)
        if funcs is None:
            raise NotImplementedError("Func {} is not supported!".format(func_enum))
        for candidate_op_t in candidate_op_types:
            func_name = funcs.get(candidate_op_t)
            if func_name is not None:
                candidate_op_types = backend_spec.get_candidate_op_types(candidate_op_t)
                break
    if len(candidate_op_types) == 0:
        raise RuntimeError(
            "Cannot find a common backend data type! candidate_op_types: {}, op_t: {}.".format(
                candidate_op_types, op_t
            )
        )
    if op_t in set(candidate_op_types):
        op_t = candidate_op_types[0]
    else:
        op_t = data_t
        candidate_op_types = backend_spec.get_candidate_op_types(op_t)

    sub_func_metadata = []
    for op in ops:
        func_enum = op._attrs["func"]
        funcs = backend_spec.func_enum_to_func_name.get(func_enum)
        func_name = None
        func_op_t = None
        for candidate_op_t in candidate_op_types:
            func_name = funcs.get(candidate_op_t)
            if func_name is not None:
                func_op_t = candidate_op_t
                break
        if func_name is None:
            raise NotImplementedError(
                "Unsupported func {} and op type {}!".format(func_enum, op_t)
            )
        sub_func_metadata.append(
            ElementwiseMetaData(
                func_name, func_op_t, op._attrs["args"], op._attrs["outputs"]
            )
        )
    return (sub_func_metadata, op_t)


def _get_types_and_sizes(
    inputs: List[Tensor],
    input_accessors: List[TensorAccessor],
    output_accessors: List[TensorAccessor],
    backend_spec: BackendSpec,
) -> Tuple[int, List[List[IntVar]], str]:
    """
    Returns Tuple(alignment, input_broadcast_sizes, dtype)
    """

    # Handle input broadcast.
    output_shape = output_accessors[0].original_shapes
    dtype = inputs[0]._attrs["dtype"]
    input_broadcast_sizes = []
    min_num_elements = None
    for input_accessor in input_accessors:
        input_shape = input_accessor.original_shapes
        broadcastable, _ = shape_utils.get_broadcast_max_shape(
            output_shape, input_shape
        )
        if not broadcastable:
            raise RuntimeError(
                "Input shape {} is not compatible with output shape {}!".format(
                    input_shape, output_shape
                )
            )
        num_rightmost_non_broadcast_elements = len(output_shape)
        extended_input_shape = list(input_shape)
        if input_shape == output_shape:
            input_broadcast_sizes.append(None)
        else:
            extended_input_shape = [IntImm(1)] * len(output_shape)
            extended_input_shape[len(output_shape) - len(input_shape) :] = input_shape
            input_broadcast_sizes.append(extended_input_shape)
            for i in reversed(range(len(extended_input_shape))):
                if extended_input_shape[i] != output_shape[i]:
                    num_rightmost_non_broadcast_elements -= i + 1
                    break
        num_elements_for_alignments = shape_utils.get_num_rightmost_static_elements(
            extended_input_shape, num_rightmost_non_broadcast_elements
        )
        if not min_num_elements:
            min_num_elements = num_elements_for_alignments
        else:
            min_num_elements = min(min_num_elements, num_elements_for_alignments)
    alignment = tensor_accessor_codegen.find_max_alignment(
        min_num_elements, dtype, output_accessors
    )
    # Note that we use the same alignment for accessing inputs and outputs, although
    # they may have different alignment requirements. We may lose perf a little bit,
    # but reduce the complexity of our jinja template. We can do some perf
    # experiments later to determine if we want to chase more perf gains.
    alignment = tensor_accessor_codegen.find_max_alignment(
        alignment, dtype, input_accessors
    )
    return alignment, input_broadcast_sizes, dtype


def _get_dynamic_dims(output_accessors: List[TensorAccessor]) -> List[IntVar]:
    res = {}
    for output_accessor in output_accessors:
        for dim in output_accessor.original_shapes:
            if not isinstance(dim, IntImm):
                res[dim._attrs["name"]] = dim
    return res.values()


def _parse_func_metadata(
    ops: List[Operator],
    inputs: List[Tensor],
    outputs: List[Tensor],
    input_accessors: List[TensorAccessor],
    output_accessors: List[TensorAccessor],
    original_inputs: List[Tensor],
    original_outputs: List[Tensor],
    backend_spec: BackendSpec,
) -> FusedElementwiseMetaData:
    alignment, input_broadcast_sizes, dtype = _get_types_and_sizes(
        inputs, input_accessors, output_accessors, backend_spec
    )
    read_type = backend_spec.get_elementwise_read_backend_type(alignment, dtype)
    op_type = backend_spec.get_elementwise_op_backend_type(alignment, dtype)
    data_type = backend_spec.dtype_to_backend_type(dtype)
    sub_func_metadata, op_type = _get_sub_func_metadata(
        ops, data_type, op_type, backend_spec
    )
    dynamic_dims = _get_dynamic_dims(output_accessors)

    return FusedElementwiseMetaData(
        inputs,
        outputs,
        input_accessors,
        output_accessors,
        original_inputs,
        original_outputs,
        read_type,
        op_type,
        data_type,
        input_broadcast_sizes,
        dynamic_dims,
        sub_func_metadata,
    )


def _gen_int_var_product_str(
    int_vars: List[IntVar],
) -> str:
    res = []
    for int_var in int_vars:
        if isinstance(int_var, IntImm):
            res.append(str(int_var._attrs["values"][0]))
        elif isinstance(int_var, IntVar):
            res.append(int_var._attrs["name"])
        else:
            raise RuntimeError(
                "A dim must be an IntVar! Current type: {}".format(type(int_var))
            )
    return " * ".join(res) if res else "1"


def _gen_input_broadcast_calculator_str(
    input_shape: List[IntVar],
    output_shape: List[IntVar],
) -> str:
    output_num_elements = []
    output_strides = []
    input_strides = []

    start_idx = 0
    for i, (input_dim, output_dim) in enumerate(zip(input_shape, output_shape)):
        if input_dim != output_dim:
            assert input_dim == IntImm(
                1
            ), "Unexpected shapes! Input: {}, output: {}".format(
                input_shape, output_shape
            )
            input_strides.append(input_shape[i:])
            output_strides.append(output_shape[i:])
            output_num_elements.append(output_shape[start_idx:])
            start_idx = i + 1
    if start_idx < len(output_shape):
        input_strides.append([IntImm(1)])
        output_strides.append([IntImm(1)])
        output_num_elements.append(output_shape[start_idx:])

    res = []
    for (output_num_element, output_stride, input_stride) in zip(
        output_num_elements, output_strides, input_strides
    ):
        res.append(
            "{} % ({}) / ({}) * ({})".format(
                "idx * N_ELEMENTS_PER_THREAD",
                _gen_int_var_product_str(output_num_element),
                _gen_int_var_product_str(output_stride),
                _gen_int_var_product_str(input_stride),
            )
        )

    return " + ".join(res)


def _gen_input_broadcast_size_str(
    input_broadcast_sizes: List[List[IntVar]],
    output_shape: List[IntVar],
) -> List[str]:
    res = []
    for input_broadcast_size in input_broadcast_sizes:
        if input_broadcast_size is None:
            res.append("")
        else:
            res.append(
                _gen_input_broadcast_calculator_str(input_broadcast_size, output_shape)
            )
    return res


def _gen_dynamic_dim_str(
    index_type: str, dynamic_dims: List[IntVar], has_type: bool
) -> str:
    type_str = index_type + " " if has_type else ""
    res = ", ".join([type_str + dim._attrs["name"] for dim in dynamic_dims])
    if res:
        res += ", "
    return res


def _gen_read_inputs_str(
    fused_elementwise_metadata: FusedElementwiseMetaData, broadcast_sizes: List[str]
):
    read_inputs = []
    for input_idx, (input_accessor, broadcast_size) in enumerate(
        zip(fused_elementwise_metadata.input_accessors, broadcast_sizes)
    ):
        input_name = f"input_tmp{input_idx}"
        data_idx = (
            "idx"
            if not broadcast_size
            else f"({broadcast_size}) / N_ELEMENTS_PER_THREAD"
        )
        get_strided_addr_str = GET_STRIDED_ADDRESS_TEMPLATE.render(
            tensor_accessor=input_accessor,
            data_ptr=input_name,
            data_t=fused_elementwise_metadata.data_t,
            read_t=fused_elementwise_metadata.read_t,
            data_idx=data_idx,
        )
        read_input = KERNEL_READ_INPUT_TEMPLATE.render(
            get_strided_address=get_strided_addr_str,
            input_name=input_name,
            input_idx=input_idx,
            read_t=fused_elementwise_metadata.read_t,
            op_t=fused_elementwise_metadata.op_t,
        )
        read_inputs.append(read_input)
    read_inputs_str = "\n".join(read_inputs)
    return read_inputs_str


def _gen_write_outputs_str(fused_elementwise_metadata: FusedElementwiseMetaData):
    write_outputs = []
    for output_idx, output_accessor in enumerate(
        fused_elementwise_metadata.output_accessors
    ):
        output_name = f"output{output_idx}"
        get_strided_addr_str = GET_STRIDED_ADDRESS_TEMPLATE.render(
            tensor_accessor=output_accessor,
            data_ptr=output_name,
            data_t=fused_elementwise_metadata.data_t,
            read_t=fused_elementwise_metadata.read_t,
            data_idx="idx",
        )
        write_out = KERNEL_WRITE_OUTPUT_TEMPLATE.render(
            get_strided_address=get_strided_addr_str,
            output_name=output_name,
            output_idx=output_idx,
        )
        write_outputs.append(write_out)
    write_outputs_str = "\n".join(write_outputs)
    return write_outputs_str


def _gen_kernel_function(
    func_attrs: Dict[str, Any],
    index_type: str,
    fused_elementwise_metadata: FusedElementwiseMetaData,
    backend_datatype_convertors: Dict[str, Dict[str, str]],
) -> str:
    output_params_decl = ",".join(
        [
            KERNEL_DECL_OUTPUT_PARAM_TEMPLATE.render(
                read_t=fused_elementwise_metadata.read_t, idx=i
            )
            for i, _ in enumerate(fused_elementwise_metadata.outputs)
        ]
    )
    input_params_decl = ",".join(
        [
            KERNEL_DECL_INPUT_PARAM_TEMPLATE.render(
                read_t=fused_elementwise_metadata.read_t, idx=i
            )
            for i, _ in enumerate(fused_elementwise_metadata.inputs)
        ]
    )

    broadcast_sizes = _gen_input_broadcast_size_str(
        fused_elementwise_metadata.input_broadcast_sizes,
        fused_elementwise_metadata.output_accessors[0].original_shapes,
    )
    read_inputs_str = _gen_read_inputs_str(fused_elementwise_metadata, broadcast_sizes)

    define_outputs = KERNEL_DEFINE_OUTPUTS_TEMPLATE.render(
        read_t=fused_elementwise_metadata.read_t,
        op_t=fused_elementwise_metadata.op_t,
        indexes=list(range(len(fused_elementwise_metadata.outputs))),
    )
    write_outputs_str = _gen_write_outputs_str(fused_elementwise_metadata)

    input_names = [
        KERNEL_TMP_INPUT_TEMPLATE.render(idx=i)
        for i, _ in enumerate(fused_elementwise_metadata.inputs)
    ]
    output_names = [
        KERNEL_TMP_OUTPUT_TEMPLATE.render(idx=i)
        for i, _ in enumerate(fused_elementwise_metadata.outputs)
    ]
    fused_funcs = gen_function_single_thread(
        fused_elementwise_metadata,
        input_names,
        output_names,
        backend_datatype_convertors,
    )

    kernel_func = KERNEL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=index_type,
        output_params=output_params_decl,
        input_params=input_params_decl,
        dynamic_dims=_gen_dynamic_dim_str(
            index_type, fused_elementwise_metadata.dynamic_dims, has_type=True
        ),
        read_inputs=read_inputs_str,
        define_outputs=define_outputs,
        write_outputs=write_outputs_str,
        fused_funcs=fused_funcs,
    )
    return kernel_func


def fused_elementwise_gen_function(
    func_attrs: Dict[str, Any],
    custom_libs: str,
    head_template: str,
    backend_spec: BackendSpec,
) -> str:
    """Generates fused_elementwise function definition."""

    ops = func_attrs["elementwise_ops"]
    inputs = func_attrs["inputs"]
    outputs = func_attrs["outputs"]
    input_accessors = func_attrs["input_accessors"]
    output_accessors = func_attrs["output_accessors"]
    original_inputs = func_attrs["original_inputs"]
    original_outputs = func_attrs["original_outputs"]
    fused_elementwise_metadata = _parse_func_metadata(
        ops,
        inputs,
        outputs,
        input_accessors,
        output_accessors,
        original_inputs,
        original_outputs,
        backend_spec,
    )
    # Dump data types into func_attr for testing purpose.
    func_attrs["read_t"] = fused_elementwise_metadata.read_t
    func_attrs["op_t"] = fused_elementwise_metadata.op_t
    func_attrs["data_t"] = fused_elementwise_metadata.data_t

    tensor_accessor_lib = tensor_accessor_codegen.get_libs()
    tensor_accessor_lib_str = "\n\n" + tensor_accessor_lib + "\n\n"

    kernel_function = _gen_kernel_function(
        func_attrs,
        backend_spec.index_type,
        fused_elementwise_metadata,
        backend_spec.backend_datatype_convertors,
    )
    output_params_decl = ",".join(
        [
            FUNC_DECL_OUTPUT_PARAM_TEMPLATE.render(idx=i)
            for i, _ in enumerate(fused_elementwise_metadata.outputs)
        ]
    )
    input_params_decl = ",".join(
        [
            FUNC_DECL_INPUT_PARAM_TEMPLATE.render(idx=i)
            for i, _ in enumerate(fused_elementwise_metadata.inputs)
        ]
    )
    kernel_call_output_params = ",".join(
        [
            KERNEL_CALL_OUTPUT_PARAM_TEMPLATE.render(
                read_t=fused_elementwise_metadata.read_t, idx=i
            )
            for i, _ in enumerate(fused_elementwise_metadata.outputs)
        ]
    )
    kernel_call_input_params = ",".join(
        [
            KERNEL_CALL_INPUT_PARAM_TEMPLATE.render(
                read_t=fused_elementwise_metadata.read_t, idx=i
            )
            for i, _ in enumerate(fused_elementwise_metadata.inputs)
        ]
    )
    constant = CONSTANT_TEMPLATE.render(
        read_t=fused_elementwise_metadata.read_t,
        op_t=fused_elementwise_metadata.op_t,
        data_t=fused_elementwise_metadata.data_t,
    )

    function = FUNC_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        head=backend_spec.header_src_template.render(extra_header=head_template),
        constant=constant,
        custom_libs=custom_libs,
        tensor_accessor_lib=tensor_accessor_lib_str,
        kernel_function=kernel_function,
        func_name=func_attrs["name"],
        output_params=output_params_decl,
        input_params=input_params_decl,
        dynamic_dims_decl=_gen_dynamic_dim_str(
            backend_spec.index_type,
            fused_elementwise_metadata.dynamic_dims,
            has_type=True,
        ),
        dynamic_dims_call=_gen_dynamic_dim_str(
            backend_spec.index_type,
            fused_elementwise_metadata.dynamic_dims,
            has_type=False,
        ),
        kernel_call_output_params=kernel_call_output_params,
        kernel_call_input_params=kernel_call_input_params,
    )
    return function


def fused_elementwise_gen_function_decl(
    func_attrs,
    backend_spec: BackendSpec,
):
    """Generates fused_elementwise function declaration."""

    func_name = func_attrs["name"]
    ops = func_attrs["elementwise_ops"]
    inputs = func_attrs["inputs"]
    outputs = func_attrs["outputs"]
    input_accessors = func_attrs["input_accessors"]
    output_accessors = func_attrs["output_accessors"]
    original_inputs = func_attrs["original_inputs"]
    original_outputs = func_attrs["original_outputs"]
    fused_elementwise_metadata = _parse_func_metadata(
        ops,
        inputs,
        outputs,
        input_accessors,
        output_accessors,
        original_inputs,
        original_outputs,
        backend_spec,
    )
    output_params_decl = ",".join(
        [
            FUNC_DECL_OUTPUT_PARAM_TEMPLATE.render(idx=i)
            for i, _ in enumerate(fused_elementwise_metadata.outputs)
        ]
    )
    input_params_decl = ",".join(
        [
            FUNC_DECL_INPUT_PARAM_TEMPLATE.render(idx=i)
            for i, _ in enumerate(fused_elementwise_metadata.inputs)
        ]
    )

    function_decl = FUNC_DECL_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        func_name=func_name,
        output_params=output_params_decl,
        input_params=input_params_decl,
        dynamic_dims=_gen_dynamic_dim_str(
            backend_spec.index_type,
            fused_elementwise_metadata.dynamic_dims,
            has_type=True,
        ),
    )
    return function_decl


def fused_elementwise_gen_function_call(
    func_attrs,
    indent: str,
    backend_spec: BackendSpec,
):
    """Generates fused_elementwise function call."""
    ops = func_attrs["elementwise_ops"]
    inputs = func_attrs["inputs"]
    outputs = func_attrs["outputs"]
    input_accessors = func_attrs["input_accessors"]
    output_accessors = func_attrs["output_accessors"]
    original_inputs = func_attrs["original_inputs"]
    original_outputs = func_attrs["original_outputs"]
    fused_elementwise_metadata = _parse_func_metadata(
        ops,
        inputs,
        outputs,
        input_accessors,
        output_accessors,
        original_inputs,
        original_outputs,
        backend_spec,
    )

    output_params = ",".join([output._attrs["name"] for output in outputs])

    input_params = ",".join([input._attrs["name"] for input in inputs])

    num_elements_calculator = _gen_int_var_product_str(
        output_accessors[0].original_shapes
    )

    return FUNC_CALL_TEMPLATE.render(
        stream=backend_spec.stream,
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
        calculate_n=num_elements_calculator,
        output_params=output_params,
        input_params=input_params,
        dynamic_dims=_gen_dynamic_dim_str(
            backend_spec.index_type,
            fused_elementwise_metadata.dynamic_dims,
            has_type=False,
        ),
        indent=indent,
    )
