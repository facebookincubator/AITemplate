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
Common functions and templates for bmm-family ops
"""
from dataclasses import dataclass

import jinja2

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common import gemm_common
from aitemplate.backend.cuda.gemm_universal import common

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703

# ARGS_PARSER is only used by profiler, so the batch is not of concern.
ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t B = std::atoi(argv[1]);
  int64_t M = std::atoi(argv[2]);
  int64_t N = std::atoi(argv[3]);
  int64_t K = std::atoi(argv[4]);

{% for dim in a_dims %}
  int64_t a_dim{{loop.index0}} = {{dim}};
{% endfor %}
{% for dim in b_dims %}
  int64_t b_dim{{loop.index0}} = {{dim}};
{% endfor %}
{% for dim in c_dims %}
  int64_t c_dim{{loop.index0}} = {{dim}};
{% endfor %}
"""
)

OUTPUT_ADDR_CALCULATOR = jinja2.Template(
    """
  int64_t output_batch_stride = {{output_batch_stride_dim}};
  int64_t output_stride = {{output_stride_dim}};
  int64_t output_offset = {{output_offset_val}}; // default to 0
    """
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
{% if has_d %}
  void*,
{% endif %}
  void*,
  uint8_t*,
{% if support_split_k %}
  int,
{% endif %}
{% for idx in range(a_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(b_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(c_ndims) %}
  int64_t*,
{% endfor %}
  cudaStream_t
);
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}{{local_dim_defs}}
{{indent}}{{func_name}}(
{% if is_profiler %}
{{indent}}    gemm_op,
{% endif %}
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{% if has_d %}
{{indent}}    {{d_ptr}},
{% endif %}
{% if has_bias %}
{{indent}}    {{bias_ptr}},
{% endif %}
{{indent}}    {{c_ptr}},
{{indent}}    global_workspace_,
{% for dim in a_dims_ptr %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in b_dims_ptr %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in c_dims_ptr %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    stream
{{indent}});
{{indent}}}
"""
)


TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  // cast to int64_t to avoid overflow
  int64_t a_ptr_sz = 1;
  {% for idx in range(a_ndims) %}
  {{indent}} {{indent}} a_ptr_sz *= static_cast<int64_t>(a_dim{{idx}});
  {% endfor %}

  int64_t b_ptr_sz = 1;
  {% for idx in range(b_ndims) %}
  {{indent}} {{indent}} b_ptr_sz *= static_cast<int64_t>(b_dim{{idx}});
  {% endfor %}

  int64_t c_ptr_sz = 1;
  {% for idx in range(c_ndims) %}
  {{indent}} {{indent}} c_ptr_sz *= static_cast<int64_t>(c_dim{{idx}});
  {% endfor %}

  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b_ptr_sz, c_ptr_sz});
  size_t one_copy_sz = a_ptr_sz + b_ptr_sz + c_ptr_sz;
{% if has_bias %}
  one_copy_sz += c_dim2;
{%endif%}
{% if has_d %}
  one_copy_sz += c_ptr_sz;
{%endif%}
  int64_t mem_pool_sz = memory_pool->ComputeMemPoolSize(one_copy_sz, ptr_max_sz);

  memory_pool->AllocateTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateTensor(b_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz, /*is_output*/true);  // c_ptr: index 2
{% if has_bias %}
  memory_pool->AllocateTensor(c_dim2, mem_pool_sz);  // bias_ptr: index 3
{% endif %}
{% if has_d %}
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz);  // d_ptr: index 3 (no bias) or 4
{% endif %}
"""
)


@dataclass
class Bmm_problem_info:
    alpha_value: float = 1
    beta_value: float = 0
    problem_size: str = "{M, N, K}"
    batch_size: str = "B"
    a_ptr: str = "a_ptr"
    b_ptr: str = "b_ptr"
    bias_ptr: str = "d_ptr"
    c_ptr: str = "c_ptr"
    a_batch_stride: str = "0"
    b_batch_stride: str = "0"
    bias_batch_stride: str = "0"
    c_batch_stride: str = "0"
    lda: str = "0"
    ldb: str = "0"
    ldbias: str = "0"
    ldc: str = "0"


def _update_stride_info(mm_info, a_shapes, b_shapes, bias_shapes=None):
    if len(a_shapes) == 2 or a_shapes[0] == 1:
        mm_info.a_batch_stride = "0"
    if len(b_shapes) == 2 or b_shapes[0] == 1:
        mm_info.b_batch_stride = "0"

    if bias_shapes is None:
        return

    if len(bias_shapes) < 3 or bias_shapes[0] == 1:
        mm_info.bias_batch_stride = "0"
    if len(bias_shapes) < 2 or all([x == 1 for x in bias_shapes[:-1]]):
        mm_info.ldbias = "0"


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kBatched,                                                         // GemmUniversalMode mode
    {{mm_info.problem_size}},                                                                           // GemmCoord problem_size
    {{mm_info.batch_size}},                                                                             // int batch_count
    {ElementComputeEpilogue({{mm_info.alpha_value}}), ElementComputeEpilogue({{mm_info.beta_value}})},  // typename EpilogueOutputOp::Params epilogue
    {{mm_info.a_ptr}},                                                                                  // void const * ptr_A
    {{mm_info.b_ptr}},                                                                                  // void const * ptr_B
    {{mm_info.bias_ptr}},                                                                               // void const * ptr_C
    {{mm_info.c_ptr}},                                                                                  // void * ptr_D
    {{mm_info.a_batch_stride}},                                                                         // int64_t batch_stride_A
    {{mm_info.b_batch_stride}},                                                                         // int64_t batch_stride_B
    {{mm_info.bias_batch_stride}},                                                                      // int64_t batch_stride_C
    {{mm_info.c_batch_stride}},                                                                         // int64_t batch_stride_D
    {{mm_info.lda}},                                                                                    // typename LayoutA::Stride::LongIndex lda
    {{mm_info.ldb}},                                                                                    // typename LayoutB::Stride::LongIndex ldb
    {{mm_info.ldbias}},                                                                                 // typename LayoutC::Stride::LongIndex ldc
    {{mm_info.ldc}},                                                                                    // typename LayoutC::Stride::LongIndex ldd
"""
)


def reverse_dim_info_mapping(dim_info_dict, source, tensor_idx):
    def _fill(arr, idx, val):
        if len(arr) <= idx:
            arr = arr + [None] * (idx - len(arr) + 1)
        arr[idx] = val
        return arr

    ret = []
    for name, dim_infos in dim_info_dict.items():
        for dim_info in dim_infos:
            if dim_info.source == source and dim_info.tensor_idx == tensor_idx:
                for dim_idx in dim_info.dim_idx:
                    ret = _fill(ret, dim_idx, name)

    if None in ret:
        raise RuntimeError(
            "dim_info_dict for source: {}, tensor_idx: {} not complete.".format(
                source, tensor_idx
            )
        )

    return ret


def get_default_problem_info(default_problem_args, **kwargs):
    """Return the default problem args"""
    problem_args = default_problem_args.copy()
    for k, v in kwargs.items():
        problem_args[k] = v

    bmm_problem_info = Bmm_problem_info(**problem_args)
    return bmm_problem_info


def make_function_strided_args(
    func_attrs,
    dim_info_dict,
    default_mm_info,
    is_permute=False,
):
    """
    Return a tuple of (problem_args, input_addr_calculator, output_addr_calculator)
    """
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )

    input_a_batch_stride_dim = default_mm_info.a_batch_stride
    input_a_stride_lda_dim = default_mm_info.lda
    input_a_offset = 0
    input_b_batch_stride_dim = default_mm_info.b_batch_stride
    input_b_stride_ldb_dim = default_mm_info.ldb
    input_b_offset = 0

    has_bias = len(func_attrs["inputs"]) == 3

    if "input_accessors" in func_attrs:
        input_a_accessor = func_attrs["input_accessors"][0]
        input_b_accessor = func_attrs["input_accessors"][1]

        if input_a_accessor.is_from_strided_tensor:
            input_a_offset = input_a_accessor.offset
            if not input_a_accessor.is_contiguous:
                a_dims = reverse_dim_info_mapping(
                    dim_info_dict, gemm_common.Source.INPUT, 0
                )

                input_a_batch_stride_dim = input_a_accessor.gen_stride_str(0, a_dims)
                input_a_stride_lda_dim = input_a_accessor.stride(1)

        if input_b_accessor.is_from_strided_tensor:
            input_b_offset = input_b_accessor.offset
            if not input_b_accessor.is_contiguous:
                b_dims = reverse_dim_info_mapping(
                    dim_info_dict, gemm_common.Source.INPUT, 1
                )
                input_b_batch_stride_dim = input_b_accessor.gen_stride_str(0, b_dims)
                input_b_stride_ldb_dim = input_b_accessor.stride(1)

        if has_bias:
            # FIXME: we don't suppor strided bias yet. Will enable it once
            # we support it.
            input_bias_accessor = func_attrs["input_accessors"][2]
            assert (
                not input_bias_accessor.is_from_strided_tensor
            ), f'strided bias is not supported for op {func_attrs["name"]}'

    input_addr_calculator = common.INPUT_ADDR_CALCULATOR.render(
        input_a_batch_stride_dim=input_a_batch_stride_dim,
        input_a_stride_dim=input_a_stride_lda_dim,
        input_a_offset_val=input_a_offset,
        input_b_batch_stride_dim=input_b_batch_stride_dim,
        input_b_stride_dim=input_b_stride_ldb_dim,
        input_b_offset_val=input_b_offset,
    )

    # bmm_permute requires a slightly different c_batch_stride and
    # output_batch_stride_dim values
    if is_permute:
        output_batch_stride_dim = default_mm_info.bias_batch_stride
        c_batch_stride = default_mm_info.c_batch_stride
    else:
        output_batch_stride_dim = default_mm_info.c_batch_stride
        c_batch_stride = "output_batch_stride"
    output_stride_ldc_dim = default_mm_info.ldc
    output_offset = 0

    if "output_accessors" in func_attrs:
        output_accessor = func_attrs["output_accessors"][0]
        if output_accessor.is_from_strided_tensor:
            output_offset = output_accessor.offset
            if not output_accessor.is_contiguous:
                c_dims = reverse_dim_info_mapping(
                    dim_info_dict, gemm_common.Source.OUTPUT, 0
                )
                output_batch_stride_dim = output_accessor.gen_stride_str(0, c_dims)
                output_stride_ldc_dim = output_accessor.stride(1)

    output_addr_calculator = OUTPUT_ADDR_CALCULATOR.render(
        output_batch_stride_dim=output_batch_stride_dim,
        output_stride_dim=output_stride_ldc_dim,
        output_offset_val=output_offset,
    )

    bmm_problem_info = Bmm_problem_info(
        alpha_value=default_mm_info.alpha_value,
        beta_value=default_mm_info.beta_value,
        a_ptr=f"({elem_input_type}*)({default_mm_info.a_ptr}) + input_a_offset",
        b_ptr=f"({elem_input_type}*)({default_mm_info.b_ptr}) + input_b_offset",
        bias_ptr=f"({elem_output_type}*)({default_mm_info.bias_ptr})",
        c_ptr=f"({elem_output_type}*)({default_mm_info.c_ptr}) + output_offset",
        a_batch_stride="input_a_batch_stride",
        b_batch_stride="input_b_batch_stride",
        bias_batch_stride=f"{default_mm_info.bias_batch_stride}",
        c_batch_stride=c_batch_stride,
        lda="input_a_stride",
        ldb="input_b_stride",
        ldbias=f"{default_mm_info.ldbias}",
        ldc="output_stride",
    )
    a_shapes = func_attrs["input_accessors"][0].original_shapes
    b_shapes = func_attrs["input_accessors"][1].original_shapes
    d_shapes = None
    if has_bias:
        d_shapes = func_attrs["input_accessors"][2].original_shapes
    _update_stride_info(bmm_problem_info, a_shapes, b_shapes, d_shapes)

    problem_args = PROBLEM_ARGS_TEMPLATE.render(
        mm_info=bmm_problem_info,
    )
    return (problem_args, input_addr_calculator, output_addr_calculator)


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    src_template,
    problem_args,
    args_parser,
    bias_ptr_arg=None,
):
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    backend_spec = CUDASpec()
    elem_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    has_d = False
    if "has_d" in func_attrs:
        has_d = func_attrs["has_d"]

    a_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    b_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    c_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    a_dims_ptr = [f"&a_dim{idx}" for idx in range(a_ndims)]
    b_dims_ptr = [f"&b_dim{idx}" for idx in range(b_ndims)]
    c_dims_ptr = [f"&c_dim{idx}" for idx in range(c_ndims)]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    has_bias = bias_ptr_arg is not None
    assert not (has_d and has_bias)
    instance_name_base = "GemmInstance"
    exec_program = common.EXEC_TEMPLATE.render(
        indent="  ",
        instance=instance_name_base,
        is_profiler=True,
        problem_args=problem_args,
    )
    input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=a_ndims,
        weight_ndims=b_ndims,
        output_ndims=c_ndims,
    )

    function_name = "bmm"
    instances = []
    benchmark_instances = []
    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        config = common.emit_instance(op, for_profiler=True)
        config_name = common.extract_config_name(config)
        instance_name = f"{instance_name_base}_{instance_idx}"
        gemm_op = f"gemm_op_{instance_idx}"
        instance = common.INSTANCE_TEMPLATE.render(
            config_name=config_name, name=instance_name, config=config
        )
        benchmark_instance = common.BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            instance_name=instance_name,
            gemm_op=gemm_op,
            gemm_op_name=op_name,
            func_name=f"benchmark_{function_name}",
            adims=a_dims_ptr,
            bdims=b_dims_ptr,
            cdims=c_dims_ptr,
        )
        instances.append(instance)
        benchmark_instances.append(benchmark_instance)
    op_func = src_template.render(
        is_profiler=True,
        instances="\n".join(instances),
        function_name=function_name,
        input_ndims=a_ndims,
        weight_ndims=b_ndims,
        output_ndims=c_ndims,
        shape_eval=shape_func,
        input_output_checks=input_output_checks,
        exec_paths=exec_program,
        has_d=has_d,
    )
    benchmark_adims = [f"a_dim{idx}" for idx in range(a_ndims)]
    benchmark_bdims = [f"b_dim{idx}" for idx in range(b_ndims)]
    benchmark_cdims = [f"c_dim{idx}" for idx in range(c_ndims)]
    func_call = FUNC_CALL_TEMPLATE.render(
        is_profiler=True,
        func_name=function_name,
        a_ptr="memory_pool->RequestTensorByIdx(0)",
        b_ptr="memory_pool->RequestTensorByIdx(1)",
        has_bias=has_bias,
        bias_ptr=bias_ptr_arg,
        c_ptr="memory_pool->RequestTensorByIdx(2)",
        d_ptr="memory_pool->RequestTensorByIdx(%d)" % (4 if has_bias else 3),
        has_d=has_d,
        a_dims_ptr=benchmark_adims,
        b_dims_ptr=benchmark_bdims,
        c_dims_ptr=benchmark_cdims,
    )
    code = common.PROFILER_TEMPLATE.render(
        op_func=op_func,
        has_bias=has_bias,
        has_d=has_d,
        args_parse=args_parser,
        function_name=function_name,
        func_call=func_call,
        name=instance_name_base,
        input_ndims=a_ndims,
        weight_ndims=b_ndims,
        output_ndims=c_ndims,
        tensor_decl=TENSOR_DECL_TEMPLATE.render(
            a_ndims=a_ndims,
            b_ndims=b_ndims,
            c_ndims=c_ndims,
            has_d=has_d,
            has_bias=has_bias,
        ),
        benchmark_instances="\n".join(benchmark_instances),
        elem_type=elem_type,
    )
    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    common.add_profiler(file_pairs, workdir, op_type, profiler_filename, code)
    # build
    return common.build_profiler(file_pairs)


def default_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    default_problem_args,
):
    """default function for generating bmm profilers"""
    a_dims = reverse_dim_info_mapping(dim_info_dict, gemm_common.Source.INPUT, 0)
    b_dims = reverse_dim_info_mapping(dim_info_dict, gemm_common.Source.INPUT, 1)
    c_dims = reverse_dim_info_mapping(dim_info_dict, gemm_common.Source.OUTPUT, 0)

    args_parser = ARGS_PARSER_TEMPLATE.render(
        a_dims=a_dims, b_dims=b_dims, c_dims=c_dims
    )

    default_mm_info = get_default_problem_info(
        default_problem_args,
        alpha_value=func_attrs.get("alpha", 1),
    )
    a_shapes = func_attrs["input_accessors"][0].original_shapes
    b_shapes = func_attrs["input_accessors"][1].original_shapes
    _update_stride_info(default_mm_info, a_shapes, b_shapes)

    problem_args = PROBLEM_ARGS_TEMPLATE.render(
        mm_info=default_mm_info,
    )

    return gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        common.SRC_TEMPLATE,
        problem_args,
        args_parser,
    )


def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    has_d = False
    if "has_d" in func_attrs:
        has_d = func_attrs["has_d"]
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        a_ndims=len(func_attrs["input_accessors"][0].original_shapes),
        b_ndims=len(func_attrs["input_accessors"][1].original_shapes),
        c_ndims=len(func_attrs["output_accessors"][0].original_shapes),
        has_d=has_d,
    )


def gen_function(
    func_attrs,
    exec_cond_template,
    problem_args,
    dim_info_dict,
    input_addr_calculator="",
    output_addr_calculator="",
):
    return common.gen_function(
        func_attrs,
        common.SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims=len(func_attrs["input_accessors"][0].original_shapes),
        weight_ndims=len(func_attrs["input_accessors"][1].original_shapes),
        output_ndims=len(func_attrs["output_accessors"][0].original_shapes),
        dim_info_dict=dim_info_dict,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=output_addr_calculator,
    )


def gen_function_call(func_attrs, indent="  ", bias_ptr_arg=None):
    a = func_attrs["inputs"][0]
    ashape = func_attrs["input_accessors"][0].original_shapes
    a_dims_ptr = [f'&{ashape[idx]._attrs["name"]}' for idx in range(len(ashape))]
    b = func_attrs["inputs"][1]
    bshape = func_attrs["input_accessors"][1].original_shapes
    b_dims_ptr = [f'&{bshape[idx]._attrs["name"]}' for idx in range(len(bshape))]
    c = func_attrs["outputs"][0]
    cshape = func_attrs["output_accessors"][0].original_shapes
    c_dims_ptr = [f'&{cshape[idx]._attrs["name"]}' for idx in range(len(cshape))]
    has_d = False
    d_ptr = None
    if "has_d" in func_attrs:
        has_d = func_attrs["has_d"]
        d_ptr = func_attrs["inputs"][2]._attrs["name"]
    has_bias = bias_ptr_arg is not None
    assert not (has_d and has_bias)

    local_dim_defs = common.gen_local_dim_defs(func_attrs, indent=indent)

    return FUNC_CALL_TEMPLATE.render(
        local_dim_defs=local_dim_defs,
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        has_bias=has_bias,
        bias_ptr=bias_ptr_arg,
        c_ptr=c._attrs["name"],
        d_ptr=d_ptr,
        has_d=has_d,
        a_dims_ptr=a_dims_ptr,
        b_dims_ptr=b_dims_ptr,
        c_dims_ptr=c_dims_ptr,
        indent=indent,
    )
