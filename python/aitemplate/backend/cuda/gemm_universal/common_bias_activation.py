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
Common codegen functions for gemm_bias_activation.
"""
import jinja2

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal import common, common_bias, gemm_rcr
from aitemplate.backend.cuda.gemm_universal.layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


EXTRA_CODE_HEADER = jinja2.Template(
    """
using elem_input_type = {{elem_input_type}};
using elem_output_type = {{elem_output_type}};
"""
)


def gemm_rcr_config(
    func_attrs,
    dtype="float16",
    include_cutlass_3x_ops=False,
):
    common.make_fproc(
        func_attrs=func_attrs,
        layout=RCR,
        include_cutlass_3x_ops=include_cutlass_3x_ops,
    )

    import cutlass_lib

    for op in func_attrs["op_instance"].values():
        if common.has_tma_epilogue(op):
            # disable residual to leave more SMEM for the mainloop
            op.C.element = cutlass_lib.library.DataType.void

            # swap the output layout to the transposed problem
            op.C.layout = cutlass_lib.library.LayoutType.ColumnMajor
            op.D.layout = cutlass_lib.library.LayoutType.ColumnMajor

            # switch to a TMA epilogue with bias
            op.epilogue_schedule = (
                cutlass_lib.library.EpilogueScheduleBiasElementwiseMapping[
                    op.epilogue_schedule
                ]
            )


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    problem_args_template,
    problem_args_template_cutlass_3x=None,
    extra_code="",
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    extra_code_header = EXTRA_CODE_HEADER.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return gemm_rcr.common_gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=common_bias.SRC_TEMPLATE,
        problem_args_template=problem_args_template,
        problem_args_template_cutlass_3x=problem_args_template_cutlass_3x,
        bias_ptr_arg="memory_pool->RequestTensorByIdx(3)",
        extra_code="\n\n".join([extra_code_header, extra_code]),
    )


def gen_function(
    func_attrs,
    problem_args_template,
    exec_cond_template,
    dim_info_dict,
    problem_args_template_cutlass_3x=None,
    extra_code="",
):
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    problem_args = problem_args_template.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    problem_args_cutlass_3x = ""
    if problem_args_template_cutlass_3x is not None:
        problem_args_cutlass_3x = problem_args_template_cutlass_3x.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
        )
    extra_code_header = EXTRA_CODE_HEADER.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return common.gen_function(
        func_attrs=func_attrs,
        src_template=common_bias.SRC_TEMPLATE,
        exec_cond_template=exec_cond_template,
        problem_args=problem_args,
        problem_args_cutlass_3x=problem_args_cutlass_3x,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
        support_split_k=True,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N",
            output_accessor=func_attrs["output_accessors"][0],
        ),
        extra_code="\n\n".join([extra_code_header, extra_code]),
    )


def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=True,
    )


def gen_function_call(func_attrs, indent="  "):
    bias = func_attrs["inputs"][2]
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        bias_ptr_arg=bias._attrs["name"],
    )
