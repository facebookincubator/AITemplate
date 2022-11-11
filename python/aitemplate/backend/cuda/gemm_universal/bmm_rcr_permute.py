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
Codegen for bmm_rcr_permute, which computes permute(A @ B + bias).
A[RowMajor], B[ColMajor], bias[RowMajor]
"""

from ... import registry
from ...backend_spec import CUDASpec
from ...common import gemm_common
from . import bmm_common, bmm_permute_common, common, common_permute

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


@registry.reg("cuda.bmm_rcr_permute.config")
def bmm_rcr_permute_config(func_attrs, dtype="float16"):
    def fproc(op):
        import cutlass_lib

        from ...backend_spec import CUDASpec

        backend_spec = CUDASpec()
        elem_type = backend_spec.dtype_to_lib_type(
            func_attrs["inputs"][0]._attrs["dtype"]
        )

        return common.default_fproc(
            op=op,
            a_layout=cutlass_lib.library.LayoutType.RowMajor,
            b_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            c_layout=cutlass_lib.library.LayoutType.RowMajor,
            elem_type=elem_type,
            epiligue_name=func_attrs["epilogue"],
            permute_layout=func_attrs["layout"],
        )

    func_attrs["op_instance"] = common_permute.extract_config(fproc, func_attrs)


@registry.reg("cuda.bmm_rcr_permute.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    a_dims = bmm_common.reverse_dim_info_mapping(
        dim_info_dict, gemm_common.Source.INPUT, 0
    )
    b_dims = bmm_common.reverse_dim_info_mapping(
        dim_info_dict, gemm_common.Source.INPUT, 1
    )
    c_dims = bmm_common.reverse_dim_info_mapping(
        dim_info_dict, gemm_common.Source.OUTPUT, 0
    )

    args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
        a_dims=a_dims, b_dims=b_dims, c_dims=c_dims
    )

    bmm_problem_info = bmm_common.Bmm_problem_info(
        alpha_value=func_attrs.get("alpha", 1),
        bias_ptr="c_ptr",
        a_batch_stride="M * K",
        b_batch_stride="N * K",
        bias_batch_stride="M * N",
        c_batch_stride="0",
        lda="K",
        ldb="K",
        ldbias="N",
        ldc="N",
    )
    a_shapes = func_attrs["input_accessors"][0].original_shapes
    b_shapes = func_attrs["input_accessors"][1].original_shapes
    bmm_common._update_stride_info(bmm_problem_info, a_shapes, b_shapes)

    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=bmm_problem_info,
    )

    return bmm_permute_common.gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        common.SRC_TEMPLATE,
        problem_args,
        args_parser,
        emit_kernel=True,
        extra_code=common_permute.EXTRA_CODE.render(),
    )


@registry.reg("cuda.bmm_rcr_permute.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )

    input_a_batch_stride_dim = "M * K"
    input_a_stride_k_dim = "K"
    input_a_offset = 0
    input_b_batch_stride_dim = "N * K"
    input_b_stride_k_dim = "K"
    input_b_offset = 0

    if "input_accessors" in func_attrs:
        input_a_accessor = func_attrs["input_accessors"][0]
        input_b_accessor = func_attrs["input_accessors"][1]

        if input_a_accessor.is_from_strided_tensor:
            input_a_offset = input_a_accessor.offset
            if not input_a_accessor.is_contiguous:
                input_a_batch_stride_dim = input_a_accessor.stride(0)
                input_a_stride_k_dim = input_a_accessor.stride(1)

        if input_b_accessor.is_from_strided_tensor:
            input_b_offset = input_b_accessor.offset
            if not input_b_accessor.is_contiguous:
                input_b_batch_stride_dim = input_b_accessor.stride(0)
                input_b_stride_k_dim = input_b_accessor.stride(1)

    input_addr_calculator = common.INPUT_ADDR_CALCULATOR.render(
        input_a_batch_stride_dim=input_a_batch_stride_dim,
        input_a_stride_dim=input_a_stride_k_dim,
        input_a_offset_val=input_a_offset,
        input_b_batch_stride_dim=input_b_batch_stride_dim,
        input_b_stride_dim=input_b_stride_k_dim,
        input_b_offset_val=input_b_offset,
    )

    output_batch_stride_dim = "M * N"
    output_stride_n_dim = "N"
    output_offset = 0

    if "output_accessors" in func_attrs:
        output_accessor = func_attrs["output_accessors"][0]
        if output_accessor.is_from_strided_tensor:
            output_offset = output_accessor.offset
            if not output_accessor.is_contiguous:
                output_batch_stride_dim = output_accessor.stride(0)
                output_stride_n_dim = output_accessor.stride(1)

    output_addr_calculator = bmm_common.OUTPUT_ADDR_CALCULATOR.render(
        output_batch_stride_dim=output_batch_stride_dim,
        output_stride_dim=output_stride_n_dim,
        output_offset_val=output_offset,
    )

    bmm_problem_info = bmm_common.Bmm_problem_info(
        alpha_value=func_attrs.get("alpha", 1),
        a_ptr="(" + elem_input_type + "*)(a_ptr) + input_a_offset",
        b_ptr="(" + elem_input_type + "*)(b_ptr) + input_b_offset",
        bias_ptr="(" + elem_output_type + "*)(c_ptr) + output_offset",
        c_ptr="(" + elem_output_type + "*)(c_ptr) + output_offset",
        a_batch_stride="input_a_batch_stride",
        b_batch_stride="input_b_batch_stride",
        bias_batch_stride="output_batch_stride",
        c_batch_stride="0",
        lda="input_a_stride",
        ldb="input_b_stride",
        ldbias="output_stride",
        ldc="output_stride",
    )
    a_shapes = func_attrs["input_accessors"][0].original_shapes
    b_shapes = func_attrs["input_accessors"][1].original_shapes
    bmm_common._update_stride_info(bmm_problem_info, a_shapes, b_shapes)

    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=bmm_problem_info,
    )

    return bmm_permute_common.gen_function(
        func_attrs,
        exec_cond_template,
        problem_args,
        dim_info_dict,
        input_addr_calculator,
        output_addr_calculator,
        extra_code=common_permute.EXTRA_CODE.render(),
    )


@registry.reg("cuda.bmm_rcr_permute.func_decl")
def gen_function_decl(func_attrs):
    return bmm_permute_common.gen_function_decl(func_attrs)


@registry.reg("cuda.bmm_rcr_permute.func_call")
def gen_function_call(func_attrs, indent="  "):
    return bmm_permute_common.gen_function_call(func_attrs, indent)


@registry.reg("cuda.bmm_rcr_permute.filter")
def function_filter(cfg, func_attrs, ab_alignment):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    ab_alignment:
        Input alignments.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common.function_filter(cfg, func_attrs, ab_alignment)
