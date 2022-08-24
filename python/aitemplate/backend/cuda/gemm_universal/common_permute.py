# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] common template for conv2d
"""
import re
from collections import OrderedDict
from hashlib import sha1

import jinja2

from ...common import gemm_common
from ...target import Target
from ..gemm_universal import common

# pylint: disable=C0301,C0415,R1705

EXTRA_CODE = jinja2.Template(
    """
#include "cutlass/gemm/device/gemm_universal_with_perm.h"
"""
)

# HACK: we don't record different permutation shape,
# because it has little impact on execution time compared.
# Therefore, no matter what permutation shape it is,
# we will use the same kernel, i.e. the first generated perm_shape
# At runtime, the kernel will be regenerated and thus the correctness will not be affected.
KERNEL_KEY_TEMPLATE = jinja2.Template(
    """
cutlass_{{opcode_class_name}}_{{extended_name}}_{{threadblock}}_{{layout}}_align_{{align_ab}}_{{align_c}}
"""
)


def kernel_name(op, func_attrs):
    """[summary]

    Parameters
    ----------
    op : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    from cutlass_lib import library

    threadblock = op.tile_description.procedural_name()
    extended_name = op.extended_name()
    opcode_class_name = library.OpcodeClassNames[
        op.tile_description.math_instruction.opcode_class
    ]
    layout = op.layout_name()
    align_ab = op.A.alignment
    align_c = op.C.alignment
    name = KERNEL_KEY_TEMPLATE.render(
        threadblock=threadblock,
        extended_name=extended_name,
        opcode_class_name=opcode_class_name,
        layout=layout,
        align_ab=align_ab,
        align_c=align_c,
    )
    return name.replace("\n", "")


def default_fproc_f16(
    *, op, a_layout, b_layout, c_layout, epiligue_name, permute_layout
):
    """[summary]

    Parameters
    ----------
    op: [type]
        [description]
    a_layout: [type]
        [description]
    b_layout: [type]
        [description]
    c_layout: [type]
        [description]
    Returns
    -------
    [type]
        [description]
    """
    import copy

    import cutlass_lib

    ret = []
    data_type = cutlass_lib.library.DataType.f16
    acc_type = cutlass_lib.library.DataType.f32
    # check target use fp16 acc
    if "use_fp16_acc" in Target.current()._kwargs:
        if Target.current()._kwargs["use_fp16_acc"]:
            acc_type = cutlass_lib.library.DataType.f16
    if (
        op.A.element == data_type
        and op.B.element == data_type
        and op.C.element == data_type
        and op.accumulator_type() == acc_type
        and op.A.layout == a_layout
        and op.B.layout == b_layout
    ):
        op = copy.deepcopy(op)
        # set output major
        op.C.layout = c_layout
        # set epilogue
        op.epilogue_functor = cutlass_lib.library.EpilogueFunctorName[epiligue_name]
        op.element_epilogue = acc_type
        op.permute_layout = cutlass_lib.library.EpiloguePermuteLayoutName[
            permute_layout
        ]
        # set C alignment
        for i in [8, 4, 2, 1]:
            op = copy.deepcopy(op)
            op.C.alignment = i
            ret.append(op)
    return ret


def extract_config(f_proc_op, func_attrs):
    """[summary]

    Parameters
    ----------
    dtype : str, optional
        [description], by default "float16"
    f_process_epilogue : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    import cutlass_lib

    op_kind = cutlass_lib.library.OperationKind.Gemm
    gemm_kind = cutlass_lib.library.GemmKind.Universal
    gemm_ops = OrderedDict()
    extract_ops = list(Target.current()._operators[op_kind].items())

    for _, value in extract_ops:
        op = value[0]
        if op.gemm_kind == gemm_kind:
            ret = f_proc_op(op)
            if len(ret) > 0:
                for op_inst in ret:
                    key = kernel_name(op_inst, func_attrs)
                    gemm_ops[key] = op_inst
    return gemm_ops


def gemm_permute_instance(op_def, func_attrs, for_profiler):
    """_summary_

    Parameters
    ----------
    op_def : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    import cutlass_lib

    op_def = common.update_alignments_in_gemm_instance(
        op_def,
        func_attrs,
        for_profiler,
        # expected to have 26 of params, the index offset of alignment value
        # in the full op_def string is 4
        kernel_config=("cutlass::gemm::device::permute::GemmUniversal", 26, 4),
    )
    shape_info = ", ".join(map(str, func_attrs["shape"]))
    layout = cutlass_lib.library.EpiloguePermuteLayoutName[func_attrs["layout"]]
    layout_class = cutlass_lib.library.EpiloguePermuteLayoutTag[layout]
    tmp = re.sub(
        r"{}".format(layout_class), "{}<{}>".format(layout_class, shape_info), op_def
    )
    return tmp


def emit_instance(
    op,
    for_profiler,
    f_instance_convertor=gemm_permute_instance,
    emit_kernel=False,
    func_attrs=None,
):
    import cutlass_lib

    emiter = cutlass_lib.gemm_operation.EmitGemmInstance()
    if emit_kernel:
        emiter = cutlass_lib.gemm_operation.EmitGemmPermuteInstance()

    op_def = emiter.emit(op)
    op_def = f_instance_convertor(op_def, func_attrs, for_profiler)
    return op_def


def gen_function(
    func_attrs,
    src_template,
    exec_cond_template,
    problem_args,
    input_ndims,
    weight_ndims,
    output_ndims,
    dim_info_dict,
    f_instance_convertor=gemm_permute_instance,
    emit_kernel=False,
    support_split_k=False,
    input_addr_calculator="",
    output_addr_calculator="",
    extra_code="",
):
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]
    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for exec_item in exec_path.values():
        fname = "f" + sha1(exec_item.exec_cond.encode()).hexdigest()
        algo = exec_item.algo
        if algo not in inst_def_flag:
            config = emit_instance(
                op_instance[algo],
                for_profiler=False,
                f_instance_convertor=f_instance_convertor,
                emit_kernel=emit_kernel,
                func_attrs=func_attrs,
            )
            inst_def_flag.add(algo)
        else:
            config = ""
        inst = common.INSTANCE_TEMPLATE.render(
            config=config, name=fname, config_name=common.extract_config_name(config)
        )
        instances[exec_item.exec_cond] = inst
        instance_decl += inst
    shape_eval_func = gemm_common.gen_shape_eval_code(
        indent=1, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        program = common.EXEC_TEMPLATE.render(
            indent="    ",
            instance=fname,
            problem_args=problem_args,
            support_split_k=support_split_k,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
    )
    return src_template.render(
        instances=instance_decl,
        function_name=func_name,
        dtype="cutlass::half_t",
        shape_eval=shape_eval_func,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=output_addr_calculator,
        input_output_checks=input_output_checks,
        exec_paths=exec_paths,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        support_split_k=support_split_k,
        has_d=common.has_d(func_attrs),
        has_d1=common.has_d1(func_attrs),
        extra_code=extra_code,
    )


def gen_profiler(
    func_attrs,
    workdir,
    dim_info_dict,
    src_template,
    problem_args_template,
    args_parser_template,
    emit_kernel=False,
    support_split_k=False,
    output_addr_calculator="",
    bias_ptr_arg=None,
    extra_code="",
):
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]

    ndims = 2
    adims = ["&a_dim" + str(i) for i in range(ndims)]
    bdims = ["&b_dim" + str(i) for i in range(ndims)]
    cdims = ["&c_dim" + str(i) for i in range(ndims)]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    file_pairs = []
    has_bias = bias_ptr_arg is not None
    for op_name, op in op_instance.items():
        config = emit_instance(
            op, for_profiler=True, emit_kernel=emit_kernel, func_attrs=func_attrs
        )
        config_name = common.extract_config_name(config)
        name = "GemmInstance"
        instance = common.INSTANCE_TEMPLATE.render(
            config_name=config_name, name=name, config=config
        )
        exec_program = common.EXEC_TEMPLATE.render(
            indent="  ",
            instance=name,
            is_profiler=True,
            support_split_k=support_split_k,
            problem_args=problem_args_template.render(),
        )
        input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
            input_ndims=ndims,
            weight_ndims=ndims,
            output_ndims=ndims,
        )
        op_func = src_template.render(
            instances=instance,
            function_name="gemm",
            input_ndims=2,
            weight_ndims=2,
            output_ndims=2,
            shape_eval=shape_func,
            input_output_checks=input_output_checks,
            exec_paths=exec_program,
            output_addr_calculator=output_addr_calculator,
            support_split_k=support_split_k,
            extra_code=extra_code,
        )
        func_call = common.FUNC_CALL_TEMPLATE.render(
            func_name="gemm",
            a_ptr="memory_pool->RequestHalfTensorByIdx(0)",
            b_ptr="memory_pool->RequestHalfTensorByIdx(1)",
            has_bias=has_bias,
            bias_ptr=bias_ptr_arg,
            c_ptr="memory_pool->RequestHalfTensorByIdx(2)",
            split_k="split_k",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
        )
        # TODO: Render args_parse by caller.
        args_parse = (
            args_parser_template
            if isinstance(args_parser_template, str)
            else args_parser_template.render()
        )
        code = common.PROFILER_TEMPLATE.render(
            op_func=op_func,
            args_parse=args_parse,
            func_call=func_call,
            name=name,
            tensor_decl=common.TENSOR_DECL_TEMPLATE.render(
                name=name, has_bias=has_bias
            ),
        )
        common.add_profiler(file_pairs, workdir, op_type, op_name, code)
    # build
    common.build_profiler(file_pairs)
