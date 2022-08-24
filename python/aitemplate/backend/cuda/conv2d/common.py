# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
common template for conv2d
"""
import re
from collections import OrderedDict
from hashlib import sha1
from typing import List

import jinja2

from ...target import Target
from ..gemm_universal.common import add_profiler, build_profiler  # noqa: F401


KERNEL_KEY_TEMPLATE = jinja2.Template(
    """
cutlass{{opcode_class}}_{{extended_name}}_{{threadblock}}_{{layout}}_align_{{align_ab}}_{{align_c}}
"""
)


def kernel_name(op):
    """[summary]
    generate cuda kernel name
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


def emit_instance(op):
    """[summary]
    emit instance
    """
    import cutlass_lib

    if hasattr(op, "binary_op"):
        emiter = cutlass_lib.conv2d_operation.EmitConv2dWithBroadcastInstance()
    else:
        emiter = cutlass_lib.conv2d_operation.EmitConv2dInstance()
    op_def = emiter.emit(op)
    return op_def


def extract_config(func_attrs, f_proc_op=None):
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
    import copy

    import cutlass_lib

    def f_proc_op_default(op):
        # import cutlass_lib
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
            and op.iterator_algorithm == cutlass_lib.library.IteratorAlgorithm.Optimized
            and op.accumulator_type() == acc_type
        ):

            op = copy.deepcopy(op)
            # set epilogue
            epilogue_name = func_attrs["epilogue"]
            op.epilogue_functor = cutlass_lib.library.EpilogueFunctorName[epilogue_name]
            op.element_epilogue = acc_type
            # set C alignment
            for i in [8, 4, 2, 1]:
                op = copy.deepcopy(op)
                op.C.alignment = i
                ret.append(op)
        return ret

    op_kind = cutlass_lib.library.OperationKind.Conv2d
    conv_kind = cutlass_lib.library.ConvKind.Fprop
    ret = []
    conv2d_ops = OrderedDict()
    extract_ops = list(Target.current()._operators[op_kind].items())

    for _, value in extract_ops:
        op = value[0]
        if op.conv_kind == conv_kind:
            if f_proc_op is None:
                ret = f_proc_op_default(op)
            else:
                ret = f_proc_op(op)
            if len(ret) > 0:
                for op_inst in ret:
                    key = kernel_name(op_inst)
                    conv2d_ops[key] = op_inst
    return conv2d_ops


def extract_config_name(config):
    """[summary]

    Parameters
    ----------
    config : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    RuntimeError
        [description]
    """
    pattern = re.compile(r"\s*using\s(.*?)\s=")
    decl = config.split("\n")[2]
    match = pattern.match(decl)
    if match is None:
        raise RuntimeError("Invalid config: \n" + config)
    return match.groups()[0]


def gen_function(
    func_attrs,
    instance_template,
    exec_template,
    src_template,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
    f_emit_instance=emit_instance,
    extra_header="",
):
    """[summary]

    Parameters
    ----------
    func_name : [type]
        [description]
    exec_path : [type]
        [description]
    exec_cond_remplate : [type]
        [description]
    shape_eval_template : [type]
        [description]
    shape_save_template : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]

    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for key, value in exec_path.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        if value not in inst_def_flag:
            config = f_emit_instance(op_instance[value])
            inst_def_flag.add(value)
        else:
            config = ""
        inst = instance_template.render(
            config=config, name=fname, config_name=extract_config_name(config)
        )
        instances[key] = inst
        instance_decl += inst
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
        x_dim3="*in_ch",
        w_dim0="*out_ch",
        w_dim1="*kernel_h",
        w_dim2="*kernel_w",
        stride="stride",
        dilate="dilation",
        pad="pad",
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
        y_dim3="*out_ch",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = ""
    for key in instances:
        fname = "f" + sha1(key.encode()).hexdigest()
        program = exec_template.render(indent="    ", instance=fname)
        exec_inst = exec_cond_remplate.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return src_template.render(
        instances=instance_decl,
        function_name=func_name,
        dtype="cutlass::half_t",
        shape_function=shape_func,
        exec_paths=exec_paths,
        extra_header=extra_header,
    )


def cal_align_ab(x_shape: List[int]) -> int:
    """[summary]

    Parameters
    ----------
    shape : List
        Tensor shape

    Returns
    -------
    alignment: int
        alignment for epilogue configuration
    """
    k = x_shape[3]  # CI
    if k % 8 == 0:
        return 8
    if k % 4 == 0:
        return 4
    if k % 2 == 0:
        return 2
    raise RuntimeError("a/b is not aligned")


def function_filter(cfg, func_attrs, x_shape):
    """[summary]
    Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    x_shape:
        Input shapes.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    ab_alignment = cal_align_ab(x_shape)
    tmp = cfg.split("_")
    align_c = int(tmp[-1])
    align_ab = int(tmp[-2])
    if align_c != func_attrs["epilogue_alignment"]:
        return False
    if align_ab != ab_alignment:
        return False
    return True
