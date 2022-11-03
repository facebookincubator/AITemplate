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
common template for conv2d
"""
import re

from collections import OrderedDict
from hashlib import sha1
from typing import List

import jinja2

from aitemplate.backend.backend_spec import CUDASpec

from ...target import Target
from ..gemm_universal.common import add_profiler, build_profiler  # noqa: F401


KERNEL_KEY_TEMPLATE = jinja2.Template(
    """
cutlass{{opcode_class}}_{{extended_name}}_{{threadblock}}_{{layout}}_align_{{align_ab}}_{{align_c}}
"""
)


def kernel_name(op):
    """generate cuda kernel name"""
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
    """emit instance"""
    import cutlass_lib

    if hasattr(op, "binary_op"):
        emiter = cutlass_lib.conv2d_operation.EmitConv2dWithBroadcastInstance()
    else:
        emiter = cutlass_lib.conv2d_operation.EmitConv2dInstance()
    op_def = emiter.emit(op)
    return op_def


def extract_config(func_attrs, f_proc_op=None):
    """Extracts cutlass config for conv kernels."""
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
    """Extracts config name from a given config."""
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
    """Function definition codegen."""
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]

    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    backend_spec = CUDASpec()
    dtype = backend_spec.dtype_to_lib_type(func_attrs["inputs"][0]._attrs["dtype"])
    for key, value in exec_path.items():
        fname = "f" + sha1(key.encode()).hexdigest()

        emit_instance = f_emit_instance(op_instance[value])
        if value not in inst_def_flag:
            inst_def_flag.add(value)
            config = emit_instance
        else:
            config = ""
        inst = instance_template.render(
            config=config, name=fname, config_name=extract_config_name(emit_instance)
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
        program = exec_template.render(
            indent=" " * 4,
            instance=fname,
            dtype=dtype,
        )
        exec_inst = exec_cond_remplate.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return src_template.render(
        instances=instance_decl,
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
        extra_header=extra_header,
    )


def cal_align_ab(x_shape: List[int]) -> int:
    """Returns input alignment."""
    k = x_shape[3]  # CI
    if k % 8 == 0:
        return 8
    if k % 4 == 0:
        return 4
    if k % 2 == 0:
        return 2
    raise RuntimeError(f"a/b is not aligned {x_shape=}")


def function_filter(cfg, func_attrs, x_shape):
    """Generates function filter.

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
