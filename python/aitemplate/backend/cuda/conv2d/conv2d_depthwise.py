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
Codegen for conv2d_depthwise.
"""
from collections import OrderedDict

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.conv2d import common
from aitemplate.backend.target import Target

# pylint: disable=C0103,C0415,W0613,C0301


def conv_dw_instance(op_def):
    op_def = op_def.replace("DefaultConv2dFprop", "DefaultDepthwiseFprop")
    op_def = op_def.replace("OpClassTensorOp", "OpClassSimt")
    idx = op_def.find("kAnalytic")
    op_def = op_def[: idx + 9] + "\n>::Kernel;\n"
    return op_def


def emit_instance(op, f_instance_convertor=conv_dw_instance):
    """Emits cutlass instance."""
    import cutlass_lib

    emiter = cutlass_lib.conv2d_operation.EmitConv2dInstance()
    op_def = emiter.emit(op)
    op_def = f_instance_convertor(op_def)
    return op_def


def apply_special_config(func_attrs, op):
    import cutlass_lib

    op.iterator_algorithm = cutlass_lib.library.IteratorAlgorithm.Analytic
    op.A.alignment = 1
    op.B.alignment = 1
    op.tile_description.stages = 2
    op.tile_description.math_instruction.instruction_shape = [1, 1, 1]
    op.tile_description.threadblock_shape[-1] = 8
    return op


def extract_config(func_attrs, dtype="float16"):
    import copy

    import cutlass_lib

    spec = CUDASpec()
    lib_dtype = spec.dtype_to_lib_type(dtype)

    if lib_dtype == "float":
        data_type = cutlass_lib.library.DataType.f32
        acc_type = cutlass_lib.library.DataType.f32
    else:
        data_type = cutlass_lib.library.DataType.f16
        acc_type = cutlass_lib.library.DataType.f32
        # check target use fp16 acc
        if "use_fp16_acc" in Target.current()._kwargs:
            if Target.current()._kwargs["use_fp16_acc"]:
                acc_type = cutlass_lib.library.DataType.f16

    def f_proc_op_special(op):
        ret = []
        if (
            op.A.element == data_type
            and op.B.element == data_type
            and op.C.element == data_type
            and op.iterator_algorithm == cutlass_lib.library.IteratorAlgorithm.Optimized
            and op.accumulator_type() == acc_type
            and op.group_mode == cutlass_lib.library.GroupMode.NoneGroup
        ):
            op = copy.deepcopy(op)
            # set epilogue
            epilogue_name = func_attrs["epilogue"]
            op.epilogue_functor = cutlass_lib.library.EpilogueFunctorName[epilogue_name]
            op.element_epilogue = acc_type
            op = apply_special_config(func_attrs, op)

            # set C alignment
            for i in [1]:
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
            ret = f_proc_op_special(op)
            if len(ret) > 0:
                for op_inst in ret:
                    key = common.kernel_name(op_inst)
                    conv2d_ops[key] = op_inst
    return conv2d_ops


@registry.reg("cuda.conv2d_depthwise.config")
def conv2d_depthwise_config(func_attrs, dtype="float16"):
    """Populates conv2d_depthwise cutlass configs into 'op_instance' field."""
    func_attrs["op_instance"] = extract_config(func_attrs, dtype)


@registry.reg("cuda.conv2d_depthwise.gen_profiler")
def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
):
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        shape_template=shape_template,
        f_emit_instance=emit_instance,
        is_depthwise=True,
        instance_name_base="DeviceConvFwdInstance",
    )


@registry.reg("cuda.conv2d_depthwise.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    """Codegen for conv2d_depthwise function."""
    return common.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        shape_eval_template=shape_eval_template,
        shape_save_template=shape_save_template,
        is_depthwise=True,
        f_emit_instance=emit_instance,
    )


@registry.reg("cuda.conv2d_depthwise.func_decl")
def conv2d_depthwise_gen_function_decl(func_attrs):
    """Codegen for conv2d_depthwise function declaration."""
    return common.gen_function_decl(
        func_attrs=func_attrs,
    )


@registry.reg("cuda.conv2d_depthwise.func_call")
def conv2d_depthwise_gen_function_call(func_attrs, indent="  "):
    """Codegen for conv2d_depthwise function call."""
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
    )


@registry.reg("cuda.conv2d_depthwise.filter")
def conv2d_depthwise_function_filter(cfg, func_attrs, x_shape):
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
    return True
