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
specialize conv2d op with few channels(< 8)
"""
from collections import OrderedDict

from ... import registry
from ...target import Target
from . import common, common_conv2d_bias_activation as cba

# pylint: disable=C0103,C0415,W0613,C0301


def apply_special_config(func_attrs, op):
    import cutlass_lib

    x = func_attrs["inputs"][0]
    in_ch = x._attrs["shape"][-1]._attrs["values"][0]

    if in_ch == 3:
        # By default we don't use it since the perf is worse than pad4+fixchannel
        op.iterator_algorithm = cutlass_lib.library.IteratorAlgorithm.FewChannels
        op.A.alignment = 1
        op.B.alignment = 1
        op.tile_description.stages = 2
    elif in_ch in [2, 4, 8]:
        op.iterator_algorithm = cutlass_lib.library.IteratorAlgorithm.FixedChannels
        op.A.alignment = in_ch
        op.B.alignment = in_ch
        op.tile_description.stages = 3
    return op


def extract_config(func_attrs):
    """extract epilogue for conv op

    Parameters
    ----------
    func_attrs : Dict
        [description] op attributes

    Returns
    -------
    [type]: Dict
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    import copy

    import cutlass_lib

    def f_proc_op_special(op):
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
            op = apply_special_config(func_attrs, op)
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
            ret = f_proc_op_special(op)
            if len(ret) > 0:
                for op_inst in ret:
                    key = common.kernel_name(op_inst)
                    conv2d_ops[key] = op_inst
    return conv2d_ops


@registry.reg("cuda.conv2d_bias_few_channels.config")
def conv2d_config(func_attrs, dtype="float16"):
    """extract configurations for profiling

    Parameters
    ----------
    func_attrs : Dict
        [description] op attributes
    dtype : str, optional
        [description] by default "float16"

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    func_attrs["op_instance"] = extract_config(func_attrs)


@registry.reg("cuda.conv2d_bias_few_channels.gen_profiler")
def gen_profiler(func_attrs, workdir, shape_template):
    """generate code for profiling"""
    cba.gen_profiler(func_attrs, workdir, shape_template)


@registry.reg("cuda.conv2d_bias_few_channels.gen_function")
def gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    """generating special conv2d kernel and all of its auxiliary functions

    Parameters
    ----------
    func_attrs : Dict
        [description] attributes of conv2d op
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
    return common.gen_function(
        func_attrs,
        cba.INSTANCE_TEMPLATE,
        cba.EXEC_TEMPLATE,
        cba.SRC_TEMPLATE,
        exec_cond_remplate,
        shape_eval_template,
        shape_save_template,
    )


@registry.reg("cuda.conv2d_bias_few_channels.func_decl")
def conv2d_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return cba.FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.conv2d_bias_few_channels.func_call")
def conv2d_gen_function_call(func_attrs, indent="  "):
    return cba.gen_function_call(func_attrs, indent)


@registry.reg("cuda.conv2d_bias_few_channels.filter")
def conv2d_function_filter(cfg, func_attrs, x_shape):
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
    return common.function_filter(cfg, func_attrs, x_shape)
