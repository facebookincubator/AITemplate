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
conv2d bias add hardswish codegen
"""
from ... import registry
from ...target import Target
from . import common, common_conv2d_bias_add_activation as cbaa

# pylint: disable=C0103,C0415,W0613,C0301


@registry.reg("cuda.conv2d_bias_add_hardswish.config")
def conv2d_config(func_attrs, dtype="float16"):
    def fproc_f16(op):
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
            and op.iterator_algorithm == cutlass_lib.library.IteratorAlgorithm.Optimized
            and op.accumulator_type() == acc_type
        ):

            op = copy.deepcopy(op)
            # set epilogue
            epilogue_name = func_attrs["epilogue"]
            op.epilogue_functor = cutlass_lib.library.EpilogueFunctorName[epilogue_name]
            op.element_epilogue = acc_type

            op.activation_op = cutlass_lib.library.EpilogueMathName["Identity"]
            op.binary_op = cutlass_lib.library.EpilogueMathName["Add"]
            op.unary_op = cutlass_lib.library.EpilogueMathName["HardSwish"]

            # set C alignment
            for i in [8, 4, 2, 1]:
                op = copy.deepcopy(op)
                op.C.alignment = i
                ret.append(op)
        return ret

    func_attrs["op_instance"] = common.extract_config(func_attrs, fproc_f16)


@registry.reg("cuda.conv2d_bias_add_hardswish.gen_profiler")
def gen_profiler(func_attrs, workdir, shape_template):
    cbaa.gen_profiler(func_attrs, workdir, shape_template)


@registry.reg("cuda.conv2d_bias_add_hardswish.gen_function")
def gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    return common.gen_function(
        func_attrs,
        cbaa.INSTANCE_TEMPLATE,
        cbaa.EXEC_TEMPLATE,
        cbaa.SRC_TEMPLATE,
        exec_cond_remplate,
        shape_eval_template,
        shape_save_template,
    )


@registry.reg("cuda.conv2d_bias_add_hardswish.func_decl")
def conv2d_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return cbaa.FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.conv2d_bias_add_hardswish.func_call")
def conv2d_gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    w = func_attrs["inputs"][1]
    b = func_attrs["inputs"][2]
    r = func_attrs["inputs"][3]
    wshape = w._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    return cbaa.FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        weight_ptr=w._attrs["name"],
        out_ptr=y._attrs["name"],
        bias_ptr=b._attrs["name"],
        res_ptr=r._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_out_ch="&" + wshape[0]._attrs["name"],
        p_in_ch="&" + xshape[3]._attrs["name"],
        p_kernel_h="&" + wshape[1]._attrs["name"],
        p_kernel_w="&" + wshape[2]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        stride=func_attrs["stride"],
        dilation=func_attrs["dilate"],
        pad=func_attrs["pad"],
        indent=indent,
    )


@registry.reg("cuda.conv2d_bias_add_hardswish.filter")
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
