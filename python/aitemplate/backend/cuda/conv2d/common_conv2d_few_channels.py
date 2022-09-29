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
common functions for conv2d op with few channels(< 8)
"""

from collections import OrderedDict

from ...target import Target
from . import common


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
