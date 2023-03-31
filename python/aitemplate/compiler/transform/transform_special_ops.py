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
Perform graph transformation specifically for gemm -> gemm_special.
Check each transform function summary for specific pattern to be transformed.
"""
from typing import Callable, List, Tuple, Type, Union

from aitemplate.backend.target import Target

from aitemplate.compiler import ops
from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.ops.gemm_special.gemm_rrr_small_nk import gemm_rrr_small_nk
from aitemplate.compiler.ops.gemm_universal.bmm_xxx import bmm_rcr
from aitemplate.compiler.ops.gemm_universal.gemm_rrr import gemm_rrr
from aitemplate.compiler.transform.transform_utils import (
    copy_src_op_attributes,
    copy_tensor_attributes,
    remove_dst_op_from_tensor,
    replace_tensor,
    sanitize_sorted_graph,
)

from aitemplate.utils.shape_utils import is_singleton_dimension

# pylint: disable=C0103,C0415,W0612


def _simple_transform_with_constraint(
    sorted_graph: List[Tensor],
    matchFunc: Callable[[Tensor], bool],
    replaceFunc: Callable[[Tensor], Union[Tensor, List[Tensor]]],
) -> List[Tensor]:
    """
    Replace ops in sorted_graph that match constraint provided by matchFunc.
    Op to be replaced is determined by matchFunc, which if true, we call replaceFunc to substitute the tensor away.


    Parameters
    ----------
    sorted_graph : List[Tensor]
        original AIT graph

    matchFunc : Callable(Tensor) -> bool
        A function that returns whether or not the operator needs to be substituted

    replaceFunc  : Callable(Tensor) -> Tensor
        A function that return the resulting tensor that replaces the original one.

    Returns
    ----------
    List[Tensor]
        AIT graph after transformation
    """

    new_sorted_graph = []
    transformed = False
    for tensor in sorted_graph:
        if matchFunc(tensor):
            new_tensors = replaceFunc(tensor)
            if isinstance(new_tensors, Tensor):
                new_sorted_graph.append(new_tensors)
            else:
                new_sorted_graph.extend(new_tensors)

            transformed = True
        else:
            new_sorted_graph.append(tensor)

    if not transformed:
        return sorted_graph
    return sanitize_sorted_graph(new_sorted_graph)


def _single_source_shape_constraint_funcs(
    src_type: Type[Operator], target_type: Type[Operator]
) -> Tuple[Callable[[Tensor], bool], Callable[[Tensor], Tensor]]:
    """
    Returns matching function and replace function for tensors that are single src_op with shape constraints.

    Parameters
    ----------
    src_type : Type[Operator]
        An operator type that can be substituted by target_type provided tensor shape constraint is satisfied.
    target_type : Type[Operator]
        An operator that can substitute src_type provided shape constraint is satisfied.
        is_valid_shape needs to be implemented for target_type

    Returns
    ----------
    Tuple[Callable[[Tensor], bool], Callable[[Tensor], Tensor]]
        A tuple of function which corresponds to a matching function and a replacing function wrt tensor provided.
    """

    def matchFunc(tensor: Tensor) -> bool:
        src_ops = tensor._attrs["src_ops"]
        if src_ops is None or len(src_ops) != 1:
            return False

        src_op = list(src_ops)[0]
        if src_op._attrs["op"] != src_type()._attrs["op"]:
            return False

        A, B = src_op._attrs["inputs"]
        if not target_type.is_valid_shape(A, B):
            return False

        return True

    def replaceFunc(old_tensor: Tensor) -> Tensor:
        src_op = list(old_tensor._attrs["src_ops"])[0]
        A, B = src_op._attrs["inputs"]

        new_op = target_type()
        new_tensor = new_op(A, B)
        copy_tensor_attributes(new_tensor, old_tensor)
        copy_src_op_attributes(new_tensor, old_tensor)
        remove_dst_op_from_tensor([A, B], src_op)
        replace_tensor(old_tensor, new_tensor)
        return new_tensor

    return (matchFunc, replaceFunc)


def _transform_bmm_rcr_n1(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Replace kernel bmm_rcr with N == 1 and K % 8 == 0 with bmm_rcr_n1

    Parameters
    ----------
    sorted_graph : List[Tensor]
        original AIT graph

    Returns
    ----------
    List[Tensor]
        AIT graph with suitable bmm_rcr substituted with bmm_rcr_n1
    """
    matchFunc, replaceFunc = _single_source_shape_constraint_funcs(
        bmm_rcr, ops.gemm_special.bmm_rcr_n1
    )

    return _simple_transform_with_constraint(sorted_graph, matchFunc, replaceFunc)


def _transform_gemm_rrr_small_nk(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Replace kernel gemm_rrr with N <= 8, K <= 16 with gemm_rrr_small_nk

    Parameters
    ----------
    sorted_graph : List[Tensor]
        original AIT graph

    Returns
    ----------
    List[Tensor]
        AIT graph with suitable gemm_rrr substituted with gemm_rrr_small_nk
    """
    matchFunc, replaceFunc = _single_source_shape_constraint_funcs(
        gemm_rrr, gemm_rrr_small_nk
    )

    return _simple_transform_with_constraint(sorted_graph, matchFunc, replaceFunc)


def _transform_1x1_conv_gemm_rcr(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Replace a 1x1 conv2d with an equivalent gemm_rcr call.
    """

    conv_to_gemm = {
        "conv2d": ops.gemm_rcr,
        "conv2d_bias": ops.gemm_rcr_bias,
        "conv2d_bias_relu": ops.gemm_rcr_bias_relu,
        "conv2d_bias_sigmoid": ops.gemm_rcr_bias_sigmoid,
        "conv2d_bias_hardswish": ops.gemm_rcr_bias_hardswish,
        "conv2d_bias_add_relu": ops.gemm_rcr_bias_add_relu,
    }

    def match_func(tensor: Tensor) -> bool:
        src_ops = tensor._attrs["src_ops"]
        if src_ops is None or len(src_ops) != 1:
            return False

        src_op = list(src_ops)[0]
        if src_op._attrs["op"] not in conv_to_gemm:
            return False

        if (
            src_op._attrs["pad"] != 0
            or src_op._attrs["dilate"] != 1
            or src_op._attrs["group"] != 1
            or src_op._attrs["stride"] != 1
        ):
            return False

        # Check that the filter is 1x1
        w_shape = src_op._attrs["inputs"][1]._attrs["shape"]
        if not is_singleton_dimension(w_shape[1]) or not is_singleton_dimension(
            w_shape[2]
        ):
            return False

        return True

    def replace_func(old_tensor: Tensor) -> List[Tensor]:
        src_op = list(old_tensor._attrs["src_ops"])[0]
        inputs = src_op._attrs["inputs"]
        X = inputs[0]
        W = inputs[1]

        # Get rid of the singleton dimensions
        W_squeeze_0 = ops.squeeze(1)(W)
        W_squeeze_1 = ops.squeeze(1)(W_squeeze_0)

        batch, HH, WW, CI = ops.size()(X)
        CO = ops.size()(W, dim=0)

        X_reshape = ops.reshape()(X, (-1, CI))
        new_op = conv_to_gemm[src_op._attrs["op"]]
        if len(inputs) == 2:
            new_tensor = new_op()(X_reshape, W_squeeze_1)
        elif len(inputs) == 3:
            new_tensor = new_op()(X_reshape, W_squeeze_1, inputs[2])
        elif len(inputs) == 4:
            new_tensor = new_op()(X_reshape, W_squeeze_1, inputs[2], inputs[3])
        else:
            raise NotImplementedError(f"Unsupported number of inputs: {len(inputs)}")

        new_tensor_reshape = ops.reshape()(new_tensor, (batch, HH, WW, CO))

        copy_tensor_attributes(new_tensor_reshape, old_tensor)
        copy_src_op_attributes(new_tensor_reshape, old_tensor)

        remove_dst_op_from_tensor(inputs, src_op)
        replace_tensor(old_tensor, new_tensor_reshape)

        return [
            W_squeeze_0,
            W_squeeze_1,
            batch,
            HH,
            WW,
            CI,
            CO,
            X_reshape,
            new_tensor,
            new_tensor_reshape,
        ]

    return _simple_transform_with_constraint(sorted_graph, match_func, replace_func)


def transform_special_ops(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """Transform generic gemm/conv ops to special ops.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph
    workdir : str, optional
        workdir, by default None

    Returns
    -------
    List[Tensor]
        Transformed graph
    """
    if Target.current().name() == "rocm":
        funcs = []
    else:
        funcs = [
            _transform_bmm_rcr_n1,
            _transform_gemm_rrr_small_nk,
        ]

    for func in funcs:
        sorted_graph = func(sorted_graph)

    # This pass doesn not always make the model run faster.
    # Turn it on with detect_target(transform_conv_to_gemm=True)
    funcs = [
        _transform_1x1_conv_gemm_rcr,
    ]
    
    if "transform_conv_to_gemm" in Target.current()._kwargs:
        if Target.current()._kwargs["transform_conv_to_gemm"]:
            for func in funcs:
                sorted_graph = func(sorted_graph)
    return sorted_graph
