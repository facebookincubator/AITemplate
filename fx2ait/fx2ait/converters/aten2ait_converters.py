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
import logging
import torch  # isort:skip
import copy
import operator
from typing import Dict, List, Tuple, Union

import numpy

from aitemplate.compiler.public import (
    avg_pool2d,
    bmm_rrr,
    chunk,
    concatenate,
    conv2d,
    conv2d_bias,
    dynamic_slice,
    elementwise,
    expand,
    FuncEnum,
    gemm_rcr,
    gemm_rcr_bias,
    gemm_rrr,
    getitem,
    int_elementwise,
    IntImm,
    IntVar,
    IntVarTensor,
    layernorm,
    max_pool2d,
    nhwc3to8,
    pad_last_dim,
    permute,
    reduce_mean,
    reduce_sum,
    reshape,
    size,
    split,
    squeeze,
    Tensor as AITTensor,
    transposed_conv2d,
    transposed_conv2d_bias,
    unsqueeze,
)
from fx2ait.converters.utils import (
    create_binary_op,
    get_positive_dim,
    identical_elem_tuple_to_int,
    nchw2nhwc,
    unify_dynamic_shape_name,
)
from fx2ait.passes.lower_basic_pass_aten import (
    aten_compose_bmm_2d,
    aten_compose_bmm_3d,
    aten_compose_chunk,
    aten_compose_getitem_slice,
    aten_compose_mm_2d,
    aten_operator_getitem,
)
from torch.fx.node import Argument, Target

from .converter_registry import ait_converter


# Logging
logger: logging.Logger = logging.getLogger(__name__)
ConverterOutput = Union[AITTensor, Tuple[AITTensor, ...], List[IntVar], IntVar]

## make sure the functions are place in alphabetic order


@ait_converter(torch.ops.aten._adaptive_avg_pool2d.default)
def aten_ops_adaptive_avg_pool2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO: @qxy11 Update once NCHW supported
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    output_size = identical_elem_tuple_to_int(args[1])
    # FIXME: try to find a way to not explose internal date like this.
    shape = [var._attrs["values"][0] for var in input_val._attrs["shape"]]
    HI, WI, CI = shape[1], shape[2], shape[3]
    if CI % 2 != 0:
        raise RuntimeError(
            f"AIT avg_pool2d expects input channel dim to align w/ a multiple of 2 but got {CI}"
        )
    if HI != WI:
        raise RuntimeError(
            f"adaptive_avg_pool2d currently only supports square input H/W but got H: {shape[1]} and W: {shape[2]}"
        )
    stride = HI // output_size
    kernel_size = HI - (output_size - 1) * stride

    return avg_pool2d(kernel_size=kernel_size, stride=stride, pad=0)(input_val)


@ait_converter(torch.ops.aten.avg_pool2d.default)
def aten_ops_avg_pool2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO: @qxy11 Update once NCHW supported
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    kernel_size = args[1]
    stride = args[2]
    padding = args[3] if len(args) > 3 else 0
    kernel_size = identical_elem_tuple_to_int(kernel_size)
    stride = identical_elem_tuple_to_int(stride)
    padding = identical_elem_tuple_to_int(padding)
    return avg_pool2d(kernel_size=kernel_size, stride=stride, pad=padding)(input_val)


@ait_converter(torch.ops.aten.batch_norm)
def aten_ops_batch_norm(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO @qxy11: Update channels-last assumption once AIT backend is updated
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    weight = args[1]
    bias = args[2]
    running_mean = args[3]
    running_var = args[4]
    eps = args[7]
    scale = elementwise(FuncEnum.DIV)(
        weight,
        elementwise(FuncEnum.ADD)(
            elementwise(FuncEnum.SQRT)(running_var),
            AITTensor(shape=[], value=eps),
        ),
    )
    running_mean = elementwise(FuncEnum.MUL)(scale, running_mean)
    bias = elementwise(FuncEnum.SUB)(bias, running_mean)
    matmul_result = elementwise(FuncEnum.MUL)(input_val, scale)
    result = elementwise(FuncEnum.ADD)(matmul_result, bias)
    return result


@ait_converter(torch.ops.aten.add.Tensor)
def aten_binary_ops_add(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    unify_dynamic_shape_name(args[0], args[1])
    kwargs = {
        "input": args[0],
        "other": args[1],
    }
    return create_binary_op(FuncEnum.ADD, args, kwargs, name)


@ait_converter(torch.ops.aten.div.Tensor)
def aten_binary_ops_div(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    kwargs = {
        "input": args[0],
        "other": args[1],
    }
    return create_binary_op(FuncEnum.DIV, args, kwargs, name)


@ait_converter(torch.ops.aten.mul.Tensor)
def aten_binary_ops_mul(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    kwargs = {
        "input": args[0],
        "other": args[1],
    }
    return create_binary_op(FuncEnum.MUL, args, kwargs, name)


@ait_converter(torch.ops.aten.sub.Tensor)
def aten_binary_ops_sub(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    kwargs = {
        "input": args[0],
        "other": args[1],
    }
    return create_binary_op(FuncEnum.SUB, args, kwargs, name)


@ait_converter(torch.ops.aten.cat.default)
def aten_ops_cat(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    tensors = args[0]
    for t in tensors:
        if not isinstance(t, AITTensor):
            raise ValueError(f"Non-tensor inputs for {name}: {tensors}")

    dim = args[1] if len(args) > 1 else 0
    if not isinstance(dim, int):
        raise ValueError(f"Unexpected {type(dim)} dim for {name}: {dim}")

    return concatenate()(tensors, dim=dim)


def _choose_conv2d_op(
    stride: int,
    pad: int,
    dilate: int,
    x: AITTensor,
    weight: AITTensor,
    bias: [AITTensor],
    transposed: [bool] = False,
) -> ConverterOutput:
    """
    Helper to choose conv2d vs. conv2d_bias op based on existence of bias
    and pad channel input dim to 4/8
    """
    if transposed:
        if bias:
            return transposed_conv2d_bias(stride=stride, pad=pad, dilate=dilate)(
                x, weight, bias
            )
        else:
            return transposed_conv2d(stride=stride, pad=pad, dilate=dilate)(x, weight)

    last_dim = x._attrs["shape"][-1]._attrs["values"][0]
    # CUDA conv channel dim weights need to align w/ a multiple of 2/4/8
    # if CI < 4, pad to 4; if 5 < CI < 8, pad to 8;
    if last_dim < 4:
        weight = pad_last_dim(len(weight._attrs["shape"]), 4)(weight)
        x = pad_last_dim(len(x._attrs["shape"]), 4)(x)
    elif last_dim in range(5, 8):
        to_8 = nhwc3to8()
        weight = to_8(weight)
        x = to_8(x)
    elif last_dim % 2 != 0:
        return RuntimeError(
            f"Conv2d is not implemented for input channel dim {last_dim}: it needs to be aligned to a multiple of 2/4/8"
        )
    if bias:
        return conv2d_bias(stride=stride, pad=pad, dilate=dilate)(x, weight, bias)
    else:
        return conv2d(stride=stride, pad=pad, dilate=dilate)(x, weight)


@ait_converter(torch.ops.aten.convolution.default)
def aten_ops_conv2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO: qxy11: Update once channels-first format is supported
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    weight = args[1]
    assert isinstance(weight, AITTensor)
    weight._attrs["data"].tensor = weight._attrs["data"].tensor.permute(0, 2, 3, 1)
    weight._attrs["shape"] = nchw2nhwc(weight._attrs["shape"])

    bias = args[2]
    if not (isinstance(bias, AITTensor) or bias is None):
        raise RuntimeError(f"Non-tensor weight for {name}: {bias}")

    stride = args[3]
    stride = identical_elem_tuple_to_int(stride)
    padding = args[4]
    padding = identical_elem_tuple_to_int(padding)
    dilation = args[5]
    dilation = identical_elem_tuple_to_int(dilation)
    transposed = args[6]
    # output_padding = args[7]
    groups = args[8]

    assert all(
        isinstance(x, int) for x in [stride, padding, dilation]
    ), "Expected int stride, padding, and dilation"

    if groups is None or groups == 1:
        if transposed:
            if bias:
                result = transposed_conv2d_bias(
                    stride=stride, pad=padding, dilate=dilation
                )(input_val, weight, bias)
            else:
                result = transposed_conv2d(stride=stride, pad=padding, dilate=dilation)(
                    input_val, weight
                )
        else:
            result = _choose_conv2d_op(
                stride, padding, dilation, input_val, weight, bias, transposed
            )
    else:
        # Grouped conv doesn't currently work on AIT CUDA, manually map
        group_size = input_val.shape()[3]._attrs["values"][0] // groups
        w_group_size = weight.shape()[0]._attrs["values"][0] // groups

        def make_slice(x, dim, start, end, step, name):
            args = []
            args.append(x)
            args.append(dim)
            args.append(start)
            args.append(end)
            args.append(step)
            return aten_ops_slice(
                target,
                args,
                None,
                name,
            )

        conv_groups = [
            _choose_conv2d_op(
                stride,
                padding,
                dilation,
                make_slice(  # input_val[:,:,:,gs*i:gs*i + gs]
                    input_val,
                    3,
                    i * group_size,
                    i * group_size + group_size,
                    1,
                    f"{name}.slice_{i}",
                ),
                make_slice(  # weights[wgs*i:wgs*i + wgs,]
                    weight,
                    0,
                    i * w_group_size,
                    i * w_group_size + w_group_size,
                    1,
                    f"{name}.weight.slice_{i}",
                ),
                None
                if bias is None
                else make_slice(  # bias[wgs*i:wgs*i + wgs,]
                    bias,
                    0,
                    i * w_group_size,
                    i * w_group_size + w_group_size,
                    1,
                    f"{name}.bias.slice_{i}",
                ),
                transposed=transposed,
            )
            for i in range(groups)
        ]
        result = concatenate()(conv_groups, dim=3)

    return result


@ait_converter(torch.ops.aten.clone.default)
def aten_unary_ops_clone(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    res = copy.deepcopy(input_val)
    res._attrs["dst_ops"].clear()
    return res


@ait_converter(torch.ops.aten.cos.default)
def aten_unary_ops_cos(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    return elementwise(FuncEnum.COS)(input_val)


@ait_converter(aten_compose_chunk)
@ait_converter(torch.ops.aten.chunk.default)
def aten_ops_chunk(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    shape = input_val.shape()
    target_dim = get_positive_dim(args[2], len(shape))
    chunks = min(args[1], shape[target_dim].value())
    assert isinstance(
        shape[target_dim], IntImm
    ), f"Cannot perform chunk on dynamic dim! Get target dim {target_dim}."

    return chunk()(input_val, chunks, dim=target_dim)


@ait_converter(torch.ops.aten.expand.default)
def aten_ops_expand(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO expand is not functional yet but only for cases with dim=-1
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Non-tensor inputs for {name}: {input_val}")

    sizes = args[1]
    if not sizes:
        raise ValueError("Expand sizes cannot be empty")

    def _is_int_list(iterable):
        return all(isinstance(dim, (int, IntVar, IntVarTensor)) for dim in iterable)

    # sizes can either be a single int list or a list of ints.
    if _is_int_list(sizes):
        shape = sizes
    elif len(sizes) == 1 and _is_int_list(sizes[0]):
        shape = sizes[0]
    else:
        raise ValueError(
            f"sizes argument can either be many ints or single int iterable, but got: {', '.join(str(type(dim)) for dim in sizes)}"
        )

    return expand()(input_val, shape)


@ait_converter(aten_operator_getitem)
def aten_ops_getitem(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    idx = args[1]

    # This case is decomposed into a few slice.Tensor ops
    # if (
    #     isinstance(idx, slice)
    #     or (isinstance(idx, Sequence) and any(isinstance(x, slice) for x in idx))
    #     or isinstance(input_val, AITTensor)
    # ):
    #     return aten_ops_slice(target, args, kwargs, name)

    if isinstance(idx, int):
        idx = get_positive_dim(idx, len(input_val))

    if all(isinstance(i, IntImm) for i in input_val):
        return operator.getitem(input_val, idx)
    else:
        return getitem()(input_val, idx)


@ait_converter(torch.ops.aten.layer_norm.default)
def aten_ops_layer_norm(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    shape = args[1]
    if shape is None or len(shape) == 0:
        raise ValueError(f"Unexpected normalized shape value in {name}: {shape}")
    weight = args[2]
    bias = args[3]
    normalized_shape = []
    if all(isinstance(i, int) for i in shape):
        for i in shape:
            normalized_shape.append(IntImm(i))
    elif all(isinstance(i, IntImm) or isinstance(i, IntVarTensor) for i in shape):
        normalized_shape = shape
    else:
        raise ValueError(f"Unexpected normalized shape value in {name}: {shape}")
    return layernorm()(input_val, weight, bias, normalized_shape)


@ait_converter(torch.ops.aten.linear)
def aten_ops_linear(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    weight = args[1]
    bias = args[2]

    assert isinstance(weight, AITTensor)
    if bias is None:
        return gemm_rcr()(input_val, weight)
    else:
        return gemm_rcr_bias()(input_val, weight, bias)


@ait_converter(torch.ops.aten.max_pool2d)
def aten_ops_max_pool2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO: @qxy11 Update once NCHW supported
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    kernel_size = args[1]
    stride = args[2]
    padding = args[3] if len(args) > 3 else 0
    kernel_size = identical_elem_tuple_to_int(kernel_size)
    stride = identical_elem_tuple_to_int(stride)
    padding = identical_elem_tuple_to_int(padding)
    return max_pool2d(kernel_size=kernel_size, stride=stride, pad=padding)(input_val)


@ait_converter(aten_compose_mm_2d)
@ait_converter(aten_compose_bmm_3d)
@ait_converter(aten_compose_bmm_2d)
@ait_converter(torch.ops.aten.addmm.default)
@ait_converter(torch.ops.aten.mm.default)
@ait_converter(torch.ops.aten.bmm.default)
def aten_ops_matmul(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    if len(args) > 2:
        bias = args[0]
        input_val = args[1]
        weight = args[2]
    else:
        bias = None
        input_val = args[0]
        weight = args[1]

    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")
    if not isinstance(weight, AITTensor):
        raise ValueError(f"Unexpected weight for {name}: {weight}")

    input_shape = input_val.shape()
    weight_shape = weight.shape()
    if len(weight_shape) == 2:
        result = gemm_rrr()(input_val, weight)
    elif len(input_shape) == 3 and len(weight_shape) == 3:
        unify_dynamic_shape_name(input_val, weight)
        result = bmm_rrr()(input_val, weight)
    else:
        raise NotImplementedError(
            f"This case is unsupported in {name}: {len(input_shape)} and {len(weight_shape)}"
        )

    if bias is not None:
        if not isinstance(bias, AITTensor):
            raise ValueError(f"Unexpected weight for {name}: {bias}")
        result = elementwise(FuncEnum.ADD)(result, bias)

    return result


@ait_converter(torch.ops.aten.mean.dim)
def aten_ops_mean(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    dims = args[1]
    keepdim = args[2] if len(args) > 2 else False
    if len(dims) == 1:
        return reduce_mean(dim=dims, keepdim=keepdim)(input_val)
    else:
        new_dims = list(dims)
        res = input_val
        for i, d in enumerate(new_dims):
            if d < 0:
                d += len(input_val.shape())
            new_dims[i] = d
            res = reduce_mean(dim=d, keepdim=True)(res)
        if not keepdim:
            new_dims = sorted(new_dims, reverse=True)
            for d in new_dims:
                res = squeeze(d)(res)
        return res


@ait_converter(torch.ops.aten.nan_to_num.default)
def aten_ops_nan_to_num(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    nan = args[1] if len(args) > 1 else None
    nan = 0 if nan is None else nan
    posinf = args[2] if len(args) > 2 else None
    posinf = numpy.finfo(numpy.float16).max if posinf is None else posinf
    neginf = args[3] if len(args) > 3 else None
    neginf = numpy.finfo(numpy.float16).min if neginf is None else neginf
    return elementwise(FuncEnum.NAN_TO_NUM)(
        input_val,
        AITTensor(value=nan, shape=[], name="nan"),
        AITTensor(value=posinf, shape=[], name="posinf"),
        AITTensor(value=neginf, shape=[], name="neginf"),
    )


@ait_converter(torch.ops.aten.split_with_sizes.default)
@ait_converter(torch.ops.aten.split.Tensor)
def aten_ops_split(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Non-tensor inputs for {name}: {input_val}")

    split_size_or_sections = args[1]
    # TODO split_size_or_sections can be IntVar and AIT does not support yet
    # if not isinstance(split_size_or_sections, (int, list)):
    #     raise ValueError(
    #         f"Unexpected value for split_size_or_sections in {name}: {split_size_or_sections}"
    #     )
    dim = args[2] if len(args) > 2 else 0
    if not isinstance(dim, int):
        raise ValueError(f"Unexpected value for dim in {name}: {dim}")

    return split()(input_val, split_size_or_sections, dim)


@ait_converter(torch.ops.aten.sym_numel)
def aten_ops_numel(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")
    shape = size()(input_val)
    res = shape[0]
    for ind, dim in enumerate(shape):
        if ind != 0:
            res = int_elementwise(FuncEnum.MUL)(res, dim)
    return res


@ait_converter(torch.ops.aten.permute.default)
def aten_ops_permute(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    permutation = args[1]
    if len(permutation) > 5:
        raise RuntimeError(f"Unsupported permutation {permutation} for {input_val}")

    return permute()(input_val, permutation)


@ait_converter(torch.ops.aten.pow.Tensor_Scalar)
def aten_ops_pow(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    exp = args[1]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    return elementwise(FuncEnum.POW)(input_val, exp)


@ait_converter(torch.ops.aten.relu.default)
def aten_ops_relu(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    return elementwise(FuncEnum.RELU)(input_val)


@ait_converter(torch.ops.aten.reshape)
@ait_converter(torch.ops.aten.view.default)
def aten_ops_reshape(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")
    shape = args[1]
    new_shape = []
    for s in shape:
        if isinstance(s, IntVarTensor) or s == -1:
            new_shape.append(s)
        elif isinstance(s, int):
            new_shape.append(IntVarTensor(IntImm(s)))
        else:
            raise RuntimeError(f"Unexpected shape type for {name}: {s} in {shape}")

    if new_shape.count(-1):
        assert new_shape.count(-1) == 1
        input_shape = size()(input_val)
        unkown_dim = input_shape[0]
        for i in range(1, len(input_shape)):
            unkown_dim = unkown_dim * input_shape[i]
        idx = new_shape.index(-1)

        for s in new_shape:
            if s != -1:
                unkown_dim = unkown_dim / s
        new_shape[idx] = unkown_dim

    return reshape()(input_val, new_shape)


@ait_converter(torch.ops.aten.sym_size)
def aten_ops_size(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")
    dim = args[1]
    return size()(input_val, dim)


@ait_converter(aten_compose_getitem_slice)
@ait_converter(torch.ops.aten.slice.Tensor)
@ait_converter(torch.ops.aten.select.int)
def aten_ops_slice(  # noqa: C901
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    idx = []
    if target == aten_compose_getitem_slice:
        for sli in args[1]:
            start = sli[1]
            end = sli[2] if len(sli) >= 3 else start + 1
            if end == 9223372036854775807:  # represents None in pt2 tracer
                end = None
            step = sli[3] if len(sli) >= 4 else None
            idx.append(slice(start, end, step))
    else:
        dim = args[1]
        start = args[2]
        end = args[3] if len(args) > 3 else start + 1
        if end == 9223372036854775807:  # represents None in pt2 tracer
            end = None
        step = args[4] if len(args) > 4 else None
        for _ in range(0, dim):
            idx.append(slice(None, None, None))
        if len(args) > 3:
            idx.append(slice(start, end, step))
        else:
            idx.append(start)

    if isinstance(input_val, (tuple, list)):
        return operator.getitem(input_val, idx)
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    rank = input_val._rank()
    op = dynamic_slice()

    def num_slice_types(slices):
        return sum(1 for s in slices if isinstance(s, slice) or isinstance(s, int))

    # Replace ellipsis with expand slices.
    num_ellipsis = rank - num_slice_types(idx)
    expand_idx = []
    for i in idx:
        if i == Ellipsis:
            # pass explicit start to guard against negative num_ellipsis
            for _ in range(0, num_ellipsis):
                expand_idx.append(slice(None, None, None))
        else:
            expand_idx.append(i)
    idx = expand_idx

    # Record indices that need to be either:
    #   (1) sequeezed if Slice-index is of int; or
    #   (2) unsqueezed if Slice-index is of None
    # Each element of the list is a tuple of (int, func), where the second item
    # is either squeeze or unsqueeze function and the first
    # item gives the index to be squeezed or unsqueezed.
    squeezable_indices = []
    # the number of the indices of type None
    num_none_indices = 0
    start, end = [], []
    for index, i in enumerate(idx):
        if i is None:
            squeezable_indices.append((index, unsqueeze))
            num_none_indices += 1
            continue
        if isinstance(i, int):
            i = get_positive_dim(i, input_val.shape()[index].value())
            # If we pass an int, we need to squeeze this dim.
            # Note that because we skip None-indices before, so we adjust
            # the index by subtracting the number of None-indices.
            squeezable_indices.append((index - num_none_indices, squeeze))
        # if idx is slice, AIT only support slice.step == 1
        # TODO remove check once slice support step != 1
        if isinstance(i, slice) and i.step not in (1, None):
            raise ValueError(
                f"Slice tensor only support step=1 case, get step={i.step}."
            )
        start.append(i.start if isinstance(i, slice) else i)
        end.append(i.stop if isinstance(i, slice) else (i + 1 if i is not None else i))

    # append hiden dim at end
    while len(start) < rank:
        start.append(0)
        end.append(None)
    output = op(input_val, start, end)
    for dim, squeeze_func in reversed(squeezable_indices):
        output = squeeze_func(dim)(output)
    return output


@ait_converter(torch.ops.aten.squeeze)
@ait_converter(torch.ops.aten.squeeze.default)
@ait_converter(torch.ops.aten.squeeze.dim)
def aten_ops_squeeze(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    dim = args[1] if len(args) > 1 else None
    if not isinstance(dim, int) and dim is not None:
        raise ValueError(f"Unexpected {type(dim)} dim for {name}: {dim}")

    return squeeze(dim)(input_val)


@ait_converter(torch.ops.aten.sum.dim_IntList)
def aten_ops_sum(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    dims = args[1]
    keepdim = args[2] if len(args) > 2 else False

    if len(dims) == 1:
        return reduce_sum(dim=dims, keepdim=keepdim)(input_val)
    else:
        new_dims = list(dims)
        res = input_val
        for i, d in enumerate(new_dims):
            if d < 0:
                d += len(input_val.shape())
            new_dims[i] = d
            res = reduce_sum(dim=d, keepdim=True)(res)
        if not keepdim:
            new_dims = sorted(new_dims, reverse=True)
            for d in new_dims:
                res = squeeze(d)(res)
        return res


@ait_converter(torch.ops.aten.hardtanh.default)
def aten_ops_hardtanh(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")
    result = input_val
    minimal = args[1] if len(args) > 1 else -1
    maximum = args[2] if len(args) > 2 else 1
    if minimal is not None:
        result = elementwise(FuncEnum.MAX)(result, AITTensor(value=minimal, shape=[]))
    if maximum is not None:
        result = elementwise(FuncEnum.MIN)(result, AITTensor(value=maximum, shape=[]))
    return result


@ait_converter(torch.ops.aten.t.default)
def aten_ops_transpose(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO: we will also support https://pytorch.org/docs/stable/generated/torch.transpose.html in the future
    # Be careful. Transpose is expensive, so we want to avoid it.
    input_val = args[0]
    permutation = [0, 2, 1]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")
    input_3d = unsqueeze(0)(input_val)
    input_3d = permute()(input_3d, permutation)
    input_2d = squeeze(0)(input_3d)
    return input_2d


@ait_converter(torch.ops.aten.unsqueeze.default)
def aten_ops_unsqueeze(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    dim = args[1]
    if not isinstance(dim, int):
        raise ValueError(f"Unexpected {type(dim)} dim for {name}: {dim}")

    return unsqueeze(dim)(input_val)


## operator for symbolic computation
@ait_converter(operator.mul)
def operator_ops_mul(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    other_val = args[1]
    if not isinstance(input_val, IntVarTensor):
        if isinstance(input_val, int):
            input_val = IntVarTensor(IntImm(input_val))
        else:
            raise ValueError(f"Unexpected input type for {name}: {input_val}")
    if not isinstance(other_val, IntVarTensor):
        if isinstance(other_val, int):
            other_val = IntVarTensor(IntImm(other_val))
        else:
            raise ValueError(f"Unexpected other input type for {name}: {other_val}")
    res = int_elementwise(FuncEnum.MUL)(input_val, other_val)
    return res


@ait_converter(operator.add)
def operator_ops_add(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    other_val = args[1]
    if not isinstance(input_val, IntVarTensor):
        if isinstance(input_val, int):
            input_val = IntVarTensor(IntImm(input_val))
        else:
            raise ValueError(f"Unexpected input type for {name}: {input_val}")
    if not isinstance(other_val, IntVarTensor):
        if isinstance(other_val, int):
            other_val = IntVarTensor(IntImm(other_val))
        else:
            raise ValueError(f"Unexpected other input type for {name}: {other_val}")
    res = int_elementwise(FuncEnum.ADD)(input_val, other_val)
    return res


@ait_converter(operator.sub)
def operator_ops_sub(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    other_val = args[1]
    if not isinstance(input_val, IntVarTensor):
        if isinstance(input_val, int):
            input_val = IntVarTensor(IntImm(input_val))
        else:
            raise ValueError(f"Unexpected input type for {name}: {input_val}")
    if not isinstance(other_val, IntVarTensor):
        if isinstance(other_val, int):
            other_val = IntVarTensor(IntImm(other_val))
        else:
            raise ValueError(f"Unexpected other input type for {name}: {other_val}")
    res = int_elementwise(FuncEnum.SUB)(input_val, other_val)
    return res


@ait_converter(operator.floordiv)
def operator_ops_floordiv(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    other_val = args[1]
    if not isinstance(input_val, IntVarTensor):
        if isinstance(input_val, int):
            input_val = IntVarTensor(IntImm(input_val))
        else:
            raise ValueError(f"Unexpected input type for {name}: {input_val}")
    if not isinstance(other_val, IntVarTensor):
        if isinstance(other_val, int):
            other_val = IntVarTensor(IntImm(other_val))
        else:
            raise ValueError(f"Unexpected other input type for {name}: {other_val}")
    res = int_elementwise(FuncEnum.DIV)(input_val, other_val)
    return res


@ait_converter(torch.ops.aten.abs.default)
def aten_unary_ops_abs(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    return elementwise(FuncEnum.ABS)(input_val)


@ait_converter(torch.ops.aten.clamp.default)
def aten_unary_ops_clamp(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    result = input_val
    minimal = args[1]
    maximum = args[2] if len(args) > 2 else None
    if minimal is not None:
        result = elementwise(FuncEnum.MAX)(result, AITTensor(value=minimal, shape=[]))
    if maximum is not None:
        result = elementwise(FuncEnum.MIN)(result, AITTensor(value=maximum, shape=[]))
    return result


@ait_converter(torch.ops.aten.log.default)
def aten_unary_ops_log(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    return elementwise(FuncEnum.LOGE)(input_val)


@ait_converter(torch.ops.aten.sigmoid.default)
def aten_unary_ops_sigmoid(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")
    return elementwise(FuncEnum.SIGMOID)(input_val)


@ait_converter(torch.ops.aten.sign.default)
def aten_unary_ops_sign(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    return elementwise(FuncEnum.SIGN)(input_val)


@ait_converter(torch.ops.aten.sin.default)
def aten_unary_ops_sin(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    return elementwise(FuncEnum.SIN)(input_val)


@ait_converter(torch.ops.aten.sqrt.default)
def aten_unary_ops_sqrt(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    return elementwise(FuncEnum.SQRT)(input_val)


@ait_converter(torch.ops.aten.tanh.default)
def aten_unary_ops_tanh(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = args[0]
    return elementwise(FuncEnum.TANH)(input_val)
