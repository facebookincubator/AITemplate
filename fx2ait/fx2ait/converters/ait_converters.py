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
import math
import operator
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from aitemplate.compiler.public import (
    avg_pool2d,
    bmm_rrr,
    chunk,
    clamp,
    concatenate,
    conv2d,
    conv2d_bias,
    conv3d,
    conv3d_bias,
    depthwise_conv3d,
    dynamic_slice,
    elementwise,
    expand,
    flatten,
    full,
    FuncEnum,
    gemm_rcr,
    gemm_rrr,
    getitem,
    group_norm,
    IntImm,
    IntVar,
    IntVarTensor,
    layernorm,
    max_pool2d,
    ndhwc3to8,
    pad_last_dim,
    permute,
    reduce_mean,
    reduce_sum,
    reshape,
    size,
    softmax,
    split,
    squeeze,
    Tensor as AITTensor,
    topk,
    transposed_conv2d,
    transposed_conv2d_bias,
    tuple_construct,
    unsqueeze,
    var,
    vector_norm,
)

from aitemplate.testing import detect_target

from fx2ait.acc_tracer import acc_ops, ait_acc_ops
from torch.fx.node import Argument, Target

from .converter_registry import ait_converter

from .utils import (
    ait_ncdhw2ndhwc,
    ait_nchw2nhwc,
    ait_ncl2nlc,
    ait_ndhwc2ncdhw,
    ait_nhwc2nchw,
    ait_nlc2ncl,
    create_binary_op,
    create_reduce_op,
    create_unary_op,
    get_positive_dim,
    identical_elem_tuple_to_int,
    ncdhw2ndhwc,
    nchw2nhwc,
    weight_ncdhw2ndhwc,
    weight_nchw2nhwc,
)

USE_ROCM = detect_target().name() == "rocm"

logger: logging.Logger = logging.getLogger(__name__)
ConverterOutput = Union[AITTensor, Tuple[AITTensor, ...], List[IntVar], IntVar]


@ait_converter(acc_ops.sigmoid)
def acc_ops_sigmoid(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")
    return elementwise(FuncEnum.SIGMOID)(input_val)


@ait_converter(acc_ops.mul)
def acc_ops_mul(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    return create_binary_op(FuncEnum.MUL, args, kwargs, name)


@ait_converter(acc_ops.div)
def acc_ops_div(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    return create_binary_op(FuncEnum.DIV, args, kwargs, name)


@ait_converter(acc_ops.floor_div)
def acc_ops_floor_div(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    return create_binary_op(FuncEnum.FLOOR_DIV, args, kwargs, name)


@ait_converter(acc_ops.add)
def acc_ops_add(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    return create_binary_op(FuncEnum.ADD, args, kwargs, name)


@ait_converter(acc_ops.sub)
def acc_ops_sub(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    return create_binary_op(FuncEnum.SUB, args, kwargs, name)


@ait_converter(acc_ops.tanh)
def acc_ops_tanh(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    return elementwise(FuncEnum.TANH)(input_val)


@ait_converter(acc_ops.sin)
def acc_ops_sin(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    return elementwise(FuncEnum.SIN)(input_val)


@ait_converter(acc_ops.cos)
def acc_ops_cos(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    return elementwise(FuncEnum.COS)(input_val)


@ait_converter(acc_ops.sqrt)
def acc_ops_sqrt(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    return elementwise(FuncEnum.SQRT)(input_val)


@ait_converter(acc_ops.clone)
def acc_ops_clone(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    # deepcopy results with an error. replace with Idnetity multiplication by 1.
    # TODO: implement __deepcopy__ / clone for AITTensor.
    one_const = AITTensor(shape=[], dtype="float16", name="one_const", value=1.0)
    identity_mul_result = elementwise(FuncEnum.MUL)(input_val, one_const)
    return identity_mul_result


@ait_converter(acc_ops.sum)
def acc_ops_sum(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    return create_reduce_op(reduce_sum, args, kwargs, name)


@ait_converter(acc_ops.mean)
def acc_ops_mean(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    return create_reduce_op(reduce_mean, args, kwargs, name)


@ait_converter(acc_ops.linear)
def acc_ops_linear(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if USE_ROCM:
        shape = input_val._attrs["shape"]
        input_val = input_val if len(shape) == 2 else reshape()(input_val, [-1, shape[-1]])
    weight = kwargs["weight"]
    assert isinstance(weight, AITTensor)
    
    result = gemm_rcr()(input_val, weight)

    bias = kwargs["bias"]
    if bias is not None:
        assert isinstance(bias, AITTensor)
        result = elementwise(FuncEnum.ADD)(result, bias)
    if USE_ROCM:
        result = result if len(shape) == 2 else reshape()(result, [shape[0], -1, result._attrs["shape"][-1]])
    return result


@ait_converter(acc_ops.unsqueeze)
def acc_ops_unsqueeze(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    dim = kwargs["dim"]
    if not isinstance(dim, int):
        raise ValueError(f"Unexpected {type(dim)} dim for {name}: {dim}")

    return unsqueeze(dim)(input_val)


@ait_converter(acc_ops.clamp)
def acc_ops_clamp(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    result = input_val
    min_val = kwargs.get("min")
    max_val = kwargs.get("max")
    return clamp()(result, min_val, max_val)


@ait_converter(acc_ops.linalg_norm)
def acc_ops_linalg_norm(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    if "ord" not in kwargs or kwargs["ord"] != 2:
        raise RuntimeError("AIT linalg_norm only supports ord=2 use case!")

    # Hard code ord_kind=2 for l2 norm
    l2_norm = vector_norm(
        ord_kind=2, dim=kwargs["dim"], keepdim=kwargs["keepdim"], dtype=None
    )

    return l2_norm(input_val)


@ait_converter(acc_ops.permute)
def acc_ops_permute(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    permutation = kwargs["permutation"]

    return permute()(input_val, permutation)


@ait_converter(acc_ops.cat)
def acc_ops_cat(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    tensors = kwargs["tensors"]
    for t in tensors:
        if not isinstance(t, AITTensor):
            raise ValueError(f"Non-tensor inputs for {name}: {tensors}")

    dim = kwargs["dim"]
    if not isinstance(dim, int):
        raise ValueError(f"Unexpected {type(dim)} dim for {name}: {dim}")

    return concatenate()(tensors, dim=dim)


@ait_converter(acc_ops.sign)
def acc_ops_sign(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    return elementwise(FuncEnum.SIGN)(input_val)


@ait_converter(acc_ops.abs)
def acc_ops_abs(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    return elementwise(FuncEnum.ABS)(input_val)


@ait_converter(acc_ops.log)
def acc_ops_log(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    return elementwise(FuncEnum.LOGE)(input_val)


@ait_converter(acc_ops.var)
def acc_ops_var(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    op = var(
        dim=kwargs["dim"],
        unbiased=kwargs["unbiased"],
        keepdim=kwargs["keepdim"],
        dtype=None,
    )
    return op(input_val)


@ait_converter(acc_ops.softmax)
def acc_ops_softmax(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    dim = kwargs["dim"]
    rank = len(input_val.shape())
    if dim < 0:
        dim = rank + dim
    if dim != rank - 1:
        for i in range(dim + 1, rank):
            unsupported = False
            if isinstance(input_val.shape()[i], IntImm):
                if input_val.shape()[i].value() != 1:
                    unsupported = True
            elif isinstance(input_val.shape()[i], IntVar):
                unsupported = True
            else:
                raise RuntimeError(
                    f"unknown dimension type={type(i)} in AITTensor={input_val}"
                )

            if unsupported:
                raise ValueError(
                    f"AIT softmax only supports dim=rank-1, got AITTensor={input_val}, "
                    f"where dim={dim}, rank={rank}"
                )
        reshape_dim = size()(input_val)[: dim + 1]
        reshape_val = reshape()(input_val, reshape_dim)
        softmax_val = softmax()(reshape_val, -1)
        return reshape()(
            softmax_val, reshape_dim + [IntVarTensor(IntImm(1))] * (rank - dim - 1)
        )

    return softmax()(input_val, dim)


@ait_converter(acc_ops.relu)
def acc_ops_relu(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    return elementwise(FuncEnum.RELU)(input_val)


@ait_converter(acc_ops.leaky_relu)
def acc_ops_leaky_relu(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")
    negative_slope = kwargs["negative_slope"]
    return elementwise(FuncEnum.LRELU)(input_val, negative_slope)


@ait_converter(acc_ops.squeeze)
def acc_ops_squeeze(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")

    dim = kwargs["dim"] if "dim" in kwargs else None
    op = squeeze(dim)
    return op(input_val)


@ait_converter(acc_ops.size)
def acc_ops_size(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    if "dim" in kwargs:
        raise NotImplementedError(
            f"In {name} found 'dim' in size() which is not supported"
        )

    return size()(input_val)


@ait_converter(acc_ops.unbind)
def acc_ops_unbind(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    dim = kwargs["dim"]
    shape = input_val.shape()
    res = []
    if dim < 0:
        dim = len(shape) + dim
    for cnt in range(shape[dim].value()):
        idx = []
        for i in range(len(shape)):
            if i != dim:
                idx.append(slice(None, None, None))
            else:
                idx.append(cnt)
        kwargs_new = {
            "input": input_val,
            "idx": tuple(idx),
        }
        res.append(acc_ops_getitem(target, args, kwargs_new, name))
    return res


@ait_converter(operator.getitem)
@ait_converter(acc_ops.getitem)
def acc_ops_getitem(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # operator.getitem does not have kwargs. We copy args to kwargs so the downstream like acc_ops_slice can use it.
    new_kwargs = dict(kwargs)
    if "input" not in kwargs:
        new_kwargs["input"] = args[0]
    if "idx" not in kwargs:
        new_kwargs["idx"] = args[1]
    kwargs = new_kwargs
    input_val = kwargs["input"]
    idx = kwargs["idx"]
    if isinstance(idx, Sequence) and any(isinstance(x, Sequence) for x in idx):
        count = 0
        dim = None
        s = None
        for d, x in enumerate(idx):
            if isinstance(x, Sequence):
                count += 1
                dim = d
                s = x
        # TODO: Because multi-list concatenations e.g. x[[0,1],[0,2]] have broadcast implications
        # which requires careful pre-conditions and complicated calculations,
        # we ignore the situation for now and may add support per request.
        assert count == 1, "Expected only one dimension with list concatenation."

        # For list concatenations, we first take slices and then concate them back
        # In terms of performance, AIT backend will take care of fusing these ops.
        groups = []
        kw = {"input": input_val}
        start_idx = 0
        end_idx = 1
        while end_idx < len(s):
            if s[end_idx] - s[start_idx] == end_idx - start_idx:
                end_idx += 1
                continue
            else:
                kw["idx"] = (
                    idx[:dim]
                    + (slice(s[start_idx], s[end_idx - 1] + 1, None),)
                    + idx[dim + 1 :]
                )
                groups.append(acc_ops_slice(target, args, kw, name))
                start_idx = end_idx
                end_idx += 1
        kw["idx"] = (
            idx[:dim]
            + (slice(s[start_idx], s[end_idx - 1] + 1, None),)
            + idx[dim + 1 :]
        )
        groups.append(acc_ops_slice(target, args, kw, name))
        return concatenate()(groups, dim=dim)

    if isinstance(idx, slice) or (
        isinstance(idx, Sequence) and any(isinstance(x, slice) for x in idx)
    ):
        return acc_ops_slice(target, args, kwargs, name)
    if isinstance(input_val, AITTensor):
        return acc_ops_slice(target, args, kwargs, name)

    if isinstance(kwargs["idx"], int):
        idx = get_positive_dim(idx, len(input_val))

    if all(isinstance(i, IntImm) for i in input_val):
        return operator.getitem(input_val, kwargs["idx"])
    else:
        return getitem()(input_val, idx)


@ait_converter(acc_ops.slice_tensor)
def acc_ops_slice(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    idx = kwargs["idx"]
    if isinstance(input_val, (tuple, list)):
        return operator.getitem(input_val, idx)
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    rank = input_val._rank()
    if not isinstance(idx, Sequence):
        idx = [idx]
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
        # TODO: fix None for a more general case.
        # unsqueeze(dim=-1) to unsqueeze the most inner dimension
        if dim > rank and squeeze_func == unsqueeze:
            dim = -1
        output = squeeze_func(dim)(output)
    return output


@ait_converter(acc_ops.reshape)
def acc_ops_reshape(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    shape = kwargs["acc_out_ty"].shape

    return reshape()(input_val, shape)


# TODO (T124248862)
# We are waiting for full support of topk including:
# actual return values
# dim,
# largest flag,
# sorted flag
@ait_converter(acc_ops.topk)
def acc_ops_topk(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Unexpected input for {name}: {input_val}")

    k = kwargs["k"]
    if not isinstance(k, int):
        raise ValueError(f"Unexpected value for k in {name}: {k}")

    dim = kwargs["dim"] if "dim" in kwargs else None
    if dim is not None and dim != -1:
        raise NotImplementedError(
            f"Found 'dim' in {name} which is not supported: {dim}"
        )

    largest = kwargs["largest"] if "largest" in kwargs else None
    if largest is not None and largest is not True:
        raise NotImplementedError(
            f"Found 'largest' in {name} which is not supported: {largest}"
        )

    # current AIT implementation only returns indices, so 'sorted' does not apply. Ignore if specified.
    sorted = kwargs["sorted"] if "sorted" in kwargs else None
    if sorted is not None:
        logger.warning("Ignoring the value of 'sorted': %s", sorted)

    result_indices = topk(k=k)(input_val)
    # current AIT implementation only returns indices. to match the torch topk return types, create dummy values
    #
    # TODO remove the hard coded dtype below, once we know whether AIT will support fp32 (thus providing an option of
    # fp16 or fp32 for values)
    return (
        AITTensor(
            shape=result_indices.shape(), dtype="float16", name=f"{name}_result_values"
        ),
        result_indices,
    )


@ait_converter(acc_ops.tuple_construct)
def acc_ops_tuple_construct(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    tensors = kwargs["tensors"]
    return tuple_construct()(*tensors)


@ait_converter(acc_ops.conv_transpose2d)
def acc_ops_conv_transpose2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    output_padding = identical_elem_tuple_to_int(kwargs["output_padding"])
    assert output_padding == 0, "output_padding is not 0!"

    input_val = ait_nchw2nhwc(kwargs["input"])
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    weight = kwargs["weight"]
    assert isinstance(weight, AITTensor)
    weight = weight_nchw2nhwc(weight)
    weight._attrs["shape"] = nchw2nhwc(weight._attrs["shape"])
    w_last_dim = weight._attrs["data"].tensor.shape[-1]

    bias = kwargs["bias"]
    assert bias is None or isinstance(bias, AITTensor)

    stride = identical_elem_tuple_to_int(kwargs["stride"])
    padding = identical_elem_tuple_to_int(kwargs["padding"])
    dilation = identical_elem_tuple_to_int(kwargs["dilation"])
    assert dilation == 1, "dilation {dilation} does not equal to 1!"
    assert all(
        isinstance(x, int) for x in [stride, padding, dilation]
    ), "Expected int stride, padding, and dilation"

    if kwargs["groups"] is None or kwargs["groups"] == 1:
        assert (
            w_last_dim % 8 == 0
        ), f"cutlass needs weight output channel={w_last_dim} is not divisble by 8! This restriction may be not valid in newer version"

        if bias:
            result = transposed_conv2d_bias(
                stride=stride, pad=padding, dilate=dilation
            )(input_val, weight, bias)
        else:
            result = transposed_conv2d(stride=stride, pad=padding, dilate=dilation)(
                input_val, weight
            )
    else:
        # Grouped conv doesn't currently work on AIT CUDA, manually map
        groups = kwargs["groups"]
        assert (
            w_last_dim * groups
        ) % 8 == 0, f"cutlass needs weight output channel={w_last_dim*groups} is not divisble by 8! This restriction may be not valid in newer version"

        group_size = input_val.shape()[3]._attrs["values"][0] // groups
        w_group_size = weight.shape()[0]._attrs["values"][0] // groups

        def get_channel_dim_slice_idx(start, end, step):
            all_none_slice = slice(None, None, None)
            return (
                all_none_slice,
                all_none_slice,
                all_none_slice,
                slice(start, end, step),
            )

        def get_batch_dim_slice_idx(start, end, step):
            return (slice(start, end, step),)

        def make_slice(x, slice_idx, name):
            return acc_ops_slice(
                target,
                args,
                {
                    "input": x,
                    "idx": slice_idx,
                },
                name,
            )

        conv_groups = [
            _choose_conv2d_op(
                stride,
                padding,
                dilation,
                make_slice(  # input_val[:,:,:,gs*i:gs*i + gs]
                    input_val,
                    get_channel_dim_slice_idx(
                        i * group_size, i * group_size + group_size, 1
                    ),
                    f"{name}.slice_{i}",
                ),
                make_slice(  # weights[wgs*i:wgs*i + wgs,]
                    weight,
                    get_batch_dim_slice_idx(
                        i * w_group_size, i * w_group_size + w_group_size, 1
                    ),
                    f"{name}.weight.slice_{i}",
                ),
                None
                if bias is None
                else make_slice(  # bias[wgs*i:wgs*i + wgs,]
                    bias,
                    get_batch_dim_slice_idx(
                        i * w_group_size, i * w_group_size + w_group_size, 1
                    ),
                    f"{name}.bias.slice_{i}",
                ),
                transposed=True,
            )
            for i in range(groups)
        ]
        result = concatenate()(conv_groups, dim=3)

    return ait_nhwc2nchw(result)


@ait_converter(acc_ops.nan_to_num)
def acc_ops_nan_to_num(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    nan = 0 if kwargs["nan"] is None else kwargs["nan"]

    def _get_dtype(dtype: str):
        if dtype in ("float", "float32"):
            return np.float32
        elif dtype == "float16":
            return np.float16
        else:
            raise NotImplementedError(f"Unsupported dtype {dtype} for nan_to_num")

    input_dtype = input_val.dtype()
    np_dtype = _get_dtype(input_dtype)
    posinf = np.finfo(np_dtype).max if kwargs["posinf"] is None else kwargs["posinf"]
    neginf = np.finfo(np_dtype).min if kwargs["neginf"] is None else kwargs["neginf"]
    return elementwise(FuncEnum.NAN_TO_NUM)(
        input_val,
        AITTensor(value=nan, shape=[], name="nan", dtype=input_dtype),
        AITTensor(value=posinf, shape=[], name="posinf", dtype=input_dtype),
        AITTensor(value=neginf, shape=[], name="neginf", dtype=input_dtype),
    )


@ait_converter(acc_ops.group_norm)
def acc_ops_group_norm(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    input_val = ait_nchw2nhwc(kwargs["input"])
    num_groups = kwargs["num_groups"]
    weight_val = kwargs["weight"]
    bias_val = kwargs["bias"]
    eps_val = kwargs["eps"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    num_channels = input_val.shape()[-1].value()
    op = group_norm(num_groups, num_channels)
    result = op(input_val, weight_val, bias_val, eps_val)
    return ait_nhwc2nchw(result)


@ait_converter(acc_ops.layer_norm)
def acc_ops_layer_norm(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    shape = kwargs["normalized_shape"]
    if shape is None or len(shape) == 0:
        raise ValueError(f"Unexpected normalized shape value in {name}: {shape}")
    weight = kwargs["weight"]
    bias = kwargs["bias"]
    eps = kwargs["eps"]
    normalized_shape = []
    if all(isinstance(i, int) for i in shape):
        for i in shape:
            normalized_shape.append(IntImm(i))
    elif all(isinstance(i, IntImm) or isinstance(i, IntVarTensor) for i in shape):
        normalized_shape = shape
    else:
        raise ValueError(f"Unexpected normalized shape value in {name}: {shape}")
    return layernorm()(input_val, weight, bias, normalized_shape, eps)


@ait_converter(acc_ops.flatten)
def acc_ops_flatten(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    start_dim = kwargs["start_dim"] if "start_dim" in kwargs else 0
    end_dim = kwargs["end_dim"] if "end_dim" in kwargs else -1

    return flatten(start_dim=start_dim, end_dim=end_dim)(input_val)


@ait_converter(acc_ops.matmul)
def acc_ops_matmul(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    lhs = kwargs["input"]
    if not isinstance(lhs, AITTensor):
        raise ValueError(f"Unexpected left operand in {name}: {lhs}")
    # TODO: ideally we shouldn't be using _rank()/_shape()
    lhs_shape = lhs.shape()
    if len(lhs_shape) < 2:
        raise ValueError(f"Not enough dims for matmul in {name}: {lhs_shape}")

    rhs = kwargs["other"]
    if not isinstance(rhs, AITTensor):
        raise ValueError(f"Unexpected right operand in {name}: {rhs}")
    # TODO: ideally we shouldn't be using _rank()/_shape()
    rhs_shape = rhs.shape()
    if len(rhs_shape) < 2:
        raise ValueError(f"Not enough dims for matmul in {name}: {rhs_shape}")

    if len(rhs_shape) == 2:
        return gemm_rrr()(lhs, rhs)
    elif len(lhs_shape) <= 3 and len(rhs_shape) <= 3:
        return bmm_rrr()(lhs, rhs)
    elif len(lhs_shape) == 4 and len(rhs_shape) == 4 and lhs_shape[1] == rhs_shape[1]:
        assert all(isinstance(i, IntImm) for i in lhs_shape[1:])
        assert all(isinstance(i, IntImm) for i in rhs_shape[1:])
        # Current AIT bmm only supports 3-dim. Use reshape to workaround.
        channel = lhs_shape[1].value()
        M = lhs_shape[2].value()
        K = lhs_shape[3].value()
        N = rhs_shape[3].value()
        if K != rhs_shape[2].value():
            raise ValueError(
                f"K dim mismatch on matmaul. Expected: [N, K] X [K, M]. Found: : [{M}, {K}] X [{rhs_shape[2].value()}, {N}]"
            )
        if isinstance(lhs_shape[0], IntImm) and (rhs_shape[0], IntImm):
            batch_size = lhs_shape[0].value()
            shape_0 = (batch_size * channel, M, K)
            shape_1 = (batch_size * channel, K, N)
            shape_2 = (batch_size, channel, M, N)
        elif isinstance(lhs_shape[0], IntVar) and isinstance(rhs_shape[0], IntVar):
            if lhs_shape[0] != rhs_shape[0]:
                raise ValueError(
                    f"Batch size mismatch on matmul. Expected: {lhs_shape[0]} == {rhs_shape[0]}"
                )
            lhs_size = size()(lhs)
            new_size = getitem()(lhs_size, 0) * getitem()(lhs_size, 1)
            shape_0 = (new_size, M, K)
            shape_1 = (new_size, K, N)
            shape_2 = (getitem()(lhs_size, 0), channel, M, N)
        else:
            raise NotImplementedError(
                f"Expected all dimension except for the batch dim to be static. Got {lhs_shape} vs. {rhs_shape}"
            )
        reshape_op_0 = reshape()(lhs, shape_0)
        reshape_op_1 = reshape()(rhs, shape_1)
        return reshape()(bmm_rrr()(reshape_op_0, reshape_op_1), shape_2)
    else:
        raise NotImplementedError(
            f"This case is unsupported in {name}: {len(lhs_shape)} and {len(rhs_shape)}"
        )


@ait_converter(acc_ops.chunk)
def acc_ops_chunk(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    shape = input_val.shape()
    target_dim = get_positive_dim(kwargs["dim"], len(shape))
    chunks = min(kwargs["chunks"], shape[target_dim].value())
    assert isinstance(
        shape[target_dim], IntImm
    ), f"Cannot perform chunk on dynamic dim! Get target dim {target_dim}."

    return chunk()(input_val, chunks, dim=target_dim)


@ait_converter(ait_acc_ops.split)
def ait_acc_ops_split(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Non-tensor inputs for {name}: {input_val}")

    split_size_or_sections = kwargs["split_size_or_sections"]
    if not isinstance(split_size_or_sections, (int, list)):
        raise ValueError(
            f"Unexpected value for split_size_or_sections in {name}: {split_size_or_sections}"
        )

    dim = kwargs["dim"]
    if not isinstance(dim, int):
        raise ValueError(f"Unexpected value for dim in {name}: {dim}")

    return split()(input_val, split_size_or_sections, dim)


@ait_converter(acc_ops.expand)
def ait_acc_ops_expand(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise ValueError(f"Non-tensor inputs for {name}: {input_val}")

    sizes = kwargs["sizes"]
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


@ait_converter(acc_ops.batch_norm)
def acc_ops_batch_norm(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    input_shape = input_val._attrs["shape"]
    input_rank = len(input_shape)
    assert 2 <= input_rank <= 5, f"expected {input_rank=} to be within [2, 5]"
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    if input_rank == 3:
        # BatchNorm1d
        input_val = ait_ncl2nlc(input_val)
    elif input_rank == 4:
        # BatchNorm2d
        input_val = ait_nchw2nhwc(input_val)
    elif input_rank == 5:
        # BatchNorm3d
        input_val = ait_ncdhw2ndhwc(input_val)

    scale = elementwise(FuncEnum.DIV)(
        kwargs["weight"],
        elementwise(FuncEnum.ADD)(
            elementwise(FuncEnum.SQRT)(kwargs["running_var"]),
            AITTensor(shape=[], value=kwargs["eps"]),
        ),
    )
    bias = elementwise(FuncEnum.SUB)(kwargs["bias"], kwargs["running_mean"])

    scale_dim_val = scale._attrs["shape"][0].value()

    # input is channel-last after permute
    input_shape = input_val._attrs["shape"]
    channel_dim = -1
    assert isinstance(
        input_shape[channel_dim], IntImm
    ), f"expected channel at {channel_dim=} in {input_shape=} to be static"
    channel_dim_val = input_shape[channel_dim].value()
    assert (
        channel_dim_val == scale_dim_val
    ), f"expected {channel_dim_val=} to be the same as {scale_dim_val=}"
    mul_result = elementwise(FuncEnum.MUL)(input_val, scale)
    result = elementwise(FuncEnum.ADD)(mul_result, bias)
    if input_rank == 3:
        # BatchNorm1d
        result = ait_nlc2ncl(result)
    elif input_rank == 4:
        # BatchNorm2d
        result = ait_nhwc2nchw(result)
    elif input_rank == 5:
        # BatchNorm3d
        result = ait_ndhwc2ncdhw(result)
    return result


def _choose_conv2d_op(
    stride: int,
    pad: int,
    dilate: int,
    x: AITTensor,
    weight: AITTensor,
    bias: AITTensor,
    transposed: bool = False,
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
    elif last_dim % 2 != 0:
        return RuntimeError(
            f"Conv2d is not implemented for input channel dim {last_dim}: it needs to be aligned to a multiple of 2/4/8"
        )
    if bias:
        return conv2d_bias(stride=stride, pad=pad, dilate=dilate)(x, weight, bias)
    else:
        return conv2d(stride=stride, pad=pad, dilate=dilate)(x, weight)


@ait_converter(acc_ops.conv2d)
def acc_ops_conv2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = ait_nchw2nhwc(kwargs["input"])
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    weight = kwargs["weight"]
    assert isinstance(weight, AITTensor)
    weight = weight_nchw2nhwc(weight)
    weight._attrs["shape"] = nchw2nhwc(weight._attrs["shape"])

    bias = kwargs["bias"]
    assert bias is None or isinstance(bias, AITTensor)

    stride = identical_elem_tuple_to_int(kwargs["stride"])
    padding = identical_elem_tuple_to_int(kwargs["padding"])
    dilation = identical_elem_tuple_to_int(kwargs["dilation"])

    assert all(
        isinstance(x, int) for x in [stride, padding, dilation]
    ), "Expected int stride, padding, and dilation"

    if kwargs["groups"] is None or kwargs["groups"] == 1:
        result = _choose_conv2d_op(stride, padding, dilation, input_val, weight, bias)
    else:
        # Grouped conv doesn't currently work on AIT CUDA, manually map
        groups = kwargs["groups"]
        group_size = input_val.shape()[3]._attrs["values"][0] // groups
        w_group_size = weight.shape()[0]._attrs["values"][0] // groups

        def get_channel_dim_slice_idx(start, end, step):
            all_none_slice = slice(None, None, None)
            return (
                all_none_slice,
                all_none_slice,
                all_none_slice,
                slice(start, end, step),
            )

        def get_batch_dim_slice_idx(start, end, step):
            return (slice(start, end, step),)

        def make_slice(x, slice_idx, name):
            return acc_ops_slice(
                target,
                args,
                {
                    "input": x,
                    "idx": slice_idx,
                },
                name,
            )

        conv_groups = [
            _choose_conv2d_op(
                stride,
                padding,
                dilation,
                make_slice(  # input_val[:,:,:,gs*i:gs*i + gs]
                    input_val,
                    get_channel_dim_slice_idx(
                        i * group_size, i * group_size + group_size, 1
                    ),
                    f"{name}.slice_{i}",
                ),
                make_slice(  # weights[wgs*i:wgs*i + wgs,]
                    weight,
                    get_batch_dim_slice_idx(
                        i * w_group_size, i * w_group_size + w_group_size, 1
                    ),
                    f"{name}.weight.slice_{i}",
                ),
                None
                if bias is None
                else make_slice(  # bias[wgs*i:wgs*i + wgs,]
                    bias,
                    get_batch_dim_slice_idx(
                        i * w_group_size, i * w_group_size + w_group_size, 1
                    ),
                    f"{name}.bias.slice_{i}",
                ),
                transposed=False,
            )
            for i in range(groups)
        ]
        result = concatenate()(conv_groups, dim=3)

    result = ait_nhwc2nchw(result)
    return result


def _choose_conv3d_op(
    stride: int,
    pad: int,
    dilate: int,
    x: AITTensor,
    weight: AITTensor,
    bias: AITTensor,
    groups: int = 1,
) -> ConverterOutput:
    """
    Helper to choose conv3d vs. depthwise_conv3d op based on existence of bias
    and groups
    """
    has_bias = bias is not None
    if groups is None or groups == 1:
        if has_bias:
            C_in = x.shape()[-1].value()

            if 3 == C_in:
                x = ndhwc3to8()(x)
                weight = ndhwc3to8()(weight)
            elif 8 != C_in:
                raise RuntimeError(
                    f"When having bias, conv3d currently only supports C_in == 3 or C_in == 8, but got C_in: {C_in}"
                )

            return conv3d_bias(stride=stride, pad=pad, dilate=dilate, group=1)(
                x, weight, bias
            )
        else:
            return conv3d(stride=stride, pad=pad, dilate=dilate, group=1)(x, weight)
    elif groups == weight._attrs["shape"][0].value():
        return depthwise_conv3d(
            stride=stride, pad=pad, dilate=dilate, group=groups, bias=has_bias
        )(x, weight, bias)
    else:
        raise RuntimeError(
            f"Currently NOT support groups != channels when groups enabled. Got C_in: {C_in} | groups: {groups} | has bias: {has_bias}"
        )


@ait_converter(acc_ops.conv3d)
def acc_ops_conv3d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = ait_ncdhw2ndhwc(kwargs["input"])
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    weight = kwargs["weight"]
    assert isinstance(weight, AITTensor)
    weight = weight_ncdhw2ndhwc(weight)
    weight._attrs["shape"] = ncdhw2ndhwc(weight._attrs["shape"])

    bias = kwargs["bias"]
    assert bias is None or isinstance(bias, AITTensor)

    stride = kwargs["stride"]
    padding = kwargs["padding"]
    dilation = kwargs["dilation"]

    groups = kwargs["groups"]

    result = _choose_conv3d_op(
        stride, padding, dilation, input_val, weight, bias, groups
    )
    return ait_ndhwc2ncdhw(result)


@ait_converter(acc_ops.max_pool3d)
def acc_ops_max_pool3d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    if (
        isinstance(kwargs["kernel_size"], tuple)
        and isinstance(kwargs["stride"], tuple)
        and isinstance(kwargs["padding"], tuple)
    ):
        kernel_size_tuple = kwargs["kernel_size"]
        stride_tuple = kwargs["stride"]
        padding_tuple = kwargs["padding"]

        assert kernel_size_tuple[0] == 1, "max_pool3d only supports kT == 1 currently"
        assert stride_tuple[0] == 1, "max_pool3d only supports sT == 1 currently"
        assert (
            padding_tuple[0] == 0
        ), "max_pool3d only supports T_padding == 0 currently"

        kernel_size = identical_elem_tuple_to_int(kernel_size_tuple[1:])
        stride = identical_elem_tuple_to_int(stride_tuple[1:])
        padding = identical_elem_tuple_to_int(padding_tuple[1:])
    elif (
        isinstance(kwargs["kernel_size"], int)
        and isinstance(kwargs["stride"], int)
        and isinstance(kwargs["padding"], int)
    ):
        kernel_size = kwargs["kernel_size"]
        stride = kwargs["stride"]
        padding = kwargs["padding"]
    else:
        raise RuntimeError("Only int or tuple types are supported")

    ceil_mode = kwargs["ceil_mode"]
    return_indices = kwargs["return_indices"]
    if ceil_mode or return_indices:
        raise RuntimeError(
            "Non-default ceil_mode/count_include_pad/divisor_override not supported yet"
        )

    N = input_val.shape()[0].value()
    C = input_val.shape()[1].value()
    D = input_val.shape()[2].value()
    H = input_val.shape()[3].value()
    W = input_val.shape()[4].value()

    reshape_op_0 = reshape()
    shape_0 = (N, C * D, H, W)
    input_val = reshape_op_0(input_val, shape_0)

    input_val = ait_nchw2nhwc(input_val)

    output = max_pool2d(kernel_size=kernel_size, stride=stride, pad=padding)(input_val)

    output = ait_nhwc2nchw(output)

    H_o = output.shape()[2].value()
    W_o = output.shape()[3].value()
    reshape_op_1 = reshape()
    shape_1 = (N, C, D, H_o, W_o)

    output = reshape_op_1(output, shape_1)
    return output


@ait_converter(acc_ops.max_pool2d)
def acc_ops_max_pool2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = ait_nchw2nhwc(kwargs["input"])
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    kernel_size = identical_elem_tuple_to_int(kwargs["kernel_size"])
    stride = identical_elem_tuple_to_int(kwargs["stride"])
    padding = identical_elem_tuple_to_int(kwargs["padding"])
    ceil_mode = kwargs["ceil_mode"]
    return_indices = kwargs["return_indices"]
    if ceil_mode or return_indices:
        raise RuntimeError(
            "Non-default ceil_mode/count_include_pad/divisor_override not supported yet"
        )
    result = max_pool2d(kernel_size=kernel_size, stride=stride, pad=padding)(input_val)
    return ait_nhwc2nchw(result)


@ait_converter(acc_ops.avg_pool2d)
def acc_ops_avg_pool2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = ait_nchw2nhwc(kwargs["input"])
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    kernel_size = identical_elem_tuple_to_int(kwargs["kernel_size"])
    stride = identical_elem_tuple_to_int(kwargs["stride"])
    padding = identical_elem_tuple_to_int(kwargs["padding"])
    ceil_mode = kwargs["ceil_mode"]
    count_include_pad = kwargs["count_include_pad"]
    divisor_override = kwargs["divisor_override"]
    if ceil_mode or not count_include_pad or divisor_override:
        raise RuntimeError(
            "Non-default ceil_mode/count_include_pad/divisor_override not supported yet"
        )
    result = avg_pool2d(kernel_size=kernel_size, stride=stride, pad=padding)(input_val)
    return ait_nhwc2nchw(result)


@ait_converter(acc_ops.adaptive_avg_pool2d)
def acc_ops_adaptive_avg_pool2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = ait_nchw2nhwc(kwargs["input"])
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    output_size = identical_elem_tuple_to_int(kwargs["output_size"])
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

    result = avg_pool2d(kernel_size=kernel_size, stride=stride, pad=0)(input_val)
    return ait_nhwc2nchw(result)


@ait_converter(acc_ops.contiguous)
def acc_ops_contiguous(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    return kwargs["input"]


@ait_converter(acc_ops.to_dtype)
def acc_ops_to_dtype(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    return kwargs["input"]


@ait_converter(acc_ops.gelu)
def acc_ops_gelu(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    # For extra speedup, you can always lower to fast_gelu
    if kwargs.get("approximate", None) == "tanh":
        result = elementwise(FuncEnum.FASTGELU)(input_val)
    else:
        result = elementwise(FuncEnum.GELU)(input_val)
    return result


@ait_converter(acc_ops.pow)
def acc_ops_pow(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    exponent = kwargs["exponent"]
    return elementwise(FuncEnum.POW)(input_val, exponent)


@ait_converter(acc_ops.tile)
def acc_ops_tile(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Unexpected input for {name}: {input_val}")
    shape_dims = list(kwargs["dims"])
    input_dim_len = len(input_val.shape())
    result = input_val
    if len(shape_dims) < input_dim_len:
        for _ in range(input_dim_len - len(shape_dims)):
            shape_dims.insert(0, 1)
    if input_dim_len < len(shape_dims):
        shape = input_val.shape()
        for _ in range(len(shape_dims) - input_dim_len):
            shape.insert(0, IntImm(1))
        result = expand()(input_val, shape)

    for i, shape in enumerate(shape_dims):
        # Avoid operate on batch_size dim
        if input_val.shape()[i]._attrs["name"] is not None:
            continue
        cat_groups = [result] * shape
        result = concatenate()(cat_groups, dim=i)
    return result


@ait_converter(math.sqrt)
def math_sqrt(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    return create_unary_op(FuncEnum.SQRT, args, kwargs, name)


@ait_converter(acc_ops.neg)
def acc_ops_neg(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    new_kwargs = kwargs.copy()
    dt = new_kwargs["input"]._attrs["dtype"]
    if dt == "float16" or dt == "float32":
        new_kwargs["other"] = float(-1)
    elif dt == "int32" or dt == "int64":
        new_kwargs["other"] = int(-1)
    else:
        raise ValueError(f"Unexpected input dtype {dt}")

    return create_binary_op(FuncEnum.MUL, args, new_kwargs, name)


@ait_converter(acc_ops.new_full)
def acc_ops_new_full(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    size = kwargs["size"]
    dtype = (
        kwargs["dtype"]
        if "dtype" in kwargs and kwargs["dtype"] is not None
        else input_val.dtype()
    )
    fill_value = kwargs["fill_value"]
    return full()(size, fill_value=fill_value, dtype=dtype)


@ait_converter(acc_ops.full_like)
def acc_ops_full_like(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    fill_value = kwargs["fill_value"]
    return full()(input_val.shape(), fill_value=fill_value, dtype=input_val.dtype())


@ait_converter(acc_ops.new_ones)
def acc_ops_new_ones(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    size = kwargs["size"]
    dtype = (
        kwargs["dtype"]
        if "dtype" in kwargs and kwargs["dtype"] is not None
        else input_val.dtype()
    )
    return full()(size, 1, dtype=dtype)


@ait_converter(acc_ops.ones_like)
def acc_ops_ones_like(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    return full()(input_val.shape(), 1, dtype=input_val.dtype())


@ait_converter(acc_ops.new_zeros)
def acc_ops_new_zeros(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    size = kwargs["size"]
    dtype = (
        kwargs["dtype"]
        if "dtype" in kwargs and kwargs["dtype"] is not None
        else input_val.dtype()
    )
    return full()(size, 0, dtype=dtype)


@ait_converter(acc_ops.zeros_like)
def acc_ops_zeros_like(
    target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")
    return full()(input_val.shape(), 0, dtype=input_val.dtype())
