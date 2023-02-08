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
import copy
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
    depthwise_conv3d,
    dynamic_slice,
    elementwise,
    expand,
    flatten,
    FuncEnum,
    gemm_rcr,
    gemm_rrr,
    getitem,
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

from fx2ait.acc_tracer import acc_ops, ait_acc_ops
from torch.fx.node import Argument, Target

from .converter_registry import ait_converter

from .utils import (
    create_binary_op,
    create_reduce_op,
    create_unary_op,
    get_positive_dim,
    identical_elem_tuple_to_int,
    ncdhw2ndhwc,
    nchw2nhwc,
    unify_dynamic_shape_name,
)

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
    res = copy.deepcopy(input_val)
    res._attrs["dst_ops"].clear()
    return res


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

    weight = kwargs["weight"]
    assert isinstance(weight, AITTensor)

    result = gemm_rcr()(input_val, weight)

    bias = kwargs["bias"]
    if bias is not None:
        assert isinstance(bias, AITTensor)
        result = elementwise(FuncEnum.ADD)(result, bias)

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

    # TODO:  unify_dynamic_shape_name is a hack to workaround AIT's dynamic shape requirement.
    # We will remove it after AIT provides vanilla support.
    for i in range(len(tensors) - 1):
        unify_dynamic_shape_name(tensors[i], tensors[i + 1])
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


@ait_converter(acc_ops.getitem)
def acc_ops_getitem(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
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
        for x in s:
            kw["idx"] = idx[:dim] + (x,) + idx[dim + 1 :]
            groups.append(unsqueeze(dim)(acc_ops_slice(target, args, kw, name)))
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

    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    weight = kwargs["weight"]
    assert isinstance(weight, AITTensor)
    weight._attrs["data"].tensor = weight._attrs["data"].tensor.permute(0, 2, 3, 1)
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

    return result


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
    normalized_shape = []
    if all(isinstance(i, int) for i in shape):
        for i in shape:
            normalized_shape.append(IntImm(i))
    elif all(isinstance(i, IntImm) or isinstance(i, IntVarTensor) for i in shape):
        normalized_shape = shape
    else:
        raise ValueError(f"Unexpected normalized shape value in {name}: {shape}")
    return layernorm()(input_val, weight, bias, normalized_shape)


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
        assert all(isinstance(i, IntImm) for i in lhs_shape)
        assert all(isinstance(i, IntImm) for i in rhs_shape)
        # Current AIT bmm only supports 3-dim. Use reshape to workaround.
        reshape_op_0 = reshape()
        batch_size = lhs_shape[0].value()
        M = lhs_shape[2].value()
        K = lhs_shape[3].value()
        channel = lhs_shape[1].value()
        shape_0 = (batch_size * channel, M, K)
        reshape_op_1 = reshape()
        N = rhs_shape[3].value()
        if K != rhs_shape[2].value():
            raise ValueError(
                f"K dim mismatch on matmaul. Expected: [N, K] X [K, M]. Found: : [{M}, {K}] X [{rhs_shape[2].value()}, {N}]"
            )

        shape_1 = (batch_size * channel, K, N)
        reshape_op_2 = reshape()
        shape_2 = (batch_size, channel, M, N)
        return reshape_op_2(
            bmm_rrr()(reshape_op_0(lhs, shape_0), reshape_op_1(rhs, shape_1)), shape_2
        )
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
    # TODO @qxy11: Update channels-last assumption once AIT backend is updated
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    scale = elementwise(FuncEnum.DIV)(
        kwargs["weight"],
        elementwise(FuncEnum.ADD)(
            elementwise(FuncEnum.SQRT)(kwargs["running_var"]),
            AITTensor(shape=[], value=kwargs["eps"]),
        ),
    )
    bias = elementwise(FuncEnum.SUB)(kwargs["bias"], kwargs["running_mean"])
    matmul_result = elementwise(FuncEnum.MUL)(input_val, scale)
    result = elementwise(FuncEnum.ADD)(matmul_result, bias)
    return result


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


@ait_converter(acc_ops.conv2d)
def acc_ops_conv2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO: qxy11: Update once channels-first format is supported
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    weight = kwargs["weight"]
    assert isinstance(weight, AITTensor)
    weight._attrs["data"].tensor = weight._attrs["data"].tensor.permute(0, 2, 3, 1)
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

    return result


def _choose_conv3d_op(
    stride: int,
    pad: int,
    dilate: int,
    x: AITTensor,
    weight: AITTensor,
    bias: [AITTensor],
    groups: int = 1,
) -> ConverterOutput:
    """
    Helper to choose conv3d vs. depthwise_conv3d op based on existence of bias
    and groups
    """
    weight._attrs["data"].tensor = weight._attrs["data"].tensor.permute(0, 2, 3, 4, 1)
    weight._attrs["shape"] = ncdhw2ndhwc(weight._attrs["shape"])

    if bias is not None:
        assert (
            groups == weight._attrs["shape"][0]
        ), "Currently only support channel == groups"
        return depthwise_conv3d(
            stride=stride, pad=pad, dilate=dilate, group=groups, bias=bias
        )(x, weight)
    else:
        assert (
            groups is None or groups == 1
        ), "Currently only support non-bias conv3d without groups"
        return conv3d(stride=stride, pad=pad, dilate=dilate)(x, weight)


@ait_converter(acc_ops.conv3d)
def acc_ops_conv3d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    input_val = kwargs["input"]
    if not isinstance(input_val, AITTensor):
        raise RuntimeError(f"Non-tensor inputs for {name}: {input_val}")

    weight = kwargs["weight"]
    assert isinstance(weight, AITTensor)

    bias = kwargs["bias"]
    assert bias is None or isinstance(bias, AITTensor)

    stride = identical_elem_tuple_to_int(kwargs["stride"])
    padding = identical_elem_tuple_to_int(kwargs["padding"])
    dilation = identical_elem_tuple_to_int(kwargs["dilation"])

    assert all(
        isinstance(x, int) for x in [stride, padding, dilation]
    ), "Expected int stride, padding, and dilation"

    groups = kwargs["groups"]

    return _choose_conv3d_op(stride, padding, dilation, input_val, weight, bias, groups)


@ait_converter(acc_ops.max_pool2d)
def acc_ops_max_pool2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO: @qxy11 Update once NCHW supported
    input_val = kwargs["input"]
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
    return max_pool2d(kernel_size=kernel_size, stride=stride, pad=padding)(input_val)


@ait_converter(acc_ops.avg_pool2d)
def acc_ops_avg_pool2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO: @qxy11 Update once NCHW supported
    input_val = kwargs["input"]
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
    return avg_pool2d(kernel_size=kernel_size, stride=stride, pad=padding)(input_val)


@ait_converter(acc_ops.adaptive_avg_pool2d)
def acc_ops_adaptive_avg_pool2d(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO: @qxy11 Update once NCHW supported
    input_val = kwargs["input"]
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

    return avg_pool2d(kernel_size=kernel_size, stride=stride, pad=0)(input_val)


@ait_converter(acc_ops.contiguous)
def acc_ops_contiguous(
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
        for i in range(input_dim_len - len(shape_dims)):
            shape_dims.insert(0, 1)
    if input_dim_len < len(shape_dims):
        shape = input_val.shape()
        for i in range(len(shape_dims) - input_dim_len):
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
