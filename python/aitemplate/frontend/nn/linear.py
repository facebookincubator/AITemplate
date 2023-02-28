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
Linear module.
"""
from aitemplate.testing import detect_target

from ...compiler import ops
from .module import Module
from .parameter import Parameter


class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_channels: size of each input sample
        out_channels: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        specialization: elementwise operation to add after the linear operation,
            Default: ``None``
        dtype: data type, default: ``float16``

    Shape:

        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in_channels}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out_channels}`.

    Attributes:

        weight: the learnable weights of the module of shape
            :math:`(\text{out_channels}, \text{in_channels})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_channels}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out_channels})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in_channels}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = Tensor(shape=[128, 20])
        >>> output = m(input)
        Tensor(shape=[128, 30])
    """

    USE_CUDA = None

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        specialization=None,
        dtype="float16",
        **kwargs,
    ):
        super().__init__()
        if Linear.USE_CUDA is None:
            Linear.USE_CUDA = detect_target().name() == "cuda"
        self.weight = Parameter(shape=[out_channels, in_channels], dtype=dtype)
        op_name = "gemm_rcr_bias" if bias else "gemm_rcr"
        if specialization is not None:
            op_name += "_" + specialization
        if bias:
            self.bias = Parameter(shape=[out_channels], dtype=dtype)
        op_func = getattr(ops, op_name)
        self._op_name = op_name
        self.op = op_func(**kwargs)
        self.use_bias = bias
        self.in_channels = in_channels

    def forward(self, *args):
        assert len(args) >= 1
        x = args[0]
        if not self.USE_CUDA:
            shape = x._attrs["shape"]
            x = x if len(shape) == 2 else ops.reshape()(x, [-1, self.in_channels])
        if len(args) == 2:
            if self.use_bias:
                inputs = [x, self.weight.tensor(), self.bias.tensor(), args[1]]
            else:
                inputs = [x, self.weight.tensor(), args[1]]
            output = self.op(*inputs)
            return output
        output = (
            self.op(x, self.weight.tensor(), bias=self.bias.tensor())
            if self.use_bias
            else self.op(x, self.weight.tensor())
        )
        return output
