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
Transposed conv2d op.
"""

import itertools
from typing import List

import jinja2

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.conv.conv2d import conv2d

from aitemplate.utils import shape_utils

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}HI = {{x_dim1}};
{{indent}}{{dtype}}WI = {{x_dim2}};
{{indent}}{{dtype}}CI = {{x_dim3}};
{{indent}}{{dtype}}CO = {{w_dim0}};
{{indent}}{{dtype}}KH = {{w_dim1}};
{{indent}}{{dtype}}KW = {{w_dim2}};
{{indent}}{{dtype}}SH = {{strideh}};
{{indent}}{{dtype}}SW = {{stridew}};
{{indent}}{{dtype}}DH = {{dilateh}};
{{indent}}{{dtype}}DW = {{dilatew}};
{{indent}}{{dtype}}PH = {{padh}};
{{indent}}{{dtype}}PW = {{padw}};
{{indent}}{{dtype}}KHEff = (KH - 1) * DH + 1;
{{indent}}{{dtype}}KWEff = (KW - 1) * DW + 1;
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}HO = (HI - 1) * SH - 2 * PH + KHEff;
{{indent}}{{dtype}}WO = (WI - 1) * SW - 2 * PW + KWEff;
"""
)


# pylint: disable=C0103
class transposed_conv2d(conv2d):
    r"""Transposed conv2d.

    Applies a 2D transposed convolution on input in shape (N, H, W, C_in) and produces output in shape (N, H_out, W_out, C_out). N is batch size, H, W are the height and width of the input images in pixels, and C is the number of channels.

    This module can be seen as the gradient of Conv2d with respect to its input. It is also known as a fractionally-strided convolution or a deconvolution (although it is not an actual deconvolution operation as it does not compute a true inverse of convolution). For more information, see the visualizations `here`_ and the `Deconvolutional Networks`_ paper.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`pad` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points.

    * :attr:`dilate` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`group` controls the number of blocked connections from input channels
      to output channels.

    Args:
        input: input tensor of shape :math:`(N , H , W, \text{in\_channels})`

        weight: filters of shape :math:`(\text{out\_channels} , K_h, K_w, \frac{\text{in\_channels}}{\text{groups}})`

    This operator uses "channels_last" data format. Below is an example and its equivalence in PyTorch:

    .. highlight:: python
    .. code-block:: python

        X = Tensor(shape=[N, H, W, C_in], dtype="float16", name="images", is_input=True)
        W = Tensor(shape=[C_out, K_h, K_w, C_in], dtype="float16", name="weight", is_input=True)
        OP = aitemplate.compiler.ops.transposed_conv2d(stride=1, pad=1, dilate=1)
        Y = OP(X, W)

    .. highlight:: python
    .. code-block:: python

        X_pt = NHWC2NCHW(X_ait)
        W_pt = NHWC2NCHW(W_ait)
        Y_pt = torch.nn.functional.conv_transpose2d(X_pt, W_pt)
        Y = NCHW2NHWC(Y_pt)

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """

    def __init__(self, stride, pad, dilate=1, group=1) -> None:
        """Transposed_conv2d constructor.

        Parameters
        ----------
        stride : int
            Stride of the convolution
        pad : int
            Size of padding to add to the input
        dilate : int, optional
            Size of spacing between kernel elements, by default 1
        group : int, optional
            Number of input channels to process to compute one output channel, by default 1
        """
        super().__init__(stride, pad, dilate=dilate, group=group)
        self._attrs["op"] = "transposed_conv2d"
        self._attrs["epilogue"] = "LinearCombination"
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE

    def _infer_shape(self, x: List[int], w: List[int]) -> List[int]:
        if x[3] != w[0] * self._attrs["group"]:
            raise RuntimeError("X/W Shape mismatch for conv2d")
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=x[3],
            w_dim0=w[3],  # for conv_transpose w = [c_in, kh, kw, c_out]
            w_dim1=w[1],
            w_dim2=w[2],
            **self._get_params_factory(),
        )
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NO"]),
            int(output["HO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor, w: Tensor) -> List[int]:
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        w_shape = [var._attrs["values"][0] for var in w._attrs["shape"]]
        self._attrs["CO"] = w_shape[3]
        self._attrs["KH"] = w_shape[1]
        self._attrs["KW"] = w_shape[2]
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape, w_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            shape_utils.gen_int_var(unique([d[0] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
        ]
        return output_shape
