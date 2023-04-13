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
Fused special_conv2d_bias_activation op.
"""
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.conv.conv2d import conv2d
from aitemplate.compiler.ops.padding import nhwc3to4, nhwc3to8


# pylint: disable=C0103
class special_conv2d_bias_activation(conv2d):
    """Special_conv2d_bias_activation.

    This operator equals to conv2d_bias_activation but has improved performance for in_channels < 8.
    """

    def __init__(self, activation, stride, pad, dilate=1, auto_padding=True) -> None:
        """Special_conv2d_bias_activation constructor.

        Parameters
        ----------
        activation : str
            Name of the activation operator
        stride : int
            Stride of the convolution
        pad : int
            Size of padding to add to the input
        dilate : int, optional
            Size of spacing between kernel elements, by default 1
        group : int, optional
            Number of input channels to process to compute one output channel, by default 1
        """
        super().__init__(stride, pad, dilate=dilate)
        self._attrs["op"] = "conv2d_bias_{act}_few_channels".format(act=activation)
        if activation == "relu":
            self._attrs["epilogue"] = "LinearCombinationRelu"
        elif activation == "hardswish":
            self._attrs["epilogue"] = "LinearCombinationHardSwish"
        elif activation == "identity":
            self._attrs["epilogue"] = "LinearCombination"
        else:
            raise NotImplementedError
        self._auto_padding = auto_padding

    def __call__(self, x: Tensor, w: Tensor, b: Tensor):
        """Call special_conv2d_bias_activation with tensors x, w, b.

        Parameters
        ----------
        x : Tensor
            in shape (N, H, W, C_in)
        w : Tensor
            in shape (C_out, K_h, K_w, C_in)
        b : Tensor
            in shape (C_out)

        Returns
        -------
        List[Tensor]
            includes the output tensor in shape (N, H_out, W_out, C_out)
        """
        if self._auto_padding:
            last_dim = x._attrs["shape"][-1]._attrs["values"][0]
            if last_dim in range(1, 4):
                x = nhwc3to4()(x)
            elif last_dim in range(5, 8):
                x = nhwc3to8()(x)
        self._attrs["inputs"] = [x, w, b]
        self._set_depth()
        output_shape = self._infer_shapes(x, w)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._extract_exec_path(x)
        self._extract_epilogue_alignment(output_shape)
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        target_attrs = ["dilate", "pad", "stride"]
        attr = {
            "activation": self._attrs["op"].split("_")[-1],
            "auto_padding": self._auto_padding,
        }

        for target_attr in target_attrs:
            if target_attr in self._attrs:
                attr[target_attr] = self._attrs[target_attr]

        return attr
