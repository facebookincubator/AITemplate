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
Base class for depthwise_conv3d.
"""
import itertools
import re
from collections import OrderedDict
from typing import Any, Dict, List

import jinja2

from .... import backend
from ....backend import registry
from ....utils import shape_utils
from ...base import IntImm, IntVar, Operator, Tensor

# pylint: disable=C0103,W0221,R1732,W0102,W1202,C0301,R1716


SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}TI = {{x_dim1}};
{{indent}}{{dtype}}HI = {{x_dim2}};
{{indent}}{{dtype}}WI = {{x_dim3}};
{{indent}}{{dtype}}CI = {{x_dim4}};
{{indent}}{{dtype}}CO = {{w_dim0}};
{{indent}}{{dtype}}KT = {{w_dim1}};
{{indent}}{{dtype}}KH = {{w_dim2}};
{{indent}}{{dtype}}KW = {{w_dim3}};
{{indent}}{{dtype}}ST = {{stride_t}};
{{indent}}{{dtype}}SH = {{stride_h}};
{{indent}}{{dtype}}SW = {{stride_w}};
{{indent}}{{dtype}}DT = {{dilate_t}};
{{indent}}{{dtype}}DH = {{dilate_h}};
{{indent}}{{dtype}}DW = {{dilate_w}};
{{indent}}{{dtype}}PT = {{pad_t}};
{{indent}}{{dtype}}PH = {{pad_h}};
{{indent}}{{dtype}}PW = {{pad_w}};
{{indent}}{{dtype}}KTEff = (KT - 1) * DT + 1;
{{indent}}{{dtype}}KHEff = (KH - 1) * DH + 1;
{{indent}}{{dtype}}KWEff = (KW - 1) * DW + 1;
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}TO = (TI + PT + PT - KTEff) {{div}} ST + 1;
{{indent}}{{dtype}}HO = (HI + PH + PH - KHEff) {{div}} SH + 1;
{{indent}}{{dtype}}WO = (WI + PW + PW - KWEff) {{div}} SW + 1;
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = NO;
{{indent}}{{y_dim1}} = TO;
{{indent}}{{y_dim2}} = HO;
{{indent}}{{y_dim3}} = WO;
{{indent}}{{y_dim4}} = CO;
"""
)

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
NI == {{x_dim0}} && TI == {{x_dim1}} && HI == {{x_dim2}} && WI == {{x_dim3}} && CI == {{x_dim4}}
"""
)

EXEC_DYN_KEY_TEMPLATE = jinja2.Template(
    """
NI >= {{x_dim0_lb}} && NI <= {{x_dim0_ub}} && TI == {{x_dim1}} && HI == {{x_dim2}} && WI == {{x_dim3}} && CI == {{x_dim4}}
"""
)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class depthwise_conv3d(Operator):
    r"""depthwise_conv3d"""

    def __init__(self, stride, pad, dilate=1, group=1, bias=False) -> None:
        """Conv3d constructor.

        Parameters
        ----------
        stride : int or tuple
            Stride of the convolution
        pad : int or tuple
            Size of padding to add to the input
        dilate : int or tuple, optional
            Size of spacing between kernel elements, by default 1
        group : int, optional
           Number of blocked connections from input
            channels to output channels, by default 1
        """
        super().__init__()
        self._attrs["op"] = "depthwise_conv3d_bias" if bias else "depthwise_conv3d"
        self._attrs["stride"] = stride
        if isinstance(stride, int):
            self._attrs["stride"] = (stride, stride, stride)
        self._attrs["pad"] = pad
        if isinstance(pad, int):
            self._attrs["pad"] = (pad, pad, pad)
        self._attrs["dilate"] = dilate
        if isinstance(dilate, int):
            self._attrs["dilate"] = (dilate, dilate, dilate)
        self._attrs["group"] = group
        self._attrs["has_profiler"] = False
        self._attrs["epilogue_alignment"] = 1
        self._attrs["epilogue"] = "LinearCombination"
        self._attrs["workspace"] = 0
        self._attrs["split_k"] = None
        self._attrs["bias"] = bias
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE
        self.exec_key_template = EXEC_KEY_TEMPLATE
        self.exec_dyn_key_template = EXEC_DYN_KEY_TEMPLATE
        self.exec_cond_template = EXEC_COND_TEMPLATE

    def _infer_shape(self, x: List[int], w: List[int]) -> List[int]:
        if x[4] != w[0] or x[4] != self._attrs["group"]:
            raise RuntimeError("Wrong inputs for depthwise_conv3d")
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            stride_t=self._attrs["stride"][0],
            stride_h=self._attrs["stride"][1],
            stride_w=self._attrs["stride"][2],
            pad_t=self._attrs["pad"][0],
            pad_h=self._attrs["pad"][1],
            pad_w=self._attrs["pad"][2],
            dilate_t=self._attrs["dilate"][0],
            dilate_h=self._attrs["dilate"][1],
            dilate_w=self._attrs["dilate"][2],
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=x[3],
            x_dim4=x[4],
            w_dim0=w[0],
            w_dim1=w[1],
            w_dim2=w[2],
            w_dim3=w[3],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NO"]),
            int(output["TO"]),
            int(output["HO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor, w: Tensor) -> List[int]:
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        w_shape = [var._attrs["values"][0] for var in w._attrs["shape"]]
        self._attrs["CO"] = w_shape[0]
        self._attrs["KT"] = w_shape[1]
        self._attrs["KH"] = w_shape[2]
        self._attrs["KW"] = w_shape[3]
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape, w_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            x._attrs["shape"][0],
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[4] for d in y_shapes])),
        ]
        return output_shape

    def _invert_exec_key(self, key):
        tmp = re.findall(r"(\d+)", key)
        return [int(x) for x in tmp]

    def _gen_exec_key(self, shape: List[int]):
        return self.exec_key_template.render(
            x_dim0=shape[0],
            x_dim1=shape[1],
            x_dim2=shape[2],
            x_dim3=shape[3],
            x_dim4=shape[4],
        ).replace("\n", "")

    def _gen_dyn_exec_key(self, dim0_lb, dim0_ub, dim1, dim2, dim3):
        return self.exec_dyn_key_template.render(
            x_dim0_lb=dim0_lb, x_dim0_ub=dim0_ub, x_dim1=dim1, x_dim2=dim2, x_dim3=dim3
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        self._attrs["exec_path"] = OrderedDict()
        for x_shape in x_shapes:
            key = self._gen_exec_key(x_shape)
            self._attrs["exec_path"][key] = ""

    def _signature(self):
        signature = "depthwise_conv3d: K=[{kt}, {kh}, {kw}], S=[{st}, {sh}, {sw}], P=[{pt}, {ph}, {pw}], CO=[{co}]".format(
            kt=self._attrs["KT"],
            kh=self._attrs["KH"],
            kw=self._attrs["KW"],
            st=self._attrs["stride"][0],
            sh=self._attrs["stride"][1],
            sw=self._attrs["stride"][2],
            pt=self._attrs["pad"][0],
            ph=self._attrs["pad"][1],
            pw=self._attrs["pad"][2],
            co=self._attrs["CO"],
        )
        return signature

    def _extract_epilogue_alignment(self, output_shape: List[IntVar]) -> None:
        epilogue_dim = output_shape[-1]
        if not isinstance(epilogue_dim, IntImm):
            raise RuntimeError("Conv output last dimension must be static!")
        shape = epilogue_dim._attrs["values"][0]
        if shape % 8 == 0:
            self._attrs["epilogue_alignment"] = 8
        elif shape % 4 == 0:
            self._attrs["epilogue_alignment"] = 4
        elif shape % 2 == 0:
            self._attrs["epilogue_alignment"] = 2

    def __call__(self, x: Tensor, w: Tensor, bias: Tensor = None) -> List[Tensor]:
        """Call depthwise_conv3d with tensors x, w

        Parameters
        ----------
        x : Tensor
            in shape (N, T, H, W, C_in)
        w : Tensor
            in shape (C_out, K_t, K_h, K_w, C_in)

        Returns
        -------
        List[Tensor]
            includes the output tensor in shape (N, T_out, H_out, W_out, C_out)
        """
        self._attrs["inputs"] = [x, w]
        if bias:
            self._attrs["inputs"].append(bias)
        self._set_depth()
        output_shape = self._infer_shapes(x, w)
        self._extract_exec_path(x)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self) -> Dict[str, Any]:
        target_attrs = ["dilate", "group", "pad", "stride", "bias"]
        attr = {}

        for target_attr in target_attrs:
            if target_attr in self._attrs:
                attr[target_attr] = self._attrs[target_attr]

        return attr

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
