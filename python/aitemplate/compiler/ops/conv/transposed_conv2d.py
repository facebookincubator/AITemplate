# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] transposed conv2d op
"""
import jinja2

from .conv2d import conv2d

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}HI = {{x_dim1}};
{{indent}}{{dtype}}WI = {{x_dim2}};
{{indent}}{{dtype}}CI = {{x_dim3}};
{{indent}}{{dtype}}CO = {{w_dim0}};
{{indent}}{{dtype}}KH = {{w_dim1}};
{{indent}}{{dtype}}KW = {{w_dim2}};
{{indent}}{{dtype}}SH = {{stride}};
{{indent}}{{dtype}}SW = {{stride}};
{{indent}}{{dtype}}DH = {{dilate}};
{{indent}}{{dtype}}DW = {{dilate}};
{{indent}}{{dtype}}PH = {{pad}};
{{indent}}{{dtype}}PW = {{pad}};
{{indent}}{{dtype}}KHEff = (KH - 1) * DH + 1;
{{indent}}{{dtype}}KWEff = (KW - 1) * DW + 1;
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}HO = (HI - 1) * SH - 2 * PH + KHEff;
{{indent}}{{dtype}}WO = (WI - 1) * SW - 2 * PW + KWEff;
"""
)

# pylint: disable=C0103
class transposed_conv2d(conv2d):
    """[summary]

    Parameters
    ----------
    conv2d : [type]
        [description]
    """

    def __init__(self, stride, pad, dilate=1, group=1) -> None:
        """[summary]

        Parameters
        ----------
        stride : [type]
            [description]
        pad : [type]
            [description]
        dilate : int, optional
            [description], by default 1
        """
        super().__init__(stride, pad, dilate=dilate, group=group)
        self._attrs["op"] = "transposed_conv2d"
        self._attrs["epilogue"] = "LinearCombination"
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
