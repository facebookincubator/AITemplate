# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from ...compiler import ops
from .module import Module
from .parameter import Parameter

# pylint: disable=C0103


class Linear(Module):
    """[summary]

    Parameters
    ----------
    Block : [type]
        [description]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        activation=None,
        dtype="float16",
    ):
        """[summary]

        Parameters
        ----------
        in_channel : [type]
            [description]
        out_channel : [type]
            [description]
        bias : [type]
            [description]
        activation : [type]
            [description]
        weight_transpose : [type]
            [description]
        dtype : str, optional
            [description], by default "float16"

        Raises
        ------
        NotImplementedError
            [description]
        """
        super().__init__()
        self.W = Parameter(shape=[out_channels, in_channels], dtype=dtype)
        op_name = "gemm_rcr"
        if bias:
            self.B = Parameter(shape=[out_channels], dtype=dtype)
            if activation == "relu":
                op_name = "gemm_rcr_bias_relu"
            elif activation == "softmax":
                op_name = "gemm_rcr_bias_softmax"
            elif activation == "add_relu":
                op_name = "gemm_rcr_bias_add_relu"
            elif activation == "sigmoid":
                op_name = "gemm_rcr_bias_sigmoid"
            elif activation == "gelu":
                op_name = "gemm_rcr_bias_fast_gelu"
            else:
                op_name = "gemm_rcr_bias"
        op_func = getattr(ops, op_name)
        self._op_name = op_name
        self.op = op_func()
        self.bias = bias
        self.in_channels = in_channels

    def forward(self, *args):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """

        assert len(args) >= 1
        # print(self._op_name)
        data = args[0]
        shape = data._attrs["shape"]
        x = data if len(shape) == 2 else ops.reshape()(data, [-1, self.in_channels])
        if len(args) == 2:

            output = self.op(x, self.W.tensor(), self.B.tensor(), args[1])
            return output
        output = (
            self.op(x, self.W.tensor(), bias=self.B.tensor())
            if self.bias
            else self.op(x, self.W.tensor())
        )
        return output
