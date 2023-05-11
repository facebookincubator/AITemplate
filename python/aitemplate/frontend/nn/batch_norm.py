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
Frontend for attention module
"""
from aitemplate.compiler.public import elementwise, FuncEnum, permute
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.parameter import Parameter


class _BatchNorm(Module):
    """BatchNorm nn module"""

    def __init__(
        self,
        num_features,
        eps=1e-5,
        dtype="float16",
        permute_input_output=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = (num_features,)
        self.dtype = dtype
        self.num_features = num_features
        self.permute_input_output = permute_input_output
        self.eps = eps
        self.weight = Parameter(shape=self.dim, dtype=dtype)
        self.bias = Parameter(shape=self.dim, dtype=dtype)
        self.running_mean = Parameter(shape=self.dim, dtype=dtype)
        self.running_var = Parameter(shape=self.dim, dtype=dtype)
        # Placeholder for setting constants, won't be used
        self.num_batches_tracked = Parameter(shape=[], value=0, dtype=dtype)

    def forward(self, *args):
        assert len(args) == 1
        x = args[0]
        self._check_input_dim(x)
        x = self._convert_input(x) if self.permute_input_output else x

        x_normalized = elementwise(FuncEnum.DIV)(
            elementwise(FuncEnum.SUB)(x, self.running_mean.tensor()),
            elementwise(FuncEnum.SQRT)(
                elementwise(FuncEnum.ADD)(self.running_var.tensor(), self.eps)
            ),
        )

        y = elementwise(FuncEnum.ADD)(
            elementwise(FuncEnum.MUL)(self.weight.tensor(), x_normalized),
            self.bias.tensor(),
        )

        y = self._convert_output(y) if self.permute_input_output else y
        return y

    def _check_input_dim(self):
        raise NotImplementedError()

    def _convert_input(self):
        raise NotImplementedError()

    def _convert_output(self):
        raise NotImplementedError()


class BatchNorm1d(_BatchNorm):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        dtype="float16",
        permute_input_output=False,
        **kwargs,
    ):
        super().__init__(num_features, eps, dtype, permute_input_output, **kwargs)

    def _check_input_dim(self, x):
        if len(x.shape()) != 2 and len(x.shape()) != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(x.shape())
            )

    def _convert_input(self, x):
        if len(x.shape()) == 3:
            return permute()(x, [0, 2, 1])
        else:
            return x

    def _convert_output(self, y):
        if len(y.shape()) == 3:
            return permute()(y, [0, 2, 1])
        else:
            return y


class BatchNorm2d(_BatchNorm):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        dtype="float16",
        permute_input_output=False,
        **kwargs,
    ):
        super().__init__(num_features, eps, dtype, permute_input_output, **kwargs)

    def _check_input_dim(self, x):
        if len(x.shape()) != 4:
            raise ValueError("expected 4D input (got {}D input)".format(x.shape()))

    def _convert_input(self, x):
        return permute()(x, [0, 2, 3, 1])

    def _convert_output(self, y):
        return permute()(y, [0, 3, 1, 2])


class BatchNorm3d(_BatchNorm):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        dtype="float16",
        permute_input_output=False,
        **kwargs,
    ):
        super().__init__(num_features, eps, dtype, permute_input_output, **kwargs)

    def _check_input_dim(self, x):
        if len(x.shape()) != 5:
            raise ValueError("expected 5D input (got {}D input)".format(x.shape()))

    def _convert_input(self, x):
        return permute()(x, [0, 2, 3, 4, 1])

    def _convert_output(self, y):
        return permute()(y, [0, 4, 1, 2, 3])
