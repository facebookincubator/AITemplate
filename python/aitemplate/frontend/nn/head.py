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

from typing import Callable

from aitemplate.compiler import ops

from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.dropout import Dropout
from aitemplate.frontend.nn.linear import Linear
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.softmax import Softmax


class SequencePool(Module):
    """
    Sequence pool produces a single embedding from a sequence of embeddings. Currently
    it supports "mean" and "cls".

    """

    def __init__(self, mode: str) -> None:
        """
        Args:
            mode (str): Optionals include "cls" and "mean". If set to "cls", it assumes
                the first element in the input is the cls token and returns it. If set
                to "mean", it returns the mean of the entire sequence.
        """
        super().__init__()
        assert mode in ["mean"], "Unsupported mode for SequencePool."
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        # TODO: Add support for cls mode.
        # if self.mode == "cls":
        #     x = x[:, 0]
        if self.mode == "mean":
            x = ops.reduce_mean(1)(x)
        else:
            raise NotImplementedError
        return x


class VisionTransformerBasicHead(Module):
    """
    Vision transformer basic head.

    ::

                                      SequencePool
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation


    The builder can be found in `create_vit_basic_head`.
    """

    def __init__(
        self,
        sequence_pool: Module = None,
        dropout: Module = None,
        proj: Module = None,
        activation: Module = None,
    ) -> None:
        """
        Args:
            sequence_pool (torch.nn.modules): pooling module.
            dropout(torch.nn.modules): dropout module.
            proj (torch.nn.modules): project module.
            activation (torch.nn.modules): activation module.
        """
        super().__init__()
        self.sequence_pool = sequence_pool
        self.dropout = dropout
        self.proj = proj
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        # Performs pooling.
        if self.sequence_pool is not None:
            x = self.sequence_pool(x)

        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)
        # Performs projection.
        if self.proj is not None:
            x = self.proj(x)
        # Performs activation.
        if self.activation is not None:
            x = self.activation(x)
        return x


def create_vit_basic_head(
    *,
    # Projection configs.
    in_features: int,
    out_features: int,
    # Pooling configs.
    seq_pool_type: str = "cls",
    # Dropout configs.
    dropout_rate: float = 0.5,
    # Activation configs.
    activation: Callable = None,
) -> Module:
    """
    Creates vision transformer basic head.

    ::


                                        Pooling
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation


    Activation examples include: ReLU, Softmax, Sigmoid, and None.
    Pool type examples include: cls, mean and none.

    Args:

        in_features: input channel size of the resnet head.
        out_features: output channel size of the resnet head.

        pool_type (str): Pooling type. It supports "cls", "mean " and "none". If set to
            "cls", it assumes the first element in the input is the cls token and
            returns it. If set to "mean", it returns the mean of the entire sequence.

        activation (callable): a callable that constructs vision transformer head
            activation layer, examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and
            None (not applying activation).

        dropout_rate (float): dropout rate.
    """
    assert seq_pool_type in ["cls", "mean", "none"]

    if seq_pool_type in ["cls", "mean"]:
        seq_pool_model = SequencePool(seq_pool_type)
    elif seq_pool_type == "none":
        seq_pool_model = None
    else:
        raise NotImplementedError

    if activation is None:
        activation_model = None
    elif activation == Softmax:
        activation_model = activation(dim=1)
    else:
        activation_model = activation()

    return VisionTransformerBasicHead(
        sequence_pool=seq_pool_model,
        dropout=Dropout(dropout_rate) if dropout_rate > 0.0 else None,
        proj=Linear(in_features, out_features),
        activation=activation_model,
    )
