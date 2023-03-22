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
The front-end definition of the padded_dense_to_jagged op.
"""
from typing import List

from aitemplate.backend import registry
from aitemplate.backend.target import Target
from aitemplate.compiler.base import IntVar, JaggedDim, Operator, Tensor
from aitemplate.compiler.ops import make_jagged


class padded_dense_to_jagged(Operator):
    """
    Returns a jagged Tensor "extracted" from the input dense Tensor,
    given the offsets list. The resulting jagged Tensor contains the
    subset of values of the input dense Tensor specified by the rank-1
    offset Tensors in the offsets_list.

    Args:
        x (Tensor): input dense tensor.
        offsets_list (List[Tensor]): the list of offsets of the resulting jagged Tensor.
        total_length (IntVar): the total length dimension of the resulting jagged Tensor.
    Returns:
        y (Tensor): a jagged Tensor extracted from the input dense Tensor x.
    """

    def __init__(
        self,
        total_length: IntVar,
    ):
        if type(total_length) != IntVar:
            raise TypeError(
                f"total_length must be IntVar, but got {type(total_length).__name__}."
            )

        super().__init__()
        self._attrs["op"] = "padded_dense_to_jagged"
        self._attrs["total_length"] = total_length

    def _infer_shape(
        self,
        x: Tensor,
        offsets_list: List[Tensor],
    ) -> List[IntVar]:
        inner_shape = x.shape()[1 + len(offsets_list) :]
        return [self._attrs["total_length"]] + inner_shape

    def _get_op_attributes(self):
        return {
            "total_length": self._attrs["total_length"],
        }

    def _args_for_pseudo_code(self):
        return [f"total_length={self._attrs['total_length']}"]

    def __call__(
        self,
        x: Tensor,
        offsets_list: List[Tensor],
    ) -> Tensor:
        x_shape = x.shape()
        if not offsets_list:
            raise ValueError("At least one offsets Tensor must be specified.")
        if len(x_shape) < len(offsets_list) + 2:
            raise ValueError(
                "The input dense Tensor x must have at least len(offsets_list) + 2 dimensions: "
                "one batch dimension, as many sequence dimensions as len(offsets_list), and "
                f"at least one inner dimension, but {len(offsets_list)=}, {x_shape=}."
            )
        if type(x_shape[0]) != IntVar:
            raise TypeError(
                f"x.shape()[0] must be IntVar, but got {type(x_shape[0]).__name__}."
            )

        self._attrs["inputs"] = [x, *offsets_list]
        self._set_depth()
        output_shape = self._infer_shape(x, offsets_list)

        # the source Tensor of the resulting jagged Tensor
        source = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [source]

        # in the AIT graph, the output of the padded_dense_to_jagged op is set to
        # the source Tensor, which is still not a jagged Tensor. The source Tensor
        # is passed through the make_jagged op to obtain the jagged Tensor returned
        # from the __call__: this way, the chain of ops in the graph looks like:
        #
        #      x --> padded_dense_to_jagged --> source --> make_jagged --> y
        #                    \--------- offsets_list ----------/

        # the resulting jagged Tensor
        jagged_output = make_jagged(
            batch_dim=x_shape[0],
            jagged_dims=[
                JaggedDim(min_value=0, max_value=dim)
                for dim in x_shape[1 : 1 + len(offsets_list)]
            ],
        )(
            source=source,
            offsets_list=offsets_list,
        )

        # we keep the resulting jagged Tensor's JaggedIntVar around,
        # as we'll need it for the back-end code generation of the
        # padded_dense_to_jagged op
        self._attrs["jagged_int_var"] = jagged_output._attrs["shape"][0]

        return jagged_output

    def gen_function(self) -> str:
        target = Target.current()
        func = registry.get(f"{target.name()}.{self._attrs['op']}.gen_function")
        return func(self._attrs)
