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
from copy import deepcopy
from dataclasses import dataclass

import jinja2

# import library

from aitemplate.utils.mk_ck_lib import library


@dataclass
class TileDesc:
    block_size: int
    m_cluster_size: int
    k_cluster_size: int
    m_slice_size: int
    k_slice_size: int
    in_src_dim: int
    in_src_size: int
    out_dst_size: int

    def __str__(self) -> str:
        values = list(self.__dict__.values())
        return "_".join([str(x) for x in values])

    def emit(self) -> str:
        args = deepcopy(self.__dict__)
        template = jinja2.Template(
            """
{%for key, value in param.items() %}
    {{value}}{% if not loop.last %},{% endif %} // {{key}}
{% endfor %}
""",
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(param=args)


@dataclass
class SoftmaxOperation:
    operation_kind: library.OperationKind
    extra_kind: int
    In: library.DataType
    Out: library.DataType
    Rank: int
    NumReduceDim: int
    tile_desc: TileDesc

    def __str__(self) -> str:
        return "{op_kind}_rank{rank}_{tile_name}".format(
            op_kind=library.OperationKindNames[self.operation_kind],
            rank=self.Rank,
            tile_name=str(self.tile_desc),
        )

    def accumulator_type(self):
        return library.DataType.f32

    def emit(self) -> str:
        template = jinja2.Template(
            """
using {{name}} = ck::tensor_operation::device::DeviceSoftmaxImpl<
    {{InDType}},
    {{AccDType}},
    {{OutDType}},
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    {{Rank}},
    {{NumReduceDim}},
    {{tile_config}}
    >;
"""
        )
        return template.render(
            name=self.__str__(),
            InDType=library.DataTypeTag[self.In],
            AccDType=library.DataTypeTag[library.DataType.f32],
            OutDType=library.DataTypeTag[self.Out],
            Rank=self.Rank,
            NumReduceDim=self.NumReduceDim,  # we only need softmax(dim=-1) at this moment
            tile_config=self.tile_desc.emit(),
        )


if __name__ == "__main__":
    A = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    B = library.TensorDesc(library.DataType.f16, library.LayoutType.ColumnMajor)
    C = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    SoftmaxOp = SoftmaxOperation(
        operation_kind=library.OperationKind.Softmax,
        extra_kind=3,
        In=library.DataType.f16,
        Out=library.DataType.f16,
        Rank=3,
        tile_desc=TileDesc(256, 8, 32, 1, 8, 1, 1, 1),
    )
    print(str(SoftmaxOp))
    print(SoftmaxOp.emit())
