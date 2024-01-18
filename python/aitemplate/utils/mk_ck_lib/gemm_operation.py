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
import enum
from copy import deepcopy
from dataclasses import dataclass
from enum import auto
from typing import List

import jinja2

from aitemplate.utils.mk_ck_lib import library

# import library


class GemmSpecialization(enum.Enum):
    GemmDefault = auto()
    MNKPadding = auto()
    MNPadding = auto()
    MNOPadding = auto()
    MNKOPadding = auto()


GemmSpecializationTag = {
    GemmSpecialization.GemmDefault: "ck::tensor_operation::device::GemmSpecialization::Default",
    GemmSpecialization.MNKPadding: "ck::tensor_operation::device::GemmSpecialization::MNKPadding",
    GemmSpecialization.MNPadding: "ck::tensor_operation::device::GemmSpecialization::MNPadding",
    GemmSpecialization.MNOPadding: "ck::tensor_operation::device::GemmSpecialization::MNOPadding",
    GemmSpecialization.MNKOPadding: "ck::tensor_operation::device::GemmSpecialization::MNKOPadding",
}


class XdlOpType(enum.Enum):
    DeviceGemmXdl_CShuffle = 1  # TODO: This sucks
    DeviceGemmMultipleD_Xdl_CShuffle = auto()
    DeviceBatchedGemmXdl = auto()
    DeviceBatchedGemmCPermuteXdl = auto()
    DeviceGemmBiasCPermute_Xdl = auto()
    DeviceBatchedContractionMultipleD_Xdl_CShuffle = auto()
    DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle = auto()
    DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle = auto()
    DeviceBatchedGemmMultiD_Xdl = auto()


XdlOpTag = {
    XdlOpType.DeviceGemmXdl_CShuffle: "ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle",
    XdlOpType.DeviceGemmMultipleD_Xdl_CShuffle: "ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle",
    XdlOpType.DeviceBatchedGemmXdl: "ck::tensor_operation::device::DeviceBatchedGemmXdl",
    XdlOpType.DeviceBatchedGemmCPermuteXdl: "ck::tensor_operation::device::DeviceBatchedGemmEPermuteXdl",
    XdlOpType.DeviceGemmBiasCPermute_Xdl: "ck::tensor_operation::device::DeviceGemmBiasEPermute_Xdl",
    XdlOpType.DeviceBatchedContractionMultipleD_Xdl_CShuffle: "ck::tensor_operation::device::DeviceBatchedContractionMultipleD_Xdl_CShuffle",
    XdlOpType.DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle: "ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle",
    XdlOpType.DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle: "ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle",
    XdlOpType.DeviceBatchedGemmMultiD_Xdl: "ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl",
}


@dataclass
class TileDesc:
    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    ak1: int
    bk1: int
    m_per_xdl: int
    n_per_xdl: int
    m_xdl_per_wave: int
    n_xdl_per_wave: int

    def __str__(self) -> str:
        values = list(self.__dict__.values())
        return "_".join([str(x) for x in values])

    def emit(self) -> str:
        args = deepcopy(self.__dict__)
        template = jinja2.Template(
            """
{%for key, value in param.items() %}
{% if value!=0 %}   {{value}}, // {{key}}
    {% endif %}
{% endfor %}
""",
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(param=args)


@dataclass
class AttnTileDesc:
    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    Gemm1NPerBlock: int
    Gemm1KPerBlock: int
    ak1: int
    bk1: int
    b1k1: int
    m_per_xdl: int
    n_per_xdl: int
    m_xdl_per_wave: int
    n_xdl_per_wave: int
    Gemm1NXdlPerWave: int

    def __str__(self) -> str:
        values = list(self.__dict__.values())
        return "_".join([str(x) for x in values])

    def emit(self) -> str:
        args = deepcopy(self.__dict__)
        template = jinja2.Template(
            """
{%for key, value in param.items() %}
{% if value!=0 %}   {{value}}, // {{key}}
    {% endif %}
{% endfor %}
""",
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(param=args)


@dataclass
class BlockTransferDesc:
    thread_cluster_length: List[int]
    thread_cluster_arrange_order: List[int]
    src_access_order: List[int]
    src_vector_dim: int
    src_scalar_per_vector: int
    dst_scalar_per_vector: int
    add_extra_dim: int
    add_extra_dim_flag: bool = False

    def __str__(self) -> str:
        args = deepcopy(self.__dict__)
        args["thread_cluster_length"] = [str(x) for x in self.thread_cluster_length]
        args["thread_cluster_arrange_order"] = [
            str(x) for x in self.thread_cluster_arrange_order
        ]
        args["src_access_order"] = [str(x) for x in self.src_access_order]

        template = jinja2.Template(
            """S{{thread_cluster_length|join('_')}}S
            _S{{thread_cluster_arrange_order|join('_')}}S
            _S{{src_access_order|join('_')}}S
            _{{src_vector_dim}}
            _{{src_scalar_per_vector}}
            _{{dst_scalar_per_vector}}
            _{{add_extra_dim}}
            """,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(**args).replace("\n", "").replace(" ", "")

    def emit(self) -> str:
        args = deepcopy(self.__dict__)
        args["thread_cluster_length"] = [str(x) for x in self.thread_cluster_length]
        args["thread_cluster_arrange_order"] = [
            str(x) for x in self.thread_cluster_arrange_order
        ]
        args["src_access_order"] = [str(x) for x in self.src_access_order]
        template = jinja2.Template(
            """
    ck::Sequence<{{thread_cluster_length|join(',')}}>, // thread_cluster_length
    ck::Sequence<{{thread_cluster_arrange_order|join(',')}}>, // thread_cluster_arrange_order
    ck::Sequence<{{src_access_order|join(',')}}>, // src_access_order
    {{src_vector_dim}}, // src_vector_dim
    {{src_scalar_per_vector}}, // src_scalar_per_vector
    {{dst_scalar_per_vector}}, // dst_scalar_per_vector
{% if add_extra_dim_flag %}
    {% if add_extra_dim==1 %}true, {% else %}false,{% endif %} //add_extra_dim
{% else %}
    {{add_extra_dim}}, // add_extra_dim
{% endif %}
"""
        )
        return template.render(**args)


@dataclass
class CBlockTransferDesc:
    m_xdl_per_wave: int
    n_xdl_per_wave: int
    m_n_block_wave_per_xdl: List[int]
    scalar_per_vector: int

    def __str__(self) -> str:
        args = deepcopy(self.__dict__)
        args["m_n_block_wave_per_xdl"] = [str(x) for x in self.m_n_block_wave_per_xdl]
        template = jinja2.Template(
            """
        {{m_xdl_per_wave}}
        _{{n_xdl_per_wave}}
        {{m_n_block_wave_per_xdl|join('_')}}S
        {{scalar_per_vector}}
        """,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(**args).replace("\n", "").replace(" ", "")

    def emit(self) -> str:
        args = deepcopy(self.__dict__)
        args["m_n_block_wave_per_xdl"] = [str(x) for x in self.m_n_block_wave_per_xdl]

        template = jinja2.Template(
            """
    {{m_xdl_per_wave}}, // m_xdl_per_wave
    {{n_xdl_per_wave}}, // n_xdl_per_wave
    ck::Sequence<{{m_n_block_wave_per_xdl|join(',')}}>, // m_n_block_wave_per_xdl
    {{scalar_per_vector}} // scalar_per_vector
    """,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(**args)


@dataclass
class MaskedCBlockTransferDesc:
    m_xdl_per_wave: int
    n_xdl_per_wave: int
    m_n_block_wave_per_xdl: List[int]
    scalar_per_vector: int
    causal_mask: int

    def __str__(self) -> str:
        args = deepcopy(self.__dict__)
        args["m_n_block_wave_per_xdl"] = [str(x) for x in self.m_n_block_wave_per_xdl]
        template = jinja2.Template(
            """
        {{m_xdl_per_wave}}
        _{{n_xdl_per_wave}}
        {{m_n_block_wave_per_xdl|join('_')}}S
        {{scalar_per_vector}}
        {% if causal_mask == 1 %}
        ck::tensor_operation::device::MaskingSpecialization::MaskOutUpperTriangle // causal_mask
        {% else %}
        ck::tensor_operation::device::MaskingSpecialization::MaskDisabled // causal_mask
        {% endif %}
        """,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(**args).replace("\n", "").replace(" ", "")

    def emit(self) -> str:
        args = deepcopy(self.__dict__)
        args["m_n_block_wave_per_xdl"] = [str(x) for x in self.m_n_block_wave_per_xdl]

        template = jinja2.Template(
            """
    {{m_xdl_per_wave}}, // m_xdl_per_wave
    {{n_xdl_per_wave}}, // n_xdl_per_wave
    ck::Sequence<{{m_n_block_wave_per_xdl|join(',')}}>, // m_n_block_wave_per_xdl
    {{scalar_per_vector}}, // scalar_per_vector
    {% if causal_mask == 1 %}
    ck::tensor_operation::device::MaskingSpecialization::MaskOutUpperTriangle // causal_mask
    {% else %}
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled // causal_mask
    {% endif %}
    """,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(**args)


@dataclass
class GemmOperation:
    operation_kind: library.OperationKind
    extra_kind: library.TensorOperation
    xdl_op_type: XdlOpType
    A: library.TensorDesc
    B: library.TensorDesc
    C: library.TensorDesc
    a_elem_op: library.TensorOperation
    b_elem_op: library.TensorOperation
    epilogue_functor: library.TensorOperation
    gemm_specialization: GemmSpecialization
    tile_desc: TileDesc
    a_block_transfer: BlockTransferDesc
    b_block_transfer: BlockTransferDesc
    b1_block_transfer: BlockTransferDesc = None
    c_block_transfer: CBlockTransferDesc = None
    ds_dtype: List[library.DataType] = None
    ds_layout: List[library.LayoutType] = None
    e_dtype: library.DataType = None
    loop_scheduler: str = ""
    pipeline: str = ""

    def __str__(self) -> str:
        io_name = "{gemm_kind}_{gemm_specialization}_{a_dtype}{b_dtype}{c_dtype}_{a_layout}{b_layout}{c_layout}".format(
            gemm_kind=library.GemmKindNames[self.operation_kind],
            gemm_specialization=self.gemm_specialization.value,
            a_dtype=library.ShortDataTypeNames[self.A.element],
            b_dtype=library.ShortDataTypeNames[self.B.element],
            c_dtype=library.ShortDataTypeNames[self.C.element],
            a_layout=library.ShortLayoutTypeNames[self.A.layout],
            b_layout=library.ShortLayoutTypeNames[self.B.layout],
            c_layout=library.ShortLayoutTypeNames[self.C.layout],
        )
        extra_tile = ""
        if self.c_block_transfer is not None:
            if self.c_block_transfer.scalar_per_vector == 4:
                extra_tile = "_C4"
            elif self.c_block_transfer.scalar_per_vector == 1:
                extra_tile = "_C1"

        tile_name = str(self.tile_desc) + extra_tile
        extra_name = (
            "_CM" if library.ShortTensorOperationNames[self.extra_kind] == "CM" else ""
        )
        return "{io_name}_{tile_name}_{epilogue_functor}_{scheduler}_{pipeline}".format(
            io_name=io_name,
            tile_name=tile_name,
            epilogue_functor=library.ShortTensorOperationNames[self.epilogue_functor]
            + extra_name,
            scheduler=library.ShortSchedulerNames.get(self.loop_scheduler, "default"),
            pipeline=library.ShortPipelineNames.get(self.pipeline, "v1"),
        )

    def accumulator_type(self):
        return library.DataType.f32

    def emit(self) -> str:
        template = jinja2.Template(
            """
using {{name}} = {{xdl_op_type}}<
{% if xdl_op_type_value==1 %}
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{CDType}},
    {{AccDType}},
    {{CShuffleDType}},
{% elif xdl_op_type_value==2 %}
    {{ALayout}},
    {{BLayout}},
    ck::Tuple<{{DsLayout}}>, // DsLayout
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{AccDType}},
    {{CShuffleDType}},
    ck::Tuple<{{DsDType}}>, // DsType
    {{CDType}},
{% elif xdl_op_type_value==3 %}
    {{ADType}},
    {{BDType}},
    {{CDType}},
    {{AccDType}},
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
{% elif xdl_op_type_value==4 %}
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{AccDType}},
    {{CShuffleDType}},
    {{CDType}},
{% elif xdl_op_type_value==5 %}
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{AccDType}},
    float, // CShuffleDType
    ck::half_t,
    ck::half_t,
{% elif xdl_op_type_value==6 %}
    {% if gemm_kind == "gemm_permute_m2n3" %}
    1, 2, 3, 1, // permute m2n3
    {% elif gemm_kind == "gemm_permute_m3n2" %}
    1, 3, 2, 1, // permute m3n2
    {% endif %}
    {{ADType}},
    {{BDType}},
    {{AccDType}},
    ck::half_t,
    {% if "PassThrough" in C_elem_op %}
    ck::Tuple<>,
    {% else %}
    ck::Tuple<ck::half_t>,
    {% endif %}
    ck::half_t,
{% elif xdl_op_type_value == 7 %}
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{BDType}},
    {{CDType}},
    {{AccDType}},
    float, // CShuffleDType,
{% elif xdl_op_type_value == 8 %}
    2, 1, 1, 1, 1,
    {{ADType}},
    {{BDType}},
    {{BDType}},
    {{CDType}},
    ck::Tuple<>,
    ck::Tuple<>,
    {{AccDType}},
    float, // CShuffleDType,
{% elif xdl_op_type_value == 9 %}
    {{ALayout}},
    {{BLayout}},
    ck::Tuple<{{DsLayout}}>, // DsLayout
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{AccDType}},
    {{CShuffleDType}},
    ck::Tuple<{{DsDType}}>, // DsType
    {{EDType}},
{% endif %}
{% if xdl_op_type_value in [7, 8] %}
    {{A_elem_op}},
    {{B_elem_op}},
    ck::tensor_operation::element_wise::ScaleAndResetNaNToMinusInfinity,
{% else %}
    {{A_elem_op}},
{% endif %}
    {{B_elem_op}},
    {{C_elem_op}},
{% if xdl_op_type_value!=3 %}
    {{GemmSpecialization}},
    {% if xdl_op_type_value==6 %}
    ck::tensor_operation::device::TensorSpecialization::Packed,
    ck::tensor_operation::device::TensorSpecialization::Packed,
    ck::tensor_operation::device::TensorSpecialization::Default,
    {% elif xdl_op_type_value==8 %}
    ck::tensor_operation::device::TensorSpecialization::Default,
    ck::tensor_operation::device::TensorSpecialization::Default,
    ck::tensor_operation::device::TensorSpecialization::Default,
    ck::tensor_operation::device::TensorSpecialization::Default,
    {% endif %}
    1,
{% endif %}
    {{tile_config}}
    {{a_block_transfer}}
    {{b_block_transfer}}
{% if xdl_op_type_value in [7, 8] %}
    {{b1_block_transfer}}
{% endif %}
{% if xdl_op_type_value!=3 %}
    {{c_block_transfer}}
{% else %}
    7, // src_dst_vector_dim
    1 // dst_scalar_per_vector
{% endif %}
{% if LoopScheduler %}
    ,{{LoopScheduler}}
{% endif %}
{% if Pipeline %}
    ,{{Pipeline}}
{% endif %}
    >;
"""
        )
        return template.render(
            name=self.__str__(),
            gemm_kind=library.GemmKindNames[self.operation_kind],
            xdl_op_type=XdlOpTag[self.xdl_op_type],
            xdl_op_type_value=self.xdl_op_type.value,  # This sucks
            ALayout=library.LayoutTag[self.A.layout],
            BLayout=library.LayoutTag[self.B.layout],
            CLayout=library.LayoutTag[self.C.layout],
            ADType=library.DataTypeTag[self.A.element],
            BDType=library.DataTypeTag[self.B.element],
            CDType=library.DataTypeTag[self.C.element],
            AccDType=library.DataTypeTag[library.DataType.f32],
            CShuffleDType=library.DataTypeTag[self.C.element],
            A_elem_op=library.TensorOperationTag[self.a_elem_op],
            B_elem_op=library.TensorOperationTag[self.b_elem_op],
            C_elem_op=library.TensorOperationTag[self.epilogue_functor],
            GemmSpecialization=GemmSpecializationTag[self.gemm_specialization],
            epilogue_func=library.ShortTensorOperationNames[self.epilogue_functor],
            tile_config=self.tile_desc.emit(),
            a_block_transfer=self.a_block_transfer.emit(),
            b_block_transfer=self.b_block_transfer.emit(),
            b1_block_transfer=""
            if self.b1_block_transfer is None
            else self.b1_block_transfer.emit(),
            c_block_transfer=self.c_block_transfer.emit()
            if self.c_block_transfer is not None
            else "",
            DsDType=",".join(
                [library.DataTypeTag[d_dtype] for d_dtype in self.ds_dtype]
            )
            if self.ds_dtype is not None
            else "",
            DsLayout=",".join(
                [library.LayoutTag[d_layout] for d_layout in self.ds_layout]
            )
            if self.ds_layout is not None
            else "",
            EDType=library.DataTypeTag[self.e_dtype]
            if self.e_dtype is not None
            else "",
            LoopScheduler=self.loop_scheduler,
            Pipeline=self.pipeline
        )


if __name__ == "__main__":
    A = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    B = library.TensorDesc(library.DataType.f16, library.LayoutType.ColumnMajor)
    C = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    GemmOp = GemmOperation(
        operation_kind=library.GemmKind.BatchGemmPermute,
        extra_kind=library.TensorOperation.PassThrough,
        xdl_op_type=XdlOpType.DeviceBatchedGemmCPermuteXdl,
        A=A,
        B=B,
        C=C,
        a_elem_op=library.TensorOperation.PassThrough,
        b_elem_op=library.TensorOperation.PassThrough,
        epilogue_functor=library.TensorOperation.PassThrough,
        gemm_specialization=GemmSpecialization.GemmDefault,
        tile_desc=TileDesc(256, 256, 128, 32, 8, 2, 32, 32, 4, 2),
        a_block_transfer=BlockTransferDesc(
            [4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1, True
        ),
        b_block_transfer=BlockTransferDesc(
            [8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 1, 0, True
        ),
        # c_block_transfer=CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8),
        ds_dtype=[library.DataType.f16],
    )
    print(str(GemmOp))
    print(GemmOp.emit())
