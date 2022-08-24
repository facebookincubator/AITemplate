# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import enum
from copy import deepcopy
from dataclasses import dataclass
from enum import auto
from typing import List

import jinja2

# import library

from . import library


class Conv2DSpecialization(enum.Enum):
    ConvFwdDefault = auto()
    ConvFwd1x1P0 = auto()
    ConvFwd1x1S1P0 = auto()
    ConvFwdOddC = auto()
    GemmDefault = auto()
    GemmPadding = auto()


Conv2DSpecializationTag = {
    Conv2DSpecialization.ConvFwdDefault: "ck::tensor_operation::device::ConvolutionForwardSpecialization::Default",
    Conv2DSpecialization.ConvFwd1x1P0: "ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0",
    Conv2DSpecialization.ConvFwd1x1S1P0: "ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0",
    Conv2DSpecialization.ConvFwdOddC: "ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC",
    Conv2DSpecialization.GemmDefault: "ck::tensor_operation::device::GemmSpecialization::Default",
    Conv2DSpecialization.GemmPadding: "ck::tensor_operation::device::GemmSpecialization::MNKPadding",
}


class XdlOpType(enum.Enum):
    DeviceConv2d_Xdl_CShuffle = auto()
    DeviceConv2d_Xdl_CShuffle_Bias_Relu = auto()
    DeviceConv2d_Xdl_CShuffle_Bias_Relu_Add = auto()
    DeviceConv2d_Xdl_CShuffle_Bias_Sigmoid = auto()
    DeviceGroupedConv2D_Xdl_CShuffle_Bias_Relu = auto()


XdlOpTag = {
    XdlOpType.DeviceConv2d_Xdl_CShuffle: "ck::tensor_operation::device::DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K",
    XdlOpType.DeviceConv2d_Xdl_CShuffle_Bias_Relu: "ck::tensor_operation::device::DeviceConv2dFwdXdl_C_Shuffle_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K",
    XdlOpType.DeviceConv2d_Xdl_CShuffle_Bias_Relu_Add: "ck::tensor_operation::device::DeviceConv2dFwdXdl_C_Shuffle_Bias_Activation_Add_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K",
    XdlOpType.DeviceConv2d_Xdl_CShuffle_Bias_Sigmoid: "ck::tensor_operation::device::DeviceConv2dFwdXdl_C_Shuffle_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K",
    XdlOpType.DeviceGroupedConv2D_Xdl_CShuffle_Bias_Relu: "ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle",
}


@dataclass
class TileDesc:
    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    k1: int
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
    {{value}}, // {{key}}
{% endfor %}
""",
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(param=args)


@dataclass
class GroupTileDesc:
    default: int
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
    {{value}}, // {{key}}
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
    {% if add_extra_dim == 1 %}true{% else %}false{% endif %}, // add_extra_dim
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
class Conv2DOperation:
    operation_kind: library.Conv2dKind
    extra_kind: library.TensorOperation
    xdl_op_type: XdlOpType
    A: library.TensorDesc
    B: library.TensorDesc
    C: library.TensorDesc
    a_elem_op: library.TensorOperation
    b_elem_op: library.TensorOperation
    epilogue_functor: library.TensorOperation
    c_data_op: library.MemoryDataOperation
    conv2d_specialization: Conv2DSpecialization
    gemm_specialization: Conv2DSpecialization
    tile_desc: TileDesc
    a_block_transfer: BlockTransferDesc
    b_block_transfer: BlockTransferDesc
    c_block_transfer: CBlockTransferDesc

    def __str__(self) -> str:
        io_name = "{conv2d_kind}_{conv2d_specialization}_{gemm_specialization}_{a_dtype}{b_dtype}{c_dtype}_{a_layout}{b_layout}{c_layout}".format(
            conv2d_kind=library.Conv2dKindNames[self.operation_kind],
            conv2d_specialization=self.conv2d_specialization.value,
            gemm_specialization=self.gemm_specialization.value,
            a_dtype=library.ShortDataTypeNames[self.A.element],
            b_dtype=library.ShortDataTypeNames[self.B.element],
            c_dtype=library.ShortDataTypeNames[self.C.element],
            a_layout=library.ShortLayoutTypeNames[self.A.layout],
            b_layout=library.ShortLayoutTypeNames[self.B.layout],
            c_layout=library.ShortLayoutTypeNames[self.C.layout],
        )
        tile_name = str(self.tile_desc)
        return "{io_name}_{tile_name}_{epilogue_functor}".format(
            io_name=io_name,
            tile_name=tile_name,
            epilogue_functor=library.ShortTensorOperationNames[self.epilogue_functor],
        )

    def accumulator_type(self):
        return library.DataType.f32

    def emit(self) -> str:

        template = jinja2.Template(
            """
using {{name}} = {{xdl_op_type}}<
    2, // nd conv
    {{InLayout}}, // InLayout
    {{WeiLayout}}, // WeiLayout
    {% if func=="PT" %}
    ck::Tuple<>,
    {% elif func=="AAR" %}
    ck::Tuple<{{OutLayout}}, {{OutLayout}}>, // BiasLayout
    {% else %}
    ck::Tuple<{{OutLayout}}>, // BiasLayout
    {% endif %}
    {{OutLayout}}, // OutLayout
    {{ADType}}, // InDataType
    {{BDType}}, // WeiDataType
    {{AccDType}}, // AccDataType
    {{CShuffleDType}}, // CShuffleDataType
    {% if func=="PT" %}
    ck::Tuple<>,
    {% elif func=="AAR" %}
    ck::Tuple<{{CDType}}, {{CDType}}>, // BiasLayout
    {% else %}
    ck::Tuple<{{CDType}}>, // BiasDataType
    {% endif %}
    {{CDType}}, // OutDataType

    {{A_elem_op}},
    {{B_elem_op}},
    {{epilogue_functor}},

    {{Conv2DSpecialization}},
    {{GemmSpecialization}},

    {{tile_config}}
    {{a_block_transfer}}
    {{b_block_transfer}}
    {{c_block_transfer}}
    >;
                """
        )
        return template.render(
            name=self.__str__(),
            xdl_op_type=XdlOpTag[self.xdl_op_type],
            InLayout=library.LayoutTag[self.A.layout],
            WeiLayout=library.LayoutTag[self.B.layout],
            OutLayout=library.LayoutTag[self.C.layout],
            ADType=library.DataTypeTag[self.A.element],
            BDType=library.DataTypeTag[self.B.element],
            CDType=library.DataTypeTag[self.C.element],
            AccDType=library.DataTypeTag[library.DataType.f32],
            CShuffleDType=library.DataTypeTag[self.C.element],
            A_elem_op=library.TensorOperationTag[self.a_elem_op],
            B_elem_op=library.TensorOperationTag[self.b_elem_op],
            epilogue_functor=library.TensorOperationTag[self.epilogue_functor],
            C_data_op=library.MemoryDataOperationTag.get(self.c_data_op, -1),
            Conv2DSpecialization=Conv2DSpecializationTag[self.conv2d_specialization],
            GemmSpecialization=Conv2DSpecializationTag[self.gemm_specialization],
            tile_config=self.tile_desc.emit(),
            a_block_transfer=self.a_block_transfer.emit(),
            b_block_transfer=self.b_block_transfer.emit(),
            c_block_transfer=self.c_block_transfer.emit(),
            func=library.ShortTensorOperationNames[self.epilogue_functor],
        )


if __name__ == "__main__":
    A = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    B = library.TensorDesc(library.DataType.f16, library.LayoutType.ColumnMajor)
    C = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    Conv2DOp = Conv2DOperation(
        operation_kind=library.Conv2dKind.Conv2d,
        xdl_op_type=XdlOpType.DeviceConv2d_Xdl_CShuffle,
        A=A,
        B=B,
        C=C,
        a_elem_op=library.TensorOperation.PassThrough,
        b_elem_op=library.TensorOperation.PassThrough,
        epilogue_functor=library.TensorOperation.PassThrough,
        c_data_op="",
        conv2d_specialization=Conv2DSpecialization.ConvFwdDefault,
        tile_desc=TileDesc(256, 256, 128, 4, 8, 32, 32, 4, 2),
        a_block_transfer=BlockTransferDesc(
            [4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1
        ),
        b_block_transfer=BlockTransferDesc(
            [4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1
        ),
        c_block_transfer=CBlockTransferDesc(1, 1, [1, 1, 32, 1, 1, 8], 8),
    )
    print(str(Conv2DOp))
    print(Conv2DOp.emit())
