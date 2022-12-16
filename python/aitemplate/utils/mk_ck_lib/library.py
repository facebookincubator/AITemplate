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
from dataclasses import dataclass
from enum import auto


class DataType(enum.Enum):
    b1 = auto()
    u4 = auto()
    u8 = auto()
    u16 = auto()
    u32 = auto()
    u64 = auto()
    s4 = auto()
    s8 = auto()
    s16 = auto()
    s32 = auto()
    s64 = auto()
    f16 = auto()
    bf16 = auto()
    f32 = auto()
    f64 = auto()
    invalid = auto()


ShortDataTypeNames = {
    DataType.s32: "i",
    DataType.f16: "h",
    DataType.f32: "s",
    DataType.f64: "d",
}

#
DataTypeNames = {
    DataType.b1: "b1",
    DataType.u4: "u4",
    DataType.u8: "u8",
    DataType.u16: "u16",
    DataType.u32: "u32",
    DataType.u64: "u64",
    DataType.s4: "s4",
    DataType.s8: "s8",
    DataType.s16: "s16",
    DataType.s32: "s32",
    DataType.s64: "s64",
    DataType.f16: "f16",
    DataType.bf16: "bf16",
    DataType.f32: "f32",
    DataType.f64: "f64",
}

DataTypeTag = {
    DataType.u8: "uint8_t",
    DataType.u16: "uint16_t",
    DataType.u32: "uint32_t",
    DataType.u64: "uint64_t",
    DataType.s8: "int8_t",
    DataType.s16: "int16_t",
    DataType.s32: "int32_t",
    DataType.s64: "int64_t",
    DataType.f16: "ck::half_t",
    DataType.bf16: "ck::bhalf_t",
    DataType.f32: "float",
    DataType.f64: "double",
}

DataTypeSize = {
    DataType.b1: 1,
    DataType.u4: 4,
    DataType.u8: 8,
    DataType.u16: 16,
    DataType.u32: 32,
    DataType.u64: 64,
    DataType.s4: 4,
    DataType.s8: 8,
    DataType.s16: 16,
    DataType.s32: 32,
    DataType.s64: 64,
    DataType.f16: 16,
    DataType.bf16: 16,
    DataType.f32: 32,
    DataType.f64: 64,
}


class LayoutType(enum.Enum):
    ColumnMajor = auto()
    RowMajor = auto()
    NWC = auto()
    KXC = auto()
    NWK = auto()
    NCW = auto()
    KCX = auto()
    NKW = auto()
    NHWC = auto()
    KYXC = auto()
    NHWK = auto()
    NCHW = auto()
    KCYX = auto()
    NKWH = auto()
    NDHWC = auto()
    KZYXC = auto()
    NDHWK = auto()
    NCDHW = auto()
    KCZYX = auto()
    NKDHW = auto()
    G_NHW_C = auto()
    G_K_YX_C = auto()
    G_NHW_K = auto()
    NHWGC = auto()
    KYXGC = auto()
    NHWGK = auto()
    GNHWC = auto()
    GKYXC = auto()
    GNHWK = auto()
    GNWC = auto()
    GKXC = auto()
    GNWK = auto()


LayoutTag = {
    LayoutType.ColumnMajor: "ck::tensor_layout::gemm::ColumnMajor",
    LayoutType.RowMajor: "ck::tensor_layout::gemm::RowMajor",
    LayoutType.NWC: "ck::tensor_layout::convolution::NWC",
    LayoutType.KXC: "ck::tensor_layout::convolution::KXC",
    LayoutType.NWK: "ck::tensor_layout::convolution::NWK",
    LayoutType.NCW: "ck::tensor_layout::convolution::NCW",
    LayoutType.KCX: "ck::tensor_layout::convolution::KCX",
    LayoutType.NKW: "ck::tensor_layout::convolution::NKW",
    LayoutType.NHWC: "ck::tensor_layout::convolution::NHWC",
    LayoutType.KYXC: "ck::tensor_layout::convolution::KYXC",
    LayoutType.NHWK: "ck::tensor_layout::convolution::NHWK",
    LayoutType.NCHW: "ck::tensor_layout::convolution::NCHW",
    LayoutType.KCYX: "ck::tensor_layout::convolution::KCYX",
    LayoutType.NKWH: "ck::tensor_layout::convolution::NKWH",
    LayoutType.NDHWC: "ck::tensor_layout::convolution::NDHWC",
    LayoutType.KZYXC: "ck::tensor_layout::convolution::KZYXC",
    LayoutType.NDHWK: "ck::tensor_layout::convolution::NDHWK",
    LayoutType.NCDHW: "ck::tensor_layout::convolution::NCDHW",
    LayoutType.KCZYX: "ck::tensor_layout::convolution::KCZYX",
    LayoutType.NKDHW: "ck::tensor_layout::convolution::NKDHW",
    LayoutType.G_NHW_C: "ck::tensor_layout::convolution::G_NHW_C",
    LayoutType.G_K_YX_C: "ck::tensor_layout::convolution::G_K_YX_C",
    LayoutType.G_NHW_K: "ck::tensor_layout::convolution::G_NHW_K",
    LayoutType.NHWGC: "ck::tensor_layout::convolution::NHWGC",
    LayoutType.KYXGC: "ck::tensor_layout::convolution::KYXGC",
    LayoutType.NHWGK: "ck::tensor_layout::convolution::NHWGK",
    LayoutType.GNHWC: "ck::tensor_layout::convolution::GNHWC",
    LayoutType.GKYXC: "ck::tensor_layout::convolution::GKYXC",
    LayoutType.GNHWK: "ck::tensor_layout::convolution::GNHWK",
    LayoutType.GNWC: "ck::tensor_layout::convolution::GNWC",
    LayoutType.GKXC: "ck::tensor_layout::convolution::GKXC",
    LayoutType.GNWK: "ck::tensor_layout::convolution::GNWK",
}

ShortLayoutTypeNames = {
    LayoutType.ColumnMajor: "N",
    LayoutType.RowMajor: "T",
    LayoutType.NWC: "NWC",
    LayoutType.KXC: "KXC",
    LayoutType.NWK: "NWK",
    LayoutType.NCW: "NCW",
    LayoutType.KCX: "KCX",
    LayoutType.NKW: "NKW",
    LayoutType.NHWC: "nhwc",
    LayoutType.KYXC: "kyxc",
    LayoutType.NHWK: "NHWK",
    LayoutType.NCHW: "NCHW",
    LayoutType.KCYX: "KCYX",
    LayoutType.NKWH: "NKWH",
    LayoutType.NDHWC: "NDHWC",
    LayoutType.KZYXC: "KZYXC",
    LayoutType.NDHWK: "NDHWK",
    LayoutType.NCDHW: "NCDHW",
    LayoutType.KCZYX: "KCZYX",
    LayoutType.NKDHW: "NKDHW",
    LayoutType.G_NHW_C: "G_NHW_C",
    LayoutType.G_K_YX_C: "G_K_YX_C",
    LayoutType.G_NHW_K: "G_NHW_K",
    LayoutType.NHWGC: "NHWGC",
    LayoutType.KYXGC: "KYXGC",
    LayoutType.NHWGK: "NHWGK",
    LayoutType.GNHWC: "GNHWC",
    LayoutType.GKYXC: "GKYXC",
    LayoutType.GNHWK: "GNHWK",
    LayoutType.GNWC: "GNWC",
    LayoutType.GKXC: "GKXC",
    LayoutType.GNWK: "GNWK",
}

#
class OperationKind(enum.Enum):
    Gemm = auto()
    Conv1d = auto()
    Conv2d = auto()
    Conv3d = auto()
    Softmax = auto()
    LayerNorm = auto()
    GroupNorm = auto()


OperationKindNames = {
    OperationKind.Gemm: "gemm",
    OperationKind.Conv1d: "conv1d",
    OperationKind.Conv2d: "conv2d",
    OperationKind.Conv3d: "conv3d",
    OperationKind.Softmax: "softmax",
    OperationKind.LayerNorm: "layernorm",
    OperationKind.GroupNorm: "groupnorm",
}


class Conv2dKind(enum.Enum):
    Conv2d = auto()
    Conv2dBiasRelu = auto()
    Conv2dBiasReluAdd = auto()
    Conv2dBiasSigmoid = auto()
    GroupConv2dBiasRelu = auto()
    TransposedConv2d = auto()
    TransposedConv2dBiasRelu = auto()


Conv2dKindNames = {
    Conv2dKind.Conv2d: "conv2d",
    Conv2dKind.Conv2dBiasRelu: "conv2d_bias_relu",
    Conv2dKind.Conv2dBiasReluAdd: "conv2d_bias_relu_add",
    Conv2dKind.Conv2dBiasSigmoid: "conv2d_bias_sigmoid",
    Conv2dKind.GroupConv2dBiasRelu: "group_conv2d_bias_relu",
    Conv2dKind.TransposedConv2d: "transposed_conv2d",
    Conv2dKind.TransposedConv2dBiasRelu: "transposed_conv2d_bias_relu",
}


class GemmKind(enum.Enum):
    Gemm = auto()
    GemmPermute = auto()
    BatchGemm = auto()
    BatchGemmPermute = auto()
    SplitKGemm = auto()
    Grouped = auto()
    BatchGemmSoftmaxGemm = auto()
    BatchGemmSoftmaxGemmPermute = auto()
    GemmPermuteM2N3 = auto()
    GemmPermuteM3N2 = auto()


GemmKindNames = {
    GemmKind.Gemm: "gemm",
    GemmKind.GemmPermute: "gemm_permute",
    GemmKind.BatchGemm: "batch_gemm",
    GemmKind.BatchGemmPermute: "batch_gemm_permute",
    GemmKind.SplitKGemm: "split_k_gemm",
    GemmKind.Grouped: "grouped_gemm",
    GemmKind.BatchGemmSoftmaxGemm: "batched_gemm_softmax_gemm",
    GemmKind.BatchGemmSoftmaxGemmPermute: "batched_gemm_softmax_gemm_permute",
    GemmKind.GemmPermuteM2N3: "gemm_permute_m2n3",
    GemmKind.GemmPermuteM3N2: "gemm_permute_m3n2",
}


class TensorOperation(enum.Enum):
    PassThrough = auto()
    Add = auto()
    AddAdd = auto()
    AddMul = auto()
    AddMulTanh = auto()
    AlphaBetaAdd = auto()
    AddRelu = auto()
    AddFastGelu = auto()
    AddTanh = auto()
    AddHardswish = auto()
    AddSwish = auto()
    AddSigmoid = auto()
    AddReluAdd = auto()
    AddAddRelu = auto()
    AddSigmoidMul = auto()
    AddSigmoidMulTanh = auto()
    AddHardswishAdd = auto()
    UnaryIdentic = auto()
    UnarySquare = auto()
    UnaryAbs = auto()
    UnarySqrt = auto()
    AddMulAdd = auto()
    AddAddAdd = auto()
    AddAddAddRelu = auto()
    Bilinear = auto()
    CausalMask = auto()


#
TensorOperationTag = {
    TensorOperation.PassThrough: "ck::tensor_operation::element_wise::PassThrough",
    TensorOperation.Add: "ck::tensor_operation::element_wise::Add",
    TensorOperation.AddAdd: "ck::tensor_operation::element_wise::AddAdd",
    TensorOperation.AddMul: "ck::tensor_operation::element_wise::AddMul",
    TensorOperation.AddMulTanh: "ck::tensor_operation::element_wise::AddMulTanh",
    TensorOperation.AlphaBetaAdd: "ck::tensor_operation::element_wise::AlphaBetaAdd",
    TensorOperation.AddRelu: "ck::tensor_operation::element_wise::AddRelu",
    TensorOperation.AddFastGelu: "ck::tensor_operation::element_wise::AddFastGelu",
    TensorOperation.AddTanh: "ck::tensor_operation::element_wise::AddTanh",
    TensorOperation.AddSigmoid: "ck::tensor_operation::element_wise::AddSigmoid",
    TensorOperation.AddHardswish: "ck::tensor_operation::element_wise::AddHardswish",
    TensorOperation.AddSwish: "ck::tensor_operation::element_wise::AddSwish",
    TensorOperation.AddReluAdd: "ck::tensor_operation::element_wise::AddReluAdd",
    TensorOperation.AddAddRelu: "ck::tensor_operation::element_wise::AddAddRelu",
    TensorOperation.AddHardswishAdd: "ck::tensor_operation::element_wise::AddHardswishAdd",
    TensorOperation.AddMulAdd: "ck::tensor_operation::element_wise::AddMulAdd",
    TensorOperation.AddAddAdd: "ck::tensor_operation::element_wise::AddAddAdd",
    TensorOperation.AddAddAddRelu: "ck::tensor_operation::element_wise::AddAddAddRelu",
    TensorOperation.AddSigmoidMul: "ck::tensor_operation::element_wise::AddSigmoidMul",
    TensorOperation.AddSigmoidMulTanh: "ck::tensor_operation::element_wise::AddSigmoidMulTanh",
    TensorOperation.UnaryIdentic: "ck::tensor_operation::element_wise::UnaryIdentic",
    TensorOperation.UnarySquare: "ck::tensor_operation::element_wise::UnarySquare",
    TensorOperation.UnaryAbs: "ck::tensor_operation::element_wise::UnaryAbs",
    TensorOperation.UnarySqrt: "ck::tensor_operation::element_wise::UnarySqrt",
    TensorOperation.Bilinear: "ck::tensor_operation::element_wise::Bilinear",
    TensorOperation.CausalMask: "True",
}

#
ShortTensorOperationNames = {
    TensorOperation.PassThrough: "PT",
    TensorOperation.Add: "A",
    TensorOperation.AddAdd: "AA",
    TensorOperation.AddMul: "AM",
    TensorOperation.AddMulTanh: "AMT",
    TensorOperation.AlphaBetaAdd: "ABA",
    TensorOperation.AddRelu: "ARu",
    TensorOperation.AddFastGelu: "AFG",
    TensorOperation.AddTanh: "AT",
    TensorOperation.AddSigmoid: "AS",
    TensorOperation.AddHardswish: "AH",
    TensorOperation.AddSwish: "ASW",
    TensorOperation.AddReluAdd: "ARA",
    TensorOperation.AddAddRelu: "AAR",
    TensorOperation.AddHardswishAdd: "AHA",
    TensorOperation.AddMulAdd: "AMA",
    TensorOperation.AddAddAdd: "AAA",
    TensorOperation.AddAddAddRelu: "AAAR",
    TensorOperation.AddSigmoidMul: "ASM",
    TensorOperation.AddSigmoidMulTanh: "ASMT",
    TensorOperation.UnaryIdentic: "UI",
    TensorOperation.UnarySquare: "USR",
    TensorOperation.UnaryAbs: "UA",
    TensorOperation.UnarySqrt: "USQ",
    TensorOperation.Bilinear: "B",
    TensorOperation.CausalMask: "CM",
}


class MemoryDataOperation(enum.Enum):
    MemorySet = auto()
    InMemoryAtomicAdd = auto()


MemoryDataOperationTag = {
    MemoryDataOperation.MemorySet: "ck::InMemoryDataOperationEnum::Set",
    MemoryDataOperation.InMemoryAtomicAdd: "ck::InMemoryDataOperationEnum::AtomicAdd",
}


@dataclass
class TensorDesc:
    element: DataType
    layout: LayoutType
