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
Extra cutlass enum, mainly for epilogue
"""
import jinja2

SRC_TEMPLATE = jinja2.Template(
    """
class EpilogueFunctor(enum.Enum):
  LinearCombination = enum_auto()
  LinearCombinationClamp = enum_auto()
  LinearCombinationRelu = enum_auto()
  LinearCombinationSigmoid = enum_auto()
  LinearCombinationTanh = enum_auto()
  LinearCombinationResidualBlock = enum_auto()
  LinearCombinationHardSwish = enum_auto()
  LinearCombinationGELU = enum_auto()
  LinearCombinationFastGELU = enum_auto()
  LinearCombinationSilu = enum_auto()
  LinearCombinationELUp1 = enum_auto()
  LeftSiLUAndMul = enum_auto()
  LeftFastGeluAndMul = enum_auto()
  Div = enum_auto()

EpilogueFunctorTag = {
  EpilogueFunctor.LinearCombination:
    'cutlass::epilogue::thread::LinearCombination',
  EpilogueFunctor.LinearCombinationClamp:
    'cutlass::epilogue::thread::LinearCombinationClamp',
  EpilogueFunctor.LinearCombinationRelu:
    'cutlass::epilogue::thread::LinearCombinationRelu',
  EpilogueFunctor.LinearCombinationSigmoid:
    'cutlass::epilogue::thread::LinearCombinationSigmoid',
  EpilogueFunctor.LinearCombinationTanh:
    'cutlass::epilogue::thread::LinearCombinationTanh',
  EpilogueFunctor.LinearCombinationResidualBlock:
    'cutlass::epilogue::thread::LinearCombinationResidualBlock',
  EpilogueFunctor.LinearCombinationHardSwish:
    'cutlass::epilogue::thread::LinearCombinationHardSwish',
  EpilogueFunctor.LinearCombinationGELU:
    'cutlass::epilogue::thread::LinearCombinationGELU',
  EpilogueFunctor.LinearCombinationFastGELU:
    'cutlass::epilogue::thread::LinearCombinationFastGELU',
  EpilogueFunctor.LinearCombinationSilu:
    'cutlass::epilogue::thread::LinearCombinationSilu',
  EpilogueFunctor.LinearCombinationELUp1:
    'cutlass::epilogue::thread::LinearCombinationELUp1',
  EpilogueFunctor.LeftSiLUAndMul:
    'cutlass::epilogue::thread::LeftSiLUAndMul',
  EpilogueFunctor.LeftFastGeluAndMul:
    'cutlass::epilogue::thread::LeftFastGeluAndMul',
  EpilogueFunctor.Div:
    'cutlass::epilogue::thread::Div',
}

EpilogueFunctorName = {
  "LinearCombination": EpilogueFunctor.LinearCombination,
  "LinearCombinationClamp": EpilogueFunctor.LinearCombinationClamp,
  "LinearCombinationRelu": EpilogueFunctor.LinearCombinationRelu,
  "LinearCombinationSigmoid": EpilogueFunctor.LinearCombinationSigmoid,
  "LinearCombinationTanh": EpilogueFunctor.LinearCombinationTanh,
  "LinearCombinationResidualBlock": EpilogueFunctor.LinearCombinationResidualBlock,
  "LinearCombinationHardSwish": EpilogueFunctor.LinearCombinationHardSwish,
  "LinearCombinationGELU": EpilogueFunctor.LinearCombinationGELU,
  "LinearCombinationFastGELU": EpilogueFunctor.LinearCombinationFastGELU,
  "LinearCombinationSilu": EpilogueFunctor.LinearCombinationSilu,
  "LinearCombinationELUp1": EpilogueFunctor.LinearCombinationELUp1,
  "LeftSiLUAndMul": EpilogueFunctor.LeftSiLUAndMul,
  "LeftFastGeluAndMul": EpilogueFunctor.LeftFastGeluAndMul,
  "Div": EpilogueFunctor.Div,
}

class EpilogueMath(enum.Enum):
  ReLu = enum_auto()
  Sigmoid = enum_auto()
  Tanh = enum_auto()
  Identity = enum_auto()
  HardSwish = enum_auto()
  Plus = enum_auto()
  Gelu = enum_auto()
  FastGelu = enum_auto()
  SiLu = enum_auto()
  ELUp1 = enum_auto()


EpilogueMathTag = {
  EpilogueMath.ReLu: 'cutlass::epilogue::thread::ReLu',
  EpilogueMath.Sigmoid: 'cutlass::epilogue::thread::Sigmoid',
  EpilogueMath.Tanh: 'cutlass::epilogue::thread::Tanh',
  EpilogueMath.Identity: 'cutlass::epilogue::thread::Identity',
  EpilogueMath.HardSwish: 'cutlass::epilogue::thread::HardSwish',
  EpilogueMath.Plus: 'cutlass::plus',
  EpilogueMath.Gelu: 'GELU',
  EpilogueMath.FastGelu: 'GELU_taylor',
  EpilogueMath.SiLu: 'cutlass::epilogue::thread::SiLu',
  EpilogueMath.ELUp1: 'cutlass::epilogue::thread::ELUp1',
}

EpilogueMathName = {
  "ReLu": EpilogueMath.ReLu,
  "Sigmoid": EpilogueMath.Sigmoid,
  "Tanh": EpilogueMath.Tanh,
  "Identity": EpilogueMath.Identity,
  "HardSwish": EpilogueMath.HardSwish,
  "Plus": EpilogueMath.Plus,
  "Add": EpilogueMath.Plus,
  "Gelu": EpilogueMath.Gelu,
  "FastGelu": EpilogueMath.FastGelu,
  "SiLu": EpilogueMath.SiLu,
  "ELUp1": EpilogueMath.ELUp1
}

class EpiloguePermuteLayout(enum.Enum):
  Permute5D_20314 = enum_auto()
  Permute4D_0213 = enum_auto()
  Permute4DBMM_0213 = enum_auto()
  # Permute3DBMM_021 = enum_auto()
  NoPermute = enum_auto()

EpiloguePermuteLayoutTag = {
  EpiloguePermuteLayout.Permute5D_20314: 'cutlass::layout::Tensor5DPermute20314RowMajor',
  EpiloguePermuteLayout.Permute4D_0213: 'cutlass::layout::Tensor4DPermute0213RowMajor',
  EpiloguePermuteLayout.Permute4DBMM_0213: 'cutlass::layout::Tensor4DPermuteBMM0213RowMajor',
  EpiloguePermuteLayout.NoPermute: 'cutlass::layout::NoPermute',
  # EpiloguePermuteLayout.Permute3DBMM_021: 'cutlass::layout::Tensor3DPermute021BMM',
}

EpiloguePermuteLayoutName = {
  "Permute5D_20314": EpiloguePermuteLayout.Permute5D_20314,
  "Permute4D_0213": EpiloguePermuteLayout.Permute4D_0213,
  "Permute4DBMM_0213": EpiloguePermuteLayout.Permute4DBMM_0213,
  "NoPermute": EpiloguePermuteLayout.NoPermute,
  # "Permute3DBMM_021": EpiloguePermuteLayout.Permute3DBMM_021,
}

class EpilogueScheduleType(enum.Enum):
  ScheduleAuto = enum_auto()
  EpilogueTransposed = enum_auto()
  NoSmemWarpSpecialized = enum_auto()
  TmaWarpSpecialized = enum_auto()
  TmaWarpSpecializedCooperative = enum_auto()
  TmaWarpSpecializedElementwiseRelu = enum_auto()
  TmaWarpSpecializedCooperativeElementwiseRelu = enum_auto()
  TmaWarpSpecializedElementwiseSigmoid = enum_auto()
  TmaWarpSpecializedCooperativeElementwiseSigmoid = enum_auto()
  TmaWarpSpecializedElementwiseSiLu = enum_auto()
  TmaWarpSpecializedCooperativeElementwiseSiLu = enum_auto()
  TmaWarpSpecializedElementwiseTanh = enum_auto()
  TmaWarpSpecializedCooperativeElementwiseTanh = enum_auto()
  TmaWarpSpecializedElementwiseHardSwish = enum_auto()
  TmaWarpSpecializedCooperativeElementwiseHardSwish = enum_auto()
  TmaWarpSpecializedElementwiseGELU = enum_auto()
  TmaWarpSpecializedCooperativeElementwiseGELU = enum_auto()
  TmaWarpSpecializedElementwiseFastGELU = enum_auto()
  TmaWarpSpecializedCooperativeElementwiseFastGELU = enum_auto()
  TmaWarpSpecializedBiasElementwise = enum_auto()
  TmaWarpSpecializedCooperativeBiasElementwise = enum_auto()
  TmaWarpSpecializedBiasElementwiseRelu = enum_auto()
  TmaWarpSpecializedCooperativeBiasElementwiseRelu = enum_auto()
  TmaWarpSpecializedBiasElementwiseSigmoid = enum_auto()
  TmaWarpSpecializedCooperativeBiasElementwiseSigmoid = enum_auto()
  TmaWarpSpecializedBiasElementwiseSiLu = enum_auto()
  TmaWarpSpecializedCooperativeBiasElementwiseSiLu = enum_auto()
  TmaWarpSpecializedBiasElementwiseTanh = enum_auto()
  TmaWarpSpecializedCooperativeBiasElementwiseTanh = enum_auto()
  TmaWarpSpecializedBiasElementwiseHardSwish = enum_auto()
  TmaWarpSpecializedCooperativeBiasElementwiseHardSwish = enum_auto()
  TmaWarpSpecializedBiasElementwiseGELU = enum_auto()
  TmaWarpSpecializedCooperativeBiasElementwiseGELU = enum_auto()
  TmaWarpSpecializedBiasElementwiseFastGELU = enum_auto()
  TmaWarpSpecializedCooperativeBiasElementwiseFastGELU = enum_auto()

EpilogueScheduleTag = {
  EpilogueScheduleType.ScheduleAuto: 'cutlass::epilogue::collective::EpilogueScheduleAuto',
  EpilogueScheduleType.EpilogueTransposed: 'cutlass::gemm::EpilogueTransposed',
  EpilogueScheduleType.NoSmemWarpSpecialized: 'cutlass::epilogue::NoSmemWarpSpecialized',
  EpilogueScheduleType.TmaWarpSpecialized: 'cutlass::epilogue::TmaWarpSpecialized',
  EpilogueScheduleType.TmaWarpSpecializedCooperative: 'cutlass::epilogue::TmaWarpSpecializedCooperative',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseRelu: 'cutlass::epilogue::TmaWarpSpecializedElementwise<cutlass::epilogue::thread::ReLu>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseRelu: 'cutlass::epilogue::TmaWarpSpecializedCooperativeElementwise<cutlass::epilogue::thread::ReLu>',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseSigmoid: 'cutlass::epilogue::TmaWarpSpecializedElementwise<cutlass::epilogue::thread::Sigmoid>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseSigmoid: 'cutlass::epilogue::TmaWarpSpecializedCooperativeElementwise<cutlass::epilogue::thread::Sigmoid>',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseSiLu: 'cutlass::epilogue::TmaWarpSpecializedElementwise<cutlass::epilogue::thread::SiLu>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseSiLu: 'cutlass::epilogue::TmaWarpSpecializedCooperativeElementwise<cutlass::epilogue::thread::SiLu>',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseTanh: 'cutlass::epilogue::TmaWarpSpecializedElementwise<cutlass::epilogue::thread::Tanh>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseTanh: 'cutlass::epilogue::TmaWarpSpecializedCooperativeElementwise<cutlass::epilogue::thread::Tanh>',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseHardSwish: 'cutlass::epilogue::TmaWarpSpecializedElementwise<cutlass::epilogue::thread::HardSwish>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseHardSwish: 'cutlass::epilogue::TmaWarpSpecializedCooperativeElementwise<cutlass::epilogue::thread::HardSwish>',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseGELU: 'cutlass::epilogue::TmaWarpSpecializedElementwise<cutlass::epilogue::thread::GELU>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseGELU: 'cutlass::epilogue::TmaWarpSpecializedCooperativeElementwise<cutlass::epilogue::thread::GELU>',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseFastGELU: 'cutlass::epilogue::TmaWarpSpecializedElementwise<cutlass::epilogue::thread::GELU_taylor>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseFastGELU: 'cutlass::epilogue::TmaWarpSpecializedCooperativeElementwise<cutlass::epilogue::thread::GELU_taylor>',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwise: 'cutlass::epilogue::TmaWarpSpecializedBiasElementwise<cutlass::epilogue::thread::Identity, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwise: 'cutlass::epilogue::TmaWarpSpecializedCooperativeBiasElementwise<cutlass::epilogue::thread::Identity, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseRelu: 'cutlass::epilogue::TmaWarpSpecializedBiasElementwise<cutlass::epilogue::thread::ReLu, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseRelu: 'cutlass::epilogue::TmaWarpSpecializedCooperativeBiasElementwise<cutlass::epilogue::thread::ReLu, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseSigmoid: 'cutlass::epilogue::TmaWarpSpecializedBiasElementwise<cutlass::epilogue::thread::Sigmoid, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseSigmoid: 'cutlass::epilogue::TmaWarpSpecializedCooperativeBiasElementwise<cutlass::epilogue::thread::Sigmoid, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseSiLu: 'cutlass::epilogue::TmaWarpSpecializedBiasElementwise<cutlass::epilogue::thread::SiLu, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseSiLu: 'cutlass::epilogue::TmaWarpSpecializedCooperativeBiasElementwise<cutlass::epilogue::thread::SiLu, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseTanh: 'cutlass::epilogue::TmaWarpSpecializedBiasElementwise<cutlass::epilogue::thread::Tanh, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseTanh: 'cutlass::epilogue::TmaWarpSpecializedCooperativeBiasElementwise<cutlass::epilogue::thread::Tanh, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseHardSwish: 'cutlass::epilogue::TmaWarpSpecializedBiasElementwise<cutlass::epilogue::thread::HardSwish, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseHardSwish: 'cutlass::epilogue::TmaWarpSpecializedCooperativeBiasElementwise<cutlass::epilogue::thread::HardSwish, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseGELU: 'cutlass::epilogue::TmaWarpSpecializedBiasElementwise<cutlass::epilogue::thread::GELU, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseGELU: 'cutlass::epilogue::TmaWarpSpecializedCooperativeBiasElementwise<cutlass::epilogue::thread::GELU, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseFastGELU: 'cutlass::epilogue::TmaWarpSpecializedBiasElementwise<cutlass::epilogue::thread::GELU_taylor, elem_input_type, cutlass::plus, false, elem_input_type>',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseFastGELU: 'cutlass::epilogue::TmaWarpSpecializedCooperativeBiasElementwise<cutlass::epilogue::thread::GELU_taylor, elem_input_type, cutlass::plus, false, elem_input_type>',
}

EpilogueScheduleSuffixes = {
  EpilogueScheduleType.ScheduleAuto: '',
  EpilogueScheduleType.EpilogueTransposed: '',
  EpilogueScheduleType.NoSmemWarpSpecialized: '_epi_nosmem',
  EpilogueScheduleType.TmaWarpSpecialized: '_epi_tma',
  EpilogueScheduleType.TmaWarpSpecializedCooperative: '_epi_tma',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseRelu: '_epi_tma_relu',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseRelu: '_epi_tma_relu',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseSigmoid: '_epi_tma_sigmoid',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseSigmoid: '_epi_tma_sigmoid',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseSiLu: '_epi_tma_silu',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseSiLu: '_epi_tma_silu',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseTanh: '_epi_tma_tanh',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseTanh: '_epi_tma_tanh',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseHardSwish: '_epi_tma_hardswish',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseHardSwish: '_epi_tma_hardswish',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseGELU: '_epi_tma_gelu',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseGELU: '_epi_tma_gelu',
  EpilogueScheduleType.TmaWarpSpecializedElementwiseFastGELU: '_epi_tma_fast_gelu',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseFastGELU: '_epi_tma_fast_gelu',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwise: '_epi_tma_bias',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwise: '_epi_tma_bias',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseRelu: '_epi_tma_bias_relu',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseRelu: '_epi_tma_bias_relu',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseSigmoid: '_epi_tma_bias_sigmoid',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseSigmoid: '_epi_tma_bias_sigmoid',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseSiLu: '_epi_tma_bias_silu',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseSiLu: '_epi_tma_bias_silu',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseTanh: '_epi_tma_bias_tanh',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseTanh: '_epi_tma_bias_tanh',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseHardSwish: '_epi_tma_bias_hardswish',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseHardSwish: '_epi_tma_bias_hardswish',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseGELU: '_epi_tma_bias_gelu',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseGELU: '_epi_tma_bias_gelu',
  EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseFastGELU: '_epi_tma_bias_fast_gelu',
  EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseFastGELU: '_epi_tma_bias_fast_gelu',
}

EpilogueScheduleMapping = {
  EpilogueScheduleType.TmaWarpSpecialized: {
    EpilogueFunctor.LinearCombinationRelu: EpilogueScheduleType.TmaWarpSpecializedElementwiseRelu,
    EpilogueFunctor.LinearCombinationSigmoid: EpilogueScheduleType.TmaWarpSpecializedElementwiseSigmoid,
    EpilogueFunctor.LinearCombinationSilu: EpilogueScheduleType.TmaWarpSpecializedElementwiseSiLu,
    EpilogueFunctor.LinearCombinationTanh: EpilogueScheduleType.TmaWarpSpecializedElementwiseTanh,
    EpilogueFunctor.LinearCombinationHardSwish: EpilogueScheduleType.TmaWarpSpecializedElementwiseHardSwish,
    EpilogueFunctor.LinearCombinationGELU: EpilogueScheduleType.TmaWarpSpecializedElementwiseGELU,
    EpilogueFunctor.LinearCombinationFastGELU: EpilogueScheduleType.TmaWarpSpecializedElementwiseFastGELU,
    EpilogueFunctor.LinearCombinationResidualBlock: EpilogueScheduleType.TmaWarpSpecialized,
  },
  EpilogueScheduleType.TmaWarpSpecializedCooperative: {
    EpilogueFunctor.LinearCombinationRelu: EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseRelu,
    EpilogueFunctor.LinearCombinationSigmoid: EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseSigmoid,
    EpilogueFunctor.LinearCombinationSilu: EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseSiLu,
    EpilogueFunctor.LinearCombinationTanh: EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseTanh,
    EpilogueFunctor.LinearCombinationHardSwish: EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseHardSwish,
    EpilogueFunctor.LinearCombinationGELU: EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseGELU,
    EpilogueFunctor.LinearCombinationFastGELU: EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseFastGELU,
    EpilogueFunctor.LinearCombinationResidualBlock: EpilogueScheduleType.TmaWarpSpecializedCooperative,
  },
}

EpilogueScheduleBiasElementwiseMapping = {
  EpilogueScheduleType.TmaWarpSpecialized: EpilogueScheduleType.TmaWarpSpecializedBiasElementwise,
  EpilogueScheduleType.TmaWarpSpecializedCooperative: EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwise,
  EpilogueScheduleType.TmaWarpSpecializedElementwiseRelu: EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseRelu,
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseRelu: EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseRelu,
  EpilogueScheduleType.TmaWarpSpecializedElementwiseSigmoid: EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseSigmoid,
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseSigmoid: EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseSigmoid,
  EpilogueScheduleType.TmaWarpSpecializedElementwiseSiLu: EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseSiLu,
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseSiLu: EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseSiLu,
  EpilogueScheduleType.TmaWarpSpecializedElementwiseTanh: EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseTanh,
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseTanh: EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseTanh,
  EpilogueScheduleType.TmaWarpSpecializedElementwiseHardSwish: EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseHardSwish,
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseHardSwish: EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseHardSwish,
  EpilogueScheduleType.TmaWarpSpecializedElementwiseGELU: EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseGELU,
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseGELU: EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseGELU,
  EpilogueScheduleType.TmaWarpSpecializedElementwiseFastGELU: EpilogueScheduleType.TmaWarpSpecializedBiasElementwiseFastGELU,
  EpilogueScheduleType.TmaWarpSpecializedCooperativeElementwiseFastGELU: EpilogueScheduleType.TmaWarpSpecializedCooperativeBiasElementwiseFastGELU,
}

"""
)


def emit_library():
    return SRC_TEMPLATE.render()
