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
  EpiloguePermuteLayout.Permute5D_20314: 'cutlass::layout::Tensor5DPermute20314',
  EpiloguePermuteLayout.Permute4D_0213: 'cutlass::layout::Tensor4DPermute0213',
  EpiloguePermuteLayout.Permute4DBMM_0213: 'cutlass::layout::Tensor4DPermuteBMM0213',
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

"""
)


def emit_library():
    return SRC_TEMPLATE.render()
