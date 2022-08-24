# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
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
  LinearCombinationResidualBlockV2 = enum_auto()
  LinearCombinationHardSwish = enum_auto()
  LinearCombinationGELU = enum_auto()
  LinearCombinationFastGELU = enum_auto()
  LinearCombinationSilu = enum_auto()

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
  EpilogueFunctor.LinearCombinationResidualBlockV2:
    'cutlass::epilogue::thread::LinearCombinationResidualBlockV2',
  EpilogueFunctor.LinearCombinationHardSwish:
    'cutlass::epilogue::thread::LinearCombinationHardSwish',
  EpilogueFunctor.LinearCombinationGELU:
    'cutlass::epilogue::thread::LinearCombinationGELU',
  EpilogueFunctor.LinearCombinationFastGELU:
    'cutlass::epilogue::thread::LinearCombinationFastGELU',
  EpilogueFunctor.LinearCombinationSilu:
    'cutlass::epilogue::thread::LinearCombinationSilu',
}

EpilogueFunctorName = {
  "LinearCombination": EpilogueFunctor.LinearCombination,
  "LinearCombinationClamp": EpilogueFunctor.LinearCombinationClamp,
  "LinearCombinationRelu": EpilogueFunctor.LinearCombinationRelu,
  "LinearCombinationSigmoid": EpilogueFunctor.LinearCombinationSigmoid,
  "LinearCombinationTanh": EpilogueFunctor.LinearCombinationTanh,
  "LinearCombinationResidualBlock": EpilogueFunctor.LinearCombinationResidualBlock,
  "LinearCombinationResidualBlockV2": EpilogueFunctor.LinearCombinationResidualBlockV2,
  "LinearCombinationHardSwish": EpilogueFunctor.LinearCombinationHardSwish,
  "LinearCombinationGELU": EpilogueFunctor.LinearCombinationGELU,
  "LinearCombinationFastGELU": EpilogueFunctor.LinearCombinationFastGELU,
  "LinearCombinationSilu": EpilogueFunctor.LinearCombinationSilu
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
  Silu = enum_auto()


EpilogueMathTag = {
  EpilogueMath.ReLu: 'cutlass::epilogue::thread::ReLu',
  EpilogueMath.Sigmoid: 'cutlass::epilogue::thread::Sigmoid',
  EpilogueMath.Tanh: 'cutlass::epilogue::thread::Tanh',
  EpilogueMath.Identity: 'cutlass::epilogue::thread::Identity',
  EpilogueMath.HardSwish: 'cutlass::epilogue::thread::HardSwish',
  EpilogueMath.Plus: 'cutlass::plus',
  EpilogueMath.Gelu: 'GELU',
  EpilogueMath.FastGelu: 'GELU_taylor',
  EpilogueMath.FastGelu: 'cutlass::epilogue::thread::Silu'
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
  "Silu": EpilogueMath.Silu
}

class EpiloguePermuteLayout(enum.Enum):
  Permute5D_20314 = enum_auto()
  Permute4D_0213 = enum_auto()
  Permute4DBMM_0213 = enum_auto()
  Permute3DBMM_021 = enum_auto()
  NoPermute = enum_auto()

EpiloguePermuteLayoutTag = {
  EpiloguePermuteLayout.Permute5D_20314: 'cutlass::layout::Tensor5DPermute',
  EpiloguePermuteLayout.Permute4D_0213: 'cutlass::layout::Tensor4DPermute',
  EpiloguePermuteLayout.Permute4DBMM_0213: 'cutlass::layout::Tensor4DPermuteBMM',
  EpiloguePermuteLayout.NoPermute: 'cutlass::layout::NoPermute',
  EpiloguePermuteLayout.Permute3DBMM_021: 'cutlass::layout::Tensor3DPermute021BMM',
}

EpiloguePermuteLayoutName = {
  "Permute5D_20314": EpiloguePermuteLayout.Permute5D_20314,
  "Permute4D_0213": EpiloguePermuteLayout.Permute4D_0213,
  "Permute4DBMM_0213": EpiloguePermuteLayout.Permute4DBMM_0213,
  "NoPermute": EpiloguePermuteLayout.NoPermute,
  "Permute3DBMM_021": EpiloguePermuteLayout.Permute3DBMM_021,
}

"""
)


def emit_library():
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
    return SRC_TEMPLATE.render()
