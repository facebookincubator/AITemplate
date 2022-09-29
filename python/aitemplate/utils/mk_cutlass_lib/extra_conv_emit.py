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

CONV_TEMPLATE = jinja2.Template(
    """
  using ${operation_name}_base =
  typename cutlass::conv::kernel::DefaultConv2d${conv_kind_name}WithBroadcast<
    ${element_a},
    ${layout_a},
    ${element_b},
    ${layout_b},
    ${element_c},
    ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m},
                             ${threadblock_shape_n},
                             ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m},
                             ${warp_shape_n},
                             ${warp_shape_k} >,
    cutlass::gemm::GemmShape<${instruction_shape_m},
                             ${instruction_shape_n},
                             ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${element_accumulator},
      ${element_epilogue},
      ${element_c},
      ${epilogue_vector_length},
      ${activation_op},
      ${binary_op},
      ${unary_op}
    >,
    ${swizzling_functor}, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator},
    ${iterator_algorithm},
    ${stride_support},
    ${align_a},
    ${align_b}
  >::Kernel;
"""
)

SRC_TEMPLATE = jinja2.Template(
    '''
class EmitConv2dWithBroadcastInstance:
  def __init__(self):
    self.template = """
    {{conv_template}}
  """

  def emit(self, operation):

    warp_shape = [int(operation.tile_description.threadblock_shape[idx]
                    / operation.tile_description.warp_count[idx]) for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
                                   / DataTypeSize[operation.C.element])

    values = {
      'operation_name': operation.procedural_name(),
      'conv_kind': ConvKindTag[operation.conv_kind],
      'conv_kind_name': ConvKindNames[operation.conv_kind].capitalize(),
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[operation.A.layout],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[operation.B.layout],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_vector_length': str(epilogue_vector_length),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'iterator_algorithm': IteratorAlgorithmTag[operation.iterator_algorithm],
      'iterator_algorithm_name': IteratorAlgorithmNames[operation.iterator_algorithm].capitalize(),
      'stride_support': StrideSupportTag[operation.stride_support],
      'math_operator': 'cutlass::arch::OpMultiplyAddComplex' if operation.is_complex() else \
      MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'activation_op': EpilogueMathTag[operation.activation_op],
      'binary_op': EpilogueMathTag[operation.binary_op],
      'unary_op': EpilogueMathTag[operation.unary_op],
    }

    return SubstituteTemplate(self.template, values)


'''
)


def emit_library():
    conv_template = CONV_TEMPLATE.render()
    return SRC_TEMPLATE.render(conv_template=conv_template)
