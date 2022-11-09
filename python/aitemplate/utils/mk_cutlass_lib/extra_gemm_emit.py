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


GEMM_SOFTMAX_TEMPLATE = jinja2.Template(
    """
    using ${operation_name}_base =
    cutlass::GemmSoftmax<
        ${element_a}, ${layout_a}, ${align_a},
        ${element_b}, ${layout_b}, ${align_b},
        ${element_c}, ${align_c},
        ${opcode_class},
        ${arch},
        ${element_accumulator},
        ${stages},
        cutlass::gemm::GemmShape<${threadblock_shape_m},
                                  ${threadblock_shape_n},
                                  ${threadblock_shape_k}>,
        cutlass::gemm::GemmShape<${warp_shape_m},
                                  ${warp_shape_n},
                                  ${warp_shape_k}>,
        cutlass::gemm::GemmShape<${instruction_shape_m},
                                  ${instruction_shape_n},
                                  ${instruction_shape_k}>,
        ${epilogue_functor},
        ${swizzling_functor}
    >;
"""
)

SRC_SOFTMAX_TEMPLATE = jinja2.Template(
    '''
class EmitGemmSoftmaxInstance:
  def __init__(self, operation_suffix = ''):
      self.operation_suffix = operation_suffix
      self.includes = [
        "cutlass/cutlass.h",
        "cutlass/numeric_types.h",
        "cutlass/arch/arch.h",
        "cutlass/arch/mma.h",
        "cutlass/layout/matrix.h",
        "cutlass/gemm/device/gemm.h",
        "cutlass/gemm/kernel/gemm_grouped.h",
        "cutlass/gemm/kernel/default_gemm_grouped.h",
        "cutlass/gemm/device/gemm_grouped.h",
        "cutlass/gemm/kernel/default_gemm.h"
      ]

      self.builtin_epilogue_functor_template = """
          ${epilogue_functor}<
            ${element_c},
            ${epilogue_vector_length},
            ${element_accumulator},
            ${element_epilogue}
          >
      """

      self.gemm_template = """
      {{gemm_template}}
      """


  def emit(self, operation):

    threadblock_shape = operation.tile_description.threadblock_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

    transpose_layouts = {
      LayoutType.ColumnMajor: LayoutType.RowMajor,
      LayoutType.RowMajor: LayoutType.ColumnMajor
    }

    instance_layout_A, instance_layout_B, instance_layout_C = \
      (operation.A.layout, operation.B.layout, operation.C.layout)
    #

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):

      epilogue_vector_length = \
        min(operation.C.alignment * DataTypeSize[operation.C.element], 128) \
          // DataTypeSize[operation.C.element]

      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()


    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
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
      'epilogue_functor': epilogue_functor,
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'align_c': str(operation.C.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation]
    }

    return SubstituteTemplate(self.gemm_template, values)


'''
)


GEMM_PERMUTE_TEMPLATE = jinja2.Template(
    """
    using ${operation_name}_base =
    cutlass::gemm::device::GemmUniversal<
        ${element_a}, ${layout_a},
        ${element_b}, ${layout_b},
        ${element_c}, ${layout_c},
        ${element_accumulator},
        ${opcode_class},
        ${arch},
        cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
        cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
        cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
        ${epilogue_functor}<
          ${element_c},
          ${epilogue_vector_length},
          ${element_accumulator},
          ${element_epilogue}
        >,
        ${swizzling_functor},
        ${stages},
        ${align_a},
        ${align_b},
        ${math_operation},
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone,
        false,  /*GatherA*/
        false,  /*GatherB*/
        false,  /*ScatterD*/
        ${permute_layout} /*PermuteDLayout*/
    >;
"""
)

SRC_PERMUTE_TEMPLATE = jinja2.Template(
    '''
class EmitGemmPermuteInstance:
  def __init__(self, operation_suffix = ''):
      self.operation_suffix = operation_suffix
      self.includes = []
      self.gemm_template = """
      {{gemm_template}}
      """


  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

    values = {
      'operation_name': operation.procedural_name(),
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
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'permute_layout': EpiloguePermuteLayoutTag[operation.permute_layout],
    }

    return SubstituteTemplate(self.gemm_template, values)


'''
)

DUAL_GEMM_TEMPLATE = jinja2.Template(
    """
    using ${operation_name}_base =
    cutlass::gemm::device::DualGemm<
        ${element_a}, ${layout_a},
        ${element_b}, ${layout_b},
        ${element_c}, ${layout_c},
        ${element_accumulator},
        ${opcode_class},
        ${arch},
        cutlass::gemm::GemmShape<${threadblock_shape_m},
                                  ${threadblock_shape_n},
                                  ${threadblock_shape_k}>,
        cutlass::gemm::GemmShape<${warp_shape_m},
                                  ${warp_shape_n},
                                  ${warp_shape_k}>,
        cutlass::gemm::GemmShape<${instruction_shape_m},
                                  ${instruction_shape_n},
                                  ${instruction_shape_k}>,
        ${epilogue_functor}
        ${epilogue_functor}
        ${epilogue_functor2}
        ${swizzling_functor},
        ${stages},
        false,
        false,
        false
    >;
"""
)

SRC_DUAL_GEMM_TEMPLATE = jinja2.Template(
    '''
class EmitDualGemmInstance:
  def __init__(self, operation_suffix = ''):
      self.operation_suffix = operation_suffix
      self.includes = []
      self.builtin_epilogue_functor_template = """
          ${epilogue_functor}<
            ${element_c},
            ${epilogue_vector_length},
            ${element_accumulator},
            ${element_epilogue},
            cutlass::epilogue::thread::ScaleType::Nothing
          >,
      """
      self.builtin_epilogue_functor2_template = """
          ${epilogue_functor2}<
            ${element_c},
            ${epilogue_vector_length},
            ${element_c},
            ${element_epilogue}
          >,
      """

      self.gemm_template = """
      {{gemm_template}}
      """


  def emit(self, operation):

    threadblock_shape = operation.tile_description.threadblock_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

    transpose_layouts = {
      LayoutType.ColumnMajor: LayoutType.RowMajor,
      LayoutType.RowMajor: LayoutType.ColumnMajor
    }

    instance_layout_A, instance_layout_B, instance_layout_C = \
      (operation.A.layout, operation.B.layout, operation.C.layout)
    #

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):

      epilogue_vector_length = \
        min(operation.C.alignment * DataTypeSize[operation.C.element], 128) \
          // DataTypeSize[operation.C.element]

      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
        'epilogue_functor2': EpilogueFunctorTag[operation.epilogue_functor2],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
      epilogue_functor2 = SubstituteTemplate(self.builtin_epilogue_functor2_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
      epilogue_functor2 = self.epilogue_functor.emit_declaration()


    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
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
      'epilogue_functor': epilogue_functor,
      'epilogue_functor2': epilogue_functor2,
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'align_c': str(operation.C.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation]
    }

    return SubstituteTemplate(self.gemm_template, values)


'''
)


def emit_library():
    template = ""
    template += SRC_SOFTMAX_TEMPLATE.render(
        gemm_template=GEMM_SOFTMAX_TEMPLATE.render()
    )
    template += SRC_PERMUTE_TEMPLATE.render(
        gemm_template=GEMM_PERMUTE_TEMPLATE.render()
    )
    template += SRC_DUAL_GEMM_TEMPLATE.render(gemm_template=DUAL_GEMM_TEMPLATE.render())
    return template
