# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Extra cutlass tiling configs for special problems
"""
import jinja2

SRC_TEMPLATE = jinja2.Template(
    """
import enum
import os.path
import shutil
import argparse

from .library import *
from .manifest import *
from .generator import CreateGemmOperator

{{func_decls}}

def GenerateSM80(manifest, args):
  {{func_calls}}
"""
)

EXTRA_GEN = {}

PERM021_FC_TEMPLATE = jinja2.Template(
    """
def {{func_name}}(manifest, args):

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024
  max_cc_smem_limited = 80

  alignment_constraints = [8, 4, 2]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128,  32, 32],  7, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

"""
)

EXTRA_GEN["perm021_fc"] = PERM021_FC_TEMPLATE


def emit_library():
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
    func_decls = ""
    func_calls = ""
    for fname, ftemplate in EXTRA_GEN.items():
        func_decls += ftemplate.render(func_name=fname)
        func_calls += "  {fname}(manifest, args)".format(fname=fname)
    return SRC_TEMPLATE.render(func_decls=func_decls, func_calls=func_calls)
