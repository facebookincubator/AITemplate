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
classic_b2b_bmm kernel codegen for CUDA.
"""
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.target import Target
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base import CausalType

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/gemm/device/gemm.h"

#include "classic_b2b_bmm/device/b2b_batched_gemm.h"

namespace {

// Hardcode these sizes for now until we get profiling ready.
constexpr int ThreadblockM = 64;
constexpr int ThreadblockK = 32;
constexpr int WarpK = 32;
constexpr int InstructionM = 16;
constexpr int InstructionN = 8;
constexpr int InstructionK = 16;

// Currently, causal mask is only supported with warp shape M of 16.
// While we ought to debug it, the warp shape M restriction is not considered
// high-priority as we do not want to make warp M much larger anyway. If you want
// to explore the perf-impact of tuning this, then you can turn off causal mask after
// gemm 0 and see what the perf result is.
constexpr int WarpM = 16;

constexpr int N0 = {{n0}};
constexpr int N1 = {{n1}};

void check_status(cutlass::Status status, int64_t m0, int64_t k0, const std::string& message) {
  if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error(
        message +
        "Function: {{function_name}}. "
        "m0: " + std::to_string(m0) +
        ", k0: " + std::to_string(k0) +
        ", n0: " + std::to_string({{n0}}) +
        ", n1: " + std::to_string({{n1}}) + "."
      );
  }
  return;
}

}  // end namespace

{{func_signature}} {
  using ElementOutput = {{elem_output_type}};
  using ElementAccumulator = {{elem_accum_type}};
  using ElementCompute = {{elem_input_type}};

  ElementCompute alpha0 = ElementCompute({{alpha0}});
  ElementCompute beta0 = ElementCompute(1);
  ElementCompute activation_alpha = ElementCompute({{alpha1}});
  {% if alpha1_divide_by_seq_len %}
  activation_alpha = activation_alpha / (ElementCompute)(static_cast<int32_t>(m0));
  {% endif %}
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(0);

  using ThreadblockShape0 = cutlass::gemm::GemmShape<ThreadblockM, N0, ThreadblockK>;
  using WarpShape0 = cutlass::gemm::GemmShape<WarpM, N0, WarpK>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<ThreadblockM, N1, ThreadblockK>;
  using WarpShape1 = cutlass::gemm::GemmShape<WarpM, N1, WarpK>;
  using InstructionShape = cutlass::gemm::GemmShape<InstructionM, InstructionN, InstructionK>;

  using EpilogueOutputOp0 =
    cutlass::epilogue::thread::LinearCombinationGeneric<
      {{epilogue_math}},
      ElementOutput,
      InstructionShape::kM * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      // Saves a little time in the epilogue by not multiplying the source by beta.
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

  using EpilogueOutputOp1 =
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::Nothing
    >;

  using B2bGemmBatched = cutlass::gemm::device::B2bGemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,
    {{has_causal}} // enable causal mask after gemm0
  >;

  cutlass::gemm::GemmCoord problem_size_0(m0, {{n0}}, k0);
  cutlass::gemm::GemmCoord problem_size_1(m0, {{n1}}, {{n0}});
  typename B2bGemmBatched::Arguments arguments{
    problem_size_0,
    problem_size_1,
    {static_cast<ElementCompute*>(query), typename B2bGemmBatched::LayoutA::Stride(problem_size_0.k())},
    problem_size_0.m() * problem_size_0.k(),
    {static_cast<ElementCompute*>(key), typename B2bGemmBatched::LayoutB::Stride(problem_size_0.k())},
    problem_size_0.n() * problem_size_0.k(),
    {static_cast<ElementCompute*>(bias), typename B2bGemmBatched::LayoutC::Stride(problem_size_0.n())},
    problem_size_0.m() * problem_size_0.n(),
    {static_cast<ElementCompute*>(value), typename B2bGemmBatched::LayoutB1::Stride(problem_size_1.n())},
    problem_size_1.n() * problem_size_1.k(),
    {static_cast<ElementCompute*>(nullptr), typename B2bGemmBatched::LayoutScaleBias::Stride(0)},
    0,
    {static_cast<ElementOutput*>(output), typename B2bGemmBatched::LayoutC::Stride(problem_size_1.n())},
    problem_size_1.m() * problem_size_1.n(),
    batch_size,
    {alpha0, beta0, activation_alpha},
    {alpha1, beta1},
  };

  B2bGemmBatched b2b_gemm_op;
  check_status(
    b2b_gemm_op.can_implement(arguments),
    m0, k0,
    "Problem sizes are not supported."
  );
  check_status(
    b2b_gemm_op.initialize(arguments),
    m0, k0,
    "classic_b2b_bmm initialization failed!"
  );
  check_status(
    b2b_gemm_op(stream),
    m0, k0,
    "classic_b2b_bmm failed to execute!"
  );
}
    """
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   void* query,
                   void* key,
                   void* value,
                   void* bias,
                   int64_t batch_size,
                   int64_t m0,
                   int64_t k0,
                   cudaStream_t stream)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{output}},
{{indent}}    {{query}}, {{key}}, {{value}}, {{bias}},
{{indent}}    {{batch_size}},
{{indent}}    {{m0}},
{{indent}}    {{k0}},
{{indent}}    stream /* default stream */
{{indent}});
    """
)


@registry.reg("cuda.classic_b2b_bmm.gen_function")
def classic_b2b_bmm_gen_function(func_attrs: Dict[str, Any]) -> str:
    """the function for generating attention kernel"""
    q, k, v, bias = func_attrs["inputs"]
    n0 = k._attrs["shape"][1]
    n1 = v._attrs["shape"][2]
    if not isinstance(n0, IntImm) or not isinstance(n1, IntImm):
        raise RuntimeError(
            f"n0 and n1 must be static dims. {func_attrs['name']=}, {n0=}, {n1=}"
        )
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    if (
        "use_fp16_acc" in Target.current()._kwargs
        and Target.current()._kwargs["use_fp16_acc"]
    ):
        elem_accum_type = "cutlass::half_t"
    else:
        elem_accum_type = "float"

    import cutlass_lib

    epilogue_math = cutlass_lib.library.EpilogueMathTag[
        cutlass_lib.library.EpilogueMathName[func_attrs["epilogue_math_name"]]
    ]

    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        elem_accum_type=elem_accum_type,
        n0=str(n0.value()),
        n1=str(n1.value()),
        has_causal=(
            "true" if func_attrs["causal_type"] != CausalType.NO_CAUSAL else "false"
        ),
        alpha0=str(func_attrs["alpha0"]),
        alpha1=str(func_attrs["alpha1"]),
        alpha1_divide_by_seq_len="true"
        if func_attrs["alpha1_divide_by_seq_len"]
        else "false",
        epilogue_math=epilogue_math,
    )


@registry.reg("cuda.classic_b2b_bmm.func_decl")
def classic_b2b_bmm_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.classic_b2b_bmm.func_call")
def classic_b2b_bmm_gen_function_call(func_attrs, indent="  "):
    """the function for generating a function call for attention"""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 4

    output_name = func_attrs["outputs"][0]._attrs["name"]
    q_name = func_attrs["inputs"][0]._attrs["name"]
    k_name = func_attrs["inputs"][1]._attrs["name"]
    v_name = func_attrs["inputs"][2]._attrs["name"]
    bias_name = func_attrs["inputs"][3]._attrs["name"]

    q_shape = func_attrs["inputs"][0]._attrs["shape"]
    batch_size = q_shape[0]._attrs["name"]
    m0 = q_shape[1]._attrs["name"]
    k0 = q_shape[2]._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        query=q_name,
        key=k_name,
        value=v_name,
        bias=bias_name,
        batch_size=batch_size,
        m0=m0,
        k0=k0,
        indent=indent,
    )
