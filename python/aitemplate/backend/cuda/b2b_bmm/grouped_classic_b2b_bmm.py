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

#include "grouped_classic_b2b_bmm/device/b2b_batched_gemm.h"

namespace {

// Hardcode these sizes for now until we get profiling ready.
constexpr int ThreadblockM = 128; // changed from 64 due to improved performance. More leads to errors.
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

constexpr int N0 = {{max_seq_len}};
constexpr int N1 = {{n1}}; // embedding size of value

void check_status(cutlass::Status status, int64_t max_seq_len, int64_t k0, const std::string& message) {
  if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error(
        message +
        "Function: {{function_name}}. "
        "max_seq_len: " + std::to_string(max_seq_len) +
        ", k0: " + std::to_string(k0) +
        ", n0: " + std::to_string({{max_seq_len}}) +
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
  activation_alpha = activation_alpha / (ElementCompute)(static_cast<int32_t>(max_seq_len));
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

  using GroupedB2bGemmBatched = cutlass::gemm::device::GroupedB2bGemmBatched<
    ElementCompute,
    cutlass::layout::RowMajor,
    ElementCompute,
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
    {{has_causal}}, // enable causal mask after gemm0
    false, // stage accumulator in shared memory
    {{offset_type}}
  >;

  cutlass::gemm::GemmCoord problem_size_0({{max_seq_len}}, {{max_seq_len}}, k0);
  cutlass::gemm::GemmCoord problem_size_1({{max_seq_len}}, {{n1}}, {{max_seq_len}});

  // Assuming BMHD dim ordering for inputs and outputs, like in FHMA style op
  // B = batch size
  // M = sequence len
  // H = num heads
  // D = embedding dims per head
  // --- Tensor shapes:
  // GEMM PROBLEM 0:
  // jagged dims seq_len is in extra <...> brackets with the batch size

  // A=query : [ <batch_size, M0=jagged_seq_len>, num_heads, K0 ]
  // B=key : [ <batch_size, N0=jagged_seq_len>, num_heads, K0 ]
  // C0=bias : [ batch_size, num_heads, max_seq_len, N0 ] # Where the batch size, head and M0 dimension may be broadcasted over
  // GEMM PROBLEM 1:
  // B1=value : [ <batch_size, K1=jagged_seq_len>, num_heads, N1 ]
  // C1=unused:  [ N1 ]
  // D1=output : [ <batch_size, M1=jagged_seq_len>, num_heads, N1 ]

  // Required equalities for grouped / jagged B2B gemm:
  // seq_len = M1 = M0 = N0 = K1;


  typename GroupedB2bGemmBatched::Arguments arguments{
    problem_size_0, // = GemmCoord problem_size_0;
    problem_size_1, // = GemmCoord problem_size_1;
    {static_cast<ElementCompute*>(query), typename GroupedB2bGemmBatched::LayoutA::Stride(num_heads * problem_size_0.k())},    // TensorRef<ElementA const, LayoutA> ref_A0;
    problem_size_0.k(),                                                                                                 // int64_t head_stride_A0;
    num_heads * problem_size_0.m() * problem_size_0.k(),                                                                // int64_t batch_stride_A0;
    {static_cast<ElementCompute*>(key), typename GroupedB2bGemmBatched::LayoutB::Stride(num_heads * problem_size_0.k())},      // TensorRef<ElementB const, LayoutB> ref_B0;
    problem_size_0.k(),                                                                                                 // int64_t head_stride_B0;
    num_heads * problem_size_0.n() * problem_size_0.k(),                                                                // int64_t batch_stride_B0;
    {static_cast<ElementCompute*>(bias), typename GroupedB2bGemmBatched::LayoutC::Stride({{bias_stride_n}})},                  // TensorRef<ElementC const, LayoutC> ref_C0;
    {{bias_stride_mn}},                                                                                                 // int64_t head_stride_C0;
    {{bias_stride_hmn}},                                                                                                // int64_t batch_stride_C0;
    {static_cast<ElementCompute*>(value), typename GroupedB2bGemmBatched::LayoutB1::Stride(num_heads * problem_size_1.n())},   // TensorRef<ElementC const, LayoutC> ref_B1;
    problem_size_1.n(),                                                                                                 // int64_t head_stride_B1;                                                                    //
    num_heads * problem_size_1.n() * problem_size_1.k(),                                                                // int64_t batch_stride_B1;
    {static_cast<ElementCompute*>(nullptr), typename GroupedB2bGemmBatched::LayoutScaleBias::Stride(0)},                       // Not used due to ScaleType::Nothing for output op 1
    0,                                                                                                                  // not used: int64_t head_stride_C1;
    0,                                                                                                                  // not used: int64_t batch_stride_C1;
    {static_cast<ElementOutput*>(output), typename GroupedB2bGemmBatched::LayoutC::Stride(num_heads * problem_size_1.n())},    // TensorRef<ElementC, LayoutC> ref_D1;
    problem_size_1.n(),                                                                                                 // int64_t head_stride_output;
    num_heads * problem_size_1.m() * problem_size_1.n(),                                                                // int64_t batch_stride_output;
    batch_size,                                                                                                         // int batch_count;
    num_heads,                                                                                                          // int num_heads
    static_cast<const {{offset_type}}*>(offsets),                                                                                       // const offset_t *
    {alpha0, beta0, activation_alpha},                                                                                  // typename EpilogueOutputOp0::Params epilogue0;
    {alpha1, beta1},                                                                                                    // typename EpilogueOutputOp1::Params epilogue1;
  };

  GroupedB2bGemmBatched b2b_gemm_op;
  check_status(
    b2b_gemm_op.can_implement(arguments),
    {{max_seq_len}}, k0,
    "Problem sizes are not supported."
  );
  check_status(
    b2b_gemm_op.initialize(arguments),
    {{max_seq_len}}, k0,
    "classic_b2b_bmm initialization failed!"
  );
  check_status(
    b2b_gemm_op(stream),
    {{max_seq_len}}, k0,
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
                   int64_t num_heads,
                   int64_t max_seq_len,
                   int64_t k0,
                   const void *offsets,
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
{{indent}}    {{num_heads}},
{{indent}}    {{max_seq_len}},
{{indent}}    {{k0}},
{{indent}}    {{offsets}},
{{indent}}    stream /* default stream */
{{indent}});
    """
)


@registry.reg("cuda.grouped_classic_b2b_bmm.gen_function")
def classic_b2b_bmm_gen_function(func_attrs: Dict[str, Any]) -> str:
    """the function for generating attention kernel"""
    q, k, v, bias = func_attrs["inputs"]
    max_seq_len = func_attrs["max_seq_len"]
    n1 = v._attrs["shape"][-1]
    jagged_intvar = q._attrs["shape"][0]
    if not isinstance(n1, IntImm):
        raise RuntimeError(f"n1 must be static dim. {func_attrs['name']=}, {n1=}")
    backend_spec = CUDASpec()
    if func_attrs["inputs"][0]._attrs["dtype"] != "float16":
        raise NotImplementedError(
            "only float16 dtype supported for now in classic_b2b_bmm op"
        )
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

    bias_shape = bias._attrs["shape"]
    bias_broadcast = [s == IntImm(1) for s in bias_shape]
    if len(bias_broadcast) == 3:
        # single head case: Add num heads dimension of size 1
        bias_broadcast = [bias_broadcast[0], True, bias_broadcast[1], bias_broadcast[2]]
    assert (
        len(bias_broadcast) == 4
    ), f"Bias shape should be of length 4, got {len(bias_broadcast)=}"

    # Calculate stride expressions for bias tensor
    # Last dimension of bias has implicit stride of 1,
    # so cannot be broadcasted over
    bias_stride_n = "problem_size_0.n()"
    bias_shape_expr = [bias_stride_n]

    # build stride expressions
    if not bias_broadcast[-2]:
        bias_shape_expr.append("problem_size_0.m()")
    bias_stride_mn = "*".join(bias_shape_expr)
    if not bias_broadcast[-3]:
        bias_shape_expr.append("num_heads")
    bias_stride_hmn = "*".join(bias_shape_expr)  # batch stride

    # Strides for broadcasted dimensions are zero
    if bias_broadcast[0]:  # query sequence len stride
        bias_stride_hmn = "0"
    if bias_broadcast[1]:  # head stride
        bias_stride_mn = "0"
    if bias_broadcast[2]:  # query sequence length stride
        bias_stride_n = "0"

    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        elem_accum_type=elem_accum_type,
        max_seq_len=str(max_seq_len),
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
        bias_stride_n=bias_stride_n,
        bias_stride_mn=bias_stride_mn,
        bias_stride_hmn=bias_stride_hmn,
        offset_type=jagged_intvar.offsets_type(),
    )


@registry.reg("cuda.grouped_classic_b2b_bmm.func_decl")
def classic_b2b_bmm_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.grouped_classic_b2b_bmm.func_call")
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

    k0 = q_shape[2]._attrs["name"]

    jagged_intvar = q_shape[0]
    batch_size_var = jagged_intvar.batch_dim()._attrs["name"]
    if len(jagged_intvar.jagged_dims()) != 1:
        raise RuntimeError(
            "Only support 1 jagged dim in grouped_classic_b2b_bmm for now! "
            f"Current jagged intvar: {jagged_intvar}"
        )
    max_seq_len_dim = jagged_intvar.jagged_dims()[0].max_value()
    max_seq_len_var = (
        str(max_seq_len_dim.value())
        if isinstance(max_seq_len_dim, IntImm)
        else max_seq_len_dim._attrs["name"]
    )
    num_heads_var = q_shape[1]._attrs["name"]
    offsets_var_name = f"{jagged_intvar.offsets_var_name()}.data[0]"
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        query=q_name,
        key=k_name,
        value=v_name,
        bias=bias_name,
        batch_size=batch_size_var,
        num_heads=num_heads_var,
        max_seq_len=max_seq_len_var,
        k0=k0,
        indent=indent,
        offsets=offsets_var_name,
        offset_type=jagged_intvar.offsets_type(),
    )
