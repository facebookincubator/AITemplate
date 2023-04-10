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
Attention kernel codegen for CUDA.
"""
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec

# pylint: disable=C0301

CUDA_CHECK = """
#ifndef CUDA_CHECK_ME_ATTN
#define CUDA_CHECK_ME_ATTN(expr, msg)                                          \\
  do {                                                                         \\
    cudaError_t status = (expr);                                               \\
    if (status != cudaSuccess) {                                               \\
      std::cerr << msg << " at " << __FILE__ << ": " << __LINE__ << std::endl; \\
      throw std::runtime_error(cudaGetErrorString(status));                    \\
    }                                                                          \\
  } while (0)
#endif // CUDA_CHECK_ME_ATTN
"""

FUNC_TEMPLATE_KERNEL_FWD = jinja2.Template(
    """
#include <iostream>
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "gemm_kernel_utils.h"

#include "mem_eff_attention/kernel_forward.h"


using namespace gemm_kernel_utils;

{{cuda_check}}

{{func_signature}}
{

    /*
    The code is based on fused_multihead_attention_fixed_seqlen.cu example in CUTLASS repo:
    https://github.com/NVIDIA/cutlass/blob/209faf7b94ce4ba573d27389fb643962e75d0581/examples/41_fused_multi_head_attention/fused_multihead_attention_fixed_seqlen.cu

    problem_sizes0 [b, m, n, k]
    [head_number * batch_size, m, mkv, k0]
    [head_number * batch_size, seq_length_q, seq_length_kv, head_size]

    problem_sizes1
    [head_number * batch_size, m, k1, mkv]
    [head_number * batch_size, seq_length_q, head_size_v, seq_length_kv]

    m = seq_len_q
    n = seq_len_kv
    k = head_size

    Q: B, M, K
    K: B, N, K
    P: B, M, N
    V: B, N, K
    O: B, M, K
    output: bs, seq_len_q, num_head, head_size
    */


    using ArchTag = cutlass::arch::Sm{{arch}};
    constexpr bool kIs64x64 = {{kIs64x64}};
    constexpr bool kSingleValueIteration = {{kSingleValueIteration}};

    // Set grid size
    constexpr int64_t kQueriesPerBlock = kIs64x64 ? 64 : 32;
    constexpr int64_t kKeysPerBlock = kIs64x64 ? 64 : 128;
    if (kIs64x64 && head_size_v > kKeysPerBlock) {
        std::cerr << "WARNING: you will get better performance with `kIs64x64=false`";
    }
    if (kSingleValueIteration && head_size_v > kKeysPerBlock) {
        std::cerr << "ERROR  : Use kSingleValueIteration to keep output in RF. " \
        "This requires to have `head_size <= kKeysPerBlock` " \
        "but head_size_v=" << head_size_v << " and kKeysPerBlock=" << kKeysPerBlock << "";
        return;
    }
    if (!kSingleValueIteration && head_size_v <= kKeysPerBlock) {
        std::cerr << "WARNING: you will get better performance with `kSingleValueIteration=true` (keeps the output in RF rather than GMEM)";
    }

    using GemmType = DefaultGemmType<ArchTag, {{elem_input_type}}>;
    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            {{elem_input_type}},
            {{elem_input_type}},
            {{elem_input_type}}, // ElementC
            float // ElementAccumulator
            >;

    // If the head_size already meets the alignment requirement, then
    // it's safe to mark mem_align to be true to maximize the alignment
    // benefit. Otherwise, assign false to it to use the minimal alignment.
    constexpr const bool mem_align =
        ({{head_size}} % DefaultConfig::kAlignmentA == 0) &&
        ({{head_size}} % DefaultConfig::kAlignmentB == 0);
    using Attention = AttentionKernel<
        {{elem_input_type}}, // scalar_t
        ArchTag,
        mem_align, // memory is aligned
        kQueriesPerBlock,
        kKeysPerBlock,
        kSingleValueIteration
    >;

    typename Attention::Params p;
    {
        // set parameters
        p.query_ptr = static_cast<{{elem_input_type}}*>(query);
        p.key_ptr = static_cast<{{elem_input_type}}*>(key);
        p.value_ptr = static_cast<{{elem_input_type}}*>(value);
        p.logsumexp_ptr = nullptr; // Only needed for bw
        p.output_accum_ptr = nullptr;

        if (!fixed_seq_length_q) {
            p.seqlens_q_ptr = lengths_q;
        }
        if (!fixed_seq_length_kv) {
            p.seqlens_k_ptr = lengths_kv;
        }

        if (Attention::kNeedsOutputAccumulatorBuffer) {
          p.output_accum_ptr = static_cast<float*>(workspace);
        }
        p.output_ptr = static_cast<{{elem_input_type}}*>(output);

        p.num_heads = num_heads;
        p.num_batches = *batch_size;
        p.head_dim = head_size;
        p.head_dim_value = head_size_v;
        p.num_queries = *seq_len_q;
        p.num_keys = *seq_len_kv;
        p.causal = is_causal;


        p.q_strideM = head_size;
        p.k_strideM = head_size;
        p.v_strideM = head_size_v;

        p.q_strideH = p.q_strideM * (*seq_len_q);
        p.k_strideH = p.k_strideM * (*seq_len_kv);
        p.v_strideH = p.v_strideM * (*seq_len_kv);
        p.o_strideH = head_size_v;
        p.q_strideB = p.q_strideH * num_heads;
        p.k_strideB = p.k_strideH * num_heads;
        p.v_strideB = p.v_strideH * num_heads;
        p.o_strideB = head_size_v * (*seq_len_q) * num_heads;
    }

    // launch kernel
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
      std::string error_msg = std::string("Got error: kernel does not support these inputs") +
           " at " + __FILE__ + ": " + std::to_string(__LINE__);
      throw std::runtime_error(error_msg);
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(err);
      return;
    }

}
    """
)


FUNC_TEMPLATE_GROUPED_FMHA = jinja2.Template(
    """
#include <vector>
#include <iostream>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "gemm_kernel_utils.h"
#include "cutlass/gemm/device/gemm_grouped.h"

#include "cutlass/fast_math.h"

#include "default_fmha_grouped.h"

using namespace gemm_kernel_utils;

{{cuda_check}}

{{func_signature}}

{
  /*
  The code is based on fused_multihead_attention_variable_seqlen.cu example in CUTLASS repo:
  https://github.com/NVIDIA/cutlass/blob/209faf7b94ce4ba573d27389fb643962e75d0581/examples/41_fused_multi_head_attention/fused_multihead_attention_variable_seqlen.cu

  problem_sizes0 [b, m, n, k]
  [head_number * batch_size, mq, mkv, k0]
  [head_number * batch_size, seq_length_q, seq_length_kv, head_size]

  problem_sizes1
  [head_number * batch_size, mq, k1, mkv]
  [head_number * batch_size, seq_length_q, head_size_v, seq_length_kv]

  m = seq_len_q
  n = seq_len_kv
  k = head_size

  Q: B, M, K
  K: B, N, K
  P: B, M, N
  V: B, N, K
  O: B, M, K
  output: bs, seq_len_q, num_head, head_size

  Note that the output shape is different from the CUTLASS example.
  */
  //
  int problem_count = (*batch_size) * num_heads;

  /////// Calculate offsets of FMHA arguments in the workspace //////

  int used_memory = 0;
  // Space for problem sizes for each problem
  int size_problem_sizes = sizeof(cutlass::gemm::GemmCoord) * problem_count;
  cutlass::gemm::GemmCoord* problem_sizes_device0 =
      static_cast<cutlass::gemm::GemmCoord*>(workspace + used_memory);
  used_memory += size_problem_sizes;
  cutlass::gemm::GemmCoord* problem_sizes_device1 =
      static_cast<cutlass::gemm::GemmCoord*>(workspace + used_memory);
  used_memory += size_problem_sizes;
  // Space for leading dimensions of tensors in each problem
  int size_ld = sizeof(int64_t) * problem_count;
  int64_t* ldq = static_cast<int64_t*>(workspace + used_memory);
  used_memory += size_ld;
  int64_t* ldk = static_cast<int64_t*>(workspace + used_memory);
  used_memory += size_ld;
  int64_t* ldv = static_cast<int64_t*>(workspace + used_memory);
  used_memory += size_ld;
  int64_t* ldo = static_cast<int64_t*>(workspace + used_memory);
  used_memory += size_ld;

  using ArchTag = cutlass::arch::Sm{{arch}};
  constexpr bool kIs64x64 = {{kIs64x64}};
  constexpr bool kSingleValueIteration = {{kSingleValueIteration}};

  // Set grid size
  constexpr int64_t kQueriesPerBlock = kIs64x64 ? 64 : 32;
  constexpr int64_t kKeysPerBlock = kIs64x64 ? 64 : 128;
  if (kIs64x64 && head_size_v > kKeysPerBlock) {
    std::cerr
        << "WARNING: you will get better performance with `kIs64x64=false`";
  }
  if (kSingleValueIteration && head_size_v > kKeysPerBlock) {
    std::cerr << "ERROR  : Use kSingleValueIteration to keep output in RF. "
                 "This requires to have `head_size <= kKeysPerBlock` "
                 "but head_size_v="
              << head_size_v << " and kKeysPerBlock=" << kKeysPerBlock << "";
    return;
  }
  if (!kSingleValueIteration && head_size_v <= kKeysPerBlock) {
    std::cerr
        << "WARNING: you will get better performance with `kSingleValueIteration=true` (keeps the output in RF rather than GMEM)";
  }

  using GemmType = DefaultGemmType<ArchTag, {{elem_input_type}}>;
  using OpClass = typename GemmType::OpClass;
  using DefaultConfig =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OpClass,
          ArchTag,
          {{elem_input_type}},
          {{elem_input_type}},
          {{elem_input_type}}, // ElementC
          float // ElementAccumulator
          >;

  // If the head_size already meets the alignment requirement, then
  // it's safe to mark mem_align to be true to maximize the alignment
  // benefit. Otherwise, assign false to it to use the minimal alignment.
  constexpr const bool mem_align = ({{head_size}} % DefaultConfig::kAlignmentA == 0) &&
      ({{head_size}} % DefaultConfig::kAlignmentB == 0);

  cutlass::gemm::kernel::GroupScheduleMode const GroupScheduleMode_ =
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly;

  using AttentionKernel = typename cutlass::gemm::kernel::DefaultFMHAGrouped<
      {{elem_input_type}}, // scalar_t
      ArchTag,
      mem_align,
      kQueriesPerBlock,
      kKeysPerBlock,
      kSingleValueIteration,
      GroupScheduleMode_>::FMHAKernel;
  using Attention = cutlass::gemm::device::GemmGrouped<AttentionKernel>;

  if (({{head_size}} % AttentionKernel::kAlignmentQ != 0) ||
      ({{head_size}} % AttentionKernel::kAlignmentK != 0)) {
    std::cerr << "Error at " << __FILE__ << ": " << __LINE__ <<
        "head_size not aligned! head_size has to be divisible by " <<
        std::to_string(AttentionKernel::kAlignmentQ) << " and " <<
        std::to_string(AttentionKernel::kAlignmentK) + ", but got {{head_size}}."
        << std::endl;
    return;
  }

  // If we need a separate buffer for output accumulation
  static bool const kNeedsOutputAccumulatorBuffer =
      Attention::GemmKernel::kNeedsOutputAccumulatorBuffer;

  // Problem sizes with actual sequence lengths
  std::vector<cutlass::gemm::GemmCoord> problem_sizes0;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes1;
  // Problem sizes with "full" sequence lengths
  std::vector<cutlass::gemm::GemmCoord> problem_sizes0_full;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes1_full;

  problem_sizes0.reserve(problem_count);
  problem_sizes1.reserve(problem_count);
  problem_sizes0_full.reserve(problem_count);
  problem_sizes1_full.reserve(problem_count);

  // Copy sequence lengths from device to host, if they are not fixed
  std::vector<int> mq_real_buf; // Target sequence lengths
  std::vector<int> mkv_real_buf; // Source sequence lengths
  if (!fixed_seq_length_q) {
    mq_real_buf.resize(*batch_size);
    CUDA_CHECK_ME_ATTN(
      cudaMemcpyAsync(
        mq_real_buf.data(), lengths_q, *batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream),
      "Error when copying target sequence lengths from device!");
  }
  if (!fixed_seq_length_kv) {
    mkv_real_buf.resize(*batch_size);
    CUDA_CHECK_ME_ATTN(
      cudaMemcpyAsync(
        mkv_real_buf.data(), lengths_kv,  *batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream),
        "Error when copying source sequence lengths from device!");
  }
  if (!fixed_seq_length_q || !fixed_seq_length_kv) {
    CUDA_CHECK_ME_ATTN(cudaStreamSynchronize(stream),
          "Error when synchronizing stream after copying sequence lengths from device!");
  }

  int mq_full = *seq_len_q;
  int mkv_full = *seq_len_kv;

  for (int i = 0; i < *batch_size; ++i) {
    // Problems belonging to the same batch share the same seq len
    // Source sequence length
    int mkv_real = fixed_seq_length_kv ? mkv_full : mkv_real_buf.at(i);
    // Target sequence length
    int mq_real = fixed_seq_length_q ? mq_full : mq_real_buf.at(i);

    int k0 = head_size;
    int k1 = head_size_v;

    // Create sizes of two GEMM problems for each of batch_size * num_heads attention problems
    for (int j = 0; j < num_heads; ++j) {
      cutlass::gemm::GemmCoord problem0(mq_real, mkv_real, k0);
      cutlass::gemm::GemmCoord problem1(mq_real, k1, mkv_real);
      problem_sizes0.push_back(problem0);
      problem_sizes1.push_back(problem1);

      cutlass::gemm::GemmCoord problem0_full(mq_full, mkv_full, k0);
      cutlass::gemm::GemmCoord problem1_full(mq_full, k1, mkv_full);
      problem_sizes0_full.push_back(problem0_full);
      problem_sizes1_full.push_back(problem1_full);
    }
  }

  // Move problem sizes to the device
  CUDA_CHECK_ME_ATTN(
    cudaMemcpyAsync(
      problem_sizes_device0,
      problem_sizes0.data(),
      size_problem_sizes,
      cudaMemcpyHostToDevice,
      stream),
    "Error when copying problem sizes 0 to device!");
  CUDA_CHECK_ME_ATTN(
    cudaMemcpyAsync(
      problem_sizes_device1,
      problem_sizes1.data(),
      size_problem_sizes,
      cudaMemcpyHostToDevice,
      stream),
    "Error when copying problem sizes 1 to device!");

  // Offsets of input, buffer, and output matrices in memory
  std::vector<int64_t> offset_Q_full;
  std::vector<int64_t> offset_K_full;
  std::vector<int64_t> offset_V_full;
  std::vector<int64_t> offset_O_full;

  // Leading dimensions of matrices of each problem
  std::vector<int64_t> ldq_host;
  std::vector<int64_t> ldk_host;
  std::vector<int64_t> ldv_host;
  std::vector<int64_t> ldo_host;
  ldq_host.resize(problem_count);
  ldk_host.resize(problem_count);
  ldv_host.resize(problem_count);
  ldo_host.resize(problem_count);

  using scalar_t = typename Attention::GemmKernel::scalar_t;
  using accum_t = typename Attention::GemmKernel::accum_t;
  using output_t = typename Attention::GemmKernel::output_t;
  using output_accum_t = typename Attention::GemmKernel::output_accum_t;

  using ElementQ = scalar_t;
  using ElementK = scalar_t;
  using ElementP = accum_t;
  using ElementAccumulator = accum_t;
  using ElementV = scalar_t;
  using ElementO = output_t;
  using ElementOAccum = output_accum_t;

  // Arrays of pointers to matrices for each problem
  int size_ptrs = sizeof(ElementQ*) * problem_count;
  ElementQ** ptr_Q = static_cast<ElementQ**>(workspace + used_memory);
  used_memory += size_ptrs;
  ElementK** ptr_K = static_cast<ElementK**>(workspace + used_memory);
  used_memory += size_ptrs;
  ElementV** ptr_V = static_cast<ElementV**>(workspace + used_memory);
  used_memory += size_ptrs;
  ElementO** ptr_O = static_cast<ElementO**>(workspace + used_memory);
  used_memory += size_ptrs;
  ElementOAccum** ptr_O_accumulate =
      static_cast<ElementOAccum**>(workspace + used_memory);
  used_memory += size_ptrs;

  int64_t total_elements_Q_full = 0;
  int64_t total_elements_K_full = 0;
  int64_t total_elements_V_full = 0;
  //int64_t total_elements_O_full = 0;
  int64_t total_elements_O_at_batch_start = 0;

  // Pointers to matrices and leading dimensions for each problem are first
  // formed on the host and then copied to the device.

  for (int32_t i_batch = 0; i_batch < *batch_size; ++i_batch) {
    int64_t total_elements_O_in_current_batch = 0;
    for (int32_t i_heads = 0; i_heads < num_heads; ++i_heads) {
      int64_t i = i_batch * num_heads + i_heads;
      auto problem0 = problem_sizes0.at(i);
      auto problem1 = problem_sizes1.at(i);

      auto problem0_full = problem_sizes0_full.at(i);
      auto problem1_full = problem_sizes1_full.at(i);

      /*
      Below we specify leading dimensions of each matix, assuming the following
      layouts and dimensions:

      using LayoutQ = cutlass::layout::RowMajor;
      using LayoutK = cutlass::layout::ColumnMajor;
      using LayoutV = cutlass::layout::RowMajor;
      using LayoutO = cutlass::layout::RowMajor;

      ldq_host.at(i) = LayoutQ::packed({problem0.m(), problem0.k()}).stride(0);
      ldk_host.at(i) = LayoutK::packed({problem0.k(), problem0.n()}).stride(0);
      ldv_host.at(i) = LayoutV::packed({problem1.k(), problem1.n()}).stride(0);
      ldo_host.at(i) = LayoutO::packed({problem1.m(), problem1.n()}).stride(0);
      */

      ldq_host.at(i) = problem0.k(); // K, rowmajor
      ldk_host.at(i) = problem0.k(); // K, columnmajor
      ldv_host.at(i) = problem1.n(); // K, rowmajor
      // Since we want output in shape [b, seq_len_q, num_head, head_size] and
      // not [b, num_head, seq_len_q, head_size], ldo is different from the
      // CUTLASS example. Each next row of O is now separated from the previous
      // one by head_size * num_heads, instead of just head_size.
      ldo_host.at(i) = problem1.n() * num_heads; // K * num_heads, rowmajor

      offset_Q_full.push_back(total_elements_Q_full);
      offset_K_full.push_back(total_elements_K_full);
      offset_V_full.push_back(total_elements_V_full);
      // To write the output in shape [b, seq_len_q, num_head, head_size]
      // instead of [b, num_head, seq_len_q, head_size], we place rows of O
      // from the same batch but different heads at stride head_size from
      // each other (and not seq_len_q * head_size).
      offset_O_full.push_back(
          total_elements_O_at_batch_start + i_heads * problem1_full.n());

      int64_t elements_Q_full = problem0_full.m() * problem0_full.k();
      int64_t elements_K_full = problem0_full.k() * problem0_full.n();
      int64_t elements_V_full = problem1_full.k() * problem1_full.n();
      int64_t elements_O_full = problem1_full.m() * problem1_full.n();

      total_elements_Q_full += elements_Q_full;
      total_elements_K_full += elements_K_full;
      total_elements_V_full += elements_V_full;
      total_elements_O_in_current_batch += elements_O_full;
    }
    total_elements_O_at_batch_start += total_elements_O_in_current_batch;
  }

  CUDA_CHECK_ME_ATTN(
    cudaMemcpyAsync(ldq, ldq_host.data(), size_ld, cudaMemcpyHostToDevice, stream),
    "Error when copying leading dimensions of Q matrices to device!");
  CUDA_CHECK_ME_ATTN(
    cudaMemcpyAsync(ldk, ldk_host.data(), size_ld, cudaMemcpyHostToDevice, stream),
    "Error when copying leading dimensions of K matrices to device!");
  CUDA_CHECK_ME_ATTN(
    cudaMemcpyAsync(ldv, ldv_host.data(), size_ld, cudaMemcpyHostToDevice, stream),
    "Error when copying leading dimensions of V matrices to device!");
  CUDA_CHECK_ME_ATTN(
    cudaMemcpyAsync(ldo, ldo_host.data(), size_ld, cudaMemcpyHostToDevice, stream),
    "Error when copying leading dimensions of O matrices to device!");

  // Buffer for output accumulation, if necessary
  float* accum_ptr = static_cast<float*>(workspace + used_memory);

  std::vector<ElementQ*> ptr_Q_host(problem_count);
  std::vector<ElementK*> ptr_K_host(problem_count);
  std::vector<ElementV*> ptr_V_host(problem_count);
  std::vector<ElementO*> ptr_O_host(problem_count);
  std::vector<ElementOAccum*> ptr_O_accumulate_host(problem_count);

  for (int32_t i = 0; i < problem_count; ++i) {
    ptr_Q_host.at(i) = static_cast<ElementQ*>(query) + offset_Q_full.at(i);
    ptr_K_host.at(i) = static_cast<ElementK*>(key) + offset_K_full.at(i);
    ptr_V_host.at(i) = static_cast<ElementV*>(value) + offset_V_full.at(i);
    ptr_O_host.at(i) = static_cast<ElementO*>(output) + offset_O_full.at(i);

    if (kNeedsOutputAccumulatorBuffer) {
      ptr_O_accumulate_host.at(i) =
        static_cast<ElementOAccum*>(accum_ptr) + offset_O_full.at(i);
    }
  }
  CUDA_CHECK_ME_ATTN(
    cudaMemcpyAsync(
      ptr_Q, ptr_Q_host.data(), size_ptrs, cudaMemcpyHostToDevice, stream),
    "Error when copying pointers to Q matrices to device!");
  CUDA_CHECK_ME_ATTN(
    cudaMemcpyAsync(
      ptr_K, ptr_K_host.data(), size_ptrs, cudaMemcpyHostToDevice, stream),
    "Error when copying pointers to K matrices to device!");
  CUDA_CHECK_ME_ATTN(
    cudaMemcpyAsync(
      ptr_V, ptr_V_host.data(), size_ptrs, cudaMemcpyHostToDevice, stream),
    "Error when copying pointers to V matrices to device!");
  CUDA_CHECK_ME_ATTN(
    cudaMemcpyAsync(
      ptr_O, ptr_O_host.data(), size_ptrs, cudaMemcpyHostToDevice, stream),
    "Error when copying pointers to O matrices to device!");

  if (kNeedsOutputAccumulatorBuffer) {
    CUDA_CHECK_ME_ATTN(
      cudaMemcpyAsync(
        ptr_O_accumulate,
        ptr_O_accumulate_host.data(),
        size_ptrs,
        cudaMemcpyHostToDevice,
        stream),
      "Error when copying pointers to accumulator buffers to device!");
  }

  int threadblock_count =
      Attention::sufficient(problem_sizes1.data(), problem_count);
  typename Attention::Arguments args(
      problem_sizes_device0,
      problem_sizes_device1,
      problem_count,
      threadblock_count,
      ptr_Q,
      ptr_K,
      nullptr, // ptr_P isn't used by grouped FMHA
      ptr_V,
      ptr_O,
      ptr_O_accumulate,
      ldq,
      ldk,
      nullptr, // ldp isn't used by grouped FMHA
      ldv,
      ldo,
      is_causal,
      problem_sizes1.data());

  Attention fmha;
  cutlass::Status status = fmha.initialize(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize CUTLASS Grouped FMHA kernel."
              << std::endl;
    return;
  }

  // Run the grouped FMHA object
  status = fmha.run(stream);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to run CUTLASS Grouped FMHA kernel." << std::endl;
    return;
  }
}

    """
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   void* query,
                   void* key,
                   void* value,
                   int64_t* batch_size,
                   int64_t* seq_len_kv,
                   int64_t* seq_len_q,
                   int num_heads,
                   int head_size,
                   int head_size_v,
                   float p_dropout,
                   float softmax_scale,
                   bool is_causal,
                   bool fixed_seq_length_kv,
                   int32_t* lengths_kv,
                   bool fixed_seq_length_q,
                   int32_t* lengths_q,
                   void* workspace,
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
{{indent}}    {{query}}, {{key}}, {{value}},
{{indent}}    {{batch_size}},
{{indent}}    {{seq_len_kv}},
{{indent}}    {{seq_len_q}},
{{indent}}    {{num_heads}},
{{indent}}    {{head_size}},
{{indent}}    {{head_size_v}},
{{indent}}    {{p_dropout}},
{{indent}}    {{softmax_scale}},
{{indent}}    {{is_causal}},
{{indent}}    {{fixed_seq_length_kv}},
{{indent}}    {{lengths_kv}},
{{indent}}    {{fixed_seq_length_q}},
{{indent}}    {{lengths_q}},
{{indent}}    global_workspace_,
{{indent}}    stream /* default stream */
{{indent}});
    """
)


@registry.reg("cuda.mem_eff_attention.gen_function")
def mem_eff_attention_gen_function(func_attrs: Dict[str, Any]) -> str:
    """the function for generating attention kernel"""
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    if func_attrs["use_grouped_fmha"]:
        func_template = FUNC_TEMPLATE_GROUPED_FMHA
    else:
        func_template = FUNC_TEMPLATE_KERNEL_FWD

    return func_template.render(
        elem_input_type=elem_input_type,
        head_size=func_attrs["head_size"],
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        kIs64x64="true" if func_attrs["head_size"] <= 64 else "false",
        kSingleValueIteration="true" if func_attrs["head_size"] <= 128 else "false",
        cuda_check=CUDA_CHECK,
        arch=func_attrs["arch"],
    )


@registry.reg("cuda.mem_eff_attention.func_decl")
def mem_eff_attention_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.mem_eff_attention.func_call")
def mem_eff_attention_gen_function_call(func_attrs, indent="  "):
    """the function for generating a function call for attention"""
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) in [3, 4, 5]

    output_name = func_attrs["outputs"][0]._attrs["name"]

    q_name = func_attrs["inputs"][0]._attrs["name"]
    k_name = func_attrs["inputs"][1]._attrs["name"]
    v_name = func_attrs["inputs"][2]._attrs["name"]

    variable_seq_length_kv = func_attrs["variable_seq_length_kv"]
    variable_seq_length_q = func_attrs["variable_seq_length_q"]

    lengths_name_kv = "nullptr"
    lengths_name_q = "nullptr"

    if variable_seq_length_kv:
        assert len(func_attrs["inputs"]) > 3
        lengths_name_kv = func_attrs["inputs"][3]._attrs["name"]
    if variable_seq_length_q:
        idx_len_q = 3 + variable_seq_length_kv
        assert len(func_attrs["inputs"]) > idx_len_q
        lengths_name_q = func_attrs["inputs"][idx_len_q]._attrs["name"]

    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    batch_size = "&" + xshape[0]._attrs["name"]
    seq_len_q = "&" + xshape[2]._attrs["name"]

    num_heads = x._attrs["shape"][1]._attrs["values"][0]
    head_size = x._attrs["shape"][3]._attrs["values"][0]
    p_dropout = func_attrs["dropout"]
    is_causal = func_attrs["causal"]
    softmax_scale = head_size ** (-0.5)

    v = func_attrs["inputs"][2]
    vshape = v._attrs["shape"]
    seq_len_kv = "&" + vshape[2]._attrs["name"]

    head_size_v = v._attrs["shape"][3]._attrs["values"][0]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        query=q_name,
        key=k_name,
        value=v_name,
        batch_size=batch_size,
        seq_len_kv=seq_len_kv,
        seq_len_q=seq_len_q,
        num_heads=num_heads,
        head_size=head_size,
        head_size_v=head_size_v,
        p_dropout=p_dropout,
        softmax_scale=softmax_scale,
        is_causal="true" if is_causal else "false",
        fixed_seq_length_kv="false" if variable_seq_length_kv else "true",
        lengths_kv=f"static_cast<int32_t*>({lengths_name_kv})",
        fixed_seq_length_q="false" if variable_seq_length_q else "true",
        lengths_q=f"static_cast<int32_t*>({lengths_name_q})",
        indent=indent,
    )
