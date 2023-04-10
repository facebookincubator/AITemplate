//  Copyright (c) Meta Platforms, Inc. and affiliates.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

#ifndef CONCATENATE_FAST_KERNEL
#define CONCATENATE_FAST_KERNEL

/////////////////////////////////////////////////////////////
// some standard includes
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

// fast tanh for the most resent hardware
#include <cutlass/fast_math.h>

////////////////////////////////////////////////////////////
// I'm trying to do as much C++ as possible in order to simplify
//   the debugging without any Python + Jinja2.

/*
////////////////////////////////////////////////////////////////////////////////////////
// Baseline C++ implementation looks the following.
// Please note that it does not include masking or TensorAccessor-objects.
// It's just a plain tensor concatenation code.
////////////////////////////////////////////////////////////////////////////////////////

// a crude representation of a tensor
struct Tensor {
    std::vector<int64_t> sizes;
    std::vector<float> data;
};

// contains a list of tensors that need to be concatenated
struct TestCase {
    std::vector<Tensor> inputs;
};

// concatDim >= 0
Tensor ConcatKernelDimN(const TestCase & tc, int64_t concatDim) {
    // this is the output tensor
    Tensor output;

    // copy sizes from the first input tensor
    output.sizes = tc.inputs[0].sizes;

    // compute the resulting number of elements for dim=concatDim
    int64_t nTotalElementsAtConcatDim = 0;
    for (const auto & tensor : tc.inputs) {
        nTotalElementsAtConcatDim += tensor.sizes[concatDim];
    }

    // save this new dimension
    output.sizes[concatDim] = nTotalElementsAtConcatDim;

    // concat all the data.
    // the overall logic is the following: we need to perform
    //   n operations, and on every iteration one copies the number
    //   of elements proportional to ncopy.

    int64_t n = 1;
    for (int64_t i = 0; i < concatDim; i++) {
        n *= output.sizes[i];
    }

    int64_t ncopy = 1;
    for (int64_t i = concatDim + 1; i < output.sizes.size(); i++) {
        ncopy *= output.sizes[i];
    }

    for (int64_t i = 0; i < n; i++) {
        for (const auto & tensor : tc.inputs) {
            // add a new chunk to the end of the output tensor data container
            output.data.insert(
                output.data.end(),
                tensor.data.cbegin() +
                    i * tensor.sizes[concatDim] * ncopy,
                tensor.data.cbegin() +
                    (i + 1) * tensor.sizes[concatDim] * ncopy);
        }
    }

    // done
    return output;
}
*/

////////////////////////////////////////////////////////////
// Here go the facilities that are responsible for post-processing,
//   such as applying tanh on top of values on a concatenated tensor.

// does no processing
template <typename DataT>
struct NoopTransform {
  using data_type = DataT;
  __device__ inline data_type operator()(const data_type value) {
    return value;
  }
};

// does tanh()
template <typename DataT>
struct TanhTransform {
  using data_type = DataT;
};

template <>
struct TanhTransform<float> {
  using data_type = float;
  __device__ inline float operator()(const float value) {
    // return tanhf(value);
    return cutlass::fast_tanh(value);
  }
};

template <>
struct TanhTransform<half> {
  using data_type = half;
  __device__ inline half operator()(const half value) {
    // return __float2half(tanhf(__half2float(value)));
    return cutlass::fast_tanh(value);
  }
};

template <>
struct TanhTransform<__nv_bfloat16> {
  using data_type = __nv_bfloat16;
  __device__ inline __nv_bfloat16 operator()(const __nv_bfloat16 value) {
    return __float2bfloat16(tanhf(__bfloat162float(value)));
  }
};

// CUDA-based hardware benefits not only from coalescing, but from
//   reading/writing in memory-aligned chunks. This template defined
//   the type which is used for reading/writing. For example, float2
//   and float4 are built-in CUDA types, so compiler will assume that
//   these types are address-aligned and issue 64-bit or 128-bit
//   read operations instead of 32-bit one.

template <int32_t AlignmentInBytes>
struct RWChunkTrait {};

template <>
struct RWChunkTrait<2> {
  using chunk_type = half;
};
template <>
struct RWChunkTrait<4> {
  using chunk_type = float;
};
template <>
struct RWChunkTrait<8> {
  using chunk_type = float2;
};
template <>
struct RWChunkTrait<16> {
  using chunk_type = float4;
};
// This one is introduced, despite current CUDA hardware
// is capable of doing only 128 bit transfers. Benchmarks
// showed that doing 2x float4 is faster than 1x or 4x.
// Maybe, 3x needs to be benchmarked as well.
template <>
struct RWChunkTrait<32> {
  using chunk_type = float4;
};

// This is a piece of tensor data that is going to be read or written.
//   The purpose is to organize read/write operations is the way to
//   maximize the number of aligned 128 bit/64 bit/32 bit reads.
// For example, AlignedChunk<half, 32> means that:
//   * our tensor works with element of datatype half
//   * it is guaranteed that it is possible to read 16 contiguous elements
//   * the address of the first element is aligned to 32 bytes
// Thus, float4 will be deduced as an underlying data type for interacting
//   with the global memory and the read/write operations will be
//   performed via 128 bit reads.
template <typename DataT, int32_t AlignmentInBytes>
struct alignas(AlignmentInBytes) AlignedChunk {
  // This is the type of the data of a tensor elements. Most likely, it is
  //   float, half or bf16.
  using data_type = DataT;
  static constexpr int32_t NElements = AlignmentInBytes / sizeof(data_type);

  // This is the type used for interacting with the global memory.
  using chunk_type = typename RWChunkTrait<AlignmentInBytes>::chunk_type;
  static constexpr int32_t NChunkElements =
      AlignmentInBytes / sizeof(chunk_type);

  using self_type = AlignedChunk<DataT, AlignmentInBytes>;

  // the data itself
  union {
    // this is for accessing and applying transformations like tanh()
    data_type data[NElements];
    // this is for reading/writing
    chunk_type chunks[NChunkElements];
  } holder;

  // read from the global memory
  __device__ inline void load(const void* const src) {
    auto srcMod = reinterpret_cast<const chunk_type*>(src);
#pragma unroll NChunkElements
    for (int32_t i = 0; i < NChunkElements; i++) {
      holder.chunks[i] = srcMod[i];
    }
  }

  // transform the elements
  template <typename TransformT>
  __device__ inline void transform() {
    TransformT transform;

#pragma unroll NElements
    for (int32_t i = 0; i < NElements; i++) {
      holder.data[i] = transform(holder.data[i]);
    }
  }

  // write to the global memory
  __device__ inline void store(void* const dst) const {
    auto dstMod = reinterpret_cast<chunk_type*>(dst);
#pragma unroll NChunkElements
    for (int32_t i = 0; i < NChunkElements; i++) {
      dstMod[i] = holder.chunks[i];
    }
  }

  // This operation is needed to merge AlignedChunk items.
  // Say, we read an input tensor as AlignedChunk<half, 8>
  //   and we write into an output tensor as AlignedChunk<half, 32>.
  //   So, it is possible to merge AlignedChunk<half, 8>[4] into
  //   a single AlignedChunk<half, 32>.
  // The compiler does nothing but just the register reassignment.
  template <typename OtherChunkT, int32_t M>
  __device__ inline void copyFrom(const OtherChunkT other[M]) {
    // TODO: this function needs to perform a type conversion
    //   if the types are different. Via if constexpr, I suppose.
    // Say, the input tensor uses half data type, and the output tensor
    //   uses float one.
    static_assert(std::is_same_v<typename OtherChunkT::data_type, data_type>);

    const data_type* otherAddr = (const data_type*)(other);
#pragma unroll NElements
    for (int32_t i = 0; i < NElements; i++) {
      holder.data[i] = otherAddr[i];
    }
  }
};

// TODO: This can be improved to have less read/write operations.
// As of now, AlignedChunk is read using the same primitive
// type all the time. Technically, it can be reorganized to have
// multiple underlying chunk types.
// Say, something like AlignedChunkPlusPlus<half, 32, 8, 16, 8>
//  that reads 8b + 16b + 8b may be used in future instead of
//  4x AlignedChunk<half, 8> that reads 8b + 8b + 8b + 8b,
//  if the alignment allows it.

// A simple 1D fixed-size array.
// One needs to be cautions about pointers, because nvcc compiler
//   does not apply __restrict correctly for the array of pointers
//   or structs.
template <typename DataT, int32_t N>
struct FSArray {
  DataT data[N];
};

// clang-format off

// The most general kernel that supports all the features, but the slowest one.
// The kernel is organized in the form so that every thread writes a single
//   ChunkOutputT value to the output tensor.
// A single write op into the output tensor is supported with the
//   one or multiple read ops from one of the input tensors.
//
// The template parameters are the following:
// * ChunkOutputT is an aligned data type which is used for
//   writing into the output tensor. We want this one to be as large as possible
//   in order to minimize the number of writing ops.
//   It is guaranteed that all the writing ops are aligned for the addresses.
//   AlignedChunk<T, M> is used for this.
// * ChunkInputT is an aligned data type which is used for
//   reading from input tensors. We want this one to be as large as possible
//   in order to minimize the number of reading ops.
//   It is guaranteed that all the reading ops are aligned for the addresses
//   of all input tensors.
//   Also, sizeof(ChunkInputT) <= sizeof(ChunkOutputT)
//   AlignedChunk<T, M> is used for this.
// * IndexT is a pointer size type. It is either int32_t or int64_t.
//   It is beneficial to use int32_t unless super-large tensors are used.
// * NInputTensors is a number of input tensors.
template <
    typename ChunkInputT,
    typename ChunkOutputT,
    typename IndexT,
    int32_t NInputTensors,
    typename TransformT>
__global__ void ConcatKernelGeneralized(
    // pointers to the data of input tensors
    const FSArray<const typename ChunkInputT::data_type*, NInputTensors>
        inputDatas,
    // TensorAccessor.original_total_elements_from_stride_dim values
    //   for input tensors, Please reference tensor_accessor.cuh file.
    const FSArray<IndexT, NInputTensors> originalTE,
    // TensorAccessor.actual_total_elements_from_stride_dim values
    //   for input tensors. Please reference tensor_accessor.cuh file.
    const FSArray<IndexT, NInputTensors> actualTE,
    // The sum of input tensor sizes for dim=concatDim.
    //   This equals to the output tensor size for dim=concatDim.
    const IndexT outputSizeAtConcatDimMultipliedByNCopy,
    // The stride for output tensor for dim=concatDim.
    //   This equals to outputSizeAtConcatDim if there were no masked inputs.
    const IndexT strideMultipliedByNCopy,
    // Postfix sum of tensor sizes for dim=concatDim.
    //   All the values are were multiplied by nCopy.
    const FSArray<IndexT, NInputTensors> concatDimPostfixSumMultipliedByNCopy,
    // Every input tensor is expected to get written on a certain
    //   offset of the output tensor. These are needed if masks are used,
    //   otherwise ones may be skipped.
    const FSArray<IndexT, NInputTensors> outputConcatDimOffsetsMultipliedByNCopy,
    // Where to write the output to.
    typename ChunkOutputT::data_type* const __restrict outputData,
    // the total amount of elements to populate in the output tensor.
    const IndexT numOutputElements) {
  // some typedefs
  using input_data_type = typename ChunkInputT::data_type;
  using output_data_type = typename ChunkOutputT::data_type;

  // put the input values into shared memory.
  __shared__ IndexT shared_concatDimPostfixSumMultipliedByNCopy[NInputTensors];
  __shared__ IndexT shared_originalTE[NInputTensors];
  __shared__ IndexT shared_actualTE[NInputTensors];
  __shared__ IndexT shared_outputConcatDimOffsetsMultipliedByNCopy[NInputTensors];
  __shared__ const input_data_type* shared_inputDatas[NInputTensors];

  if (threadIdx.x == 0) {
#pragma unroll NInputTensors
    for (int32_t i = 0; i < NInputTensors; i++) {
      shared_concatDimPostfixSumMultipliedByNCopy[i] = concatDimPostfixSumMultipliedByNCopy.data[i];
    }
  } else if (threadIdx.x == 1) {
#pragma unroll NInputTensors
    for (int32_t i = 0; i < NInputTensors; i++) {
      shared_originalTE[i] = originalTE.data[i];
    }
  } else if (threadIdx.x == 2) {
#pragma unroll NInputTensors
    for (int32_t i = 0; i < NInputTensors; i++) {
      shared_actualTE[i] = actualTE.data[i];
    }
  } else if (threadIdx.x == 3) {
#pragma unroll NInputTensors
    for (int32_t i = 0; i < NInputTensors; i++) {
      shared_outputConcatDimOffsetsMultipliedByNCopy[i] = outputConcatDimOffsetsMultipliedByNCopy.data[i];
    }
  } else if (threadIdx.x == 4) {
#pragma unroll NInputTensors
    for (int32_t i = 0; i < NInputTensors; i++) {
      shared_inputDatas[i] = inputDatas.data[i];
    }
  }

  __syncthreads();

  // Every thread handles a single ChunkOutputT element, or
  // ChunkOutputT::NElements of an output tensor;
  const IndexT tid = ((IndexT)blockIdx.x * (IndexT)blockDim.x + threadIdx.x) *
      ChunkOutputT::NElements;
  if (tid >= numOutputElements) {
    return;
  }

  // calculate the location of the output tensor a current thread is writing to:
  //   outputRowIdx is the row
  //   outputColumnIdx is the column

  const IndexT outputRowIdx = tid / (outputSizeAtConcatDimMultipliedByNCopy);

  // Find the input tensor to use
  const IndexT offset = tid % (outputSizeAtConcatDimMultipliedByNCopy);
  int32_t inputTensorIdx = 0;
#pragma unroll NInputTensors
  for (int32_t i = 1; i < NInputTensors; i++) {
    inputTensorIdx = (offset < shared_concatDimPostfixSumMultipliedByNCopy[i - 1]) ? inputTensorIdx : i;
  }

  const IndexT subtract = (inputTensorIdx == 0) ? 0 : shared_concatDimPostfixSumMultipliedByNCopy[inputTensorIdx - 1];
  const IndexT outputColumnIdx = offset - subtract;

  // Load the TensorAccessor.original_total_elements_from_stride_dim and
  //   TensorAccessor.actual_total_elements_from_stride_dim values
  //   for the current tensor.

  IndexT originalTEValue = shared_originalTE[inputTensorIdx];
  IndexT actualTEValue = shared_actualTE[inputTensorIdx];

  // Calculate the contiguous access index of the current input tensor
  IndexT readPositionContiguous =
        (inputTensorIdx == 0) ?
        shared_concatDimPostfixSumMultipliedByNCopy[0] :
        (shared_concatDimPostfixSumMultipliedByNCopy[inputTensorIdx] - shared_concatDimPostfixSumMultipliedByNCopy[inputTensorIdx - 1]);
  readPositionContiguous = outputRowIdx * readPositionContiguous + outputColumnIdx;

  // Get the pointer to data of the input tensor
  const input_data_type* __restrict inputData = shared_inputDatas[inputTensorIdx];

  // Ok, what's the number of read operations from an input tensor
  //   needed for a single write operation for the output tensor?
  constexpr int32_t N_READ_OPS = ChunkOutputT::NElements / ChunkInputT::NElements;

  // Allocate a temporary buffer and perform all these read ops
  ChunkInputT inputValues[N_READ_OPS];

  // don't merge these two branches, it is slower
  if (actualTEValue != originalTEValue) {
    TensorAccessor inputTA{0, false, 0, originalTEValue, actualTEValue};

#pragma unroll N_READ_OPS
    for (int32_t i = 0; i < N_READ_OPS; i++) {
      // each read op reads ChunkInputT::NElements elements from an input tensor
      const input_data_type* const __restrict srcp =
        inputTA.template get<const input_data_type, const input_data_type>(
          inputData,
          readPositionContiguous + i * ChunkInputT::NElements
        );

      inputValues[i].load(srcp);
    }
  }
  else {
    TensorAccessor inputTA{0, true, 0, 0, 0};

#pragma unroll N_READ_OPS
    for (int32_t i = 0; i < N_READ_OPS; i++) {
      // each read op reads ChunkInputT::NElements elements from an input tensor
      const input_data_type* const __restrict srcp =
        inputTA.template get<const input_data_type, const input_data_type>(
          inputData,
          readPositionContiguous + i * ChunkInputT::NElements
        );

      inputValues[i].load(srcp);
    }
  }

  // combine all the input data
  ChunkOutputT outputChunk;
  outputChunk.template copyFrom<ChunkInputT, N_READ_OPS>(inputValues);

  // transform
  outputChunk.template transform<TransformT>();

  // Find a destination offset for the output tensor
  IndexT outputOffsetMultipliedByNCopy = shared_outputConcatDimOffsetsMultipliedByNCopy[inputTensorIdx];
  ChunkOutputT* const __restrict outputAddr = reinterpret_cast<ChunkOutputT*>(outputData);

  // perform a write operation
  const IndexT outputWritePosition = outputRowIdx * strideMultipliedByNCopy + outputColumnIdx;

  const IndexT op = outputOffsetMultipliedByNCopy + outputWritePosition;
  outputChunk.store(outputAddr + op / ChunkOutputT::NElements);
}

// utility functions
size_t getAlignment(const void* const inputData) {
    uintptr_t ptr = (uintptr_t)(inputData);
    if ((ptr % 32) == 0) { return 32; }
    if ((ptr % 16) == 0) { return 16; }
    if ((ptr % 8) == 0) { return 8; }
    if ((ptr % 4) == 0) { return 4; }
    if ((ptr % 2) == 0) { return 2; }

    return 1;
}

size_t getAlignment(const size_t n) {
    if ((n % 32) == 0) { return 32; }
    if ((n % 16) == 0) { return 16; }
    if ((n % 8) == 0) { return 8; }
    if ((n % 4) == 0) { return 4; }
    if ((n % 2) == 0) { return 2; }

    return 1;
}

// clang-format on

//
template <
    typename ChunkInputT,
    typename ChunkOutputT,
    typename IndexT,
    int32_t NInputTensors,
    size_t NRank,
    typename TransformT>
void concatenateFastLauncher(
    const int64_t* inputDim[],
    const void* const inputData[NInputTensors],
    const int64_t inputConcatDimOffsets[],
    const int64_t originalTE[],
    const int64_t actualTE[],
    const int64_t outputDim[NRank],
    const int64_t outputConcatDimOffsets[],
    void* const outputData,
    const size_t concatDim,
    char* func_name,
    cudaStream_t stream) {
  // some typedefs
  using input_data_type = typename ChunkInputT::data_type;
  using output_data_type = typename ChunkOutputT::data_type;

  // assign input tensors
  FSArray<const input_data_type*, NInputTensors> inputDataFS;
  for (size_t iTensor = 0; iTensor < NInputTensors; iTensor++) {
    inputDataFS.data[iTensor] =
        reinterpret_cast<const input_data_type*>(inputData[iTensor]);
  }

  // compute ncopy
  int64_t ncopy = 1;
  for (size_t i = concatDim + 1; i < NRank; i++) {
    ncopy *= outputDim[i];
  }

  // copy
  FSArray<IndexT, NInputTensors> inputConcatDimOffsetsFS;
  for (size_t j = 0; j < NInputTensors; j++) {
    inputConcatDimOffsetsFS.data[j] = inputConcatDimOffsets[j];
  }

  FSArray<IndexT, NInputTensors> originalTEFS;
  for (size_t j = 0; j < NInputTensors; j++) {
    originalTEFS.data[j] = originalTE[j];
  }

  FSArray<IndexT, NInputTensors> actualTEFS;
  for (size_t j = 0; j < NInputTensors; j++) {
    actualTEFS.data[j] = actualTE[j];
  }

  FSArray<IndexT, NInputTensors> outputConcatDimOffsetsMultipliedByNCopyFS;
  for (size_t j = 0; j < NInputTensors; j++) {
    outputConcatDimOffsetsMultipliedByNCopyFS.data[j] =
        outputConcatDimOffsets[j] * ncopy;
  }

  // compute postfix sum.
  FSArray<IndexT, NInputTensors> concatDimPostfixSumMultipliedByNCopy;
  {
    int64_t current = 0;
    for (size_t j = 0; j < NInputTensors; j++) {
      auto dim = inputDim[j][concatDim];
      current += dim;
      concatDimPostfixSumMultipliedByNCopy.data[j] = (IndexT)(current * ncopy);
    }
  }

  // this is the number of elements that needs to be filled on
  //   dim=concatDim. Basically, it is the sum of available elements
  //   on dim=concatDim for all of the inputs.
  // also, this is the number of ncopy-sized chunks that needs to be processed
  //   per single row of an output tensor. So, every row processes
  //   nElementsAtConcatDim * ncopy elements.
  int64_t nElementsAtConcatDim = 0;
  for (size_t j = 0; j < NInputTensors; j++) {
    auto dim = inputDim[j][concatDim];
    nElementsAtConcatDim += dim;
  }

  // the total number of output elements that needs to be processed
  int64_t numOutputElements = 1;
  // the number of rows...
  for (int32_t iRank = 0; iRank < concatDim; iRank++) {
    numOutputElements *= outputDim[iRank];
  }

  // ... multiplied by the number of elements per row
  numOutputElements *= nElementsAtConcatDim;
  numOutputElements *= ncopy;

  if (numOutputElements == 0) {
    // nothing to do
    return;
  }

  // this is the stride for dim=concatDim. Basically, the amount of
  //   memory allocated for a single output tensor row.
  // stride != nElementsAtConcatDim if some inputs were originally masked out.
  int64_t stride = outputDim[concatDim];

  // run the CUDA kernel
  const int32_t nThreadsPerBlock = 128;
  const int64_t effNumOutputElements =
      numOutputElements / ChunkOutputT::NElements;
  const int32_t nBlocks =
      (effNumOutputElements + nThreadsPerBlock - 1) / nThreadsPerBlock;

  // // tell some debug information
  // printf(
  //     "I am %s v2 with %ld elements, %d inputs, %zd ChunkInputT, "
  //     "%zd ChunkOutputT, "
  //     "%zd InputDataT, %zd OutputDataT\n",
  //     func_name,
  //     effNumOutputElements,
  //     (int32_t)NInputTensors,
  //     sizeof(ChunkInputT),
  //     sizeof(ChunkOutputT),
  //     sizeof(input_data_type),
  //     sizeof(output_data_type));

  ConcatKernelGeneralized<
      ChunkInputT,
      ChunkOutputT,
      IndexT,
      NInputTensors,
      TransformT><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
      inputDataFS,
      originalTEFS,
      actualTEFS,
      nElementsAtConcatDim * ncopy,
      stride * ncopy,
      concatDimPostfixSumMultipliedByNCopy,
      outputConcatDimOffsetsMultipliedByNCopyFS,
      reinterpret_cast<output_data_type*>(outputData),
      numOutputElements);
}

template <
    typename InputDataT,
    typename OutputDataT,
    size_t NInputTensors,
    size_t NRank,
    typename TransformT>
void invoke_concatenate_fast(
    const int64_t* inputDim[],
    const void* const inputData[NInputTensors],
    const TensorAccessor* inputTensorAccessors[NInputTensors],
    const int64_t outputDim[NRank],
    const int64_t outputConcatDimOffsets[],
    void* const outputData,
    const size_t concatDim,
    char* func_name,
    cudaStream_t stream) {
  // check the input parameters
  if (NInputTensors == 0 || NRank == 0) {
    return;
  }
  if (outputData == nullptr) {
    throw std::runtime_error("output is nullptr!");
  }

  // every thread in a kernel may copy up to ncopy elements
  //   in a single copy operation
  int64_t ncopy = 1;
  for (size_t i = concatDim + 1; i < NRank; i++) {
    ncopy *= outputDim[i];
  }

  // Compute the alignment of our output dataset
  // the alignment of the base address
  size_t alignmentOutput = getAlignment(outputData);

  // The alignment for the amount of copied data
  alignmentOutput = std::min(
      alignmentOutput,
      getAlignment(outputDim[concatDim] * ncopy * sizeof(OutputDataT)));

  // Input tensor i will be copied to a location that starts from
  //   a column outputConcatDimOffsets[i]. Compute its alignment
  for (size_t i = 0; i < NInputTensors; i++) {
    alignmentOutput = std::min(
        alignmentOutput,
        getAlignment(outputConcatDimOffsets[i] * ncopy * sizeof(OutputDataT)));
  }

  //
  const void* inputDataWithOffsets[NInputTensors];

  int64_t originalTE[NInputTensors];
  int64_t actualTE[NInputTensors];

  int64_t inputConcatDimOffsets[NInputTensors];

  //
  size_t alignmentInputs = 65536;
  for (size_t i = 0; i < NInputTensors; i++) {
    alignmentInputs = std::min(
        alignmentInputs,
        getAlignment(inputDim[i][concatDim] * ncopy * sizeof(InputDataT)));
    alignmentOutput = std::min(
        alignmentOutput,
        getAlignment(inputDim[i][concatDim] * ncopy * sizeof(OutputDataT)));
  }

  for (size_t j = 0; j < NInputTensors; j++) {
    const auto* tensorAccessor = inputTensorAccessors[j];

    // recompute the inputData with respect to offset
    inputDataWithOffsets[j] =
        ((InputDataT*)inputData[j]) + tensorAccessor->offset;

    // alter its alignment
    alignmentInputs =
        std::min(alignmentInputs, getAlignment(inputDataWithOffsets[j]));

    // is input tensor implies a contiguous access?
    if (tensorAccessor->is_contiguous) {
      // yes
      inputConcatDimOffsets[j] = inputDim[j][concatDim];

      originalTE[j] = inputDim[j][concatDim];
      actualTE[j] = inputDim[j][concatDim];
    } else {
      // no
      if (tensorAccessor->stride_dim == -1) {
        throw std::runtime_error(
            "Unsupported negative tensorAccessor stride_dim value!");
      } else {
        inputConcatDimOffsets[j] =
            tensorAccessor->actual_total_elements_from_stride_dim;
        originalTE[j] = tensorAccessor->original_total_elements_from_stride_dim;
        actualTE[j] = tensorAccessor->actual_total_elements_from_stride_dim;

        // ncopy?
        alignmentInputs = std::min(
            alignmentInputs, getAlignment(originalTE[j] * sizeof(InputDataT)));
        alignmentInputs = std::min(
            alignmentInputs, getAlignment(actualTE[j] * sizeof(InputDataT)));
      }
    }
  }

  if (alignmentOutput < alignmentInputs) {
    // // TODO: this is a possible optimization, bcz the current kernel
    // supports N reads ops per 1 write op, but not 1 read op per N write ops.
    // printf(
    //     "SHRINK, AlignmentInputs = %zd, AlignmentOutput = %zd\n",
    //     (size_t)alignmentInputs,
    //     (size_t)alignmentOutput);
    alignmentInputs = alignmentOutput;
  }

  if (alignmentInputs == 1) {
    // unsupported yet. todo
    throw std::runtime_error("Unsupported input tensors alignment!");
  }
  if (alignmentOutput == 1) {
    // unsupported yet. todo
    throw std::runtime_error("Unsupported output tensor alignment!");
  }

#define LAUNCHER(ALIGNMENT_INPUT, ALIGNMENT_OUTPUT, INDEX_T)            \
  if (alignmentOutput == ALIGNMENT_OUTPUT &&                            \
      alignmentInputs == ALIGNMENT_INPUT) {                             \
    if constexpr (                                                      \
        sizeof(InputDataT) <= ALIGNMENT_INPUT &&                        \
        sizeof(OutputDataT) <= ALIGNMENT_OUTPUT) {                      \
      using InputChunkT = AlignedChunk<InputDataT, ALIGNMENT_INPUT>;    \
      using OutputChunkT = AlignedChunk<OutputDataT, ALIGNMENT_OUTPUT>; \
      concatenateFastLauncher<                                          \
          InputChunkT,                                                  \
          OutputChunkT,                                                 \
          INDEX_T,                                                      \
          NInputTensors,                                                \
          NRank,                                                        \
          TransformT>(                                                  \
          inputDim,                                                     \
          inputDataWithOffsets,                                         \
          inputConcatDimOffsets,                                        \
          originalTE,                                                   \
          actualTE,                                                     \
          outputDim,                                                    \
          outputConcatDimOffsets,                                       \
          outputData,                                                   \
          concatDim,                                                    \
          func_name,                                                    \
          stream);                                                      \
      return;                                                           \
    }                                                                   \
  }

  // compute the limit of the number of elements in output tensor
  int64_t numOutputElements = 1;
  for (size_t iRank = 0; iRank < NRank; iRank++) {
    numOutputElements *= outputDim[iRank];
  }

  if (numOutputElements == 0) {
    // no elements to process
    return;
  }

  // TODO: rework the following if condition.
  // 1. This value is a constexpr value, because all the
  // input & output tensor sizes are known to a template generator.
  // This improvement should reduce the compilation speed 2x.
  // 2. Strided tensors might need a special handling.
  if (!can_use_32bit_index_math(numOutputElements)) {
    using index_type = int64_t;

    LAUNCHER(32, 32, index_type);
    LAUNCHER(16, 32, index_type);
    LAUNCHER(8, 32, index_type);
    LAUNCHER(4, 32, index_type);
    LAUNCHER(2, 32, index_type);

    LAUNCHER(16, 16, index_type);
    LAUNCHER(8, 16, index_type);
    LAUNCHER(4, 16, index_type);
    LAUNCHER(2, 16, index_type);

    LAUNCHER(8, 8, index_type);
    LAUNCHER(4, 8, index_type);
    LAUNCHER(2, 8, index_type);

    LAUNCHER(4, 4, index_type);
    LAUNCHER(2, 4, index_type);

    LAUNCHER(2, 2, index_type);
  } else {
    using index_type = int32_t;

    LAUNCHER(32, 32, index_type);
    LAUNCHER(16, 32, index_type);
    LAUNCHER(8, 32, index_type);
    LAUNCHER(4, 32, index_type);
    LAUNCHER(2, 32, index_type);

    LAUNCHER(16, 16, index_type);
    LAUNCHER(8, 16, index_type);
    LAUNCHER(4, 16, index_type);
    LAUNCHER(2, 16, index_type);

    LAUNCHER(8, 8, index_type);
    LAUNCHER(4, 8, index_type);
    LAUNCHER(2, 8, index_type);

    LAUNCHER(4, 4, index_type);
    LAUNCHER(2, 4, index_type);

    LAUNCHER(2, 2, index_type);
  }

  // no launcher was found
  throw std::runtime_error("Unsupported concat kernel specialization!");
}

#endif
