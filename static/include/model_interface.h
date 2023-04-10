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
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

// We compile all models with -fvisibility=hidden. Any symbols that need to be
// exposed in the final shared library must be declared with AIT_EXPORT to make
// them visible.

#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define AIT_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define AIT_EXPORT __declspec(dllexport)
#else
#define AIT_EXPORT
#endif
#endif

struct AITemplateModelOpaque {};
using AITemplateModelHandle = AITemplateModelOpaque*;

enum class AITemplateError : int {
  AITemplateSuccess = 0,
  AITemplateFailure = 1,
};

#define AIT_ERROR_CHECK(call)                                             \
  if ((call) != AITemplateError::AITemplateSuccess) {                     \
    throw std::runtime_error(                                             \
        std::string(#call " API call failed at ") + __FILE__ + ", line" + \
        std::to_string(__LINE__));                                        \
  }

struct AITemplateParamShape {
  AITemplateParamShape() : shape_data(nullptr), size(0) {}
  AITemplateParamShape(const int64_t* shape_data_in, size_t size_in)
      : shape_data(shape_data_in), size(size_in) {}

  const int64_t* shape_data;
  size_t size;

  size_t Numel() const {
    return std::accumulate(
        shape_data, shape_data + size, (int64_t)1, std::multiplies<int64_t>());
  }
};

enum class AITemplateDtype {
  kUnset = 0,
  kHalf,
  kFloat,
  kInt,
  kLong,
  kBool,
  kBFloat16,
};

struct AITData {
  AITData() : ptr(nullptr), dtype(AITemplateDtype::kUnset) {}

  AITData(
      void* ptr_in,
      const AITemplateParamShape& shape_in,
      AITemplateDtype dtype_in)
      : ptr(ptr_in), shape(shape_in), dtype(dtype_in) {}

  void* ptr;
  AITemplateParamShape shape;
  AITemplateDtype dtype;
};

inline size_t AITemplateDtypeSizeBytes(AITemplateDtype dtype) {
  switch (dtype) {
    case AITemplateDtype::kHalf:
    case AITemplateDtype::kBFloat16:
      return 2;
    case AITemplateDtype::kFloat:
      return 4;
    case AITemplateDtype::kInt:
      return 4;
    case AITemplateDtype::kLong:
      return 8;
    case AITemplateDtype::kBool:
      return 1;
    case AITemplateDtype::kUnset:
      throw std::runtime_error("Unset dtype has no size!");
  }
  throw std::runtime_error("dtype handling is not implemented!");
}

struct AITemplateStreamOpaque {};
using AITemplateStreamHandle = AITemplateStreamOpaque*;

// Allocator to use for GPU mallocs and frees. Allocations will only happen
// when the ModelContainer is created.
class AITemplateAllocator {
 public:
  virtual void* Allocate(size_t nbytes) = 0;
  virtual void Free(void* ptr) = 0;

  virtual ~AITemplateAllocator() = default;
};

// Some custom allocators are provided. They can be created by passing
// an enum into the AITemplateAllocatorCreate() function.
enum class AITemplateAllocatorType {
  // The default allocator just uses the backend's default malloc/free.
  kDefault = 0,
  // The tracking allocator is like the default allocator, but it keeps
  // track of how many bytes it has allocated. Mainly used for testing.
  kTracking,
};

extern "C" {

// Create a ModelContainer. See model_container.h for all the details.
// Some important high-level notes:
// * If allocator is null, a default allocator is used (forwards to
//   {cuda/hip}{Malloc/Free}).
// * We assume that the allocator lives at least as long as the ModelContainer.
AIT_EXPORT AITemplateError AITemplateModelContainerCreate(
    AITemplateModelHandle* ret,
    size_t num_runtimes,
    AITemplateAllocator* allocator = nullptr);

AIT_EXPORT AITemplateError
AITemplateModelContainerDelete(AITemplateModelHandle handle);

AIT_EXPORT AITemplateError AITemplateModelContainerSetConstant(
    AITemplateModelHandle handle,
    const char* name,
    const AITData* tensor);

AIT_EXPORT AITemplateError AITemplateModelContainerSetManyConstants(
    AITemplateModelHandle handle,
    const char** names,
    const AITData* tensors,
    size_t num_tensors);

AIT_EXPORT AITemplateError AITemplateModelContainerSetDoubleBufferConstant(
    AITemplateModelHandle handle,
    AITemplateStreamHandle stream_handle,
    const char* name,
    const AITData* tensor);

AIT_EXPORT AITemplateError AITemplateModelContainerSetManyDoubleBufferConstants(
    AITemplateModelHandle handle,
    AITemplateStreamHandle stream_handle,
    const char** names,
    const AITData* tensors,
    size_t num_tensors);

AIT_EXPORT AITemplateError AITemplateModelContainerGetNumConstants(
    AITemplateModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    size_t* num_constants_out);

AIT_EXPORT AITemplateError AITemplateModelContainerGetConstantNames(
    AITemplateModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    const char** constant_names_out);

AIT_EXPORT AITemplateError AITemplateModelContainerRun(
    AITemplateModelHandle handle,
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    AITemplateStreamHandle stream_handle,
    bool sync,
    bool graph_mode,
    int64_t** output_shapes_out);

// Like AITemplateModelContainerRun, but expects outputs to be allocated on the
// host. Does an extra sync/copy at the end to copy them over. Warning: don't
// use this! It's not optimal with respect to performance. It's here for use if
// you need it for debugging.
AIT_EXPORT AITemplateError AITemplateModelContainerRunWithOutputsOnHost(
    AITemplateModelHandle handle,
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    AITemplateStreamHandle stream_handle,
    bool graph_mode,
    int64_t** output_shapes_out);

/// Do per op profile and write the profiling report to file.
AIT_EXPORT AITemplateError AITemplateModelContainerProfile(
    AITemplateModelHandle handle,
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    AITemplateStreamHandle stream_handle,
    size_t num_iters,
    const char* filename);

AIT_EXPORT AITemplateError AITemplateModelContainerBenchmark(
    AITemplateModelHandle handle,
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    AITemplateStreamHandle stream_handle,
    bool graph_mode,
    size_t count,
    size_t num_threads,
    bool use_unique_stream_per_thread,
    float* runtime_ms,
    int64_t** output_shapes_out);

AIT_EXPORT AITemplateError AITemplateModelContainerGetNumInputs(
    AITemplateModelHandle handle,
    size_t* num_inputs_out);

AIT_EXPORT AITemplateError AITemplateModelContainerGetInputName(
    AITemplateModelHandle handle,
    size_t input_idx,
    const char** input_name_out);

AIT_EXPORT AITemplateError AITemplateModelContainerGetMaximumInputShape(
    AITemplateModelHandle handle,
    size_t input_idx,
    AITemplateParamShape* shape);

AIT_EXPORT AITemplateError AITemplateModelContainerGetInputDtype(
    AITemplateModelHandle handle,
    size_t input_idx,
    AITemplateDtype* input_dtype);

AIT_EXPORT AITemplateError AITemplateModelContainerGetNumOutputs(
    AITemplateModelHandle handle,
    size_t* num_outputs_out);

AIT_EXPORT AITemplateError AITemplateModelContainerGetOutputName(
    AITemplateModelHandle handle,
    size_t output_idx,
    const char** output_name_out);

AIT_EXPORT AITemplateError AITemplateModelContainerGetMaximumOutputShape(
    AITemplateModelHandle handle,
    size_t output_idx,
    AITemplateParamShape* shape_out);

AIT_EXPORT AITemplateError AITemplateModelContainerGetOutputDtype(
    AITemplateModelHandle handle,
    size_t output_idx,
    AITemplateDtype* out);

AIT_EXPORT AITemplateError AITemplateModelContainerGetNumRuntimes(
    AITemplateModelHandle handle,
    size_t* num_runtimes_out);

AIT_EXPORT AITemplateError AITemplateModelContainerFoldConstants(
    AITemplateModelHandle handle,
    AITemplateStreamHandle stream_handle,
    bool sync);

AIT_EXPORT AITemplateError AITemplateModelContainerFoldConstantsInDoubleBuffer(
    AITemplateModelHandle handle,
    AITemplateStreamHandle stream_handle,
    bool sync);

AIT_EXPORT AITemplateError
AITemplateModelContainerSwapConstants(AITemplateModelHandle handle);

AIT_EXPORT AITemplateError AITemplateAllocatorCreate(
    AITemplateAllocator** allocator_out,
    AITemplateAllocatorType allocator_type);

AIT_EXPORT AITemplateError
AITemplateAllocatorDelete(AITemplateAllocator* allocator_out);

// Get the number of bytes allocated; mainly used for testing.
AIT_EXPORT AITemplateError AITemplateTrackingAllocatorGetNumBytes(
    AITemplateAllocator* allocator,
    size_t* num_bytes_out);

} // extern "C"
