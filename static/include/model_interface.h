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

struct AITemplateParamShape {
  AITemplateParamShape() : shape_data(nullptr), size(0) {}
  AITemplateParamShape(const int64_t* shape_data_in, size_t size_in)
      : shape_data(shape_data_in), size(size_in) {}

  const int64_t* shape_data;
  size_t size;

  size_t Numel() const {
    return std::accumulate(
        shape_data, shape_data + size, 1, std::multiplies<int64_t>());
  }
};

enum class AITemplateDtype {
  kUnset = 0,
  kHalf,
  kFloat,
  kInt,
  kLong,
};

struct AITemplateTensor {
  AITemplateTensor() : ptr(nullptr), dtype(AITemplateDtype::kUnset) {}

  AITemplateTensor(
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
      return 2;
    case AITemplateDtype::kFloat:
      return 4;
    case AITemplateDtype::kInt:
      return 4;
    case AITemplateDtype::kLong:
      return 8;
    case AITemplateDtype::kUnset:
      throw std::runtime_error("Unset dtype has no size!");
  }
}

struct AITemplateStreamOpaque {};
using AITemplateStreamHandle = AITemplateStreamOpaque*;

extern "C" {

AIT_EXPORT AITemplateError
AITemplateModelContainerCreate(AITemplateModelHandle* ret, size_t num_runtimes);

AIT_EXPORT AITemplateError
AITemplateModelContainerDelete(AITemplateModelHandle handle);

AIT_EXPORT AITemplateError AITemplateModelContainerSetConstant(
    AITemplateModelHandle handle,
    const char* name,
    const void* src,
    size_t size);

AIT_EXPORT AITemplateError AITemplateModelContainerRun(
    AITemplateModelHandle handle,
    const AITemplateTensor* inputs,
    size_t num_inputs,
    AITemplateTensor* outputs,
    size_t num_outputs,
    AITemplateStreamHandle stream_handle,
    bool sync,
    bool graph_mode,
    int64_t** output_shapes_out);

// Like AITemplateModelContainerRun, but expects outputs to be allocated on the
// host. Does an extra sync/copy at the end to copy them over. Warning: don't
// use this! It's not optimal with respect to performance. It's here for use by
// internal constant folding passes.
AIT_EXPORT AITemplateError AITemplateModelContainerRunWithOutputsOnHost(
    AITemplateModelHandle handle,
    const AITemplateTensor* inputs,
    size_t num_inputs,
    AITemplateTensor* outputs,
    size_t num_outputs,
    AITemplateStreamHandle stream_handle,
    bool graph_mode,
    int64_t** output_shapes_out);

AIT_EXPORT AITemplateError AITemplateModelContainerBenchmark(
    AITemplateModelHandle handle,
    const AITemplateTensor* inputs,
    size_t num_inputs,
    AITemplateTensor* ouputs,
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

} // extern "C"
