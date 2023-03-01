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
#include "model_interface.h"
#include <iostream>
#include <unordered_map>
#include "model-generated.h"
#include "model_container.h"

// Important: don't let exceptions escape the functions below.
// They can cause problems when -fvisibility=hidden. But more
// importantly, they can crash the program if they try to cross
// the language boundary into Python.
#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)     \
  try {                                          \
    __VA_ARGS__                                  \
  } catch (const std::exception& e) {            \
    LOG(ERROR) << "Error: " << e.what();         \
    return AITemplateError::AITemplateFailure;   \
  } catch (...) {                                \
    LOG(ERROR) << "Unknown exception occurred."; \
    return AITemplateError::AITemplateFailure;   \
  }                                              \
  return AITemplateError::AITemplateSuccess;

#define RETURN_ERROR_IF_NULL(var)                          \
  if (var == nullptr) {                                    \
    LOG(ERROR) << "Variable " << #var << " can't be null"; \
    return AITemplateError::AITemplateFailure;             \
  }

namespace ait {
namespace {
class DefaultAllocator : public AITemplateAllocator {
 public:
  void* Allocate(size_t n_bytes) override {
    void* result;
    DEVICE_CHECK(DeviceMalloc(&result, n_bytes));
    return result;
  }

  void Free(void* ptr) override {
    DEVICE_CHECK(FreeDeviceMemory(ptr));
  }
};

class TrackingAllocator : public DefaultAllocator {
 public:
  void* Allocate(size_t n_bytes) override {
    auto* result = DefaultAllocator::Allocate(n_bytes);
    num_bytes_ += n_bytes;
    return result;
  }

  size_t NumBytesAllocated() const {
    return num_bytes_;
  }

 private:
  size_t num_bytes_ = 0;
};

DefaultAllocator default_allocator;
} // namespace
} // namespace ait

extern "C" {

AITemplateError AITemplateModelContainerCreate(
    AITemplateModelHandle* ret,
    size_t num_runtimes,
    AITemplateAllocator* allocator) {
  if (num_runtimes == 0) {
    LOG(ERROR) << "num_runtimes must be positive, but got 0";
    return AITemplateError::AITemplateFailure;
  }
  RETURN_ERROR_IF_NULL(ret)
  AITemplateAllocator& allocator_ref =
      allocator == nullptr ? ait::default_allocator : *allocator;
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* m = ait::CreateModelContainer(num_runtimes, allocator_ref);
    *ret = reinterpret_cast<AITemplateModelHandle>(m);
  })
}

AITemplateError AITemplateModelContainerDelete(AITemplateModelHandle handle) {
  RETURN_ERROR_IF_NULL(handle)
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
    delete m;
  });
}

AITemplateError AITemplateModelContainerSetConstant(
    AITemplateModelHandle handle,
    const char* name,
    const AITData* tensor) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(tensor)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->SetConstant(name, *tensor); })
}

AIT_EXPORT AITemplateError AITemplateModelContainerSetManyConstants(
    AITemplateModelHandle handle,
    const char** names,
    const AITData* tensors,
    size_t num_tensors) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { m->SetManyConstants(names, tensors, num_tensors); })
}

AITemplateError AITemplateModelContainerSetDoubleBufferConstant(
    AITemplateModelHandle handle,
    AITemplateStreamHandle stream_handle,
    const char* name,
    const AITData* tensor) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(tensor)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  auto stream = reinterpret_cast<ait::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { m->SetDoubleBufferConstant(name, *tensor, stream); })
}

AIT_EXPORT AITemplateError AITemplateModelContainerSetManyDoubleBufferConstants(
    AITemplateModelHandle handle,
    AITemplateStreamHandle stream_handle,
    const char** names,
    const AITData* tensors,
    size_t num_tensors) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  auto stream = reinterpret_cast<ait::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { m->SetManyDoubleBufferConstants(names, tensors, num_tensors, stream); })
}

AITemplateError AITemplateModelContainerGetNumConstants(
    AITemplateModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    size_t* num_constants_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(num_constants_out)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (constant_folding_inputs_only) {
      *num_constants_out =
          m->GetNumConstantFoldingInputs(unbound_constants_only);
    } else {
      *num_constants_out = m->GetNumConstants(unbound_constants_only);
    }
  })
}

AITemplateError AITemplateModelContainerGetConstantNames(
    AITemplateModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    const char** constant_names_out) {
  RETURN_ERROR_IF_NULL(handle)
  // WriteAllConstantNamesTo() will handle nullptr checks on constant_names_out.
  // Passing nullptr is allowed if there are 0 constants!
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    m->WriteAllConstantNamesTo(
        constant_names_out,
        unbound_constants_only,
        constant_folding_inputs_only);
  })
}

AITemplateError AITemplateModelContainerRun(
    AITemplateModelHandle handle,
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    AITemplateStreamHandle stream_handle,
    bool sync,
    bool graph_mode,
    int64_t** output_shapes_out) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  auto stream = reinterpret_cast<ait::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    m->Run(
        inputs,
        num_inputs,
        outputs,
        num_outputs,
        stream,
        sync,
        graph_mode,
        output_shapes_out);
  })
}

AITemplateError AITemplateModelContainerRunWithOutputsOnHost(
    AITemplateModelHandle handle,
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    AITemplateStreamHandle stream_handle,
    bool graph_mode,
    int64_t** output_shapes_out) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  auto stream = reinterpret_cast<ait::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    m->RunWithOutputsOnHost(
        inputs,
        num_inputs,
        outputs,
        num_outputs,
        stream,
        graph_mode,
        output_shapes_out);
  })
}

AITemplateError AITemplateModelContainerProfile(
    AITemplateModelHandle handle,
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    AITemplateStreamHandle stream_handle,
    size_t num_iters,
    const char* filename) {
  RETURN_ERROR_IF_NULL(handle);
  RETURN_ERROR_IF_NULL(filename);
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  auto stream = reinterpret_cast<ait::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    m->Profile(
        inputs, num_inputs, outputs, num_outputs, stream, num_iters, filename);
  })
}

AITemplateError AITemplateModelContainerBenchmark(
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
    int64_t** output_shapes_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(runtime_ms)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  auto stream = reinterpret_cast<ait::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *runtime_ms = m->Benchmark(
        inputs,
        num_inputs,
        outputs,
        num_outputs,
        stream,
        graph_mode,
        count,
        num_threads,
        use_unique_stream_per_thread,
        output_shapes_out);
  })
}

AITemplateError AITemplateModelContainerGetNumInputs(
    AITemplateModelHandle handle,
    size_t* num_inputs_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(num_inputs_out)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *num_inputs_out = m->NumInputs(); })
}

AITemplateError AITemplateModelContainerGetInputName(
    AITemplateModelHandle handle,
    size_t input_idx,
    const char** input_name_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(input_name_out)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *input_name_out = m->InputName(input_idx); })
}

AITemplateError AITemplateModelContainerGetMaximumInputShape(
    AITemplateModelHandle handle,
    size_t input_idx,
    AITemplateParamShape* shape) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(shape)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *shape = m->MaxInputShape(input_idx); })
}

AITemplateError AITemplateModelContainerGetInputDtype(
    AITemplateModelHandle handle,
    size_t input_idx,
    AITemplateDtype* input_dtype) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(input_dtype)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *input_dtype = m->InputDtype(input_idx); })
}

AITemplateError AITemplateModelContainerGetNumOutputs(
    AITemplateModelHandle handle,
    size_t* num_outputs_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(num_outputs_out)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *num_outputs_out = m->NumOutputs(); })
}

AITemplateError AITemplateModelContainerGetOutputName(
    AITemplateModelHandle handle,
    size_t output_idx,
    const char** output_name_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(output_name_out)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *output_name_out = m->OutputName(output_idx); })
}

AITemplateError AITemplateModelContainerGetMaximumOutputShape(
    AITemplateModelHandle handle,
    size_t output_idx,
    AITemplateParamShape* shape_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(shape_out)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *shape_out = m->MaxOutputShape(output_idx); })
}

AITemplateError AITemplateModelContainerGetOutputDtype(
    AITemplateModelHandle handle,
    size_t output_idx,
    AITemplateDtype* dtype_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(dtype_out)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *dtype_out = m->OutputDtype(output_idx); })
}

AITemplateError AITemplateModelContainerGetNumRuntimes(
    AITemplateModelHandle handle,
    size_t* num_runtimes_out) {
  RETURN_ERROR_IF_NULL(num_runtimes_out)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *num_runtimes_out = m->GetNumRuntimes(); })
}

AITemplateError AITemplateModelContainerFoldConstants(
    AITemplateModelHandle handle,
    AITemplateStreamHandle stream_handle,
    bool sync) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  auto stream = reinterpret_cast<ait::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->FoldConstants(stream, sync, false); })
}

AITemplateError AITemplateModelContainerFoldConstantsInDoubleBuffer(
    AITemplateModelHandle handle,
    AITemplateStreamHandle stream_handle,
    bool sync) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  auto stream = reinterpret_cast<ait::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->FoldConstants(stream, sync, true); })
}

AITemplateError AITemplateModelContainerSwapConstants(
    AITemplateModelHandle handle) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->SwapConstants(); })
}

AITemplateError AITemplateAllocatorCreate(
    AITemplateAllocator** allocator_out,
    AITemplateAllocatorType allocator_type) {
  RETURN_ERROR_IF_NULL(allocator_out);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    switch (allocator_type) {
      case AITemplateAllocatorType::kDefault:
        *allocator_out = new ait::DefaultAllocator();
        break;
      case AITemplateAllocatorType::kTracking:
        *allocator_out = new ait::TrackingAllocator();
        break;
      default:
        throw std::runtime_error("Unrecognized allocator type");
    }
  });
}

AITemplateError AITemplateAllocatorDelete(AITemplateAllocator* allocator) {
  RETURN_ERROR_IF_NULL(allocator);
  delete allocator;
  return AITemplateError::AITemplateSuccess;
}

AITemplateError AITemplateTrackingAllocatorGetNumBytes(
    AITemplateAllocator* allocator,
    size_t* num_bytes_out) {
  RETURN_ERROR_IF_NULL(allocator);
  RETURN_ERROR_IF_NULL(num_bytes_out);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* tracking_allocator = dynamic_cast<ait::TrackingAllocator*>(allocator);
    if (tracking_allocator == nullptr) {
      throw std::runtime_error("Allocator was not a tracking allocator!");
    }
    *num_bytes_out = tracking_allocator->NumBytesAllocated();
  });
}

} // extern "C"
