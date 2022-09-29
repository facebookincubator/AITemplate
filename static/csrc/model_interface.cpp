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

extern "C" {

AITemplateError AITemplateModelContainerCreate(
    AITemplateModelHandle* ret,
    size_t num_runtimes) {
  if (num_runtimes == 0) {
    LOG(ERROR) << "num_runtimes must be positive, but got 0";
    return AITemplateError::AITemplateFailure;
  }
  RETURN_ERROR_IF_NULL(ret)
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* m = ait::CreateModelContainer(num_runtimes);
    *ret = reinterpret_cast<AITemplateModelHandle>(m);
    return AITemplateError::AITemplateSuccess;
  })
}

AITemplateError AITemplateModelContainerDelete(AITemplateModelHandle handle) {
  RETURN_ERROR_IF_NULL(handle)
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* m = reinterpret_cast<ait::ModelContainer*>(handle);
    delete m;
    return AITemplateError::AITemplateSuccess;
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
} // extern "C"
