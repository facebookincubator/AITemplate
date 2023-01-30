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

// This file is used for generating a standalone executable for a model.
// It only invokes the C++ model interface. We can directly invoke the
// generated executable without going through Python bindings. Because it
// aims for assisting debugging, we make a number of simplifications:
//   * we use the maximum input shapes;
//   * we only generate random inputs with a fixed seed;
//   * we assume that outputs exist on the host;
//   * we disable graph_mode;
//   * etc...
// Once the file is copied into the intemediate working dir (e.g.,
// ./tmp/test_gemm_rcr) along with other files, users are free to make any
// changes to the code. We do not try to predict users' actions.

#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "macros.h"
#include "model_interface.h"
#include "raii_wrapper.h"

using namespace ait;

template <typename T>
static void make_random_integer_values(
    std::mt19937& rnd_generator,
    T* h_data,
    size_t numel,
    T lb,
    T ub) {
  std::uniform_int_distribution<> dist(lb, ub);
  for (size_t i = 0; i < numel; i++) {
    h_data[i] = static_cast<T>(dist(rnd_generator));
  }
}

static void make_random_float_values(
    std::mt19937& rnd_generator,
    float* h_data,
    size_t numel,
    float lb,
    float ub) {
  std::uniform_real_distribution<> dist(lb, ub);
  for (size_t i = 0; i < numel; i++) {
    h_data[i] = static_cast<float>(dist(rnd_generator));
  }
}

static void make_random_float16_values(
    std::mt19937& rnd_generator,
    half* h_data,
    size_t numel,
    float lb,
    float ub) {
  std::uniform_real_distribution<> dist(lb, ub);
  for (size_t i = 0; i < numel; i++) {
    float v = static_cast<float>(dist(rnd_generator));
    h_data[i] = __float2half_rn(v);
  }
}

static void make_random_bfloat16_values(
    std::mt19937& rnd_generator,
    bfloat16* h_data,
    size_t numel,
    float lb,
    float ub) {
  std::uniform_real_distribution<> dist(lb, ub);
  for (size_t i = 0; i < numel; i++) {
    float v = static_cast<float>(dist(rnd_generator));
    h_data[i] = __float2bfloat16_rn(v);
  }
}

static GPUPtr make_random_data(
    AITemplateAllocator& allocator,
    std::mt19937& rnd_generator,
    const AITemplateParamShape& shape,
    const AITemplateDtype& dtype) {
  size_t numel = shape.Numel();
  size_t num_bytes = numel * AITemplateDtypeSizeBytes(dtype);
  void* h_data;
  DEVICE_CHECK(DeviceMallocHost(&h_data, num_bytes));
  switch (dtype) {
    case AITemplateDtype::kInt:
      make_random_integer_values<int>(
          rnd_generator,
          static_cast<int*>(h_data),
          numel,
          /*lb*/ -10,
          /*ub*/ 10);
      break;
    case AITemplateDtype::kLong:
      make_random_integer_values<int64_t>(
          rnd_generator,
          static_cast<int64_t*>(h_data),
          numel,
          /*lb*/ -10,
          /*ub*/ 10);
      break;
    case AITemplateDtype::kFloat:
      make_random_float_values(
          rnd_generator,
          static_cast<float*>(h_data),
          numel,
          /*lb*/ 1.0,
          /*ub*/ 2.0);
      break;
    case AITemplateDtype::kBFloat16:
      make_random_bfloat16_values(
          rnd_generator,
          static_cast<bfloat16*>(h_data),
          numel,
          /*lb*/ 1.0,
          /*ub*/ 2.0);
      break;
    case AITemplateDtype::kHalf:
      make_random_float16_values(
          rnd_generator,
          static_cast<half*>(h_data),
          numel,
          /*lb*/ 1.0,
          /*ub*/ 2.0);
      break;
    case AITemplateDtype::kBool:
      make_random_integer_values<bool>(
          rnd_generator, static_cast<bool*>(h_data), numel, /*lb*/ 0, /*ub*/ 1);
      break;
    default:
      throw std::runtime_error("unsupported dtype for making random data");
  }

  GPUPtr d_ptr = RAII_DeviceMalloc(num_bytes, allocator);
  DEVICE_CHECK(CopyToDevice(d_ptr.get(), h_data, num_bytes));

  // free memory
  DEVICE_CHECK(FreeDeviceHostMemory(h_data));

  return d_ptr;
}

using OutputDataPtr = std::unique_ptr<void, std::function<void(void*)>>;

struct OutputData {
  OutputData(
      OutputDataPtr& data_in,
      std::unique_ptr<int64_t[]>& shape_ptr_in,
      int shape_size_in,
      int index_in,
      AITemplateDtype dtype_in,
      const char* name_in)
      : data(std::move(data_in)),
        shape_ptr(std::move(shape_ptr_in)),
        shape_size(shape_size_in),
        index(index_in),
        dtype(dtype_in),
        name(name_in) {}

  OutputData(OutputData&& other) noexcept
      : data(std::move(other.data)),
        shape_ptr(std::move(other.shape_ptr)),
        shape_size(other.shape_size),
        index(other.index),
        dtype(other.dtype),
        name(std::move(other.name)) {}

  OutputDataPtr data;
  std::unique_ptr<int64_t[]> shape_ptr;
  int shape_size;
  int index;
  AITemplateDtype dtype;
  std::string name;
};

static AITemplateError run(
    AITemplateModelHandle handle,
    AITemplateAllocator& allocator,
    std::vector<OutputData>& outputs) {
  size_t num_outputs = 0;
  AITemplateModelContainerGetNumOutputs(handle, &num_outputs);

  outputs.reserve(num_outputs);
  std::vector<AITData> ait_outputs;
  ait_outputs.reserve(num_outputs);
  std::vector<int64_t*> ait_output_shapes_out;
  ait_output_shapes_out.reserve(num_outputs);

  for (unsigned i = 0; i < num_outputs; i++) {
    const char* name;
    AITemplateModelContainerGetOutputName(handle, i, &name);
    AITemplateParamShape shape;
    AITemplateModelContainerGetMaximumOutputShape(handle, i, &shape);
    AITemplateDtype dtype;
    AITemplateModelContainerGetOutputDtype(handle, i, &dtype);

    std::unique_ptr<int64_t[]> shape_ptr =
        std::make_unique<int64_t[]>(shape.size);
    ait_output_shapes_out.push_back(shape_ptr.get());
    size_t num_bytes = shape.Numel() * AITemplateDtypeSizeBytes(dtype);
    void* h_data;
    DEVICE_CHECK(DeviceMallocHost(&h_data, num_bytes));
    ait_outputs.emplace_back(h_data, shape, dtype);
    auto deleter = [](void* data) { FreeDeviceHostMemory(data); };
    OutputDataPtr h_output_ptr(h_data, deleter);
    outputs.emplace_back(
        h_output_ptr, shape_ptr, (int)shape.size, (int)i, dtype, name);
  }

  size_t num_inputs = 0;
  AITemplateModelContainerGetNumInputs(handle, &num_inputs);
  // Holding unique_ptr(s) that will be auto-released.
  std::vector<GPUPtr> input_ptrs;
  input_ptrs.reserve(num_inputs);

  std::map<std::string, unsigned> input_name_to_index;
  std::vector<AITData> inputs(num_inputs);
  std::mt19937 rnd_generator(1234);
  // set up the name-to-index map each input
  for (unsigned i = 0; i < num_inputs; i++) {
    const char* name;
    AITemplateModelContainerGetInputName(handle, i, &name);
    input_name_to_index.insert({name, i});
    std::cout << "input: " << name << ", at idx: " << i << "\n";

    AITemplateParamShape shape;
    AITemplateModelContainerGetMaximumInputShape(handle, i, &shape);
    AITemplateDtype dtype;
    AITemplateModelContainerGetInputDtype(handle, i, &dtype);
    // This file aims for helping debugging so we make the code logic
    // simple. Instead of asking the user to pass input names along with
    // shapes, we just use the shape with the largest dimension values
    // to make a random input. Once this code is copied into the test's
    // tmp folder, the person who will be diagnosing the issue could make any
    // changes to the code. We don't force us to predict the user's behavior.
    input_ptrs.emplace_back(
        make_random_data(allocator, rnd_generator, shape, dtype));
    inputs[i] = AITData(input_ptrs.back().get(), shape, dtype);
  }

  bool graph_mode = false;
  auto stream = RAII_StreamCreate(/*non_blocking=*/true);
  return AITemplateModelContainerRunWithOutputsOnHost(
      handle,
      inputs.data(),
      num_inputs,
      ait_outputs.data(),
      num_outputs,
      reinterpret_cast<AITemplateStreamHandle>(stream.get()),
      graph_mode,
      ait_output_shapes_out.data());
}

int main() {
  AITemplateModelHandle handle;
  AITemplateModelContainerCreate(&handle, /*num_runtimes*/ 1);
  AITemplateAllocator* allocator;
  AIT_ERROR_CHECK(
      AITemplateAllocatorCreate(&allocator, AITemplateAllocatorType::kDefault));

  auto deleter = [](void* data) { FreeDeviceHostMemory(data); };

  std::vector<OutputData> outputs;
  AIT_ERROR_CHECK(run(handle, *allocator, outputs));

  // print out something
  for (const auto& output : outputs) {
    std::cout << "output: " << output.name << " at idx: " << output.index
              << " with shape: ";
    for (int i = 0; i < output.shape_size; i++) {
      std::cout << output.shape_ptr[i] << ",";
    }
    std::cout << "\n";
  }

  AIT_ERROR_CHECK(AITemplateAllocatorDelete(allocator));
  // We are done and delete the handle.
  AITemplateModelContainerDelete(handle);
  return 0;
}
