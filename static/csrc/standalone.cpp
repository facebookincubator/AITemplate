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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
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

template <typename T>
void read_element(std::ifstream& fh, T& elem) {
  if (!fh.good()) {
    throw std::runtime_error("Input stream is not in good state.");
  }
  fh.read(reinterpret_cast<char*>(&elem), sizeof(T));
  if (fh.fail()) {
    throw std::runtime_error("Failed to read binary data");
  }
}

struct AITStandaloneTestcase {
  std::vector<AITData> expected_outputs;
  std::vector<AITData> host_outputs;
  std::vector<AITData> gpu_outputs;

  std::vector<int64_t*> ait_output_shapes_out;

  std::vector<AITData>
      inputs; // this will be filled the AITData instances for the inputs

  std::vector<int64_t> shape_data_owner;
  std::vector<GPUPtr> gpu_data_owner;

  const std::string test_data_path; // path to test data file
  AITemplateModelHandle& handle;
  AITemplateAllocator& allocator;

  float atol;
  float rtol;

  AITStandaloneTestcase(
      const char* test_data_path_,
      AITemplateModelHandle& handle_, // model handle
      AITemplateAllocator& allocator_)
      : handle(handle_),
        allocator(allocator_),
        test_data_path(test_data_path_) {
    _load();
  }

  void _load() { // relative error tolerance
    size_t num_outputs = 0;
    size_t num_inputs = 0;
    AITemplateModelContainerGetNumInputs(handle, &num_inputs);
    AITemplateModelContainerGetNumOutputs(handle, &num_outputs);
    ait_output_shapes_out.reserve(num_outputs);
    expected_outputs.reserve(num_outputs);
    host_outputs.reserve(num_outputs);
    gpu_outputs.reserve(num_outputs);
    std::ifstream fh(test_data_path);
    read_element(fh, atol); // absolute error tolerance
    read_element(fh, rtol); // relative error tolerance

    gpu_data_owner.reserve(num_inputs + num_outputs);
    ait_output_shapes_out.reserve(num_outputs);

    std::map<std::string, unsigned> input_name_to_index;
    size_t total_dim_count =
        0; // the sum of shape.ndims for all input and output tensors
    // calculate total_dim_count
    for (unsigned i = 0; i < num_inputs; i++) {
      AITemplateParamShape shape;
      AITemplateModelContainerGetMaximumInputShape(handle, i, &shape);
      total_dim_count += shape.size;
    }
    for (unsigned i = 0; i < num_outputs; i++) {
      AITemplateParamShape shape;
      AITemplateModelContainerGetMaximumOutputShape(handle, i, &shape);
      total_dim_count += shape.size * 2; // allocation required twice
    }
    // this is just a vector that owns the memory for the shape.shape_data
    // values
    shape_data_owner.reserve(total_dim_count);
    size_t shape_offset = 0; // offset into the shape_data_owner array
    for (unsigned i = 0; i < num_inputs; i++) {
      // for each input tensor
      const char* name;
      AITemplateModelContainerGetInputName(handle, i, &name);
      AITemplateDtype dtype;
      AITemplateModelContainerGetInputDtype(handle, i, &dtype);
      size_t dtype_size = AITemplateDtypeSizeBytes(dtype);
      AITemplateParamShape shape;
      AITemplateModelContainerGetMaximumInputShape(handle, i, &shape);

      input_name_to_index.insert({name, i});
      std::cout << "Loading input: " << name << ", at idx: " << i;

      // Read metadata for test case
      unsigned int read_dtype;
      unsigned int read_dtype_size;
      unsigned int read_ndims;
      size_t read_total_tensor_bytes;
      read_element(fh, read_dtype);
      std::cout << ", dtype=" << read_dtype;
      read_element(fh, read_dtype_size);
      std::cout << ", sizeof(dtype)=" << read_dtype_size;
      read_element(fh, read_ndims);
      std::cout << ", ndims=" << read_ndims;

      if (static_cast<AITemplateDtype>(read_dtype) != dtype) {
        throw std::runtime_error(
            "Mismatch between dtype of input in testcase data and in model");
      }

      if (dtype_size != static_cast<size_t>(read_dtype_size)) {
        throw std::runtime_error(
            "Mismatch between sizeof(dtype) in testcase data and in model");
      }

      // Obtain maximum shape from model and verify the testcase data has valid
      // shape
      if (read_ndims != shape.size) {
        throw std::runtime_error(
            "Mismatch between number of input dimensions in testcase data and in model");
      }
      std::cout << ", shape=(";
      for (unsigned j = 0; j < read_ndims; j++) {
        size_t dim;
        read_element(fh, dim);
        shape_data_owner.push_back(dim);
        std::cout << dim << ", ";
        if (dim > shape.shape_data[j]) {
          throw std::runtime_error(
              "Shape in testcase data exceeds maximum shape.");
        }
      }
      std::cout << ")";

      // Set the shape of the input to the actual, and not the maximum shape.
      // the previous shape.shape_data may not be deleted as it's owned by the
      // model.
      shape.shape_data = shape_data_owner.data() + shape_offset;
      shape_offset += read_ndims; // move offset to the next unused space

      // total number of bytes of tensor raw data
      read_element(fh, read_total_tensor_bytes);

      size_t numel = shape.Numel();
      size_t num_bytes = numel * AITemplateDtypeSizeBytes(dtype);
      std::cout << ", total_tensor_bytes=" << read_total_tensor_bytes
                << " - model expects " << num_bytes << "\n";
      if (num_bytes != read_total_tensor_bytes) {
        throw std::runtime_error("Tensor data total size mismatch.");
      }
      // allocate memory for tensor raw data on host
      void* h_data;
      DEVICE_CHECK(DeviceMallocHost(&h_data, num_bytes));
      // read tensor raw data from file
      fh.read(reinterpret_cast<char*>(h_data), read_total_tensor_bytes);
      // Allocate corresponding device memory and copy tensor raw data to device
      gpu_data_owner.emplace_back(RAII_DeviceMalloc(num_bytes, allocator));
      DEVICE_CHECK(
          CopyToDevice(gpu_data_owner.back().get(), h_data, num_bytes));

      // free host memory for tensor
      DEVICE_CHECK(FreeDeviceHostMemory(h_data));

      inputs.push_back(AITData(gpu_data_owner.back().get(), shape, dtype));
    }
    std::cout << "Finished loading testcase inputs."
              << "\n";
    if (fh.peek() == std::ifstream::traits_type::eof()) {
      std::cout << "No expected outputs in testcase."
                << "\n";
      return;
    }
    if (inputs.size() != num_inputs) {
      throw std::runtime_error("Number of inputs mismatches with expected.");
    }
    // read expected outputs from file
    for (unsigned i = 0; i < num_outputs; i++) {
      // for each input tensor
      const char* name;
      AITemplateModelContainerGetOutputName(handle, i, &name);
      AITemplateDtype dtype;
      AITemplateModelContainerGetOutputDtype(handle, i, &dtype);
      size_t dtype_size = AITemplateDtypeSizeBytes(dtype);
      AITemplateParamShape shape;
      AITemplateModelContainerGetMaximumOutputShape(handle, i, &shape);
      AITemplateParamShape max_shape;
      AITemplateModelContainerGetMaximumOutputShape(handle, i, &max_shape);

      size_t max_numel = shape.Numel();
      size_t max_num_bytes = max_numel * AITemplateDtypeSizeBytes(dtype);

      gpu_data_owner.emplace_back(RAII_DeviceMalloc(max_num_bytes, allocator));
      gpu_outputs.push_back(
          AITData(gpu_data_owner.back().get(), max_shape, dtype));

      std::cout << "Loading expected output: " << name << ", at idx: " << i;

      // Read metadata for test case
      unsigned int read_dtype;
      unsigned int read_dtype_size;
      unsigned int read_ndims;
      size_t read_total_tensor_bytes;
      read_element(fh, read_dtype);
      std::cout << ", dtype=" << read_dtype;
      read_element(fh, read_dtype_size);
      std::cout << ", sizeof(dtype)=" << read_dtype_size;
      read_element(fh, read_ndims);
      std::cout << ", ndims=" << read_ndims;

      if (static_cast<AITemplateDtype>(read_dtype) != dtype) {
        throw std::runtime_error(
            "Mismatch between dtype of input in testcase data and in model");
      }

      if (dtype_size != static_cast<size_t>(read_dtype_size)) {
        throw std::runtime_error(
            "Mismatch between sizeof(dtype) in testcase data and in model");
      }

      // Obtain maximum shape from model and verify the testcase data has valid
      // shape
      if (read_ndims != shape.size) {
        throw std::runtime_error(
            "Mismatch between number of input dimensions in testcase data and in model");
      }
      std::cout << ", shape=(";
      for (unsigned j = 0; j < read_ndims; j++) {
        size_t dim;
        read_element(fh, dim);
        shape_data_owner.push_back(dim);
        std::cout << dim << ", ";
        if (dim > shape.shape_data[j]) {
          throw std::runtime_error(
              "Shape in testcase data exceeds maximum shape.");
        }
      }
      std::cout << ")";

      // Set the shape of the input to the actual, and not the maximum shape.
      // the previous shape.shape_data may not be deleted as it's owned by the
      // model.
      shape.shape_data = shape_data_owner.data() + shape_offset;
      shape_offset += read_ndims; // move offset to the next unused space

      // total number of bytes of tensor raw data
      read_element(fh, read_total_tensor_bytes);

      size_t numel = shape.Numel();
      size_t num_bytes = numel * AITemplateDtypeSizeBytes(dtype);
      std::cout << ", total_tensor_bytes=" << read_total_tensor_bytes
                << " - model expects " << num_bytes << "\n";
      if (num_bytes != read_total_tensor_bytes) {
        throw std::runtime_error("Tensor data total size mismatch.");
      }
      // allocate memory for tensor raw data on host
      void* h_data_expected;
      void* h_data;
      DEVICE_CHECK(
          DeviceMallocHost(&h_data, max_num_bytes)); // max size required here
      DEVICE_CHECK(DeviceMallocHost(&h_data_expected, num_bytes));

      // read tensor raw data from file
      fh.read(
          reinterpret_cast<char*>(h_data_expected), read_total_tensor_bytes);

      // ---
      // Memory to place output tensors on host
      host_outputs.emplace_back(h_data, shape, dtype);
      ait_output_shapes_out.push_back(shape_data_owner.data());
      shape_offset += read_ndims;
      expected_outputs.emplace_back(h_data_expected, shape, dtype);
    }
  }

  AITemplateError run(
      AITemplateModelHandle handle,
      AITemplateAllocator& allocator) {
    bool graph_mode = false;
    auto stream = RAII_StreamCreate(/*non_blocking=*/true);

    return AITemplateModelContainerRunWithOutputsOnHost(
        handle,
        inputs.data(),
        inputs.size(),
        host_outputs.data(),
        host_outputs.size(),
        reinterpret_cast<AITemplateStreamHandle>(stream.get()),
        graph_mode,
        ait_output_shapes_out.data());
  }

  float benchmark(
      AITemplateModelHandle handle,
      AITemplateAllocator& allocator,
      size_t count,
      size_t num_threads) {
    bool graph_mode = false;
    auto stream = RAII_StreamCreate(/*non_blocking=*/true);
    float runtime_ms = -999.0f;
    AITemplateError err = AITemplateModelContainerBenchmark(
        handle,
        inputs.data(),
        inputs.size(),
        gpu_outputs.data(),
        gpu_outputs.size(),
        reinterpret_cast<AITemplateStreamHandle>(stream.get()),
        graph_mode,
        count,
        num_threads,
        true,
        &runtime_ms,
        ait_output_shapes_out.data());
    if (err != AITemplateError::AITemplateSuccess) {
      std::cout << "Benchmark failed with error " << static_cast<int>(err)
                << std::endl;
      return -1.0f;
    }
    return runtime_ms;
  }

  bool compare_results_to_expected() {
    bool passed = true;
    size_t num_outputs = 0;
    AITemplateModelContainerGetNumOutputs(handle, &num_outputs);
    for (unsigned output_idx = 0; output_idx < num_outputs; ++output_idx) {
      switch (expected_outputs[output_idx].dtype) {
        case AITemplateDtype::kInt:
          passed = passed and _compare_results_to_expected<int32_t>(output_idx);
          break;
        case AITemplateDtype::kLong:
          passed = passed and _compare_results_to_expected<int64_t>(output_idx);
          break;
        case AITemplateDtype::kFloat:
          passed = passed and _compare_results_to_expected<float>(output_idx);
          break;
        case AITemplateDtype::kBFloat16:
          passed =
              passed and _compare_results_to_expected<bfloat16>(output_idx);
          break;
        case AITemplateDtype::kHalf:
          passed = passed and _compare_results_to_expected<half>(output_idx);
          break;
        case AITemplateDtype::kBool:
          passed = passed and _compare_results_to_expected<bool>(output_idx);
          break;
        default:
          std::cerr << "Unsupported output dtype! "
                    << static_cast<int>(expected_outputs[output_idx].dtype)
                    << std::endl;
          throw std::runtime_error("unsupported dtype for comparisons");
      }
    }
    return passed;
  }

  template <typename T>
  bool _compare_results_to_expected(unsigned output_idx) {
    unsigned ndims = host_outputs[output_idx].shape.size;
    // check the actual output shape
    for (unsigned i = 0; i < ndims; ++i) {
      if (expected_outputs[output_idx].shape.shape_data[i] !=
          ait_output_shapes_out[output_idx][i]) {
        std::cout
            << "Mismatch between expected output shape and actual shape after inference of output #"
            << i << " at dimension " << i << " expected shape[i]=="
            << host_outputs[output_idx].shape.shape_data[i]
            << " actual shape[i]==" << ait_output_shapes_out[output_idx][i]
            << std::endl;
        return false;
      }
    }
    size_t numel = host_outputs[output_idx].shape.Numel();
    T* data = reinterpret_cast<T*>(host_outputs[output_idx].ptr);
    T* expected_data = reinterpret_cast<T*>(expected_outputs[output_idx].ptr);
    size_t violations = 0;
    int worst_idx = -1;
    double worst_abs_diff = 0.0;

    for (size_t i = 0; i < numel; ++i) {
      double val = static_cast<double>(data[i]);
      double expected = static_cast<double>(expected_data[i]);
      double actual_diff = std::abs(val - expected);
      double tolerated_diff = atol +
          rtol * std::abs(expected); // as defined by torch.testing.assert_close
      if (actual_diff > worst_abs_diff) {
        worst_abs_diff = actual_diff;
      }
      if (actual_diff > tolerated_diff) {
        violations++;
      }
    }
    if (violations > 0) {
      std::cout
          << "Actual output and expected output are not equal for output with index "
          << output_idx << " of " << numel << " elements, " << violations
          << " differed by more than the tolerance of atol=" << atol
          << " and rtol=" << rtol << rtol << "\n";
      return false;
    }
    return true;
  }
};

int run_testcase(const char* input_file, bool benchmark) {
  std::cout << "Starting single test run with input " << input_file << "\n";
  {
    AITemplateModelHandle handle;
    AITemplateModelContainerCreate(&handle, /*num_runtimes*/ 1);
    AITemplateAllocator* allocator;
    AIT_ERROR_CHECK(AITemplateAllocatorCreate(
        &allocator, AITemplateAllocatorType::kDefault));

    auto deleter = [](void* data) { FreeDeviceHostMemory(data); };
    AITStandaloneTestcase test(input_file, handle, *allocator);

    AIT_ERROR_CHECK(test.run(handle, *allocator));
    std::cout << "Finished test run with input " << input_file << "\n";
    int retval = -1;
    if (!test.compare_results_to_expected()) {
      std::cout << "Test failed. " << std::endl;
      return 1;
    }
    std::cout << "Test succeeded. " << std::endl;
  }
  if (benchmark) {
    std::cout << "Benchmarking with testcase " << input_file << "\n";
    AITemplateModelHandle handle;
    AITemplateModelContainerCreate(&handle, /*num_runtimes*/ 1);
    AITemplateAllocator* allocator;
    AIT_ERROR_CHECK(AITemplateAllocatorCreate(
        &allocator, AITemplateAllocatorType::kDefault));

    auto deleter = [](void* data) { FreeDeviceHostMemory(data); };
    AITStandaloneTestcase benchmarker(input_file, handle, *allocator);
    float runtime_ms = benchmarker.benchmark(handle, *allocator, 10, 1);
    if (runtime_ms >= 0.0) {
      std::cout << "Benchmark result: " << input_file
                << " repetitions: 10, ms/iter: " << runtime_ms << "\n";
    }
  }

  return 0;
}

int run_with_random_inputs() {
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

int main(int argc, char* argv[]) {
  try {
    if (argc <= 1) {
      std::cout
          << "No action provided on commandline. Running model with random maximum size inputs."
          << std::endl;

      return run_with_random_inputs();
    }
    std::string action(argv[1]);
    if ((action == "--help") or (action == "help")) {
      std::cout << "AITemplate standalone test runner usage:" << std::endl
                << " run with random input:   " << argv[0] << std::endl
                << " run single tests:        " << argv[0]
                << " test <testcase-file-1> ... <testcase-file-N>" << std::endl
                << " run tests and benchmark: " << argv[0]
                << " benchmark <testcase-file-1> ... <testcase-file-N>"
                << std::endl;
    }
    if ((action == "test") or (action == "benchmark")) {
      if (argc < 3) {
        std::cout
            << "Invalid number of arguments. Require at least one test case as argument"
            << std::endl;
      }
      int failure_count = 0;
      for (int i = 2; i < argc; i++) {
        if (run_testcase(argv[i], action == "benchmark") != 0) {
          failure_count++;
        }
      }
      if (failure_count == 0) {
        std::cout << "All tests succeeded." << std::endl;
      } else {
        std::cout << "Failed tests: " << failure_count << " of " << (argc - 2)
                  << std::endl;
      }
      return failure_count;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return -99;
  }
}
