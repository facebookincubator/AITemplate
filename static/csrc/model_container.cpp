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
#include "model_container.h"

#include "device_functions-generated.h"
#include "raii_wrapper.h"

namespace {
std::string GetEnumString(AITemplateDtype dtype) {
  switch (dtype) {
    case AITemplateDtype::kUnset:
      return "kUnset";
    case AITemplateDtype::kHalf:
      return "kHalf";
    case AITemplateDtype::kFloat:
      return "kFloat";
    case AITemplateDtype::kInt:
      return "kInt";
    case AITemplateDtype::kLong:
      return "kLong";
    default:
      return "unknown";
  }
}
} // namespace

namespace ait {

ModelContainer::ModelContainer(
    size_t num_models,
    size_t num_inputs,
    size_t num_outputs,
    size_t num_bound_constants,
    size_t num_unbound_constants,
    size_t params_size,
    AITemplateAllocator& allocator)
    : ModelContainerBase(
          num_inputs,
          num_outputs,
          num_bound_constants,
          num_unbound_constants,
          params_size,
          allocator),
      allocator_(allocator),
      num_inputs_(num_inputs),
      num_outputs_(num_outputs) {
  if (num_models == 0) {
    throw std::runtime_error("Number of models must be positive");
  }
  dmlc::InitLogging("aitemplate"); // TODO(xxx): render network name
  int runtime_version;
  int driver_version;
  DEVICE_CHECK(GetDriverVersion(&driver_version));
  DEVICE_CHECK(GetRuntimeVersion(&runtime_version));
  LOG(INFO) << "Device Runtime Version: " << runtime_version
            << "; Driver Version: " << driver_version;

  int dev_id;
  DevicePropertyType prop;
  DEVICE_CHECK(GetDevice(&dev_id));
  DEVICE_CHECK(GetDeviceProperties(&prop, dev_id));

  bool useDebugLogging = false;
  if (auto var = std::getenv("LOGLEVEL")) {
    if (var[0] == 'd' || var[0] == 'D') {
      useDebugLogging = true;
    }
  }
  LOG(INFO)
      << (useDebugLogging ? PrintDebugDeviceProperties(prop)
                          : PrintInfoDeviceProperties(prop));

  LOG(INFO) << "Init AITemplate Runtime with " << num_models << " concurrency";
  models_.reserve(num_models);
  available_models_.reserve(num_models);

  auto* constants_ptr = static_cast<uint8_t*>(constants_primary_.get());
  for (size_t i = 0; i < num_models; ++i) {
    models_.push_back(Model::Create(allocator, constants_ptr));
    available_models_.push_back(models_.back().get());
  }

  constant_folder_ = ConstantFolder::Create(allocator, constants_ptr);

  // Wire up the constant folder's outputs to our constant buffer.
  size_t constant_idx = 0;
  for (auto offset : constant_folding_outputs_offsets_) {
    constant_folder_->SetOutput(constants_ptr + offset, constant_idx++);
  }
}

void ModelContainer::Run(
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    StreamType stream,
    bool sync,
    bool graph_mode,
    int64_t** output_shapes_out) {
  std::shared_lock constants_lk(constants_sync_mutex_);
  if (!constant_folded_once_) {
    // We don't require users to manually call FoldConstants the first time.
    // Note that if this throws (due to an unset constant, for example)
    // constant_folded_once_ will not be set.
    constants_lk.unlock();
    std::unique_lock constants_unique_lk(constants_sync_mutex_);
    // Check again, another thread may have updated after we unlocked.
    if (!constant_folded_once_) {
      FoldConstantsImpl(stream);
    }
    constants_unique_lk.unlock();
    constants_lk.lock();
  }
  auto* model = GetAvailableModel();
  try {
    PrepareForRun(model, inputs, num_inputs, outputs, num_outputs);
    model->Run(stream, graph_mode);
  } catch (...) {
    std::lock_guard lk(models_mutex_);
    available_models_.push_back(model);
    throw;
  }

  if (output_shapes_out) {
    for (size_t i = 0; i < num_outputs; ++i) {
      auto* out_shape = output_shapes_out[i];
      model->GetOutputShape(i, out_shape);
    }
  }

  {
    std::lock_guard lk(models_mutex_);
    pending_models_.push_back(model);
  }
  pending_models_available_.notify_one();
  if (sync) {
    StreamSynchronize(stream);
  }
}

void ModelContainer::Profile(
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    StreamType stream,
    size_t num_iters,
    const char* filename) {
  auto* model = GetAvailableModel();
  if (filename == nullptr) {
    throw;
  }
  try {
    PrepareForRun(model, inputs, num_inputs, outputs, num_outputs);
    model->Profile(stream, num_iters, filename);
  } catch (...) {
    std::lock_guard lk(models_mutex_);
    available_models_.push_back(model);
    throw;
  }

  {
    std::lock_guard lk(models_mutex_);
    pending_models_.push_back(model);
  }
  pending_models_available_.notify_one();
}

void ModelContainer::RunWithOutputsOnHost(
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    StreamType stream,
    bool graph_mode,
    int64_t** output_shapes_out) {
  std::vector<std::pair<GPUPtr, size_t>> owned_outputs_ptrs;
  std::vector<AITData> owned_outputs;
  owned_outputs_ptrs.reserve(num_outputs);
  owned_outputs.reserve(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    size_t num_bytes = MaxOutputStorageBytes(i);
    owned_outputs_ptrs.emplace_back(
        RAII_DeviceMalloc(num_bytes, allocator_), num_bytes);
    owned_outputs.emplace_back(
        owned_outputs_ptrs.back().first.get(),
        outputs[i].shape,
        outputs[i].dtype);
  }

  Run(inputs,
      num_inputs,
      owned_outputs.data(),
      num_outputs,
      stream,
      /*sync=*/false,
      graph_mode,
      output_shapes_out);

  for (size_t i = 0; i < num_outputs; ++i) {
    auto& owned_output = owned_outputs_ptrs[i];
    auto& ptr = owned_output.first;
    auto num_bytes = owned_output.second;
    DEVICE_CHECK(CopyToHost(outputs[i].ptr, ptr.get(), num_bytes, stream));
  }

  DEVICE_CHECK(StreamSynchronize(stream));
}

float ModelContainer::Benchmark(
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    StreamType stream,
    bool graph_mode,
    size_t count,
    size_t num_threads,
    bool use_unique_stream_per_thread,
    int64_t** output_shapes_out) {
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }

  std::shared_lock constants_lk(constants_sync_mutex_);
  if (!constant_folded_once_) {
    constants_lk.unlock();
    std::unique_lock constants_unique_lk(constants_sync_mutex_);
    // Check again, another thread may have updated after we unlocked.
    if (!constant_folded_once_) {
      FoldConstantsImpl(stream);
    }
    constants_unique_lk.unlock();
    constants_lk.lock();
  }

  if (num_threads == 1) {
    return BenchmarkImpl(
               inputs,
               num_inputs,
               outputs,
               num_outputs,
               stream,
               graph_mode,
               count,
               output_shapes_out) /
        count;
  }
  // Clone the outputs, each thread needs its own set
  std::vector<std::vector<GPUPtr>> per_thread_outputs_ptrs;
  std::vector<std::vector<AITData>> per_thread_outputs;
  std::vector<StreamPtr> per_thread_streams;
  per_thread_outputs_ptrs.reserve(num_threads - 1);
  per_thread_outputs.reserve(num_threads - 1);

  if (use_unique_stream_per_thread) {
    per_thread_streams.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      per_thread_streams.push_back(RAII_StreamCreate(/*non_blocking=*/true));
    }
  }

  for (size_t i = 1; i < num_threads; ++i) {
    std::vector<GPUPtr> cloned_outputs_ptrs;
    std::vector<AITData> cloned_outputs;

    cloned_outputs_ptrs.reserve(num_outputs);
    cloned_outputs.reserve(num_outputs);

    for (size_t j = 0; j < num_outputs; ++j) {
      size_t num_bytes = MaxOutputStorageBytes(j);
      cloned_outputs_ptrs.emplace_back(
          RAII_DeviceMalloc(num_bytes, allocator_));
      auto* new_pointer = cloned_outputs_ptrs.back().get();
      DEVICE_CHECK(
          DeviceToDeviceCopy(new_pointer, outputs[j].ptr, num_bytes, stream));
      cloned_outputs.emplace_back(
          new_pointer, outputs[j].shape, outputs[j].dtype);
    }
    per_thread_outputs_ptrs.push_back(std::move(cloned_outputs_ptrs));
    per_thread_outputs.push_back(std::move(cloned_outputs));
  }
  DEVICE_CHECK(StreamSynchronize(stream));

  auto get_stream = [stream, use_unique_stream_per_thread, &per_thread_streams](
                        size_t thread_idx) {
    if (!use_unique_stream_per_thread) {
      return stream;
    }
    return per_thread_streams[thread_idx].get();
  };

  auto thread_func = [&](size_t thread_idx) {
    AITData* thread_outputs =
        thread_idx == 0 ? outputs : per_thread_outputs[thread_idx - 1].data();
    StreamType thread_stream = get_stream(thread_idx);
    auto* thread_output_shapes_out =
        thread_idx == 0 ? output_shapes_out : nullptr;
    return BenchmarkImpl(
        inputs,
        num_inputs,
        thread_outputs,
        num_outputs,
        thread_stream,
        graph_mode,
        count,
        thread_output_shapes_out);
  };

  std::vector<std::future<float>> futures;
  futures.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, thread_func, i));
  }

  auto max_time = std::accumulate(
      futures.begin(), futures.end(), 0.f, [](float cur_val, auto& future) {
        return std::max(future.get(), cur_val);
      });

  // Verify that all the outputs are the same
  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_size = MaxOutputStorageBytes(i);
    auto output_host = std::make_unique<uint8_t[]>(output_size);
    // NB: technically, we don't have to copy to host here, but using
    // std::memcmp is easier than writing a kernel that does comparisons
    // for both backends, and performance is not important here.
    DEVICE_CHECK(
        CopyToHost(output_host.get(), outputs[i].ptr, output_size, stream));
    DEVICE_CHECK(StreamSynchronize(stream));

    for (size_t thread_idx = 1; thread_idx < num_threads; ++thread_idx) {
      auto* thread_output = per_thread_outputs[thread_idx - 1][i].ptr;
      auto thread_output_host = std::make_unique<uint8_t[]>(output_size);
      auto thread_stream = get_stream(thread_idx);
      DEVICE_CHECK(CopyToHost(
          thread_output_host.get(), thread_output, output_size, thread_stream));
      DEVICE_CHECK(StreamSynchronize(thread_stream));
      if (std::memcmp(
              output_host.get(), thread_output_host.get(), output_size)) {
        throw std::runtime_error(
            "Output " + std::to_string(i) +
            " did not match for a spawned thread!");
      }
    }
  }
  auto total_num_iters = num_threads * count;
  return max_time / total_num_iters;
}

void ModelContainer::SetConstantImpl(
    const char* name,
    const AITData& tensor,
    bool double_buffer,
    StreamType stream) {
  auto unbound_it = unbound_constant_name_to_idx_.find(name);
  auto bound_it = bound_constant_name_to_idx_.find(name);
  if (unbound_it != unbound_constant_name_to_idx_.end()) {
    auto constant_idx = unbound_it->second + num_inputs_ + num_outputs_;
    ValidateParamDtype(tensor.dtype, constant_idx);

    CHECK_VECTOR_ACCESS(max_param_storage_bytes_, constant_idx)
    auto expected_num_bytes = max_param_storage_bytes_[constant_idx];
    auto actual_num_bytes =
        tensor.shape.Numel() * AITemplateDtypeSizeBytes(tensor.dtype);
    if (expected_num_bytes != actual_num_bytes) {
      throw std::runtime_error(
          std::string(
              "SetConstant did not receive correct number of bytes for unbound constant ") +
          name + ": expected " + std::to_string(expected_num_bytes) +
          " but got " + std::to_string(actual_num_bytes) +
          ". Check that the provided tensor's shape is correct.");
    }
  } else if (bound_it != bound_constant_name_to_idx_.end()) {
    auto constant_idx = bound_it->second;
    ValidateBoundConstantDtype(tensor.dtype, constant_idx);

    CHECK_VECTOR_ACCESS(bound_constant_size_, constant_idx)
    auto expected_num_bytes = bound_constant_size_[constant_idx];
    auto actual_num_bytes =
        tensor.shape.Numel() * AITemplateDtypeSizeBytes(tensor.dtype);
    if (expected_num_bytes != actual_num_bytes) {
      throw std::runtime_error(
          std::string(
              "SetConstant did not receive correct number of bytes for bound constant ") +
          name + ": expected " + std::to_string(expected_num_bytes) +
          " but got " + std::to_string(actual_num_bytes) +
          ". Check that the provided tensor's shape is correct.");
    }
  } else {
    LOG(WARNING) << "Called SetConstant on " << name
       << " but can't find in either bound or unbound constant set";
    return;
  }

  auto* src = tensor.ptr;
  bool is_constant_folder_ =
      constant_folding_inputs_.find(name) != constant_folding_inputs_.end() ||
      constant_folding_optional_inputs_.find(name) !=
          constant_folding_optional_inputs_.end();

  if (!double_buffer) {
    // If we don't use double_buffer, we can just SetConstant.
    if (!is_constant_folder_) {
      for (auto& model : models_) {
        model->SetConstant(name, src);
      }
    } else {
      constant_folder_->SetConstant(name, src);
    }
  } else {
    // If we use double buffer, we identify whether it's a bounded constant or
    // not. If it's unbounded, just hold the pointer. It it's bounded, we copy
    // it into the constant buffer.
    if (unbound_it != unbound_constant_name_to_idx_.end()) {
      if (is_constant_folder_) {
        constant_folder_->SetConstant(name, src);
      } else {
        model_constants_[std::string(name)] = src;
      }
    } else {
      // Constant to be set is bounded, preload into constant buffer.
      uint8_t* constants_ptr = GetInactiveConstantsBuffer();
      size_t idx = bound_it->second;
      // TODO: check whether src is host or device memory.
      DEVICE_CHECK(DeviceToDeviceCopy(
          constants_ptr + bound_constant_offsets_[idx],
          src,
          bound_constant_size_[idx],
          stream));
    }
  }

  buffer_state_ = BufferState::CONSTANTS_UPDATED;
}

void ModelContainer::SetConstant(const char* name, const AITData& tensor) {
  std::lock_guard lk(constants_sync_mutex_);
  WaitForAllModels(/*include_constant_folder=*/true);
  SetConstantImpl(name, tensor);
}

void ModelContainer::SetManyConstants(
    const char** names,
    const AITData* tensors,
    size_t num_tensors) {
  if (num_tensors == 0) {
    return;
  }

  if (tensors == nullptr) {
    throw std::runtime_error("Tensor array cannot be null");
  }

  std::lock_guard lk(constants_sync_mutex_);
  WaitForAllModels(/*include_constant_folder=*/true);

  for (size_t i = 0; i < num_tensors; ++i) {
    const char* name = names[i];
    if (name == nullptr) {
      throw std::runtime_error("Constant name cannot be null");
    }
    const auto& tensor = tensors[i];
    SetConstantImpl(names[i], tensor);
  }
}

void ModelContainer::SwapConstantFolderBuffer() {
  uint8_t* constants_ptr = GetInactiveConstantsBuffer();
  constant_folder_->ResetConstants(constants_ptr);
  size_t constant_idx = 0;
  for (auto offset : constant_folding_outputs_offsets_) {
    constant_folder_->SetOutput(constants_ptr + offset, constant_idx++);
  }
}

uint8_t* ModelContainer::GetInactiveConstantsBuffer() {
  uint8_t* constants_ptr{nullptr};
  if (use_constants_primary_buffer_) {
    if (constants_secondary_ == nullptr) {
      constants_secondary_ = RAII_DeviceMalloc(constants_size_, allocator_);
    }
    constants_ptr = static_cast<uint8_t*>(constants_secondary_.get());
  } else {
    constants_ptr = static_cast<uint8_t*>(constants_primary_.get());
  }
  return constants_ptr;
}

void ModelContainer::SetDoubleBufferConstant(
    const char* name,
    const AITData& tensor,
    StreamType stream) {
  std::lock_guard lk(constants_double_buffer_mutex_);
  SetConstantImpl(name, tensor, /* double_buffer */ true, stream);
}

void ModelContainer::SetManyDoubleBufferConstants(
    const char** names,
    const AITData* tensors,
    size_t num_tensors,
    StreamType stream) {
  if (num_tensors == 0) {
    return;
  }

  if (tensors == nullptr) {
    throw std::runtime_error("Tensor array cannot be null");
  }

  std::lock_guard lk(constants_double_buffer_mutex_);
  for (size_t i = 0; i < num_tensors; ++i) {
    const char* name = names[i];
    if (name == nullptr) {
      throw std::runtime_error("Constant name cannot be null");
    }
    const auto& tensor = tensors[i];
    SetConstantImpl(names[i], tensor, /* double_buffer */ true, stream);
  }
}

size_t ModelContainer::NumInputs() const {
  return num_inputs_;
}

const char* ModelContainer::InputName(size_t input_idx) const {
  CHECK(input_idx < num_inputs_);
  CHECK_VECTOR_ACCESS(param_names_, input_idx)
  return param_names_[input_idx];
}

AITemplateParamShape ModelContainer::MaxInputShape(size_t input_idx) const {
  CHECK(input_idx < num_inputs_);
  CHECK_VECTOR_ACCESS(max_param_shapes_, input_idx)
  auto& input_shape = max_param_shapes_[input_idx];
  return AITemplateParamShape{input_shape.data(), input_shape.size()};
}

AITemplateDtype ModelContainer::InputDtype(size_t input_idx) const {
  CHECK(input_idx < num_inputs_);
  CHECK_VECTOR_ACCESS(param_dtypes_, input_idx)
  return param_dtypes_[input_idx];
}

size_t ModelContainer::NumOutputs() const {
  return num_outputs_;
}

const char* ModelContainer::OutputName(size_t output_idx) const {
  auto idx = output_idx + num_inputs_;
  CHECK_VECTOR_ACCESS(param_names_, idx)
  return param_names_[idx];
}

AITemplateParamShape ModelContainer::MaxOutputShape(size_t output_idx) const {
  auto idx = output_idx + num_inputs_;
  CHECK_VECTOR_ACCESS(max_param_shapes_, idx)
  auto& out_shape = max_param_shapes_[idx];
  return AITemplateParamShape{out_shape.data(), out_shape.size()};
}

AITemplateDtype ModelContainer::OutputDtype(size_t output_idx) const {
  auto idx = output_idx + num_inputs_;
  CHECK_VECTOR_ACCESS(param_dtypes_, idx)
  return param_dtypes_[idx];
}

size_t ModelContainer::MaxOutputStorageBytes(size_t output_idx) const {
  auto idx = output_idx + num_inputs_;
  CHECK_VECTOR_ACCESS(max_param_storage_bytes_, idx)
  return max_param_storage_bytes_[idx];
}

void ModelContainer::WaitForAllModels(bool include_constant_folder) {
  // Wait for all on-going inferences to finish.
  for (auto* model : pending_models_) {
    try {
      model->WaitForCompletion();
      // Something has gone horribly wrong if we hit these catch cases, but
      // there's not much we can do about it. Just put the model back into the
      // pool and carry on with folding.
    } catch (std::exception& e) {
      LOG(WARNING)
          << "Model threw exception when waiting for inference to finish: "
          << e.what() << ". Ignoring and continuing constant folding.";
    } catch (...) {
      LOG(WARNING)
          << "Model threw unknown exception when waiting for inference to finish. Ignoring and continuing constant foldng.";
    }
    available_models_.push_back(model);
  }

  if (include_constant_folder) {
    try {
      constant_folder_->WaitForCompletion();
    } catch (...) {
      LOG(WARNING)
          << "Constant folder threw exception while waiting for completion, ignoring.";
    }
  }
}

void ModelContainer::FoldConstantsImpl(StreamType stream, bool double_buffer) {
  if (constant_folded_once_) {
    // We do not set the buffer state if this is the initial constant folding.
    buffer_state_ = BufferState::CONSTANTS_FOLDED;
  }

  if (double_buffer) {
    SwapConstantFolderBuffer();
  } else {
    // NB: No need to acquire models_mutex_ here. We're guaranteed that nothing
    //     will be concurrently messing with the Model vectors while we hold
    //     the constants_sync_mutex_ in unique mode. See model_container.h for
    //     the full explanation.
    WaitForAllModels();
  }
  // We might have already started constant folding, make sure it finishes
  // first. It's OK if we throw here, there's no state to restore.
  // We just won't finish the folding and will need to do it again.
  constant_folder_->WaitForCompletion();
  if (double_buffer) {
    std::lock_guard constants_unique_lk(constants_double_buffer_mutex_);
    constant_folder_->Run(stream, /*graph_mode=*/false);
  } else {
    constant_folder_->Run(stream, /*graph_mode=*/false);
  }
  constant_folded_once_ = true;
}

void ModelContainer::FoldConstants(
    StreamType stream,
    bool sync,
    bool double_buffer) {
  if (double_buffer) {
    FoldConstantsImpl(stream, double_buffer);
  } else {
    std::lock_guard constant_folding_lk(constants_sync_mutex_);
    FoldConstantsImpl(stream);
  }
  if (sync) {
    DEVICE_CHECK(StreamSynchronize(stream));
  }
}

void ModelContainer::SwapConstants() {
  if (buffer_state_ != BufferState::CONSTANTS_FOLDED) {
    LOG(WARNING) << "Called SwapConstants without calling FoldConstants().";
    return;
  }
  std::unique_lock constants_unique_lk(constants_double_buffer_mutex_);
  uint8_t* constants_ptr = GetInactiveConstantsBuffer();
  use_constants_primary_buffer_ = !use_constants_primary_buffer_;

  for (auto& model : models_) {
    model->ResetConstants(constants_ptr);
  }
  for (auto& [name, src] : model_constants_) {
    for (auto& model : models_) {
      model->SetConstant(name.c_str(), src);
    }
  }

  model_constants_.clear();
  buffer_state_ = BufferState::CLEAN;
}

size_t ModelContainer::GetNumConstants(bool unbound_constants_only) const {
  if (unbound_constants_only) {
    return unbound_constant_name_to_idx_.size();
  } else {
    return unbound_constant_name_to_idx_.size() +
        bound_constant_name_to_idx_.size();
  }
}

size_t ModelContainer::GetNumConstantFoldingInputs(
    bool unbound_constants_only) const {
  if (unbound_constants_only) {
    return constant_folding_inputs_.size();
  } else {
    return constant_folding_inputs_.size() +
        constant_folding_optional_inputs_.size();
  }
}

void ModelContainer::WriteAllConstantNamesTo(
    const char** constant_names_out,
    bool unbound_constants_only,
    bool constant_folding_inputs_only) const {
  size_t num_to_write = constant_folding_inputs_only
      ? GetNumConstants(unbound_constants_only)
      : GetNumConstantFoldingInputs(unbound_constants_only);
  if (constant_names_out == nullptr && num_to_write != 0) {
    throw std::runtime_error("constant_names_out cannot be nullptr.");
  }
  size_t idx = 0;
  for (auto& [name, _] : unbound_constant_name_to_idx_) {
    if (!constant_folding_inputs_only ||
        constant_folding_inputs_.find(name) != constant_folding_inputs_.end()) {
      constant_names_out[idx++] = name.c_str();
    }
  }
  if (!unbound_constants_only) {
    for (auto& [name, _] : bound_constant_name_to_idx_) {
      if (!constant_folding_inputs_only ||
          constant_folding_optional_inputs_.find(name) !=
              constant_folding_optional_inputs_.end()) {
        constant_names_out[idx++] = name.c_str();
      }
    }
  }
}

void ModelContainer::PrepareForRun(
    Model* model,
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs) {
  if (num_inputs != num_inputs_) {
    auto msg = "Got wrong number of inputs; expected " +
        std::to_string(num_inputs_) + ", got " + std::to_string(num_inputs);
    throw std::runtime_error(std::move(msg));
  }
  if (num_inputs > 0 && inputs == nullptr) {
    throw std::runtime_error("inputs cannot be null");
  }
  if (num_outputs != num_outputs_) {
    auto msg = "Got wrong number of outputs; expected " +
        std::to_string(num_outputs_) + ", got " + std::to_string(num_outputs);
    throw std::runtime_error(std::move(msg));
  }
  if (num_outputs > 0 && outputs == nullptr) {
    throw std::runtime_error("outputs cannot be null");
  }
  for (size_t i = 0; i < num_inputs_; ++i) {
    auto& input = inputs[i];
    ValidateParamDtype(input.dtype, i);
    model->SetInput(input.ptr, input.shape, i);
  }

  for (size_t i = 0; i < num_outputs_; ++i) {
    auto& output = outputs[i];
    ValidateParamDtype(output.dtype, i + num_inputs_);
    model->SetOutput(output.ptr, i);
  }
}

Model* ModelContainer::GetAvailableModel() {
  std::unique_lock lk(models_mutex_);
  if (available_models_.empty()) {
    ReclaimFinishedModels(lk);
  }
  auto* result = available_models_.back();
  available_models_.pop_back();
  return result;
}

void ModelContainer::ReclaimFinishedModels(std::unique_lock<std::mutex>& lk) {
  // Put any complete models at the end
  auto it = std::stable_partition(
      pending_models_.begin(), pending_models_.end(), [](Model* m) {
        return m->IsPending();
      });

  if (it != pending_models_.end()) {
    // Move all available models to the pool.
    available_models_.insert(
        available_models_.end(), it, pending_models_.end());
    pending_models_.erase(it, pending_models_.end());
    return;
  }

  pending_models_available_.wait(
      lk, [this]() { return !pending_models_.empty(); });
  // There are no available workspaces! We have to wait on one.
  auto* model = pending_models_.front();
  pending_models_.pop_front();
  lk.unlock();
  try {
    model->WaitForCompletion();
  } catch (...) {
    lk.lock();
    available_models_.push_back(model);
    throw;
  }
  lk.lock();
  available_models_.push_back(model);
}

void ModelContainer::ValidateParamDtype(AITemplateDtype dtype, size_t idx)
    const {
  CHECK_VECTOR_ACCESS(param_dtypes_, idx)
  if (dtype != param_dtypes_[idx]) {
    throw std::runtime_error(
        "Got wrong dtype for param " + std::to_string(idx) + "; expected " +
        GetEnumString(param_dtypes_[idx]) + ", got " + GetEnumString(dtype));
  }
}

void ModelContainer::ValidateBoundConstantDtype(
    AITemplateDtype dtype,
    size_t idx) const {
  CHECK_VECTOR_ACCESS(bound_constant_dtypes_, idx)
  if (dtype != bound_constant_dtypes_[idx]) {
    throw std::runtime_error(
        "Got wrong dtype for param " + std::to_string(idx) + "; expected " +
        GetEnumString(bound_constant_dtypes_[idx]) + ", got " +
        GetEnumString(dtype));
  }
}

float ModelContainer::BenchmarkImpl(
    const AITData* inputs,
    size_t num_inputs,
    AITData* outputs,
    size_t num_outputs,
    StreamType stream,
    bool graph_mode,
    size_t count,
    int64_t** output_shapes_out) {
  auto* model = GetAvailableModel();
  float runtime_ms = 0.;
  auto start_event = RAII_CreateEvent();
  auto end_event = RAII_CreateEvent();
  try {
    PrepareForRun(model, inputs, num_inputs, outputs, num_outputs);
    DEVICE_CHECK(EventRecord(start_event.get(), stream));

    for (size_t i = 0; i < count; ++i) {
      model->Run(stream, graph_mode);
    }
  } catch (...) {
    std::lock_guard lk(models_mutex_);
    available_models_.push_back(model);
    throw;
  }
  if (output_shapes_out) {
    for (size_t i = 0; i < num_outputs; ++i) {
      auto* out_shape = output_shapes_out[i];
      model->GetOutputShape(i, out_shape);
    }
  }
  // Push the model back into the pool before synchronizing the event
  // to exercise the concurrency code
  {
    std::lock_guard lk(models_mutex_);
    pending_models_.push_back(model);
  }
  pending_models_available_.notify_one();

  DEVICE_CHECK(EventRecord(end_event.get(), stream));
  DEVICE_CHECK(EventSynchronize(end_event.get()));
  DEVICE_CHECK(
      EventElapsedTime(&runtime_ms, start_event.get(), end_event.get()));
  LOG(INFO) << "Benchmark runtime ms/iter: " << runtime_ms / count;
  return runtime_ms;
}

} // namespace ait
