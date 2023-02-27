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

#include <stdexcept>
#include <string>

namespace ait {

inline void DeviceCheckLastError(const char* file, int line) {
  auto device_error = GetLastError();
  if (device_error != GetDeviceSuccess()) {
    std::string msg = std::string("Got error: ") + GetErrorString(device_error) +
        " enum: " + std::to_string(device_error) + " at " + file + ": " +
        std::to_string(line);
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }
}

// This serves as a base class for AIT runtime objects, e.g. the compiled
// model and the constant folder. It uses CRTP as a mechanism to call into
// a few base class methods (dynamic dispatch is not needed in ModelContainer,
// so there's no need to add a vtable). Inheriting classes should implement
// the following methods:
// - RunImpl(StreamType):    The bulk of the compiled model's kernel invocations
//                           go here.
// - SetUpInputsOutputs():   Check the provided input/output pointers dtypes &
//                           sizes
// - DeviceToDeviceCopies(): Called at the end of infernece, copy views of
//                           inputs/constants to the provided output pointer.
//
// In practice, inheriting classes are generated via MODEL_TEMPLATE in
// python/aitemplate/backend/main_templates.py.
template <typename ModelType>
class ModelBase {
 protected:
  // Should not be constructed directly, use the base class' factory function
  // instead.
  ModelBase(
      size_t blob_size,
      size_t workspace_size,
      size_t unique_workspace_size,
      size_t num_inputs,
      size_t num_outputs,
      size_t num_unbound_constants,
      uint8_t* constants,
      AITemplateAllocator& allocator)
      : blob_(RAII_DeviceMalloc(blob_size, allocator)),
        workspace_(RAII_DeviceMalloc(workspace_size, allocator)),
        params_(num_inputs + num_outputs + num_unbound_constants),
        num_inputs_(num_inputs),
        num_outputs_(num_outputs),
        constants_(constants) {
    global_workspace_ =
        static_cast<uint8_t*>(workspace_.get()) + unique_workspace_size;
    unique_workspace_ = static_cast<uint8_t*>(workspace_.get());
    DEVICE_CHECK(GetDevice(&device_idx_))
    DEVICE_CHECK(CreateEvent(&run_finished_));
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
    DEVICE_CHECK(cudaDeviceGetAttribute(
        &max_smem_size_, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_idx_));
#endif
    DEVICE_CHECK(GetDeviceProperties(&device_properties_, device_idx_));
    DEVICE_CHECK(StreamCreate(&graph_capture_stream_, /*non_blocking=*/true));
  }

 public:
  ~ModelBase() {
    if (run_finished_ != nullptr) {
      DestroyEvent(run_finished_);
    }
    if (graph_capture_stream_ != nullptr) {
      StreamDestroy(graph_capture_stream_);
    }
    if (graph_exec_ != nullptr) {
      GraphExecDestroy(graph_exec_);
    }
  }

  ModelBase(ModelBase&&) = delete;
  ModelBase& operator=(ModelBase&&) = delete;
  ModelBase(const ModelBase&) = delete;
  ModelBase& operator=(const ModelBase&) = delete;

  void Run(StreamType stream, bool graph_mode) {
    auto* model = static_cast<ModelType*>(this);
    model->SetUpInputsOutputs();
    if (target_has_graph_mode && graph_mode) {
      RunAsGraph(stream);
    } else {
      model->RunImpl(stream);
    }
    model->DeviceToDeviceCopies(stream);
    DEVICE_CHECK(EventRecord(run_finished_, stream));
  }

  void Profile(StreamType stream, size_t iters, const std::string& filename) {
    auto* model = static_cast<ModelType*>(this);
    model->SetUpInputsOutputs();
    model->ProfileImpl(stream, iters, filename);
  }

  bool IsPending() {
    auto query = QueryEvent(run_finished_);
    if (query == GetDeviceNotReady()) {
      return true;
    }
    if (query != GetDeviceSuccess()) {
      LOG(WARNING) << "Pending model run did not finish successfully. Error: "
                   << GetErrorString(query);
    }
    return false;
  }

  void WaitForCompletion() {
    DEVICE_CHECK(EventSynchronize(run_finished_));
  }

  size_t NumInputs() const {
    return num_inputs_;
  }

  size_t NumOutputs() const {
    return num_outputs_;
  }

  void SetParam(const void* src, size_t param_idx) {
    CHECK_VECTOR_ACCESS(params_, param_idx)
    // const_cast is not ideal here, but it is unfortunately
    // necessary:
    // 1) We store outputs and inputs in the same vector,
    //    and outputs cannot be const.
    // 2) Most of the codegen is not const-correct (most ops
    //    require non-const pointers). So even if we put const
    //    pointers into params, a const_cast would be required
    //    somewhere else.
    params_[param_idx].ptr = const_cast<void*>(src);
  }

  void SetInput(
      const void* src,
      const AITemplateParamShape& shape,
      size_t idx) {
    SetInputShape(shape, idx);
    SetParam(src, idx);
  }

  void SetOutput(void* src, size_t idx) {
    SetParam(src, idx + num_inputs_);
  }

  // Write the (possibly dynamic) output shape to the given pointer.
  // Note that this should be called _after_ the shape inference in
  // Run() is finished. output_shape_out should be able to store
  // at least GetOutputMaximumShape(idx).size values.
  void GetOutputShape(size_t idx, int64_t* output_shape_out) {
    const auto param_idx = idx + num_inputs_;
    CHECK_VECTOR_ACCESS(params_, param_idx);
    const auto& shape_ptrs = params_[param_idx].shape_ptrs;
    for (size_t i = 0; i < shape_ptrs.size(); ++i) {
      output_shape_out[i] = shape_ptrs[i].GetValue();
    }
  }

  void SetConstant(const char* name, const void* src) {
    auto it = constant_name_to_ptr_.find(name);
    if (it == constant_name_to_ptr_.end()) {
      throw std::out_of_range(std::string("Could not find constant ") + name);
    }
    const void** ptr = it->second;
    *ptr = src;
  }

 private:
  void SetInputShape(const AITemplateParamShape& shape, size_t idx) {
    auto& param = params_[idx];
    if (shape.size != param.shape_ptrs.size()) {
      throw std::runtime_error(
          "[SetInputShape] Got wrong param shape for input " +
          std::to_string(idx) + "; expected " +
          std::to_string(param.shape_ptrs.size()) + ", got " +
          std::to_string(shape.size));
    }
    for (size_t i = 0; i < param.shape_ptrs.size(); ++i) {
      param.shape_ptrs[i].SetValue(shape.shape_data[i], param.name);
    }
  }

  DeviceError EndCapture(GraphType* graph_ptr) {
    auto err = StreamEndCapture(graph_capture_stream_, graph_ptr);
    if (err != GetDeviceSuccess()) {
      // If we can't take the stream out of capture mode, something is probably
      // wrong with CUDA graph for this model (e.g. there might have been an
      // illegal capture mode operation). Disable graph mode to avoid such
      // issues in future iterations.
      target_has_graph_mode = false;
      LOG(WARNING) << "Graph capture failed to end. Disabling graph mode.";
      return err;
    }
    return GetDeviceSuccess();
  }

  void RunAsGraph(StreamType stream) {
    DEVICE_CHECK(StreamBeginCapture(graph_capture_stream_, /*global=*/false));
    try {
      static_cast<ModelType*>(this)->RunImpl(graph_capture_stream_);
    } catch (...) {
      GraphType graph;
      // No need to DEVICE_CHECK here, we want to see the original exception.
      EndCapture(&graph);
      if (graph != nullptr && GraphDestroy(graph) != GetDeviceSuccess()) {
        LOG(WARNING)
            << "Graph destruction failed while handling exception! Memory will be leaked.";
      }
      throw;
    }

    // The following function ends the capture and creates a graph
    // inside a unique_ptr that cleans up it when it goes out of scope.
    // Note that it throws an exception if EndCapture fails.
    auto graph = RAII_EndCaptureAndCreateGraph(
        [this](GraphType* graph_ptr) { return EndCapture(graph_ptr); });

    if (graph_exec_ == nullptr) {
      DEVICE_CHECK(GraphInstantiate(&graph_exec_, graph.get()));
    } else if (
        GraphExecUpdate(graph_exec_, graph.get()) != GetDeviceSuccess()) {
      // Consume the last cuda error, which may affect the next GraphExecLaunch
      // call.
      GetLastError();
      DEVICE_CHECK(GraphExecDestroy(graph_exec_));
      DEVICE_CHECK(GraphInstantiate(&graph_exec_, graph.get()));
    }

    DEVICE_CHECK(GraphExecLaunch(graph_exec_, stream));
  }

 protected:
  int device_idx_;
  int max_smem_size_{0};
  DevicePropertyType device_properties_;
  // This event tracks when the inference is finished
  // so that this Model may be reclaimed by its owning
  // ModelContainer.
  EventType run_finished_;
  // A blob of memory used for storing intermediate tensors.
  GPUPtr blob_;
  // Memory for constants that were folded into the *.so. Unowned by Model,
  // owned by ModelContainer.
  // TODO: make this const. It can't be const right now because we derive
  // tensor pointers from it, and no tensor pointers are const.
  uint8_t* constants_;
  size_t num_inputs_;
  size_t num_outputs_;

  // The workspace blob is used as scratch memory. See
  // _generate_workspace in memory planning for more information.
  GPUPtr workspace_;
  uint8_t* global_workspace_{nullptr};
  uint8_t* unique_workspace_{nullptr};

  class ParamDim {
   public:
    ParamDim(int64_t lower_bound, int64_t upper_bound, int64_t* value)
        : lower_bound_(lower_bound), upper_bound_(upper_bound), value_(value) {}

    void SetValue(int64_t new_value, const char* name = nullptr) {
      if (new_value < lower_bound_ || new_value > upper_bound_) {
        throw std::out_of_range(
            "[SetValue] Dimension got value out of bounds; expected value to be in [" +
            std::to_string(lower_bound_) + ", " + std::to_string(upper_bound_) +
            "], but got " + std::to_string(new_value) +
            (name ? ". Variable name: " + std::string(name) : "") + ".");
      }
      *value_ = new_value;
    }

    int64_t GetValue() const {
      return *value_;
    }

   private:
    int64_t lower_bound_;
    int64_t upper_bound_;
    int64_t* value_;
  };

  struct ParamInfo {
    void* ptr = nullptr;
    // TODO add offset
    const char* name;
    std::vector<ParamDim> shape_ptrs;
  };

  // Contains info for all tensors marked as inputs
  // or outputs. The first num_inputs elements are the inputs.
  // Constants are not included.
  std::vector<ParamInfo> params_;

  GraphExecType graph_exec_ = nullptr;
  StreamType graph_capture_stream_;

  std::unordered_map<std::string, const void**> constant_name_to_ptr_;
};

} // namespace ait
