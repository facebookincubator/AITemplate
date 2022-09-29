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
This file contains class definitions used in the generated main.cu file.
"""
import jinja2


MODEL_TEMPLATE = jinja2.Template(
    """
#pragma once
#include "logging.h"
#include "device_functions-generated.h"
#include "model_interface.h"
#include "raii_wrapper.h"
#include "macros.h"
#include <algorithm>
#include <deque>
#include <string>
#include <unordered_map>
#include <math.h>

{{ function_decl }}

#define CHECK_VECTOR_ACCESS(vector, idx)                                  \\
  if (idx >= vector.size()) {                                             \\
    throw std::out_of_range(                                              \\
        "[__func__]: index out of range, " #vector ".size()=" +           \\
        std::to_string(vector.size()) + ", got " + std::to_string(idx));  \\
  }

namespace ait {
namespace {
void DeviceCheckLastError(const char* file, int line) {
  auto device_error = GetLastError();
  if (device_error != GetDeviceSuccess()) {
    std::string msg = std::string("Got error: ") + GetLastErrorString() +
                      " enum: " + std::to_string(device_error) +
                      " at " + file + ": " + std::to_string(line);
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }
}
}

// Model is the class that actually performs inference. It owns memory for
// intermediate tensors and dynamic dimensions. Constants are owned by
// the model's owning container object, and input/output memory is owned
// by the user.
// Once an inference run has started, it is not safe to re-use the Model
// until the run has finished!
class Model {
  public:
  Model(
      size_t blob_size,
      size_t workspace_size,
      size_t num_inputs,
      size_t num_outputs,
      size_t num_unbound_constants,
      uint8_t* constants)
      : blob(RAII_DeviceMalloc(blob_size)),
        workspace(RAII_DeviceMalloc(workspace_size)),
        params(num_inputs + num_outputs + num_unbound_constants),
        num_inputs(num_inputs),
        constants(constants) {
      dmlc::InitLogging("aitemplate"); // TODO(xxx): render network name
      LOG(INFO) << "Init AITemplate Runtime.";
      global_workspace = static_cast<uint8_t*>(workspace.get()) + {{ unique_workspace_size }};
      unique_workspace = static_cast<uint8_t*>(workspace.get());
      DEVICE_CHECK(GetDevice(&device_idx))
      DEVICE_CHECK(CreateEvent(&run_finished));
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
      DEVICE_CHECK(cudaDeviceGetAttribute(
        &max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_idx));
#endif
      DEVICE_CHECK(GetDeviceProperties(&device_properties, device_idx));
      DEVICE_CHECK(StreamCreate(&graph_capture_stream, /*non_blocking=*/true));

  {{ set_up_constants }}
      auto* blob_ptr = static_cast<uint8_t*>(blob.get());
  {{ tensor_slice }}
  {{ tensor_map_set }}
  {{ set_up_param_dynamic_shapes }}
    }

    ~Model() {
      DestroyEvent(run_finished);
      StreamDestroy(graph_capture_stream);
      if (graph_exec != nullptr) {
        GraphExecDestroy(graph_exec);
      }
      if (graph != nullptr) {
        GraphDestroy(graph);
      }
    }

    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void SetUpInputsOutputs() {
        {{ set_inputs }}
    }

    void DeviceToDeviceCopies(StreamType stream) {
  {{ device_to_device_copies }}
    }

    void Run(StreamType stream, bool graph_mode) {
      SetUpInputsOutputs();
      if (target_has_graph_mode && graph_mode) {
        RunAsGraph(stream);
      } else {
        RunImpl(stream);
      }
      DEVICE_CHECK(EventRecord(run_finished, stream));
    }

    void RunImpl(StreamType stream) {
  {% for func in function_seq %}
  {{ func }}
      DeviceCheckLastError(__FILE__, __LINE__);
  {% endfor %}
      DeviceToDeviceCopies(stream);
    }

    bool IsPending() {
      auto query = QueryEvent(run_finished);
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
      DEVICE_CHECK(EventSynchronize(run_finished));
    }

    size_t NumInputs() const {
      return num_inputs;
    }

    size_t NumOutputs() const {
      return params.size() - num_inputs;
    }

    void SetParam(const void* src, size_t param_idx) {
      CHECK_VECTOR_ACCESS(params, param_idx)
      // const_cast is not ideal here, but it is unfortunately
      // necessary:
      // 1) We store outputs and inputs in the same vector,
      //    and outputs cannot be const.
      // 2) Most of the codegen is not const-correct (most ops
      //    require non-const pointers). So even if we put const
      //    pointers into params, a const_cast would be required
      //    somewhere else.
      params[param_idx].ptr = const_cast<void*>(src);
    }

    void SetInput(const void* src, const AITemplateParamShape& shape, size_t idx) {
      SetInputShape(shape, idx);
      SetParam(src, idx);
    }

    void SetOutput(void* src, size_t idx) {
      SetParam(src, idx + num_inputs);
    }

    // Write the (possibly dynamic) output shape to the given pointer.
    // Note that this should be called _after_ the shape inference in
    // Run() is finished. output_shape_out should be able to store
    // at least GetOutputMaximumShape(idx).size values.
    void GetOutputShape(size_t idx, int64_t* output_shape_out) {
      const auto param_idx = idx + num_inputs;
      CHECK_VECTOR_ACCESS(params, param_idx);
      const auto& shape_ptrs = params[param_idx].shape_ptrs;
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
      auto& param = params[idx];
      if (shape.size != param.shape_ptrs.size()) {
        throw std::runtime_error(
          "[SetInputShape] Got wrong param shape for input " + std::to_string(idx) +
          "; expected " + std::to_string(param.shape_ptrs.size()) + ", got " +
          std::to_string(shape.size));
      }
      for (size_t i = 0; i < param.shape_ptrs.size(); ++i) {
        param.shape_ptrs[i].SetValue(shape.shape_data[i]);
      }
    }

    void RunAsGraph(StreamType stream) {
      DEVICE_CHECK(StreamBeginCapture(graph_capture_stream));
      try {
        RunImpl(graph_capture_stream);
      } catch (...) {
        DEVICE_CHECK(StreamEndCapture(graph_capture_stream, &graph));
        throw;
      }
      DEVICE_CHECK(StreamEndCapture(graph_capture_stream, &graph));

      if (graph_exec == nullptr) {
        DEVICE_CHECK(GraphInstantiate(&graph_exec, graph));
      } else if (GraphExecUpdate(graph_exec, graph) != GetDeviceSuccess()) {
        DEVICE_CHECK(GraphExecDestroy(graph_exec));
        DEVICE_CHECK(GraphInstantiate(&graph_exec, graph));
      }

      DEVICE_CHECK(GraphExecLaunch(graph_exec, stream));
    }

    int device_idx;
    int max_smem_size{0};
    DevicePropertyType device_properties;
    // This event tracks when the inference is finished
    // so that this Model may be reclaimed by its owning
    // ModelContainer.
    EventType run_finished;
    // A blob of memory used for storing intermediate tensors.
    GPUPtr blob;
    // Memory for constants that were folded into the *.so. Unowned by Model,
    // owned by ModelContainer.
    // TODO: make this const. It can't be const right now because we derive
    // tensor pointers from it, and no tensor pointers are const.
    uint8_t* constants;
    size_t num_inputs;

    // The workspace blob is used as scratch memory. See
    // _generate_workspace in memory planning for more information.
    GPUPtr workspace;
    uint8_t* global_workspace{nullptr};
    uint8_t* unique_workspace{nullptr};

    class ParamDim {
      public:
        ParamDim(int64_t lower_bound, int64_t upper_bound, int64_t* value) :
          lower_bound_(lower_bound),
          upper_bound_(upper_bound),
          value_(value) {}

        void SetValue(int64_t new_value) {
          if (new_value < lower_bound_ || new_value > upper_bound_) {
            throw std::out_of_range(
              "[SetValue] Dimension got value out of bounds; expected value to be in [" +
              std::to_string(lower_bound_) + ", " + std::to_string(upper_bound_) + "], but got " +
              std::to_string(new_value)
            );
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
    std::vector<ParamInfo> params;

    GraphExecType graph_exec = nullptr;
    GraphType graph = nullptr;
    StreamType graph_capture_stream;

    std::unordered_map<std::string, const void**> constant_name_to_ptr_;

    constexpr static bool target_has_graph_mode = {{ target_has_graph_mode }};

{{ tensor_decl }}
{{ dim_decl }}
{{ function_state }}
};
} // namespace ait
"""
)

MODEL_CONTAINER_TEMPLATE = jinja2.Template(
    """
#include "model_container.h"
#include "owned_constants.h"

namespace ait {
namespace {
// Contains the metadata for each constant.
constexpr std::array<ConstantInfo, {{ num_constants }}> owned_constants = {
  {{ owned_constants_init }}
};
} // namespace

ModelContainerBase::ModelContainerBase(
    size_t num_inputs,
    size_t num_outputs,
    size_t num_unbound_constants,
    size_t params_size)
    : constants_(RAII_DeviceMalloc(params_size)),
      num_params_(num_inputs + num_outputs + num_unbound_constants),
      param_names_(num_params_),
      param_dtypes_(num_params_),
      max_param_shapes_(num_params_),
      max_param_numel_(num_params_),
      max_param_storage_bytes_(num_params_) {
{{ set_up_constant_names }}
{{ set_up_param_names }}
{{ set_up_param_dtypes }}
{{ set_up_output_shapes }}
  for (size_t i = 0; i < num_params_; ++i) {
    max_param_numel_[i] = std::accumulate(
      max_param_shapes_[i].begin(),
      max_param_shapes_[i].end(),
      1,
      std::multiplies<int64_t>()
    );
    max_param_storage_bytes_[i] = max_param_numel_[i] * AITemplateDtypeSizeBytes(param_dtypes_[i]);
  }

  auto* constants_ptr = static_cast<uint8_t*>(constants_.get());
  DEVICE_CHECK(DeviceMemset(constants_ptr, 0, params_size));
  const auto binary_constants_bin_size = static_cast<size_t>(_binary_constants_bin_end - _binary_constants_bin_start);
  for (auto& constant_info : owned_constants) {
    auto* dst = constants_ptr + constant_info.internal_offset;
    if (constant_info.data_offset + constant_info.num_bytes > binary_constants_bin_size) {
      throw std::runtime_error(std::string("Copying constant ") + constant_info.name + " would overflow constant buffer");
    }
    DEVICE_CHECK(CopyToDevice(dst, _binary_constants_bin_start + constant_info.data_offset, constant_info.num_bytes));
  }
}

ModelContainer* CreateModelContainer(size_t num_runtimes) {
  // num_runtimes, blob_size, workspace_size, num_inputs, num_outputs, num_unbound_constants, param_size
  return new ModelContainer(num_runtimes, {{blob_size}}, {{workspace_size}}, {{num_inputs}}, {{num_outputs}}, {{num_unbound_constants}}, {{param_size}});
}
} // namespace ait
"""
)
