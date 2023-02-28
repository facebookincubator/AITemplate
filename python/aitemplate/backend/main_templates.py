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
{% if debug_header %}
#include "debug_utility.h"
{% endif %}
#include "logging.h"
#include "device_functions-generated.h"
#include "model_interface.h"
#include "raii_wrapper.h"
#include "model.h"
#include "macros.h"
#include "jagged.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <math.h>

{{ function_decl }}

namespace ait {

// Model is the class that actually performs inference. It owns memory for
// intermediate tensors and dynamic dimensions. Constants are owned by
// the model's owning container object, and input/output memory is owned
// by the user.
// Once an inference run has started, it is not safe to re-use the Model
// until the run has finished!
class {{model_name}} : public ModelBase<{{model_name}}> {
  public:
    {{model_name}}(
        size_t blob_size,
        size_t workspace_size,
        size_t unique_workspace_size,
        size_t num_inputs,
        size_t num_outputs,
        size_t num_unbound_constants,
        uint8_t* constants,
        AITemplateAllocator& allocator)
        : ModelBase(
            blob_size,
            workspace_size,
            unique_workspace_size,
            num_inputs,
            num_outputs,
            num_unbound_constants,
            constants,
            allocator) {
    {{ set_up_constants }}
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
    {{ tensor_slice }}
    {{ tensor_map_set }}
    {{ set_up_param_dynamic_shapes }}
      }

    void SetUpInputsOutputs() {
        {{ set_inputs }}
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
        {{ reset_constants }}
    }

    void DeviceToDeviceCopies(StreamType stream) {
  {{ device_to_device_copies }}
    }

    void RunImpl(StreamType stream) {
        {% if profiler_annotation %}
        RAII_ProfilerRange _raiiAITProfilerRange("main_start");
        {% endif %}
  {% for func in function_seq %}
  {{ func }}
      DeviceCheckLastError(__FILE__, __LINE__);
  {% endfor %}
      DeviceToDeviceCopies(stream);
    }

    void ProfileImpl(StreamType stream, size_t iters, const std::string& filename) {
      std::ofstream ss(filename);
      if (!ss) {
        throw std::runtime_error(std::string("Could not open file ") + filename);
      }

      int deviceId;
      char* L2CacheSlab = nullptr;
      DevicePropertyType deviceProperties;
      GetDevice(&deviceId);
      GetDeviceProperties(&deviceProperties, deviceId);
      const size_t L2SizeInBytes = deviceProperties.l2CacheSize;
      DeviceMalloc((void**) &L2CacheSlab, L2SizeInBytes);

      ss << "{\\n";
      {% for func_name, func, input_sizes, output_sizes in per_op_profiler_seq %}
      {
        std::cout << "Profiling: " << "{{ func_name }}" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            {{ func }}
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\\"" << "{{ func_name }}" << "\\": { \\"ms_per_iter\\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \\"qps\\": " << 1000 * iters / milliseconds
           << ", \\"input_sizes\\": " << "{{ input_sizes | replace("'", '\\\\"') }}"
           << ", \\"output_sizes\\": " << "{{ output_sizes | replace("'", '\\\\"') }}"
           << " } ";
        {% if loop.last %}
          ss << "\\n";
        {% else %}
          ss << ",\\n";
        {% endif %}
      }
      {% endfor %}
      ss << "}\\n";

      DeviceToDeviceCopies(stream);
      std::cout << "AIT per op profiling finished." << std::endl;
      FreeDeviceMemory(L2CacheSlab);
    }

    static std::unique_ptr<{{model_name}}> Create(
      AITemplateAllocator& allocator,
      uint8_t* constants
    ) {
      return std::make_unique<{{model_name}}>(
          {{ blob_size }},
          {{ workspace_size }},
          {{ unique_workspace_size }},
          {{ num_inputs }},
          {{ num_outputs }},
          {{ num_unbound_constants }},
          constants,
          allocator
      );
    }

  private:
{{ tensor_decl }}
{{ dim_decl }}
{{ jagged_decl }}
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
    size_t num_bound_constants,
    size_t num_unbound_constants,
    size_t params_size,
    AITemplateAllocator& allocator)
    : constants_size_(params_size),
      constants_primary_(RAII_DeviceMalloc(constants_size_, allocator)),
      constants_secondary_(nullptr),
      use_constants_primary_buffer_(true),
      buffer_state_(BufferState::CLEAN),
      bound_constant_size_(num_bound_constants),
      bound_constant_dtypes_(num_bound_constants),
      num_params_(num_inputs + num_outputs + num_unbound_constants),
      param_names_(num_params_),
      param_dtypes_(num_params_),
      max_param_shapes_(num_params_),
      max_param_numel_(num_params_),
      max_param_storage_bytes_(num_params_) {
{{ set_up_constant_names }}
{{ set_up_param_names }}
{{ set_up_param_dtypes }}
{{ set_up_bound_constant_dtypes }}
{{ set_up_bound_constant_size }}
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
{{ set_up_constant_offsets }}
{{ set_up_constant_folding_inputs }}

  auto* constants_ptr = static_cast<uint8_t*>(constants_primary_.get());
  const auto binary_constants_bin_size = static_cast<size_t>(_binary_constants_bin_end - _binary_constants_bin_start);
  for (auto& constant_info : owned_constants) {
    auto* dst = constants_ptr + constant_info.internal_offset;
    if (constant_info.data_offset + constant_info.num_bytes > binary_constants_bin_size) {
      throw std::runtime_error(std::string("Copying constant ") + constant_info.name + " would overflow constant buffer");
    }
    DEVICE_CHECK(CopyToDevice(dst, _binary_constants_bin_start + constant_info.data_offset, constant_info.num_bytes));
  }
}

ModelContainer* CreateModelContainer(size_t num_runtimes, AITemplateAllocator& allocator) {
  // num_runtimes, blob_size, workspace_size, num_inputs, num_outputs, num_unbound_constants, param_size, allocator
  return new ModelContainer(num_runtimes, {{num_inputs}}, {{num_outputs}}, {{num_bound_constants}}, {{num_unbound_constants}}, {{param_size}}, allocator);
}
} // namespace ait
"""
)
