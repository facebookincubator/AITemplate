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

#include "model-generated.h"
#include "model_interface.h"
#include "raii_wrapper.h"

#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <numeric>
#include <unordered_map>

namespace ait {

// ModelContainer inherits from this class; its implementation is
// generated at compilation time. Most of the ModelContainer
// logic does not need codegen; anything that does should be put
// into this class instead.
class ModelContainerBase {
 public:
  ModelContainerBase(
      size_t num_inputs,
      size_t num_outputs,
      size_t num_unbound_constants,
      size_t params_size,
      AITemplateAllocator& allocator);

 protected:
  // The set of unbound constants/weights/parameters. These are constants which
  // have no value at compile time and do not participate in constant folding.
  // They must be set via SetConstant prior to inference.
  std::unordered_map<std::string, size_t> unbound_constant_name_to_idx_;

  // a single piece of memory for all constants
  GPUPtr constants_;

  // size of the containers below: # inputs + # outputs + # unbound constants.
  size_t num_params_;

  // These entries correspond to inputs/outputs/unbound constants in order;
  // inputs first, then outputs, then constants.
  std::vector<const char*> param_names_;
  std::vector<std::vector<int64_t>> max_param_shapes_;
  std::vector<AITemplateDtype> param_dtypes_;

  // NB: technically these could be derived from both the max shape and
  // the dytpe, but it's easier to just cache them.
  std::vector<size_t> max_param_storage_bytes_;
  std::vector<size_t> max_param_numel_;
};

// This creates a new ModelContainer; its implementation is also
// codegened (the parameters passed to the ctor are determined
// at compilation time)
class ModelContainer;
ModelContainer* CreateModelContainer(
    size_t num_runtimes,
    AITemplateAllocator& allocator);

// Each ModelContainer contains num_models Models. Inference runs
// can be started by invoking Run() with lists of pre-allocated
// input/output tensors. GetOutputMaximumShape() can be used to
// determine how much memory is required for each output.
//
// If there are N tensors marked with is_output=True,
// the user will always be expected to pass N output pointers -
// extra copies will occur if the outputs are views of constants,
// inputs, or other outputs in this case to avoid surprises.
//
// Use stream = nullptr for default stream. ModelContainer/Model does not
// create or own any stream. The user is expected to create and manage streams.
//
// We can support at most num_models concurrent inferences.
// Run() takes a stream to run the inference on. For example,
// to start up two inferences on different streams concurrently,
// we can do this:
//
// model_container.Run(inputs0, num_inputs, outputs0, num_ouputs, stream0, ...);
// model_container.Run(inputs1, num_inputs, outputs1, num_ouputs, stream1, ...);
// StreamSynchronize(stream0);
// StreamSynchronize(stream1);
//
// Note that if there are no models available for inference, Run() will block
// until one becomes available.
//
// ModelContainer optionally takes an allocator argument, which it will use to
// allocate the space for the buffers used for intermediate tensors and
// constants. If it is nullptr, the default allocator will be used (e.g. just
// {cuda/hip}{Malloc/Free}).
// Important: we assume that the allocator lives until the ModelContainer is
// destroyed. The default allocator has a static lifetime.
class ModelContainer : ModelContainerBase {
 public:
  ModelContainer(
      size_t num_models,
      size_t blob_size,
      size_t workspace_size,
      size_t num_inputs,
      size_t num_outputs,
      size_t num_unbound_constants,
      size_t params_size,
      AITemplateAllocator& allocator);

  void Run(
      const AITData* inputs,
      size_t num_inputs,
      AITData* outputs,
      size_t num_outputs,
      StreamType stream,
      bool sync,
      bool graph_mode,
      int64_t** output_shapes_out);

  void RunWithOutputsOnHost(
      const AITData* inputs,
      size_t num_inputs,
      AITData* outputs,
      size_t num_outputs,
      StreamType stream,
      bool graph_mode,
      int64_t** output_shapes_out);

  float Benchmark(
      const AITData* inputs,
      size_t num_inputs,
      AITData* outputs,
      size_t num_outputs,
      StreamType stream,
      bool graph_mode,
      size_t count,
      size_t num_threads,
      bool use_unique_stream_per_thread,
      int64_t** output_shapes_out);

  void SetConstant(const char* name, const AITData& tensor);

  size_t NumInputs() const;
  size_t NumOutputs() const;

  const char* InputName(size_t input_idx) const;
  const char* OutputName(size_t output_idx) const;

  AITemplateParamShape MaxOutputShape(size_t output_idx) const;
  AITemplateDtype OutputDtype(size_t output_idx) const;
  size_t MaxOutputStorageBytes(size_t output_idx) const;

  size_t GetNumRuntimes() const {
    return models_.size();
  }

 private:
  void PrepareForRun(
      Model* model,
      const AITData* inputs,
      size_t num_inputs,
      AITData* outputs,
      size_t num_outputs);

  Model* GetAvailableModel();
  void ReclaimFinishedModels(std::unique_lock<std::mutex>& lk);
  void ValidateDtype(AITemplateDtype dtype, size_t idx) const;

  float BenchmarkImpl(
      const AITData* inputs,
      size_t num_inputs,
      AITData* outputs,
      size_t num_outputs,
      StreamType stream,
      bool graph_mode,
      size_t count,
      int64_t** output_shapes_out);

  AITemplateAllocator& allocator_;

  std::vector<Model> models_;
  std::vector<Model*> available_models_;
  std::deque<Model*> pending_models_;

  // Guards accesses to available/pending models.
  std::mutex models_mutex_;
  // Notified whenever a model is put into pending_models_.
  std::condition_variable pending_models_available_;

  size_t num_inputs_;
  size_t num_outputs_;
};

} // namespace ait
