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

#include "constant_folder-generated.h"
#include "model-generated.h"
#include "model_interface.h"
#include "raii_wrapper.h"

#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace ait {

enum class BufferState {
  CLEAN = 0,
  CONSTANTS_UPDATED = 1,
  CONSTANTS_FOLDED = 2
};

// ModelContainer inherits from this class; its implementation is
// generated at compilation time. Most of the ModelContainer
// logic does not need codegen; anything that does should be put
// into this class instead.
class ModelContainerBase {
 public:
  ModelContainerBase(
      size_t num_inputs,
      size_t num_outputs,
      size_t num_bound_constants,
      size_t num_unbound_constants,
      size_t params_size,
      AITemplateAllocator& allocator);

 protected:
  // The set of bounded constants/weights/parameters. These are constants which
  // have value during compile time. We maintain it's size, and unlike unbound
  // constants, we do not need to check whether they are set via SetConstant
  // prior to inference.
  std::unordered_map<std::string, size_t> bound_constant_name_to_idx_;

  // The set of unbound constants/weights/parameters. These are constants which
  // have no value at compile time and do not participate in constant folding.
  // They must be set via SetConstant prior to inference.
  std::unordered_map<std::string, size_t> unbound_constant_name_to_idx_;

  // The names of all tensors that are required for constant folding, but are
  // not necessarily in the final graph.
  // constant_folding_optional_inputs_ are those that has initial value during
  // compile time.
  std::unordered_set<std::string> constant_folding_inputs_;
  std::unordered_set<std::string> constant_folding_optional_inputs_;

  // Offsets here correspond to the offsets of constants that were the outputs
  // of constant folding. The indices are guaranteed to map to the correct
  // indices in constant_folder_.
  std::vector<size_t> constant_folding_outputs_offsets_;
  // Offsets here correspond to the offsets of constants for bounded constants.
  std::vector<size_t> bound_constant_offsets_;

  // size for constants_ GPUPtr
  size_t constants_size_;
  // Pieces of memory for holding all constants, controled by
  // use_constants_primary_buffer_
  GPUPtr constants_primary_;
  GPUPtr constants_secondary_;
  bool use_constants_primary_buffer_;
  // State of whether SetConstants/FoldConstants was called.
  BufferState buffer_state_;
  // Mapping for constant names to pointer
  std::unordered_map<std::string, const void*> model_constants_;

  // size of the containers below: # inputs + # outputs + # unbound constants.
  size_t num_params_;

  // These entries correspond to inputs/outputs/unbound constants in order;
  // inputs first, then outputs, then constants.
  std::vector<const char*> param_names_;
  std::vector<std::vector<int64_t>> max_param_shapes_;
  std::vector<AITemplateDtype> param_dtypes_;

  // These are entries used for bound constants.
  std::vector<size_t> bound_constant_size_;
  std::vector<AITemplateDtype> bound_constant_dtypes_;

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
// model_container.Run(inputs0, n_inputs, outputs0, n_outputs, stream0, ...);
// model_container.Run(inputs1, n_inputs, outputs1, n_outputs, stream1, ...);
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
      size_t num_inputs,
      size_t num_outputs,
      size_t num_bound_constants,
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

  void Profile(
      const AITData* inputs,
      size_t num_inputs,
      AITData* outputs,
      size_t num_outputs,
      StreamType stream,
      size_t num_iters,
      const char* filename);

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
  void SetManyConstants(
      const char** names,
      const AITData* tensors,
      size_t num_tensors);

  uint8_t* GetInactiveConstantsBuffer();
  void SetDoubleBufferConstant(
      const char* name,
      const AITData& tensor,
      StreamType stream = 0);
  void SetManyDoubleBufferConstants(
      const char** names,
      const AITData* tensors,
      size_t num_tensors,
      StreamType stream = 0);

  size_t NumInputs() const;
  size_t NumOutputs() const;

  const char* InputName(size_t input_idx) const;
  const char* OutputName(size_t output_idx) const;

  AITemplateParamShape MaxInputShape(size_t input_idx) const;
  AITemplateParamShape MaxOutputShape(size_t output_idx) const;

  AITemplateDtype InputDtype(size_t input_idx) const;
  AITemplateDtype OutputDtype(size_t output_idx) const;

  size_t MaxOutputStorageBytes(size_t output_idx) const;

  size_t GetNumRuntimes() const {
    return models_.size();
  }

  void FoldConstants(StreamType stream, bool sync, bool double_buffer = false);
  void SwapConstants();

  size_t GetNumConstants(bool unbound_constants_only = true) const;
  size_t GetNumConstantFoldingInputs(bool unbound_constants_only = true) const;

  // Write all constant names to the array pointed to by names_out.
  // This function assumes that names_out has enough space to hold
  // at least GetNumConstants() pointers. The strings written
  // are guaranteed to live as long as their owning ModelContainer.
  void WriteAllConstantNamesTo(
      const char** names_out,
      bool unbound_constants_only,
      bool constant_folding_inputs_only) const;

 private:
  void WaitForAllModels(bool include_constant_folder = false);
  void FoldConstantsImpl(StreamType stream, bool double_buffer = false);
  void SetConstantImpl(
      const char* name,
      const AITData& tensor,
      bool use_secondary_buffer = false,
      StreamType stream = 0);
  void SwapConstantFolderBuffer();

  void PrepareForRun(
      Model* model,
      const AITData* inputs,
      size_t num_inputs,
      AITData* outputs,
      size_t num_outputs);

  Model* GetAvailableModel();
  void ReclaimFinishedModels(std::unique_lock<std::mutex>& lk);
  void ValidateParamDtype(AITemplateDtype dtype, size_t idx) const;
  void ValidateBoundConstantDtype(AITemplateDtype dtype, size_t idx) const;

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

  std::vector<std::unique_ptr<Model>> models_;
  std::unique_ptr<ConstantFolder> constant_folder_;
  std::vector<Model*> available_models_;
  std::deque<Model*> pending_models_;

  // Guards accesses to available/pending models.
  std::mutex models_mutex_;
  // Notified whenever a model is put into pending_models_.
  std::condition_variable pending_models_available_;
  // Prevents constant folding or SetConstants on main models from starting
  // while there are ongoing inferences (and vice versa). FoldConstants() and
  // SetConstants acquires in unique mode, Run()/Benchmark() acquire in shared
  // mode.
  //
  // Since constants_sync_mutex_ is acquired in shared mode for the entire
  // duration of Run()/Benchmark(), there is no need to acquire models_mutex_
  // while constants_sync_mutex_ is acquired in unique mode.
  // Why complicate things with two locks? The system is designed with the
  // assumption that concurrent inferences are common. We don't want to acquire
  // models_mutex_ uniquely for the entire duration of Run(), because that
  // prevents concurrent inferences from happening while kernels are being
  // queued.
  std::shared_mutex constants_sync_mutex_;
  // constants_double_buffer_mutex_ is separate from constants_sync_mutex since
  // when we use double buffer, it won't affect the main model.
  std::shared_mutex constants_double_buffer_mutex_;

  size_t num_inputs_;
  size_t num_outputs_;

  bool constant_folded_once_ = false;
};

} // namespace ait
