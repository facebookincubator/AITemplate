#pragma once

#include "model_interface.h" // @manual=//aitemplate/AITemplate/static/include:aitemplate

#include <dlfcn.h>
#include <torch/torch.h> // @manual=//caffe2:torch-cpp
#include <memory>
#include <optional>

#ifdef FBCODE_AIT
#include "folly/container/F14Map.h"
#endif

namespace torch::aitemplate {

class AITemplatePyTorchCachingAllocator : public AITemplateAllocator {
 public:
  AITemplatePyTorchCachingAllocator();
  void* Allocate(size_t num_bytes) override;
  void Free(void* ptr) override;

 private:
  c10::Allocator* cuda_allocator_;
};

class AITModelImpl {
 public:
  explicit AITModelImpl(
      const std::string& model_path,
      std::vector<std::string> input_names,
      std::vector<std::string> output_names,
      c10::optional<at::ScalarType> input_dtype,
      c10::optional<at::ScalarType> output_dtype,
      int64_t num_runtimes = 2,
      bool use_cuda_graph = false);

  ~AITModelImpl() {
    if (model_handle_) {
      deleteFunc_(model_handle_);
    }
  }

  std::vector<torch::Tensor> forward(std::vector<torch::Tensor>& inputs);

  void profile(
      std::vector<torch::Tensor>& inputs,
      const std::string& filename,
      size_t num_iters);

  // If we need to move or copy this object, then we should just
  // define a unique_ptr with deleter for the handle.
  AITModelImpl(const AITModelImpl&) = delete;
  AITModelImpl& operator=(const AITModelImpl&) = delete;

  static void registerLibraryNameToPathMap(
      std::unordered_map<std::string, std::string> map);

  static const std::string& getFullPathForLibraryName(const std::string& name);

  static bool getDeserializePickledModel();

  static void setDeserializePickledModel(bool deserializePickledModel);

  /*
   * Returns a path to .so file (either relative or absolute).
   */
  const std::string& libraryPath() const {
    return library_path_;
  }

  void setUseCudaGraph(bool use_cuda_graph) {
    use_cuda_graph_ = use_cuda_graph;
  }

  bool getUseCudaGraph() const {
    return use_cuda_graph_;
  }

  const std::string& libraryBasename() const {
    return library_basename_;
  }

  const std::vector<std::string>& inputNames() const {
    return input_names_;
  }

  const std::vector<std::string>& outputNames() const {
    return output_names_;
  }

  const c10::optional<at::ScalarType> floatingPointInputDtype() const {
    return floating_point_input_dtype_;
  }

  const c10::optional<at::ScalarType> floatingPointOutputDtype() const {
    return floating_point_output_dtype_;
  }

  void updateConstantsWithWeights(
      const std::unordered_map<std::string, torch::Tensor>& weights);

  void swapConstants();

 private:
  // @lint-ignore CLANGTIDY facebook-hte-NonPodStaticDeclaration
  static thread_local std::unordered_map<std::string, std::string>
      name_to_path_map_;
  static thread_local bool deserialize_pickled_model_;

  struct DlcloseDeleter {
    void operator()(void* p) const {
      if (p) {
        dlclose(p);
      }
    }
  };

  std::vector<AITData> processInputs(
      std::vector<torch::Tensor>& inputs,
      std::vector<torch::Tensor>& inputs_contig);

  std::vector<torch::Tensor> processOutputs(
      std::vector<c10::intrusive_ptr<c10::StorageImpl>>&
          output_index_to_output_storage_impl,
      const std::vector<std::vector<int64_t>>& output_shapes);

  void allocateOutputs(
      std::vector<c10::intrusive_ptr<c10::StorageImpl>>&
          output_index_to_output_storage_impl,
      std::vector<AITData>& ait_outputs,
      std::vector<std::vector<int64_t>>& output_shapes,
      std::vector<int64_t*>& output_shape_ptrs,
      const c10::Device& device);

  const std::unique_ptr<void, DlcloseDeleter> handle_ = nullptr;
  AITemplateModelHandle model_handle_;

  decltype(&AITemplateModelContainerDelete) deleteFunc_ = nullptr;
  decltype(&AITemplateModelContainerRun) runFunc_ = nullptr;
  decltype(&AITemplateModelContainerProfile) profileFunc_ = nullptr;
  decltype(&AITemplateModelContainerGetOutputName) getOutputNameFunc_ = nullptr;
  decltype(&AITemplateModelContainerGetMaximumOutputShape)
      getMaximumOutputShapeFunc_ = nullptr;
  decltype(&AITemplateModelContainerGetOutputDtype) getOutputDtypeFunc_ =
      nullptr;
  decltype(&AITemplateModelContainerSetManyDoubleBufferConstants)
      setManyConstantsDoubleBufferFunc_ = nullptr;
  decltype(&AITemplateModelContainerFoldConstants) foldConstantsFunc_ = nullptr;
  decltype(&AITemplateModelContainerGetConstantNames) getConstantNamesFunc_ =
      nullptr;
  decltype(&AITemplateModelContainerGetNumConstants) getNumConstantsFunc_ =
      nullptr;
  decltype(&AITemplateModelContainerSwapConstants) swapConstantsFunc_ = nullptr;
  decltype(&AITemplateModelContainerFoldConstantsInDoubleBuffer)
      foldConstantsDoubleBufferFunc_ = nullptr;

  const std::string library_basename_;
  const std::string library_path_;
  const std::vector<std::string> input_names_;
  const std::vector<std::string> output_names_;
  const c10::optional<at::ScalarType> floating_point_input_dtype_;
  const c10::optional<at::ScalarType> floating_point_output_dtype_;
#ifdef FBCODE_AIT
  folly::F14FastMap<const char*, size_t> input_name_to_index_;
  folly::F14FastMap<const char*, size_t> output_name_to_index_;
#else
  std::unordered_map<std::string, size_t> input_name_to_index_;
  std::unordered_map<std::string, size_t> output_name_to_index_;
#endif

  // Whether to use CUDA graph when launching the model. Defaults to
  // FLAGS_ait_model_enable_cuda_graph, but can be overridden by
  // setUseCudaGraph().
  bool use_cuda_graph_;

  AITemplatePyTorchCachingAllocator allocator_;
};
} // namespace torch::aitemplate
