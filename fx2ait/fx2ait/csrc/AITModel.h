#pragma once

#include <torch/torch.h> // @manual=//caffe2:torch-cpp
#include "AITModelImpl.h"

namespace torch::aitemplate {

class AITModel : public torch::CustomClassHolder {
 public:
  explicit AITModel(
      const std::string& model_path,
      std::vector<std::string> input_names,
      std::vector<std::string> output_names,
      c10::optional<at::ScalarType> input_dtype,
      c10::optional<at::ScalarType> output_dtype,
      int64_t num_runtimes = 2,
      bool use_cuda_graph = false)
      : aitModelImpl_(
            model_path,
            input_names,
            output_names,
            input_dtype,
            output_dtype) {}

  ~AITModel() {}

  // If we need to move or copy this object, then we should just
  // define a unique_ptr with deleter for the handle.
  AITModel(const AITModel&) = delete;
  AITModel& operator=(const AITModel&) = delete;

  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs) {
    return aitModelImpl_.forward(inputs);
  }

  void profile(
      std::vector<torch::Tensor> inputs,
      const std::string& filename,
      int64_t num_iters) {
    TORCH_CHECK_GE(num_iters, 0);
    aitModelImpl_.profile(inputs, filename, static_cast<size_t>(num_iters));
  }

  const std::string& libraryPath() const {
    return aitModelImpl_.libraryPath();
  }

  void setUseCudaGraph(bool use_cuda_graph) {
    aitModelImpl_.setUseCudaGraph(use_cuda_graph);
  }

  bool getUseCudaGraph() const {
    return aitModelImpl_.getUseCudaGraph();
  }

  std::string serialize() const;

  static void loadAsTorchClass();

 private:
  AITModelImpl aitModelImpl_;
};

} // namespace torch::aitemplate
