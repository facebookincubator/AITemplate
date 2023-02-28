#include "AITModelImpl.h" // @manual

#include <type_traits>

#include <dlfcn.h>
#include <sstream>

#include "ATen/Context.h" // @manual
#ifdef __HIP_PLATFORM_HCC__
#include "ATen/hip/HIPContext.h"
#include "c10/core/CPUAllocator.h"
#include "c10/hip/HIPStream.h"
#else
#include "ATen/cuda/CUDAContext.h"
#include "c10/core/CPUAllocator.h"
#include "c10/cuda/CUDAStream.h"
#endif

#ifdef FBCODE_AIT
#include "folly/MapUtil.h"
#endif

namespace torch::aitemplate {

AITemplatePyTorchCachingAllocator::AITemplatePyTorchCachingAllocator() {
  #ifndef __HIP_PLATFORM_HCC__
  at::globalContext().lazyInitCUDA();
  #endif
  cuda_allocator_ = at::cuda::getCUDADeviceAllocator();
  TORCH_CHECK(cuda_allocator_ != nullptr);
}

void* AITemplatePyTorchCachingAllocator::Allocate(size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }
  return cuda_allocator_->raw_allocate(num_bytes);
}

void AITemplatePyTorchCachingAllocator::Free(void* ptr) {
  if (!ptr) {
    return;
  }
  cuda_allocator_->raw_deallocate(ptr);
}

namespace {
template <typename T>
struct GetLastArgType;

template <typename T>
struct tag {
  using type = T;
};

template <typename Function, typename... Args>
struct GetLastArgType<Function(Args...)> {
  using last_arg_type = typename decltype((tag<Args>{}, ...))::type;
};

template <typename T>
struct AITCallImpl;

#define AIT_CHECK(status)                           \
  TORCH_CHECK(                                      \
      status == AITemplateError::AITemplateSuccess, \
      "an AITemplate function failed")

template <>
struct AITCallImpl<AITemplateError(AITemplateModelHandle)> {
  // Special version for a function with no result.
  void operator()(
      AITemplateError (*f)(AITemplateModelHandle),
      AITemplateModelHandle handle) {
    AIT_CHECK(f(handle));
  }
};

template <typename... Args>
struct AITCallImpl<AITemplateError(AITemplateModelHandle*, Args...)> {
  // Special version for ModelContainer creation
  void operator()(
      AITemplateError (*f)(AITemplateModelHandle*, Args...),
      AITemplateModelHandle* handle,
      Args... args) {
    AIT_CHECK(f(handle, args...));
  }
};

template <typename... Args>
struct AITCallImpl<AITemplateError(AITemplateModelHandle, Args...)> {
  using Function = AITemplateError(AITemplateModelHandle, Args...);
  template <typename... ArgsWithoutLastArgument>
  auto operator()(
      Function* f,
      AITemplateModelHandle handle,
      ArgsWithoutLastArgument... args) {
    std::remove_pointer_t<typename GetLastArgType<Function>::last_arg_type>
        result;
    AIT_CHECK(f(handle, args..., &result));
    return result;
  }
};

template <typename Function, typename... Args>
auto AITCall(Function* f, AITemplateModelHandle handle, Args... args) {
  return AITCallImpl<Function>()(f, handle, args...);
}

template <typename Function>
auto AITCallCreate(
    Function* f,
    AITemplateModelHandle* handle,
    size_t num_runtimes,
    AITemplateAllocator* allocator = nullptr) {
  return AITCallImpl<Function>()(f, handle, num_runtimes, allocator);
}

std::string getFileBasename(const std::string& filename) {
  const auto slash = filename.rfind('/');
  return slash != std::string::npos ? filename.substr(slash + 1) : filename;
}

} // namespace

AITModelImpl::AITModelImpl(
    const std::string& model_path,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    c10::optional<at::ScalarType> input_dtype,
    c10::optional<at::ScalarType> output_dtype,
    int64_t num_runtimes,
    bool use_cuda_graph)
    : handle_(dlopen(model_path.c_str(), RTLD_NOW | RTLD_LOCAL)),
      library_basename_(getFileBasename(model_path)),
      library_path_(model_path),
      input_names_(std::move(input_names)),
      output_names_(std::move(output_names)),
      floating_point_input_dtype_(input_dtype),
      floating_point_output_dtype_(output_dtype),
      use_cuda_graph_(use_cuda_graph) {
  LOG(INFO) << "Loading .so lib " << model_path;
  TORCH_CHECK(handle_, "could not dlopen ", model_path, ": ", dlerror());
  TORCH_CHECK(num_runtimes > 0, "num_runtimes must be positive");

  // It's not clear what stream we want to use yet. Create a new one.
  // We could alternatively use the default stream, but that could cause extra
  // synchronization.
#ifdef __HIP_PLATFORM_HCC__
  hipStream_t creation_stream;
  TORCH_CHECK(
      hipStreamCreateWithFlags(&creation_stream, hipStreamNonBlocking) ==
      hipSuccess);

  using StreamGuard = std::unique_ptr<
      std::remove_pointer_t<hipStream_t>,
      decltype(&hipStreamDestroy)>;
  StreamGuard creation_stream_guard{creation_stream, hipStreamDestroy};
#else
  cudaStream_t creation_stream;
  TORCH_CHECK(
      cudaStreamCreateWithFlags(&creation_stream, cudaStreamNonBlocking) ==
      cudaSuccess);

  using StreamGuard = std::unique_ptr<
      std::remove_pointer_t<cudaStream_t>,
      decltype(&cudaStreamDestroy)>;
  StreamGuard creation_stream_guard{creation_stream, cudaStreamDestroy};
#endif

#define LOAD_SYMBOL(var, name_str)                                       \
  var = reinterpret_cast<decltype(var)>(dlsym(handle_.get(), name_str)); \
  TORCH_CHECK(var, "could not dlsym " name_str);

#define LOAD_SYMBOL_WARN(var, name_str)                                  \
  var = reinterpret_cast<decltype(var)>(dlsym(handle_.get(), name_str)); \
  if (!var) {                                                            \
    LOG(WARNING) << "Could not dlsym " << name_str;                      \
  }

  LOAD_SYMBOL(deleteFunc_, "AITemplateModelContainerDelete");
  LOAD_SYMBOL(runFunc_, "AITemplateModelContainerRun");
  LOAD_SYMBOL(getOutputNameFunc_, "AITemplateModelContainerGetOutputName");
  LOAD_SYMBOL(
      getMaximumOutputShapeFunc_,
      "AITemplateModelContainerGetMaximumOutputShape");
  LOAD_SYMBOL(getOutputDtypeFunc_, "AITemplateModelContainerGetOutputDtype");

  // It's possible that these functions are not loaded in .so file.
  // Making these function possible to load as nullptr and check when using.
  // Once all relevant packages have been updated, we can just use
  // LOAD_SYMBOL.
  LOAD_SYMBOL_WARN(
      setManyConstantsDoubleBufferFunc_,
      "AITemplateModelContainerSetManyDoubleBufferConstants");
  LOAD_SYMBOL_WARN(foldConstantsFunc_, "AITemplateModelContainerFoldConstants");
  LOAD_SYMBOL_WARN(
      getConstantNamesFunc_, "AITemplateModelContainerGetConstantNames");
  LOAD_SYMBOL_WARN(
      getNumConstantsFunc_, "AITemplateModelContainerGetNumConstants");
  LOAD_SYMBOL_WARN(swapConstantsFunc_, "AITemplateModelContainerSwapConstants");
  LOAD_SYMBOL_WARN(
      foldConstantsDoubleBufferFunc_,
      "AITemplateModelContainerFoldConstantsInDoubleBuffer");

  // It's possible that we have new field added in AITemplateModelContainer,
  // But we can be using a new AITModel to load an old AITemplateModelContainer.
  // The newly added method are usually non-critical, so we issue warning
  // instead of hard exception.
  LOAD_SYMBOL_WARN(profileFunc_, "AITemplateModelContainerProfile");

  // We never call these functions again after the constructor returns, so
  // there's no point in caching them in member variables.
  decltype(&AITemplateModelContainerCreate) createFunc;
  decltype(&AITemplateModelContainerGetInputName) getInputNameFunc;
  decltype(&AITemplateModelContainerGetNumInputs) getNumInputsFunc;
  decltype(&AITemplateModelContainerGetNumOutputs) getNumOutputsFunc;
  LOAD_SYMBOL(createFunc, "AITemplateModelContainerCreate");
  LOAD_SYMBOL(getInputNameFunc, "AITemplateModelContainerGetInputName");
  LOAD_SYMBOL(getNumInputsFunc, "AITemplateModelContainerGetNumInputs");
  LOAD_SYMBOL(getNumOutputsFunc, "AITemplateModelContainerGetNumOutputs");
#undef LOAD_SYMBOL

  AITCallCreate(createFunc, &model_handle_, num_runtimes, &allocator_);

  // TODO: this check is optional so we don't break backwards comptability.
  // Once all relevant packages have been updated, we can just use
  // LOAD_SYMBOL.
  if (foldConstantsFunc_ != nullptr) {
    AIT_CHECK(foldConstantsFunc_(
        model_handle_,
        /*stream=*/reinterpret_cast<AITemplateStreamOpaque*>(creation_stream),
        /*sync=*/true));
  }

  const auto num_inputs = AITCall(getNumInputsFunc, model_handle_);
  const auto num_outputs = AITCall(getNumOutputsFunc, model_handle_);

  for (const auto idx : c10::irange(num_inputs)) {
    input_name_to_index_.emplace(
        AITCall(getInputNameFunc, model_handle_, idx), idx);
  }
  for (const auto idx : c10::irange(num_outputs)) {
    output_name_to_index_.emplace(
        AITCall(getOutputNameFunc_, model_handle_, idx), idx);
  }
}

namespace {
at::ScalarType AITemplateDtypeToTorchDtype(AITemplateDtype ait_dtype) {
  switch (ait_dtype) {
    case AITemplateDtype::kHalf:
      return torch::kHalf;
    case AITemplateDtype::kFloat:
      return torch::kFloat;
    case AITemplateDtype::kInt:
      return torch::kInt;
    case AITemplateDtype::kLong:
      return torch::kLong;
    case AITemplateDtype::kBool:
      return torch::kBool;
    case AITemplateDtype::kBFloat16:
      return torch::kBFloat16;
    case AITemplateDtype::kUnset:
      TORCH_CHECK(false, "Unset AITemplate dtype");
  }
}

AITemplateDtype TorchDtypeToAITemplateDtype(at::ScalarType torch_dtype) {
  switch (torch_dtype) {
    case torch::kHalf:
      return AITemplateDtype::kHalf;
    case torch::kFloat:
      return AITemplateDtype::kFloat;
    case torch::kInt:
      return AITemplateDtype::kInt;
    case torch::kLong:
      return AITemplateDtype::kLong;
    case torch::kBool:
      return AITemplateDtype::kBool;
    case torch::kBFloat16:
      return AITemplateDtype::kBFloat16;
    default:
      TORCH_CHECK(false, "Unknown or unsupported torch dtype");
  }
}

AITData torchToAitData(const torch::Tensor& tensor) {
  return AITData{
      tensor.data_ptr(),
      AITemplateParamShape{tensor.sizes().data(), tensor.sizes().size()},
      TorchDtypeToAITemplateDtype(tensor.scalar_type())};
}

} // namespace

void AITModelImpl::allocateOutputs(
    std::vector<c10::intrusive_ptr<c10::StorageImpl>>&
        output_index_to_output_storage_impl,
    std::vector<AITData>& ait_outputs,
    std::vector<std::vector<int64_t>>& output_shapes,
    std::vector<int64_t*>& output_shape_ptrs,
    const c10::Device& device) {
  RECORD_USER_SCOPE("AITModel::AllocateOutputs");
  const auto num_outputs = output_name_to_index_.size();
  output_index_to_output_storage_impl.resize(num_outputs);
  const c10::DeviceGuard device_guard(device);
  ait_outputs.reserve(num_outputs);
  for (const auto output_index : c10::irange(num_outputs)) {
    const auto shape =
        AITCall(getMaximumOutputShapeFunc_, model_handle_, output_index);
    auto output_ndim = shape.size;
    output_shapes.emplace_back(output_ndim, 0);
    output_shape_ptrs.emplace_back(output_shapes.back().data());

    size_t size_bytes = 0;
    AITemplateDtype ait_dtype = AITemplateDtype::kUnset;
    ait_dtype = AITCall(getOutputDtypeFunc_, model_handle_, output_index);
    TORCH_CHECK(
        ait_dtype != AITemplateDtype::kUnset,
        "Unset dtype for AITemplate output ",
        AITCall(getOutputNameFunc_, model_handle_, output_index));
    const auto dtype = AITemplateDtypeToTorchDtype(ait_dtype);
    const auto size_array_ref = c10::IntArrayRef(shape.shape_data, shape.size);
    size_bytes = at::detail::computeStorageNbytesContiguous(
        size_array_ref, scalarTypeToTypeMeta(dtype).itemsize());
    c10::Allocator* const allocator = at::cuda::getCUDADeviceAllocator();
    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator->allocate(size_bytes),
        allocator,
        /*resizable=*/true);
    ait_outputs.emplace_back(
        storage_impl->unsafe_data<void>(), shape, ait_dtype);
    output_index_to_output_storage_impl[output_index] = std::move(storage_impl);
  }
}

std::vector<torch::Tensor> AITModelImpl::processOutputs(
    std::vector<c10::intrusive_ptr<c10::StorageImpl>>&
        output_index_to_output_storage_impl,
    const std::vector<std::vector<int64_t>>& output_shapes) {
  std::vector<torch::Tensor> outputs;
  outputs.reserve(output_names_.size());
  for (const auto& output_name : output_names_) {
    const auto output_idx = output_name_to_index_.at(output_name);

    // Now take the storage and jam it into a Tensor that has its shape set
    // to the actual shape.
    const auto ait_dtype =
        AITCall(getOutputDtypeFunc_, model_handle_, output_idx);
    // This should never fail as we checked it the first time around...
    TORCH_CHECK(
        ait_dtype != AITemplateDtype::kUnset,
        "Unset dtype for AITemplate output ",
        AITCall(getOutputNameFunc_, model_handle_, output_idx));
    const auto dtype = AITemplateDtypeToTorchDtype(ait_dtype);

    auto output = at::detail::make_tensor_base<c10::TensorImpl>(
        std::move(output_index_to_output_storage_impl.at(output_idx)),
        #ifdef __HIP_PLATFORM_HCC__
        c10::DispatchKeySet(c10::DispatchKey::HIP),
        #else
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        #endif
        scalarTypeToTypeMeta(dtype));
    const auto& size = output_shapes.at(output_idx);
    if (size.size() != 1 || size[0] != 0) {
      output.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }

    if (floating_point_output_dtype_ != c10::nullopt &&
        output.is_floating_point()) {
      outputs.emplace_back(output.to(*floating_point_output_dtype_));
    } else {
      outputs.emplace_back(std::move(output));
    }
  }
  return outputs;
}

std::vector<AITData> AITModelImpl::processInputs(
    std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& inputs_contig) {
  RECORD_USER_SCOPE("AITModel::ProcessInputs");
  const auto num_inputs = input_name_to_index_.size();
  std::vector<AITData> ait_inputs;
  TORCH_CHECK(
      inputs.size() == num_inputs,
      "User passed ",
      inputs.size(),
      " inputs, but the model expects ",
      num_inputs);
  ait_inputs.resize(inputs.size());
  for (int python_input_idx = 0; python_input_idx < input_names_.size();
       python_input_idx++) {
    auto input_name = input_names_[python_input_idx];
    const auto ait_input_idx = input_name_to_index_.at(input_name);
    auto& input = inputs[python_input_idx];
    if (floating_point_input_dtype_ != c10::nullopt &&
        input.is_floating_point()) {
      // Need to keep input alive; cannot just stash result of to()
      // call in a local!
      input = input.to(*floating_point_input_dtype_);
    }
    inputs_contig.push_back(input.contiguous());
    auto& input_contig = inputs_contig.back();
    auto input_shape_array_ref = input_contig.sizes();
    ait_inputs[ait_input_idx] = AITData{
        input_contig.data_ptr(),
        AITemplateParamShape{
            input_shape_array_ref.data(), input_shape_array_ref.size()},
        TorchDtypeToAITemplateDtype(input.scalar_type())};
  }
  return ait_inputs;
}

std::vector<torch::Tensor> AITModelImpl::forward(
    std::vector<torch::Tensor>& inputs) {
  RECORD_USER_SCOPE("AITModel::Forward");
  TORCH_CHECK(!inputs.empty());
  const auto device = inputs[0].device();

  // Process inputs
  std::vector<torch::Tensor> inputs_contig;
  std::vector<AITData> ait_inputs = processInputs(inputs, inputs_contig);

  // Allocate outputs
  std::vector<c10::intrusive_ptr<c10::StorageImpl>>
      output_index_to_output_storage_impl;
  std::vector<AITData> ait_outputs;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t*> output_shape_ptrs;
  allocateOutputs(
      output_index_to_output_storage_impl,
      ait_outputs,
      output_shapes,
      output_shape_ptrs,
      device);

  std::vector<torch::Tensor> outputs;
  {
    #ifdef __HIP_PLATFORM_HCC__
    const auto& cuda_stream = at::hip::getCurrentHIPStream(device.index());
    #else
    const auto& cuda_stream = at::cuda::getCurrentCUDAStream(device.index());
    #endif
    const auto stream_id = cuda_stream.stream();
    // TODO: remove casting after fixing API
    AITemplateStreamHandle stream_handle =
        reinterpret_cast<AITemplateStreamHandle>(stream_id);
    RECORD_USER_SCOPE("AITModel::AITRuntime");
    if (runFunc_(
            model_handle_,
            ait_inputs.data(),
            ait_inputs.size(),
            ait_outputs.data(),
            ait_outputs.size(),
            /* stream = */ stream_handle,
            /* sync = */ false,
            use_cuda_graph_,
            output_shape_ptrs.data()) != AITemplateError::AITemplateSuccess) {
      std::stringstream ss;
      ss << "AITModel run failed with input spec: ";
      for (const auto& i : inputs) {
        ss << i.sizes() << ":" << i.dtype() << ", ";
      }
      TORCH_CHECK(false, ss.str());
    }

    // Process outputs
    outputs =
        processOutputs(output_index_to_output_storage_impl, output_shapes);
  }
  return outputs;
}

void AITModelImpl::profile(
    std::vector<torch::Tensor>& inputs,
    const std::string& filename,
    size_t num_iters) {
  TORCH_CHECK(!inputs.empty());
  TORCH_CHECK(
      profileFunc_,
      "Check whether the loaded AITModelContainer.so contains Profile().");
  const auto device = inputs[0].device();

  // Process inputs
  std::vector<torch::Tensor> inputs_contig;
  std::vector<AITData> ait_inputs = processInputs(inputs, inputs_contig);

  // Allocate outputs
  std::vector<c10::intrusive_ptr<c10::StorageImpl>>
      output_index_to_output_storage_impl;
  std::vector<AITData> ait_outputs;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t*> output_shape_ptrs;
  allocateOutputs(
      output_index_to_output_storage_impl,
      ait_outputs,
      output_shapes,
      output_shape_ptrs,
      device);

  {
    #ifdef __HIP_PLATFORM_HCC__
    const auto& cuda_stream = at::hip::getCurrentHIPStream(device.index());
    #else
    const auto& cuda_stream = at::cuda::getCurrentCUDAStream(device.index());
    #endif
    const auto stream_id = cuda_stream.stream();
    // TODO: remove casting after fixing API
    AITemplateStreamHandle stream_handle =
        reinterpret_cast<AITemplateStreamHandle>(stream_id);
    if (profileFunc_(
            model_handle_,
            ait_inputs.data(),
            ait_inputs.size(),
            ait_outputs.data(),
            ait_outputs.size(),
            /* stream = */ stream_handle,
            num_iters,
            filename.c_str()) != AITemplateError::AITemplateSuccess) {
      std::stringstream ss;
      ss << "AITModel profile failed with input spec: ";
      for (const auto& i : inputs) {
        ss << i.sizes() << ":" << i.dtype() << ", ";
      }
      TORCH_CHECK(false, ss.str());
    }
  }
}

thread_local std::unordered_map<std::string, std::string>
    AITModelImpl::name_to_path_map_;

thread_local bool AITModelImpl::deserialize_pickled_model_{true};

void AITModelImpl::registerLibraryNameToPathMap(
    std::unordered_map<std::string, std::string> map) {
  std::ostringstream ss;
  ss << "{\n";
  for (const auto& [k, v] : map) {
    ss << "  " << k << " => " << v << ",\n";
  }
  ss << "}";

  LOG(INFO) << "Registering .so lib paths: " << ss.str();
  name_to_path_map_ = std::move(map);
}

const std::string& AITModelImpl::getFullPathForLibraryName(
    const std::string& name) {
  const std::string* path = nullptr;
#ifdef FBCODE_AIT
  path = folly::get_ptr(name_to_path_map_, name);
#else
  auto it = name_to_path_map_.find(name);
  if (it != name_to_path_map_.end()) {
    path = &(it->second);
  }
#endif
  std::ostringstream ss;
  ss << "{\n";
  for (const auto& [k, v] : name_to_path_map_) {
    ss << "  " << k << " => " << v << ",\n";
  }
  ss << "}";
  TORCH_CHECK(
      path != nullptr,
      "could not find full path for AITemplate model .so named ",
      name,
      ". available paths: ",
      ss.str());
  return *path;
}

bool AITModelImpl::getDeserializePickledModel() {
  return deserialize_pickled_model_;
}

// Set thread local boolean to disable real loading from .so file
// for reusing the same module later on
void AITModelImpl::setDeserializePickledModel(bool deserializePickledModel) {
  deserialize_pickled_model_ = deserializePickledModel;
}

// Function to update constants in place with double buffering as well as fold
// constants. The weights supplied must be the exact same number of the current
// contants loaded in the AITModel. This call should only set the unused buffer
// in the model for both direct used constants and folded constants. The weights
// will not take effect until swapConstants is being called
void AITModelImpl::updateConstantsWithWeights(
    const std::unordered_map<std::string, torch::Tensor>& weights) {
  TORCH_CHECK(
      getNumConstantsFunc_,
      "getNumConstantsFunc_ not loaded, can not do in place update");
  TORCH_CHECK(
      getConstantNamesFunc_,
      "getConstantNamesFunc_ not loaded, can not do in place update");
  TORCH_CHECK(
      setManyConstantsDoubleBufferFunc_,
      "setManyConstantsDoubleBufferFunc_ not loaded, can not do in place update");
  TORCH_CHECK(
      foldConstantsDoubleBufferFunc_,
      "foldConstantsDoubleBufferFunc_ not loaded, can not do in place update");
  VLOG(1) << "AITModelImpl in place update for weights";
  const auto numConstants =
      AITCall(getNumConstantsFunc_, model_handle_, false, false);
  TORCH_CHECK(
      numConstants == weights.size(),
      "Number of constants loaded ",
      numConstants,
      " mismatched with number of new constants provided ",
      weights.size());
  std::vector<const char*> constantNames(numConstants, nullptr);
  AIT_CHECK(
      getConstantNamesFunc_(model_handle_, false, false, constantNames.data()));
  std::vector<AITData> constants;
  // TODO: Add check from caller side to make sure the weights are matched with
  // loaded constants for sizes and shapes
  for (const auto& name : constantNames) {
    auto it = weights.find(name);
    TORCH_CHECK(
        it != weights.end(),
        "could not find the constant named ",
        name,
        " in predictor supplied weights, ",
        "failing this round of weight update");
    constants.emplace_back(torchToAitData(it->second));
  }
#ifdef __HIP_PLATFORM_HCC__
  hipStream_t constants_stream;
  TORCH_CHECK(
      hipStreamCreateWithFlags(&constants_stream, hipStreamNonBlocking) ==
      hipSuccess);

  using StreamGuard = std::unique_ptr<
      std::remove_pointer_t<hipStream_t>,
      decltype(&hipStreamDestroy)>;
  StreamGuard constants_stream_guard{constants_stream, hipStreamDestroy};
#else
  cudaStream_t constants_stream;
  TORCH_CHECK(
      cudaStreamCreateWithFlags(&constants_stream, cudaStreamNonBlocking) ==
      cudaSuccess);

  using StreamGuard = std::unique_ptr<
      std::remove_pointer_t<cudaStream_t>,
      decltype(&cudaStreamDestroy)>;
  StreamGuard constants_stream_guard{constants_stream, cudaStreamDestroy};
#endif
  AIT_CHECK(setManyConstantsDoubleBufferFunc_(
      model_handle_,
      /*stream=*/reinterpret_cast<AITemplateStreamOpaque*>(constants_stream),
      constantNames.data(),
      constants.data(),
      numConstants));
  VLOG(1) << "Completed on setting constants in double buffers";
  AIT_CHECK(foldConstantsDoubleBufferFunc_(
      model_handle_,
      /*stream=*/reinterpret_cast<AITemplateStreamOpaque*>(constants_stream),
      /*sync=*/true));
  VLOG(1) << "Completed the constants folding process in double buffering";
}

// Swap the constants stored in the double bufferings for both model level and
// folded constants, this will take effect immediately to make this AITModel run
// with new weights
void AITModelImpl::swapConstants() {
  TORCH_CHECK(
      swapConstantsFunc_,
      "swapConstantsFunc_ not loaded, can not do in place update");
  AIT_CHECK(swapConstantsFunc_(model_handle_));
}
} // namespace torch::aitemplate
