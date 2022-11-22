/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/************** One flow generic permute implementation **************/
// https://github.com/Oneflow-Inc/oneflow/blob/f0e9d38b2ba4ac535fd6de5dbeca4e3d2051de23/oneflow/core/ep/cuda/primitive/permute.cu
// The following code fixed a bug in the original code related to vector
// read/write.

template <typename T, int N>
class NdIndexOffsetHelper {
 public:
  CUTLASS_HOST_DEVICE NdIndexOffsetHelper() = default;

  template <class... Ts>
  CUTLASS_HOST_DEVICE explicit NdIndexOffsetHelper(T d0, Ts... dims) {
    constexpr int n = 1 + sizeof...(dims);
    static_assert(n <= N, "");
    T dims_arr[n] = {d0, static_cast<T>(dims)...};
    InitStrides(dims_arr, n);
  }

  CUTLASS_HOST_DEVICE explicit NdIndexOffsetHelper(const T* dims) {
    InitStrides(dims, N);
  }

  template <typename U>
  CUTLASS_HOST_DEVICE explicit NdIndexOffsetHelper(const U* dims) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) {
      dims_arr[i] = dims[i];
    }
    InitStrides(dims_arr, N);
  }

  CUTLASS_HOST_DEVICE explicit NdIndexOffsetHelper(const T* dims, int n) {
    InitStrides(dims, n);
  }

  template <typename U>
  CUTLASS_HOST_DEVICE explicit NdIndexOffsetHelper(const U* dims, int n) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        dims_arr[i] = dims[i];
      }
    }
    InitStrides(dims_arr, n);
  }

  ~NdIndexOffsetHelper() = default;

  CUTLASS_HOST_DEVICE T NdIndexToOffset(const T* index) const {
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      offset += index[i] * stride_[i];
    }
    return offset;
  }

  CUTLASS_HOST_DEVICE T NdIndexToOffset(const T* index, int n) const {
    assert(n <= N);
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        offset += index[i] * stride_[i];
      }
    }
    return offset;
  }

  template <class... Ts>
  CUTLASS_HOST_DEVICE T NdIndexToOffset(T d0, Ts... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T index[n] = {d0, others...};
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) {
      offset += index[i] * stride_[i];
    }
    if (n == N) {
      offset += index[n - 1];
    } else {
      offset += index[n - 1] * stride_[n - 1];
    }
    return offset;
  }

  CUTLASS_HOST_DEVICE void OffsetToNdIndex(T offset, T* index) const {
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) {
      const T idx = remaining / stride_[i];
      index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    index[N - 1] = remaining;
  }

  CUTLASS_HOST_DEVICE void OffsetToNdIndex(T offset, T* index, int n) const {
    assert(n <= N);
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        const T idx = remaining / stride_[i];
        index[i] = idx;
        remaining = remaining - idx * stride_[i];
      }
    }
  }

  template <class... Ts>
  CUTLASS_HOST_DEVICE void OffsetToNdIndex(T offset, T& d0, Ts&... others)
      const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T* index[n] = {&d0, &others...};
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) {
      const T idx = remaining / stride_[i];
      *index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    if (n == N) {
      *index[n - 1] = remaining;
    } else {
      *index[n - 1] = remaining / stride_[n - 1];
    }
  }

  CUTLASS_HOST_DEVICE constexpr int Size() const {
    return N;
  }

 protected:
  CUTLASS_HOST_DEVICE void InitStrides(const T* dims, const int n) {
    for (int i = n - 1; i < N; ++i) {
      stride_[i] = 1;
    }
    for (int i = n - 2; i >= 0; --i) {
      stride_[i] = dims[i + 1] * stride_[i + 1];
    }
  }

  T stride_[N];
};

template <size_t num_dims, typename IndexType>
struct PermuteKernelParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> dst_index_helper;
  int permutation[num_dims]{};
  IndexType count{};
  const void* src{};
  void* dst{};
};

template <size_t num_dims, typename IndexType>
PermuteKernelParams<num_dims, IndexType> MakePermuteParams(
    const int64_t* src_dims,
    const void* src,
    const int* permutation,
    void* dst,
    size_t count) {
  PermuteKernelParams<num_dims, IndexType> params;
  params.src_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(src_dims);
  int64_t dst_dims[num_dims];
  for (size_t i = 0; i < num_dims; ++i) {
    dst_dims[i] = src_dims[permutation[i]];
  }
  params.dst_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(dst_dims);
  for (size_t i = 0; i < num_dims; ++i) {
    params.permutation[i] = permutation[i];
  }
  params.src = src;
  params.dst = dst;
  params.count = static_cast<IndexType>(count);
  return params;
}

template <size_t num_dims, size_t movement_size, typename IndexType>
__global__ void PermuteKernel(PermuteKernelParams<num_dims, IndexType> params) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  const T* src = reinterpret_cast<const T*>(params.src);
  T* dst = reinterpret_cast<T*>(params.dst);

  IndexType src_index[num_dims];
  IndexType dst_index[num_dims];

  IndexType start_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType step = blockDim.x * gridDim.x;
  for (IndexType i = start_idx; i < params.count; i += step) {
    params.dst_index_helper.OffsetToNdIndex(i, dst_index);
#pragma unroll
    for (size_t dim = 0; dim < num_dims; ++dim) {
      src_index[params.permutation[dim]] = dst_index[dim];
    }
    IndexType src_offset = params.src_index_helper.NdIndexToOffset(src_index);
    dst[i] = src[src_offset];
  }
}

// SimplifyPermutation can be added to further improve perf

template <size_t max_movement_size>
size_t GetMovementSize(
    size_t elem_size,
    size_t num_dims,
    const int64_t* src_dims,
    const void* src,
    const int* permutation,
    void* dst) {
  static_assert(
      max_movement_size > 0 &&
          (max_movement_size & (max_movement_size - 1)) == 0,
      "");
  CHECK_GT(elem_size, 0);
  CHECK_EQ((elem_size & (elem_size - 1)), 0);
  CHECK_EQ(max_movement_size % elem_size, 0);

  if (permutation[num_dims - 1] == num_dims - 1) {
    const int64_t last_dim_size = src_dims[num_dims - 1] * elem_size;
    auto src_ptr = reinterpret_cast<std::uintptr_t>(src);
    auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst);
    for (size_t size = max_movement_size; size > elem_size; size /= 2) {
      if (last_dim_size % size == 0 && src_ptr % size == 0 &&
          dst_ptr % size == 0) {
        return size;
      }
    }
  }
  return elem_size;
}

const int32_t kCudaThreadsNumPerBlock = 512;
const int32_t kCudaMaxBlocksNum = 8192;

inline int64_t BlocksNum4ThreadsNum(
    const int64_t n,
    const int64_t num_threads_per_block = kCudaThreadsNumPerBlock) {
  CHECK_GT(n, 0);
  return std::min(
      (n + num_threads_per_block - 1) / num_threads_per_block,
      static_cast<int64_t>(kCudaMaxBlocksNum));
}

template <size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(
    const int64_t* src_dims,
    const void* src,
    const int* permutation,
    void* dst,
    size_t count,
    cudaStream_t cuda_stream) {
  PermuteKernelParams<num_dims, IndexType> params =
      MakePermuteParams<num_dims, IndexType>(
          src_dims, src, permutation, dst, count);

  PermuteKernel<num_dims, movement_size, IndexType>
      <<<BlocksNum4ThreadsNum(params.count),
         std::min((int64_t)kCudaThreadsNumPerBlock, (int64_t)params.count),
         0,
         cuda_stream>>>(params);
}

template <size_t num_dims, size_t movement_size>
void DispatchIndexType(
    int64_t* src_dims,
    const void* src,
    const int* permutation,
    void* dst,
    cudaStream_t stream) {
  // Vector read/write.
  // This fixed a bug in the original oneflow code.
  src_dims[num_dims - 1] = src_dims[num_dims - 1] * 2 / movement_size;

  size_t count = 1;
  for (size_t i = 0; i < num_dims; ++i) {
    count *= src_dims[i];
  }
  if (count < std::numeric_limits<int32_t>::max()) {
    LaunchKernel<num_dims, movement_size, int32_t>(
        src_dims, src, permutation, dst, count, stream);
  } else {
    LaunchKernel<num_dims, movement_size, int64_t>(
        src_dims, src, permutation, dst, count, stream);
  }
}

template <size_t num_dims>
void DispatchMovementSize(
    size_t movement_size,
    int64_t* src_dims,
    const void* src,
    const int* permutation,
    void* dst,
    cudaStream_t stream) {
  void (*func)(
      int64_t* /*src_dims*/,
      const void* /*src*/,
      const int* /*permutation*/,
      void* /*dst*/,
      cudaStream_t /*stream*/) = nullptr;
  if (movement_size == 1) {
    func = DispatchIndexType<num_dims, 1>;
  } else if (movement_size == 2) {
    func = DispatchIndexType<num_dims, 2>;
  } else if (movement_size == 4) {
    func = DispatchIndexType<num_dims, 4>;
  } else if (movement_size == 8) {
    func = DispatchIndexType<num_dims, 8>;
  } else if (movement_size == 16) {
    func = DispatchIndexType<num_dims, 16>;
  } else {
    throw std::runtime_error("unsupported movement_size for permute");
  }
  func(src_dims, src, permutation, dst, stream);
}

template <size_t num_dims, size_t elem_size>
void invokePermute(
    void* dst,
    const void* src,
    int64_t* src_dims,
    const int* permutation,
    cudaStream_t stream) {
  if (!dst) {
    throw std::runtime_error("dst is NULL!");
  }
  if (!src) {
    throw std::runtime_error("src is NULL!");
  }

  // 2 bytes/half * 8 halves
  constexpr size_t kMaxMovementSize = 16;
  const size_t movement_size = GetMovementSize<kMaxMovementSize>(
      elem_size, num_dims, src_dims, src, permutation, dst);
  DispatchMovementSize<num_dims>(
      movement_size, src_dims, src, permutation, dst, stream);
}
