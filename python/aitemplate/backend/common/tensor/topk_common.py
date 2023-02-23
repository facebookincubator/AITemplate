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
topk kernel codegen.
"""

import os
from typing import Any, Dict, List, Tuple

import jinja2

# pylint: disable=C0301

FUNC_CALL_INT64_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int64_t*>({{name}})")

FUNC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

{{kernel}}

}  // namespace

{{func_signature}}
{
    topk_launcher<{{dtype}}>(stream, elem_cnt, instance_size, instance_num, top_k, input, workspace, output);
}
    """
)

PROFILER_TEMPLATE = jinja2.Template(
    """
#include <iostream>
{{header_files}}

size_t GLOBAL_WORKSPACE_SIZE = 0;

namespace {

{{kernel}}

}  // namespace

int main(int argc, char** argv) {
  int elem_cnt = std::stoi(argv[1]);
  int instance_size = std::stoi(argv[2]);
  int instance_num = std::stoi(argv[3]);

  float runtime_ms = 0;
  const int64_t sorted_in_aligned_bytes = GetAlignedSize(elem_cnt * sizeof({{dtype}}));
  const int64_t indices_aligned_bytes = GetAlignedSize(elem_cnt * sizeof(int64_t));
  const int64_t sorted_indices_aligned_bytes = indices_aligned_bytes;
  int64_t temp_storage_bytes = InferTempStorageForSortPairsDescending<{{dtype}}, int64_t>(instance_size, instance_num);
  GLOBAL_WORKSPACE_SIZE  =  GetAlignedSize(sorted_in_aligned_bytes + indices_aligned_bytes + sorted_indices_aligned_bytes + temp_storage_bytes);
  std::cout << "TIME:" << runtime_ms << std::endl;
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(int64_t* output,
                   const void* input,
                   const {{index_type}} elem_cnt,
                   const {{index_type}} instance_size,
                   const {{index_type}} instance_num,
                   const {{index_type}} top_k,
                   uint8_t* workspace,
                   {{prefix}}Stream_t stream)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{output}}, {{input}},
{{indent}}    {{elem_cnt}},
{{indent}}    {{instance_size}},
{{indent}}    {{instance_num}},
{{indent}}    {{top_k}},
{{indent}}    global_workspace_, stream /* default stream */
{{indent}});
    """
)

KERNEL_TEMPLATE = jinja2.Template(
    """
const int32_t kThreadsNumPerBlock = 256;
const int32_t kMaxBlocksNum = 8192;

#define GPU_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

inline size_t GetAlignedSize(size_t size) {
  const size_t kAlignSize = 512;
  return (size + kAlignSize - 1) / kAlignSize * kAlignSize;
}

template <typename T>
struct NumericTraits;

template<>
struct NumericTraits<half> {
  __host__ __device__
  static half zero() {
    return 0;
  }

  __host__ __device__
  static half one() {
    uint16_t ret = 0x3c00;
    return *reinterpret_cast<half*>(&ret);
  }

  __host__ __device__
  static half min() {
    uint16_t ret = 0xfbff;
    return *reinterpret_cast<half*>(&ret);
  }

  __host__ __device__
  static half max() {
    uint16_t ret = 0x7bff;
    return *reinterpret_cast<half*>(&ret);
  }
};

template<>
struct NumericTraits<float> {

  __host__ __device__
  static float zero() {
    return 0.0;
  }

  __host__ __device__
  static float one() {
    return 1.0;
  }

  __host__ __device__
  static float min() {
    uint32_t ret = 0xff7fffff;
    return *reinterpret_cast<float*>(&ret);
  }

  __host__ __device__
  static float max() {
    uint32_t ret = 0x7f7fffff;
    return *reinterpret_cast<float*>(&ret);
  }
};

template <typename T>
T PowOf2Floor(T val, int64_t max_power) {
  T max_floor = static_cast<T>(std::pow(2, max_power));
  val = std::min(val, max_floor);
  T ret = (T) 1;
  while (true) {
    ret *= 2;
    if (ret >= val) {
      return ret == val ? ret : ret / 2;
    }
  }
}

template <typename T>
T PowOf2Ceil(T val, int64_t max_power) {
  T max_ceil = static_cast<T>(std::pow(2, max_power));
  val = std::min(val, max_ceil);
  T ret = (T) 1;
  while (true) {
    ret *= 2;
    if (ret >= val) {
      return ret;
    }
  }
}

template <typename T, typename Compare>
__device__ void BitonicSwap(
    T* data,
    const int64_t i,
    const int64_t j,
    const bool dir,
    const Compare& comp) {
  if (comp(data[i], data[j]) == dir) {
    T tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
  }
}

class MultiplyFunctor final {
 public:
  MultiplyFunctor(int32_t num_col) : num_col_(num_col) {}
  __host__ __device__ __forceinline__ int32_t operator()(int32_t idx) const {
    return idx * num_col_;
  }

 private:
  int32_t num_col_;
};

template <typename KeyType, typename ValueType>
size_t InferTempStorageForSortPairsDescending(
    int32_t num_row,
    int32_t num_col) {
  using SegmentOffsetIter = {{cub}}::TransformInputIterator<
      int32_t,
      MultiplyFunctor,
      {{cub}}::CountingInputIterator<int32_t>>;

  {{cub}}::CountingInputIterator<int32_t> counting_iter(0);
  MultiplyFunctor multiply_functor(num_col);
  SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

  size_t temp_storage_bytes = 0;
  auto err = {{cub}}::DeviceSegmentedRadixSort::
      SortPairsDescending<KeyType, ValueType, SegmentOffsetIter>(
          /* d_temp_storage */ nullptr,
          /* temp_storage_bytes */ temp_storage_bytes,
          /* d_keys_in */ nullptr,
          /* d_keys_out */ nullptr,
          /* d_values_in */ nullptr,
          /* d_values_out */ nullptr,
          /* num_items */ num_row * num_col,
          /* num_segments */ num_row,
          /* d_begin_offsets */ segment_offset_iter,
          /* d_end_offsets */ segment_offset_iter + 1,
          /* begin_bit */ 0,
          /* end_bit */ sizeof(KeyType) * 8,
          /* stream */ 0);

  return temp_storage_bytes;
}

template <typename KeyType, typename ValueType>
void SortPairsDescending(
    const KeyType* keys_ptr,
    const ValueType* values_ptr,
    int32_t num_row,
    int32_t num_col,
    void* temp_storage_ptr,
    int32_t temp_storage_bytes,
    KeyType* sorted_keys_ptr,
    ValueType* sorted_values_ptr,
    {{prefix}}Stream_t stream) {
  size_t rt_inferred_temp_storage_bytes =
      InferTempStorageForSortPairsDescending<KeyType, ValueType>(
          num_row, num_col);

  using SegmentOffsetIter = {{cub}}::TransformInputIterator<
      int32_t,
      MultiplyFunctor,
      {{cub}}::CountingInputIterator<int32_t>>;

  {{cub}}::CountingInputIterator<int32_t> counting_iter(0);
  MultiplyFunctor multiply_functor(num_col);
  SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

  auto err = {{cub}}::DeviceSegmentedRadixSort::SortPairsDescending(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_keys_in */ keys_ptr,
      /* d_keys_out */ sorted_keys_ptr,
      /* d_values_in */ values_ptr,
      /* d_values_out */ sorted_values_ptr,
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offset_iter,
      /* d_end_offsets */ segment_offset_iter + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ stream);
}

template <typename T, typename Compare>
__device__ void
BitonicSort(T* data, const int64_t elem_cnt, const Compare& comp) {
  // The element count of instance should be pow-of-2
  assert(elem_cnt > 0 && !(elem_cnt & (elem_cnt - 1)));

  // Generate a bitonic sequence from input
  for (int64_t size = 2; size <= elem_cnt / 2; size *= 2) {
    // Merge 2 bitonic sequences of length 'size' into a bitonic sequence of
    // length '2 * size'
    for (int64_t stride = size / 2; stride > 0; stride /= 2) {
      for (int64_t swap_id = threadIdx.x; swap_id < elem_cnt / 2;
           swap_id += blockDim.x) {
        // Change dir at intervals of 'size / 2' swaps
        const bool dir = swap_id & (size / 2);
        // Locate the pair {pos, pos + stride} which is going te be swaped if
        // needed
        const int pos = 2 * swap_id - (swap_id & (stride - 1));

        BitonicSwap(data, pos, pos + stride, dir, comp);

        __syncthreads();
      }
    }
  }

  // Sort the bitonic sequence
  for (int64_t stride = elem_cnt / 2; stride > 0; stride /= 2) {
    for (int64_t swap_id = threadIdx.x; swap_id < elem_cnt / 2;
         swap_id += blockDim.x) {
      // Locate the pair {pos, pos + stride} which is going te be swaped if
      // needed
      const int pos = 2 * swap_id - (swap_id & (stride - 1));

      BitonicSwap(data, pos, pos + stride, false, comp);

      __syncthreads();
    }
  }
}

template <typename T>
class Entry final {
 public:
  __device__ __forceinline__ Entry(int64_t index, T value)
      : index_(index), value_(value) {}

  __device__ __forceinline__ int64_t GetIndex() const {
    return index_;
  }
  __device__ __forceinline__ T GetValue() const {
    return value_;
  }
  __device__ __forceinline__ void SetIndex(int64_t index) {
    index_ = index;
  }
  __device__ __forceinline__ void SetValue(T value) {
    value_ = value;
  }

  __device__ __forceinline__ bool operator<(const Entry& entry) const {
    return (value_ < entry.GetValue()) ||
        (value_ == entry.GetValue() && index_ > entry.GetIndex());
  }
  __device__ __forceinline__ bool operator>(const Entry& entry) const {
    return (value_ > entry.GetValue()) ||
        (value_ == entry.GetValue() && index_ < entry.GetIndex());
  }

 private:
  int64_t index_;
  T value_;
};

template <typename T>
class MinHeap final {
 public:
  __device__ __forceinline__ MinHeap(
      Entry<T>* data,
      const int64_t heap_size,
      const int64_t init_index,
      const T init_value)
      : data_(data), heap_size_(heap_size) {
    for (int64_t i = 0; i < heap_size; ++i) {
      data_[i].SetIndex(init_index);
      data_[i].SetValue(init_value);
    }
  }
  __device__ __forceinline__ Entry<T>& Top() {
    return data_[0];
  }
  __device__ __forceinline__ void Swap(const int64_t i, const int64_t j) {
    auto tmp = data_[j];
    data_[j] = data_[i];
    data_[i] = tmp;
  }
  __device__ __forceinline__ void MinHeapify(int64_t index) {
    while (true) {
      const int64_t left = 2 * index + 1;
      const int64_t right = 2 * index + 2;
      int64_t min = index;
      if (left < heap_size_ && data_[left] < data_[min]) {
        min = left;
      }
      if (right < heap_size_ && data_[right] < data_[min]) {
        min = right;
      }
      if (min == index) {
        return;
      }
      Swap(min, index);
      index = min;
    }
  }

 private:
  Entry<T>* data_;
  int64_t heap_size_;
};

template <typename T>
class TmpBufferManager final {
 public:
  TmpBufferManager(int64_t capacity, void* ptr, const int64_t N)
      : capacity_{capacity},
        sorted_in_elem_cnt_{N},
        indices_elem_cnt_{sorted_in_elem_cnt_},
        sorted_indices_elem_cnt_{sorted_in_elem_cnt_} {
    const int64_t sorted_in_aligned_bytes =
        GetAlignedSize(sorted_in_elem_cnt_ * sizeof(T));
    const int64_t indices_aligned_bytes =
        GetAlignedSize(indices_elem_cnt_ * sizeof(int64_t));
    const int64_t sorted_indices_aligned_bytes = indices_aligned_bytes;
    sorted_in_ptr_ = reinterpret_cast<T*>(ptr);
    indices_ptr_ = reinterpret_cast<int64_t*>(
        reinterpret_cast<char*>(sorted_in_ptr_) + sorted_in_aligned_bytes);
    sorted_indices_ptr_ = reinterpret_cast<int64_t*>(
        reinterpret_cast<char*>(indices_ptr_) + indices_aligned_bytes);
    temp_storage_ptr_ = reinterpret_cast<void*>(
        reinterpret_cast<char*>(sorted_indices_ptr_) +
        sorted_indices_aligned_bytes);
    temp_storage_bytes_ = capacity_ - sorted_in_aligned_bytes -
        indices_aligned_bytes - sorted_indices_aligned_bytes;
  }
  ~TmpBufferManager() = default;

  T* SortedInPtr() const {
    return sorted_in_ptr_;
  }
  int64_t* IndicesPtr() const {
    return indices_ptr_;
  }
  int64_t* SortedIndicesPtr() const {
    return sorted_indices_ptr_;
  }
  void* TempStoragePtr() const {
    return temp_storage_ptr_;
  }

  int64_t TempStorageBytes() const {
    return temp_storage_bytes_;
  }

 private:
  int64_t capacity_;

  T* sorted_in_ptr_;
  int64_t* indices_ptr_;
  int64_t* sorted_indices_ptr_;
  void* temp_storage_ptr_;

  int64_t sorted_in_elem_cnt_;
  int64_t indices_elem_cnt_;
  int64_t sorted_indices_elem_cnt_;
  int64_t temp_storage_bytes_;
};

__global__ void InitializeIndices(
    int64_t elem_cnt,
    int64_t* indices_ptr,
    int64_t instance_size) {
  GPU_KERNEL_LOOP(i, elem_cnt) {
    indices_ptr[i] = i % instance_size;
  };
}

template <typename T>
__global__ void GetOutput(
    int64_t top_k,
    int64_t instance_num,
    int64_t instance_size,
    int64_t* indices_ptr,
    T* output) {
  for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < instance_num;
       j += blockDim.y * gridDim.y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < top_k;
         i += blockDim.x * gridDim.x) {
      output[top_k * j + i] = indices_ptr[instance_size * j + i];
    }
  }
}

template <typename T>
__global__ void HeapTopKKernel(
    const T* in_ptr,
    const int64_t instance_num,
    const int64_t instance_size,
    const int64_t k,
    const int64_t heap_size,
    const int64_t init_index,
    const T init_value,
    int64_t* out_ptr) {
  extern __shared__ char smem[];
  auto* shared_entries = reinterpret_cast<Entry<T>*>(smem);

  // Divide elements to be sorted into disjoint sets (# of sets == # of heaps).
  // Each thread in the thread block manipulates one heap to select top
  // heap_size entries from corresponding set
  const T* input = in_ptr + blockIdx.x * instance_size;
  auto heap = MinHeap<T>(
      shared_entries + threadIdx.x * heap_size,
      heap_size,
      init_index,
      init_value);
  for (int64_t i = threadIdx.x; i < instance_size; i += blockDim.x) {
    auto entry = Entry<T>(i, input[i]);
    if (entry > heap.Top()) {
      heap.Top() = entry;
      heap.MinHeapify(0);
    }
  }

  __syncthreads();

  // Merge all heaps into a unified, sorted array
  BitonicSort(
      shared_entries,
      blockDim.x * heap_size,
      [](const Entry<T>& x, const Entry<T>& y) { return x > y; });

  // Write top_k elements in sorted array to output
  for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
    (out_ptr + blockIdx.x * k)[i] = shared_entries[i].GetIndex();
  }
}
// ALIGNPTR
int64_t* alignPtr(int64_t* ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return (int64_t*)addr;
}

inline int32_t BlocksNum4ThreadsNum(const int32_t n) {
  return std::min(
      (n + kThreadsNumPerBlock - 1) / kThreadsNumPerBlock,
      kMaxBlocksNum);
}

template <typename T>
void topk_launcher(
    {{prefix}}Stream_t stream,
    const int elem_cnt,
    const int instance_size,
    const int instance_num,
    const int top_k,
    const void* input,
    void* workspace,
    void* output) {
  const int32_t k = std::min(top_k, instance_size);

  if (top_k < 100) {
    const int32_t kMaxSharedMemoryByteSize = 48 << 10;

    // Use as many heaps as possible (# of heaps == # of threads used in thread
    // block). Limitation 1: size of shared memory We also need heap_size *
    // num_heap to be pow-of-2 which is necessary for bitonic sort
    const int64_t heap_size = PowOf2Ceil(k, 16);
    int32_t num_heap = PowOf2Floor(
        kMaxSharedMemoryByteSize / (heap_size * sizeof(Entry<T>)), 16);
    // Limitation 2: # of threads in thread block
    num_heap = std::min(num_heap, kThreadsNumPerBlock);

    HeapTopKKernel<T>
        <<<instance_num,
           num_heap,
           num_heap * heap_size * sizeof(Entry<T>),
           stream>>>(
            (const T*)input,
            instance_num,
            instance_size,
            k,
            heap_size,
            std::numeric_limits<int64_t>::max(),
            NumericTraits<T>::min(),
            (int64_t*)output);

  } else {
    const uintptr_t ALIGNMENT = 32;
    int64_t* vworkspace = alignPtr((int64_t*)workspace, ALIGNMENT);
    T* tmp_buffer = (T*)vworkspace;

    TmpBufferManager<T> buf_manager(
        static_cast<int64_t>(elem_cnt), tmp_buffer, elem_cnt);

    InitializeIndices<<<
        BlocksNum4ThreadsNum(elem_cnt),
        kThreadsNumPerBlock,
        0,
        stream>>>(elem_cnt, buf_manager.IndicesPtr(), instance_size);

    SortPairsDescending(
        (const T*)input,
        buf_manager.IndicesPtr(),
        instance_num,
        instance_size,
        buf_manager.TempStoragePtr(),
        buf_manager.TempStorageBytes(),
        buf_manager.SortedInPtr(),
        buf_manager.SortedIndicesPtr(),
        stream);

    {{prefix}}Memcpy2DAsync(
        (int64_t*)output,
        k * sizeof(int64_t),
        buf_manager.SortedIndicesPtr(),
        instance_size * sizeof(int64_t),
        k * sizeof(int64_t),
        instance_num,
        {{prefix}}MemcpyDefault,
        stream);
  }
}
    """
)


def gen_function(func_attrs: Dict[str, Any], header_files: str, backend_spec) -> str:
    """Generates function.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    header_files : str
        Includes the header files for a backend.
    backend_spec : class
        Specifies the backend configurations.

    Returns
    -------
    str
        Rendered function.
    """
    index_type = backend_spec.index_type
    prefix = backend_spec.prefix
    dtype = backend_spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])
    return FUNC_TEMPLATE.render(
        header_files=header_files,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], index_type=index_type, prefix=prefix
        ),
        kernel=KERNEL_TEMPLATE.render(cub=backend_spec.cub, prefix=prefix),
        dtype=dtype,
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    """Generates function decl.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec : class
        Specifies the backend configurations.

    Returns
    -------
    str
        Rendered function decl.
    """
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=backend_spec.index_type,
            prefix=backend_spec.prefix,
        ),
    ).strip()


def gen_function_call(func_attrs: Dict[str, Any], backend_spec, indent="  ") -> str:
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec : class
        Specifies the backend configurations.
    indent : str, optional
        Indent for template, by default "  ".

    Returns
    -------
    str
        Rendered function call.
    """
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 1

    output_name = FUNC_CALL_INT64_PARAM_TEMPLATE.render(
        name=func_attrs["outputs"][0]._attrs["name"]
    )
    input_name = func_attrs["inputs"][0]._attrs["name"]

    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]

    elem_cnt = 1
    for shape in xshape:
        elem_cnt *= shape._attrs["values"][0]
    instance_size = xshape[-1]._attrs["values"][0]
    instance_num = elem_cnt // instance_size

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        input=input_name,
        elem_cnt=elem_cnt,
        instance_size=instance_size,
        instance_num=instance_num,
        top_k=func_attrs["topK"],
        indent=indent,
    )


def add_profiler(
    file_pairs: List[Tuple[str, str]],
    workdir: str,
    op_type: str,
    output_name: str,
    code: str,
):
    prefix = os.path.join(workdir, "profiler", op_type)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    src_path = os.path.join(prefix, output_name + ".cu")
    obj_path = os.path.join(prefix, output_name)
    if os.path.exists(obj_path):
        return
    with open(src_path, "w") as f:
        f.write(code)
    file_pairs.append((src_path, obj_path))


def gen_profiler(
    func_attrs: Dict[str, Any], workdir: str, header_files: str, backend_spec
):
    """Generates code for topk profiling.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    workdir: str
        Target directory for generated C++ source code files
    header_files : str
        Includes the header files for a backend.
    backend_spec : class
        Specifies the backend configurations.

    Returns
    -------
    None
    """
    # If topK is less than 100, disable profiling since our implementation does not need it.
    if func_attrs["topK"] < 100:
        func_attrs["has_profiler"] = False
        return

    op_type = func_attrs["op"]
    file_pairs = []
    index_type = backend_spec.index_type
    prefix = backend_spec.prefix
    dtype = backend_spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])

    code = PROFILER_TEMPLATE.render(
        header_files=header_files,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], index_type=index_type, prefix=prefix
        ),
        kernel=KERNEL_TEMPLATE.render(cub=backend_spec.cub, prefix=prefix),
        dtype=dtype,
    )
    op_name = func_attrs["op"]
    add_profiler(file_pairs, workdir, op_type, op_name, code)
    return file_pairs
