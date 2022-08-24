#ifndef AIT_TENSOR_ACCESSOR_CUH
#define AIT_TENSOR_ACCESSOR_CUH

// Returns a strided address based on a base pointer, an index and strided
// information.
// DATA_T: tensor data type.
// READ_T: actual data type used when reading data. e.g. for a "half"
// tensor, READ_T could be uint4 when all data is aligned.
// data: A base pointer in READ_T type.
// idx: read index in terms of READ_T.
// offset, original_total_elements_from_stride_dim and
// actual_total_elements_from_stride_dim are the corresponding data member
// values of TensorAccessor.
template <typename DATA_T, typename READ_T, bool is_contiguous>
__device__ __forceinline__ READ_T* get_strided_address(
    READ_T* data,
    int64_t idx,
    int64_t offset,
    int64_t original_total_elements_from_stride_dim,
    int64_t actual_total_elements_from_stride_dim) {
  if constexpr (is_contiguous) {
    return reinterpret_cast<READ_T*>(reinterpret_cast<DATA_T*>(data) + offset) +
        idx;
  } else {
    constexpr int N_ELEMENTS_PER_READ = sizeof(READ_T) / sizeof(DATA_T);
    int64_t data_idx = idx * N_ELEMENTS_PER_READ;
    int64_t num_rows = data_idx / original_total_elements_from_stride_dim;
    int64_t row_offset = data_idx % original_total_elements_from_stride_dim;
    data_idx =
        num_rows * actual_total_elements_from_stride_dim + row_offset + offset;
    return reinterpret_cast<READ_T*>(
        reinterpret_cast<DATA_T*>(data) + data_idx);
  }
}

// A TensorAccessor which handles strided tensor access underneath.
struct TensorAccessor {
  int64_t offset{0};
  bool is_contiguous{true};

  int stride_dim{-1};
  int64_t original_total_elements_from_stride_dim{-1};
  int64_t actual_total_elements_from_stride_dim{-1};

  // Returns an address based on a base pointer and an index.

  // DATA_T: tensor data type.
  // READ_T: actual data type used when reading data. e.g. for a "half"
  // tensor, READ_T could be uint4 when all data is aligned.
  // data: A base pointer in READ_T type.
  // idx: read index in terms of READ_T.
  template <typename DATA_T, typename READ_T>
  __device__ inline READ_T* get(READ_T* data, int64_t idx) const {
    if (is_contiguous) {
      return get_strided_address<DATA_T, READ_T, true>(
          data,
          idx,
          offset,
          original_total_elements_from_stride_dim,
          actual_total_elements_from_stride_dim);
    } else {
      return get_strided_address<DATA_T, READ_T, false>(
          data,
          idx,
          offset,
          original_total_elements_from_stride_dim,
          actual_total_elements_from_stride_dim);
    }
  }
};
#endif
