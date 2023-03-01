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
argmax kernel codegen.
"""

import os
from typing import Any, Dict, List, Tuple

import jinja2

# pylint: disable=C0301

FUNC_CALL_FP16_PARAM_TEMPLATE = jinja2.Template(
    "reinterpret_cast<half*>(&({{name}}->raw()))"
)

FUNC_CALL_INT64_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int64_t*>({{name}})")

FUNC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

{{kernel}}

}  // namespace

{{func_signature}}
{

    argmax_launcher<{{dtype}}>(stream, elem_cnt, instance_size, instance_num, input, workspace, output);
}
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

class TmpBufferManager final {
 public:
  TmpBufferManager(int32_t capacity, void* ptr, int32_t instance_num)
      : capacity_{capacity}, key_value_out_elem_cnt_{instance_num} {
    const int32_t key_value_out_aligned_bytes = GetAlignedSize(
        key_value_out_elem_cnt_ * sizeof({{cub}}::KeyValuePair<int32_t, T>));

    key_value_out_ptr_ = reinterpret_cast<{{cub}}::KeyValuePair<int32_t, T>*>(ptr);
    temp_storage_ptr_ = reinterpret_cast<void*>(
        reinterpret_cast<char*>(key_value_out_ptr_) +
        key_value_out_aligned_bytes);

    temp_storage_bytes_ = capacity_ - key_value_out_aligned_bytes;
  }
  ~TmpBufferManager() = default;

  {{cub}}::KeyValuePair<int32_t, T>* KeyValueOutPtr() const {
    return key_value_out_ptr_;
  }
  void* TempStoragePtr() const {
    return temp_storage_ptr_;
  }

  int32_t TempStorageBytes() const {
    return temp_storage_bytes_;
  }

 private:
  int32_t capacity_;

  {{cub}}::KeyValuePair<int32_t, T>* key_value_out_ptr_;
  void* temp_storage_ptr_;

  int32_t key_value_out_elem_cnt_;
  int32_t temp_storage_bytes_;
};

class MultiplyFunctor final {
 public:
  MultiplyFunctor(int32_t num_col) : num_col_(num_col) {}
  __host__ __device__ __forceinline__ int32_t operator()(int32_t idx) const {
    return idx * num_col_;
  }

 private:
  int32_t num_col_;
};

template <typename T>

size_t InferTempStorageForArgMax(int32_t num_row, int32_t num_col) {
  using SegmentOffsetIter = {{cub}}::TransformInputIterator<
      int32_t,
      MultiplyFunctor,
      {{cub}}::CountingInputIterator<int32_t>>;

  {{cub}}::CountingInputIterator<int32_t> counting_iter(0);
  MultiplyFunctor multiply_functor(num_col);
  SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

  size_t temp_storage_bytes = 0;
  auto err = {{cub}}::DeviceSegmentedReduce::
      ArgMax<T*, {{cub}}::KeyValuePair<int32_t, T>*, SegmentOffsetIter>(
          /* d_temp_storage */ nullptr,
          /* temp_storage_bytes */ temp_storage_bytes,
          /* d_in */ nullptr,
          /* d_out */ nullptr,
          /* num_segments */ num_row,
          /* d_begin_offsets */ segment_offset_iter,
          /* d_end_offsets */ segment_offset_iter + 1,

          /* stream */ 0);
  return temp_storage_bytes;
}

template <typename T>
void ArgMax(
    const T* in_ptr,
    int32_t num_row,
    int32_t num_col,
    void* temp_storage_ptr,
    int32_t temp_storage_bytes,
    {{cub}}::KeyValuePair<int32_t, T>* out_ptr,
    {{prefix}}Stream_t stream) {
  size_t rt_inferred_temp_storage_bytes =
      InferTempStorageForArgMax<T>(num_row, num_col);

  using SegmentOffsetIter = {{cub}}::TransformInputIterator<
      int32_t,
      MultiplyFunctor,
      {{cub}}::CountingInputIterator<int32_t>>;

  {{cub}}::CountingInputIterator<int32_t> counting_iter(0);
  MultiplyFunctor multiply_functor(num_col);
  SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

  auto err = {{cub}}::DeviceSegmentedReduce::ArgMax(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_in */ in_ptr,
      /* d_out */ out_ptr,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offset_iter,
      /* d_end_offsets */ segment_offset_iter + 1,
      /* stream */ stream);
}

template <typename T>
__global__ void WriteKeysToOutput(
    const int32_t instance_num,
    const int32_t instance_size,
    const {{cub}}::KeyValuePair<int32_t, T>* key_value_out_ptr,
    int64_t* out_ptr) {
  GPU_KERNEL_LOOP(i, instance_num) {
    out_ptr[i] = key_value_out_ptr[i].key{% if is_hipcub %} - instance_size * i{% endif %};
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
void argmax_launcher(
    {{prefix}}Stream_t stream,
    const {{index_type}} elem_cnt,
    const {{index_type}} instance_size,
    const {{index_type}} instance_num,
    const void* input,
    void* workspace,
    void* output) {
  const uintptr_t ALIGNMENT = 32;
  int64_t* vworkspace = alignPtr((int64_t*)workspace, ALIGNMENT);
  T* tmp_buffer = (T*)vworkspace;

  TmpBufferManager<T> buffer_manager(
      static_cast<int64_t>(elem_cnt), tmp_buffer, instance_num);

  ArgMax(
      (const T*)input,
      instance_num,
      instance_size,
      buffer_manager.TempStoragePtr(),
      buffer_manager.TempStorageBytes(),
      buffer_manager.KeyValueOutPtr(),
      stream);

  WriteKeysToOutput<T>
      <<<BlocksNum4ThreadsNum(instance_num),
         kThreadsNumPerBlock,
         0,
         stream>>>(
          instance_num, instance_size, buffer_manager.KeyValueOutPtr(), (int64_t*)output);
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
  int instance_size = std::stoi(argv[1]);
  int instance_num = std::stoi(argv[2]);

  float runtime_ms = 0;
  int32_t key_value_out_bytes = GetAlignedSize(instance_num * sizeof({{cub}}::KeyValuePair<int32_t, {{dtype}}>));
  size_t temp_storage_bytes = InferTempStorageForArgMax<{{dtype}}>(instance_num, instance_size);
  GLOBAL_WORKSPACE_SIZE  =  GetAlignedSize(key_value_out_bytes + temp_storage_bytes);

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
{{indent}}    global_workspace_, stream /* default stream */
{{indent}});
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
    cub = backend_spec.cub

    dtype = backend_spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])

    return FUNC_TEMPLATE.render(
        header_files=header_files,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=index_type,
            prefix=prefix,
            dtype=dtype,
        ),
        kernel=KERNEL_TEMPLATE.render(
            cub=cub, index_type=index_type, prefix=prefix, is_hipcub=(cub == "hipcub")
        ),
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
    dtype = backend_spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=backend_spec.index_type,
            prefix=backend_spec.prefix,
            dtype=dtype,
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

    dtype = backend_spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])

    output_name = FUNC_CALL_INT64_PARAM_TEMPLATE.render(
        name=func_attrs["outputs"][0]._attrs["name"]
    )
    input_name = backend_spec.cast_to_ptr_template.render(
        name=func_attrs["inputs"][0]._attrs["name"],
        dtype=dtype,
    )

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
    """Generates code for argmax profiling.

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
    op_type = func_attrs["op"]
    file_pairs = []
    index_type = backend_spec.index_type
    prefix = backend_spec.prefix
    cub = backend_spec.cub

    dtype = backend_spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])

    code = PROFILER_TEMPLATE.render(
        header_files=header_files,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=index_type,
            prefix=prefix,
            dtype=dtype,
        ),
        kernel=KERNEL_TEMPLATE.render(
            cub=cub, index_type=index_type, prefix=prefix, is_hipcub=(cub == "hipcub")
        ),
        cub=cub,
        dtype=dtype,
    )
    op_name = func_attrs["op"]
    add_profiler(file_pairs, workdir, op_type, op_name, code)
    return file_pairs
