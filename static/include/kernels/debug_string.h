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
#include <cutlass/cutlass.h>
#include "kat_printf.h"

// Helper functions for debug logging, these
// make it easier to create meaningful debug log entries, especially
// from within CUDA code.
template <size_t SIZE = 255>
struct DebugString {
  char buffer[SIZE + 1];
  const size_t size;
  size_t pos;

  CUTLASS_HOST_DEVICE DebugString() : size{SIZE}, pos{0} {
    buffer[size] = '\0';
    buffer[0] = '\0';
  }

  CUTLASS_HOST_DEVICE void reset() {
    pos = 0;
    buffer[size] = '\0';
    buffer[0] = '\0';
  }

  CUTLASS_HOST_DEVICE void terminate() {
    if (pos < size) {
      buffer[pos] = '\0';
    }
  }

  CUTLASS_DEVICE int snprintf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    int ret = 0;
    if (size > pos) {
      ret = kat::vsnprintf(buffer + pos, size - pos - 1, format, args);
      pos += ret;
      if (pos >= size) {
        pos = size - 1;
      }
    }
    va_end(args);
    this->terminate();
    return ret;
  }

  CUTLASS_DEVICE int append_str(const char* str) {
    int spos = 0;
    while ((str[spos] != '\0') && (pos < size - 1)) {
      buffer[pos++] = str[spos++];
    }
    this->terminate();
    return spos;
  }

  CUTLASS_DEVICE int append_str(const char* str, int max_len) {
    int spos = 0;
    while ((str[spos] != '\0') && (pos < size - 1)) {
      buffer[pos++] = str[spos++];
      if (spos >= max_len) {
        break;
      }
    }
    this->terminate();
    return spos;
  }

  template <typename T>
  CUTLASS_DEVICE void append_float_array(T* arr, int start, int n) {
    this->append_str("[");
    for (std::size_t i = start; i < start + n; ++i) {
      if (i != start)
        this->append_str(", ");
      this->snprintf("%f", static_cast<float>(arr[i]));
    }
    this->append_str("]");
    this->terminate();
  }

  template <typename T>
  CUTLASS_DEVICE void append_int_array(T* arr, int start, int n) {
    this->append_str("[");
    for (std::size_t i = start; i < start + n; ++i) {
      if (i != start)
        this->append_str(", ");
      this->snprintf("%d", static_cast<int>(arr[i]));
    }
    this->append_str("]");
  }

  template <typename T>
  CUTLASS_DEVICE void append_float_array(T& arr, int start, int n) {
    this->append_str("[");
    for (std::size_t i = start; i < start + n; ++i) {
      if (i != start)
        this->append_str(", ");
      this->snprintf("%f", static_cast<float>(arr[i]));
    }
    this->append_str("]");
    this->terminate();
  }

  template <typename T>
  CUTLASS_DEVICE void append_int_array(T& arr, int start, int n) {
    this->append_str("[");
    for (std::size_t i = start; i < start + n; ++i) {
      if (i != start)
        this->append_str(", ");
      this->snprintf("%d", static_cast<int>(arr[i]));
    }
    this->append_str("]");
  }

  template <typename T>
  CUTLASS_DEVICE void append_float_array_from_ptr_to_array(
      T* arr,
      int start,
      int n) {
    // cutlass TileAccessIterator.get returns a pointer to a cutlass::Array,
    // which cannot be passed to the above functions without copying
    this->append_str("[");
    for (std::size_t i = start; i < start + n; ++i) {
      if (i != start)
        this->append_str(", ");
      this->snprintf("%f", static_cast<float>((*arr)[i]));
    }
    this->append_str("]");
    this->terminate();
  }

  template <typename T>
  CUTLASS_DEVICE void append_int_array_from_ptr_to_array(
      T* arr,
      int start,
      int n) {
    // cutlass TileAccessIterator.get returns a pointer to a cutlass::Array,
    // which cannot be passed to the above functions without copying
    this->append_str("[");
    for (std::size_t i = start; i < start + n; ++i) {
      if (i != start)
        this->append_str(", ");
      this->snprintf("%d", static_cast<int>((*arr)[i]));
    }
    this->append_str("]");
  }

  CUTLASS_DEVICE
  void append_threadinfo() {
    this->snprintf("thread(%d/%d", (int)threadIdx.x, (int)blockDim.x);
    if ((blockDim.y > 1) or (blockDim.z > 1)) {
      this->snprintf(
          ", %d/%d, %d/%d", threadIdx.y, blockDim.y, threadIdx.z, blockDim.z);
    }
    this->snprintf(") grid=(%d/%d", blockIdx.x, gridDim.x);
    if ((gridDim.y > 1) or (gridDim.z > 1)) {
      this->snprintf(
          ", %d/%d, %d/%d)", blockIdx.y, gridDim.y, blockIdx.z, gridDim.z);
    }
    this->append_str(")");
  }

  template <typename... Args>
  CUTLASS_DEVICE void append_arg_types(Args... args) {
    const char* pretty = __PRETTY_FUNCTION__; // special compiler-defined macro

    const char* start = kat::strchr(pretty, '[');
    const char* end = kat::strrchr(pretty, ']');
    size_t len = end - start;
    len = (len > size - pos) ? size - pos : len;
    this->append_str(start, len);
  }

  template <typename... Args>
  CUTLASS_DEVICE void append_types() {
    const char* pretty = __PRETTY_FUNCTION__; // special compiler-defined macro

    const char* start = kat::strchr(pretty, '[');
    const char* end = kat::strrchr(pretty, ']');
    size_t len = end - start;
    len = (len > size - pos) ? size - pos : len;
    this->append_str(start, len);
  }

  CUTLASS_HOST_DEVICE
  void println() {
    printf("%s\n", buffer);
  }
};
