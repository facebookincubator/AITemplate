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
#pragma once
#include "device_functions-generated.h"

namespace {
template <typename T>
__global__ void outputs_checker(const T* tensor, int64_t elem_cnt) {
  for (int64_t i = 0; i < elem_cnt; i++) {
    float v = (float)(*(tensor + i));
    if (i != 0) {
      printf(", ");
    }
    printf("%f", v);
  }
  printf("\n");
}

} // namespace

namespace ait {
void InvokeInfAndNanChecker(
    const half* tensor,
    const char* tensor_name,
    int64_t elem_cnt,
    ait::StreamType stream);

template <typename T>
void InvokeOutputsChecker(
    const T* tensor,
    const char* tensor_name,
    int64_t elem_cnt,
    ait::StreamType stream) {
  printf("Tensor (%s) output:\n", tensor_name);
  outputs_checker<<<1, 1, 0, stream>>>(tensor, elem_cnt);
  ait::StreamSynchronize(stream);
}
} // namespace ait
