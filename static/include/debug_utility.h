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

namespace ait {
void InvokeInfAndNanChecker(
    const half* tensor,
    const char* tensor_name,
    int64_t elem_cnt,
    ait::StreamType stream);

void InvokeOutputsChecker(
    const half* tensor,
    const char* tensor_name,
    int64_t elem_cnt,
    ait::StreamType stream);
} // namespace ait
