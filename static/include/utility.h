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
#include "device_functions-generated.h"
#include "model_interface.h"

// Utility functions for allocating, freeing, and manipulating GPU memory.
// These are useful to have around for Python clients - it allows users
// to allocate AIT tensors without depending on any extra Python libraries.

namespace ait {
enum class AITemplateMemcpyKind {
  HostToDevice = 0,
  DeviceToHost,
  DeviceToDevice,
};
} // namespace ait

extern "C" {

AIT_EXPORT
AITemplateError AITemplateDeviceMalloc(
    void** ptr_out,
    size_t size,
    ait::StreamType stream = 0,
    bool sync = true);

AIT_EXPORT
AITemplateError AITemplateDeviceFree(
    void* ptr,
    ait::StreamType stream = 0,
    bool sync = true);

AIT_EXPORT
AITemplateError AITemplateMemcpy(
    void* dst,
    const void* src,
    size_t count,
    ait::AITemplateMemcpyKind kind,
    ait::StreamType stream = 0,
    bool sync = true);
}
