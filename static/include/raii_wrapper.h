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
// Some helpful unique_ptr instantiations and factory functions for CUDA types
#include <memory>
#include <type_traits>

#include "device_functions-generated.h"
#include "macros.h"

namespace ait {

// RAII wrapper for owned GPU memory. Not that the underlying calls
// to malloc/free are synchronous for simplicity.
using GPUPtr = std::unique_ptr<void, decltype(&FreeDeviceMemory)>;

using StreamPtr = std::
    unique_ptr<std::remove_pointer<StreamType>::type, decltype(&StreamDestroy)>;

using EventPtr = std::
    unique_ptr<std::remove_pointer<EventType>::type, decltype(&DestroyEvent)>;

inline GPUPtr RAII_DeviceMalloc(size_t num_bytes) {
  void* output;
  DEVICE_CHECK(DeviceMalloc(&output, num_bytes));
  return GPUPtr(output, FreeDeviceMemory);
}

inline StreamPtr RAII_StreamCreate(bool non_blocking = false) {
  StreamType stream;
  DEVICE_CHECK(StreamCreate(&stream, non_blocking));
  return StreamPtr(stream, StreamDestroy);
}

inline EventPtr RAII_CreateEvent() {
  EventType event;
  DEVICE_CHECK(CreateEvent(&event));
  return EventPtr(event, DestroyEvent);
}

} // namespace ait
