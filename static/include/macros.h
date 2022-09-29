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
// This file contains macros that are shared across many source files.
#include <stdexcept>
#include <string>

#include "device_functions-generated.h"

#define DEVICE_CHECK(call)                                           \
  if ((call) != GetDeviceSuccess()) {                                \
    throw std::runtime_error(                                        \
        #call " API call failed: " + GetLastErrorString() + " at " + \
        __FILE__ + ", line" + std::to_string(__LINE__));             \
  }

#define LAUNCH_CHECK() DEVICE_CHECK(GetLastError())
