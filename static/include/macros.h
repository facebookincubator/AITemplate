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
