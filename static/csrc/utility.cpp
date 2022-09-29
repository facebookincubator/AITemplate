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
#include "utility.h"
#include "logging.h"

#define FAIL_IF_ERROR(expr)                       \
  if ((expr) != ait::GetDeviceSuccess()) {        \
    LOG(ERROR) << "Call " << #expr << " failed."; \
    return AITemplateError::AITemplateFailure;    \
  }

AITemplateError AITemplateDeviceMalloc(
    void** ptr_out,
    size_t size,
    ait::StreamType stream,
    bool sync) {
  FAIL_IF_ERROR(ait::DeviceMallocAsync(ptr_out, size, stream));
  if (sync) {
    FAIL_IF_ERROR(ait::StreamSynchronize(stream));
  }
  return AITemplateError::AITemplateSuccess;
}

AITemplateError AITemplateDeviceFree(
    void* ptr,
    ait::StreamType stream,
    bool sync) {
  FAIL_IF_ERROR(ait::FreeDeviceMemoryAsync(ptr, stream));
  if (sync) {
    FAIL_IF_ERROR(ait::StreamSynchronize(stream));
  }
  return AITemplateError::AITemplateSuccess;
}

AITemplateError AITemplateMemcpy(
    void* dst,
    const void* src,
    size_t count,
    ait::AITemplateMemcpyKind kind,
    ait::StreamType stream,
    bool sync) {
  switch (kind) {
    case ait::AITemplateMemcpyKind::HostToDevice:
      FAIL_IF_ERROR(ait::CopyToDevice(dst, src, count, stream));
      break;
    case ait::AITemplateMemcpyKind::DeviceToHost:
      FAIL_IF_ERROR(ait::CopyToHost(dst, src, count, stream));
      break;
    case ait::AITemplateMemcpyKind::DeviceToDevice:
      FAIL_IF_ERROR(ait::DeviceToDeviceCopy(dst, src, count, stream));
      break;
  }
  if (sync) {
    FAIL_IF_ERROR(ait::StreamSynchronize(stream));
  }
  return AITemplateError::AITemplateSuccess;
}
