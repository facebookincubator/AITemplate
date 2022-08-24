#pragma once

#include <string>

#include <half.hpp>
#include <stdlib.h>
#include <cstdlib>
#include <initializer_list>
#include "include/ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/utility/print.hpp"
#include "library/include/ck/library/utility/device_memory.hpp"
#include "library/include/ck/library/utility/host_tensor.hpp"
#include "library/include/ck/library/utility/host_tensor_generator.hpp"

namespace ait {

using DeviceError = hipError_t;
using DevicePropertyType = hipDeviceProp_t;
using StreamType = hipStream_t;
using EventType = hipEvent_t;
using GraphType = hipGraph_t;
using GraphExecType = hipGraphExec_t;
using Handle = void*;

inline DeviceError GetDevice(int* device_idx) {
  return hipGetDevice(device_idx);
}

inline DeviceError GetDeviceProperties(
    DevicePropertyType* prop,
    int device_idx) {
  return hipGetDeviceProperties(prop, device_idx);
}

inline DeviceError StreamCreate(StreamType* stream, bool non_blocking = false) {
  auto flags = non_blocking ? hipStreamNonBlocking : hipStreamDefault;
  return hipStreamCreateWithFlags(stream, flags);
}

inline DeviceError StreamBeginCapture(StreamType stream, bool global = true) {
  auto capture_mode =
      global ? hipStreamCaptureModeGlobal : hipStreamCaptureModeThreadLocal;
  return hipStreamBeginCapture(stream, capture_mode);
}

inline DeviceError StreamEndCapture(StreamType stream, GraphType* graph) {
  return hipStreamEndCapture(stream, graph);
}

inline DeviceError StreamDestroy(StreamType stream) {
  return hipStreamDestroy(stream);
}

inline DeviceError GraphInstantiate(
    GraphExecType* graph_exec,
    GraphType graph) {
  return hipGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0);
}

inline DeviceError GraphDestroy(GraphType graph) {
  return hipGraphDestroy(graph);
}

inline DeviceError GraphExecUpdate(GraphExecType graph_exec, GraphType graph) {
  // We don't have hipGraphExecUpdate in some versions of rocm
  return hipErrorUnknown;
}

inline DeviceError GraphExecDestroy(GraphExecType graph_exec) {
  return hipGraphExecDestroy(graph_exec);
}

inline DeviceError GraphExecLaunch(
    GraphExecType graph_exec,
    StreamType stream) {
  return hipGraphLaunch(graph_exec, stream);
}

inline DeviceError
CopyToDevice(Handle dst, const void* src, size_t size, StreamType stream = 0) {
  return hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream);
}

inline DeviceError
CopyToHost(Handle dst, const void* src, size_t size, StreamType stream = 0) {
  return hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, stream);
}

inline DeviceError DeviceToDeviceCopy(
    Handle dst,
    const void* src,
    size_t size,
    StreamType stream = 0) {
  return hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream);
}

inline DeviceError FreeDeviceMemory(Handle src) {
  return hipFree(src);
}

inline DeviceError DeviceMalloc(Handle* dst, size_t size) {
  return hipMalloc(dst, size);
}

inline DeviceError GetDeviceSuccess() {
  return hipSuccess;
}

inline DeviceError DeviceMemset(Handle src, int value, size_t size) {
  return hipMemset(src, value, size);
}

inline DeviceError GetLastError() {
  return hipGetLastError();
}

inline std::string GetLastErrorString() {
  return hipGetErrorString(hipGetLastError());
}

inline DeviceError StreamSynchronize(StreamType stream) {
  return hipStreamSynchronize(stream);
}

inline DeviceError CreateEvent(EventType* event) {
  return hipEventCreate(event);
}

inline DeviceError DestroyEvent(EventType event) {
  return hipEventDestroy(event);
}

inline DeviceError EventRecord(EventType event, StreamType stream = 0) {
  return hipEventRecord(event, stream);
}

inline DeviceError EventSynchronize(EventType event) {
  return hipEventSynchronize(event);
}

inline DeviceError EventElapsedTime(float* ms, EventType start, EventType end) {
  return hipEventElapsedTime(ms, start, end);
}

inline DeviceError QueryEvent(EventType event) {
  return hipEventQuery(event);
}

inline const char* GetErrorString(DeviceError err) {
  return hipGetErrorString(err);
}

inline DeviceError GetDeviceNotReady() {
  return hipErrorNotReady;
}

} // namespace ait
