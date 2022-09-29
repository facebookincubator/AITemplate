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

#include <string>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

namespace ait {

using DeviceError = cudaError_t;
using DevicePropertyType = cudaDeviceProp;
using StreamType = cudaStream_t;
using EventType = cudaEvent_t;
using GraphType = cudaGraph_t;
using GraphExecType = cudaGraphExec_t;
using Handle = void*;

inline DeviceError GetDevice(int* device_idx) {
  return cudaGetDevice(device_idx);
}

inline DeviceError GetDeviceProperties(
    DevicePropertyType* prop,
    int device_idx) {
  return cudaGetDeviceProperties(prop, device_idx);
}

inline DeviceError StreamCreate(StreamType* stream, bool non_blocking = false) {
  auto flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
  return cudaStreamCreateWithFlags(stream, flags);
}

inline DeviceError StreamBeginCapture(StreamType stream, bool global = true) {
  auto capture_mode =
      global ? cudaStreamCaptureModeGlobal : cudaStreamCaptureModeThreadLocal;
  return cudaStreamBeginCapture(stream, capture_mode);
}

inline DeviceError StreamEndCapture(StreamType stream, GraphType* graph) {
  return cudaStreamEndCapture(stream, graph);
}

inline DeviceError StreamDestroy(StreamType stream) {
  return cudaStreamDestroy(stream);
}

inline DeviceError GraphInstantiate(
    GraphExecType* graph_exec,
    GraphType graph) {
  return cudaGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0);
}

inline DeviceError GraphDestroy(GraphType graph) {
  return cudaGraphDestroy(graph);
}

inline DeviceError GraphExecUpdate(GraphExecType graph_exec, GraphType graph) {
  cudaGraphExecUpdateResult update;
  return cudaGraphExecUpdate(graph_exec, graph, nullptr, &update);
}

inline DeviceError GraphExecDestroy(GraphExecType graph_exec) {
  return cudaGraphExecDestroy(graph_exec);
}

inline DeviceError GraphExecLaunch(
    GraphExecType graph_exec,
    StreamType stream) {
  return cudaGraphLaunch(graph_exec, stream);
}

inline DeviceError CopyToDevice(
    Handle dst,
    const void* src,
    size_t size,
    StreamType stream = 0) {
  return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}

inline DeviceError CopyToHost(
    Handle dst,
    const void* src,
    size_t size,
    StreamType stream = 0) {
  return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}

inline DeviceError DeviceToDeviceCopy(
    Handle dst,
    const void* src,
    size_t size,
    StreamType stream = 0) {
  return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
}

inline DeviceError FreeDeviceMemory(Handle src) {
  return cudaFree(src);
}

inline DeviceError FreeDeviceMemoryAsync(Handle src, StreamType stream = 0) {
  return cudaFreeAsync(src, stream);
}

inline DeviceError DeviceMalloc(Handle* dst, size_t size) {
  return cudaMalloc(dst, size);
}

inline DeviceError DeviceMallocAsync(
    Handle* dst,
    size_t size,
    StreamType stream = 0) {
  return cudaMallocAsync(dst, size, stream);
}

inline DeviceError GetDeviceSuccess() {
  return cudaSuccess;
}

inline DeviceError DeviceMemset(Handle src, int value, size_t size) {
  return cudaMemset(src, value, size);
}

inline DeviceError GetLastError() {
  return cudaGetLastError();
}

inline std::string GetLastErrorString() {
  return cudaGetErrorString(cudaGetLastError());
}

inline DeviceError StreamSynchronize(StreamType stream) {
  return cudaStreamSynchronize(stream);
}

inline DeviceError CreateEvent(EventType* event) {
  return cudaEventCreate(event);
}

inline DeviceError DestroyEvent(EventType event) {
  return cudaEventDestroy(event);
}

inline DeviceError EventRecord(EventType event, StreamType stream = 0) {
  return cudaEventRecord(event, stream);
}

inline DeviceError EventSynchronize(EventType event) {
  return cudaEventSynchronize(event);
}

inline DeviceError EventElapsedTime(float* ms, EventType start, EventType end) {
  return cudaEventElapsedTime(ms, start, end);
}

inline DeviceError QueryEvent(EventType event) {
  return cudaEventQuery(event);
}

inline const char* GetErrorString(DeviceError err) {
  return cudaGetErrorString(err);
}

inline DeviceError GetDeviceNotReady() {
  return cudaErrorNotReady;
}

} // namespace ait
