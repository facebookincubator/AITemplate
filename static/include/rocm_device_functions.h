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

// #include <half.hpp>
#include <stdlib.h>
#include <cstdlib>
#include <initializer_list>
#include "include/ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/utility/print.hpp"
#include "library/include/ck/library/utility/device_memory.hpp"
#include "library/include/ck/library/utility/host_tensor.hpp"
#include "library/include/ck/library/utility/host_tensor_generator.hpp"

namespace ait {

inline thread_local bool target_has_graph_mode = false;

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

inline std::string PrintArchFeatureFlags(const hipDeviceArch_t& arch) {
  std::ostringstream oss;
  oss << "\n     Has 32-bit integer atomics for global memory: "
      << (arch.hasGlobalInt32Atomics ? "yes" : "no")
      << "\n     Has 32-bit float atomic exch for global memory: "
      << (arch.hasGlobalFloatAtomicExch ? "yes" : "no")
      << "\n     Has 32-bit integer atomics for shared memory: "
      << (arch.hasSharedInt32Atomics ? "yes" : "no")
      << "\n     Has 32-bit float atomic exch for shared memory: "
      << (arch.hasSharedFloatAtomicExch ? "yes" : "no")
      << "\n     Has 32-bit float atomic add in global and shared memory: "
      << (arch.hasFloatAtomicAdd ? "yes" : "no")
      << "\n     Has 64-bit integer atomics for global memory: "
      << (arch.hasGlobalInt64Atomics ? "yes" : "no")
      << "\n     Has 64-bit integer atomics for shared memory: "
      << (arch.hasSharedInt64Atomics ? "yes" : "no")
      << "\n     Has double-precision floating point: "
      << (arch.hasDoubles ? "yes" : "no")
      << "\n     Has warp vote instructions (__any, __all): "
      << (arch.hasWarpVote ? "yes" : "no")
      << "\n     Has warp ballot instructions (__ballot): "
      << (arch.hasWarpBallot ? "yes" : "no")
      << "\n     Has warp shuffle operations. (__shfl_*): "
      << (arch.hasWarpShuffle ? "yes" : "no")
      << "\n     Has funnel two words into one with shift&mask caps: "
      << (arch.hasFunnelShift ? "yes" : "no")
      << "\n     Has __threadfence_system: "
      << (arch.hasThreadFenceSystem ? "yes" : "no")
      << "\n     Has __syncthreads_count, syncthreads_and, syncthreads_or: "
      << (arch.hasSyncThreadsExt ? "yes" : "no")
      << "\n     Has surface functions: "
      << (arch.hasSurfaceFuncs ? "yes" : "no")
      << "\n     Grid and group dims are 3D (rather than 2D): "
      << (arch.has3dGrid ? "yes" : "no")
      << "\n     Has dynamic parallelism: "
      << (arch.hasDynamicParallelism ? "yes" : "no");
      return oss.str();
}

inline std::string PrintInfoDeviceProperties(const DevicePropertyType& prop) {
  std::ostringstream oss;
  oss << "Hardware accelerator device properties: "
      << "\n  Device: "
      << "\n     ASCII string identifying device: " << prop.name
      << "\n     Major compute capability: " << prop.major
      << "\n     Minor compute capability: " << prop.minor
      << "\n     AMD GCN Arch Value: " << prop.gcnArch
      << "\n     PCI bus ID of the device: " << prop.pciBusID
      << "\n     PCI device ID of the device: " << prop.pciDeviceID
      << "\n  Memory limits: "
      << "\n     Constant memory available on device in bytes: "
      << prop.totalConstMem
      << "\n     Global memory available on device in bytes: "
      << prop.totalGlobalMem
      << "\n     Global memory bus width in bits: " << prop.memoryBusWidth
      << "\n     Size of L2 cache in bytes: " << prop.l2CacheSize
      << "\n     Shared memory available per block in bytes: "
      << prop.sharedMemPerBlock
      << "\n     Maximum Shared Memory Per Multiprocessor in bytes: "
      << prop.maxSharedMemoryPerMultiProcessor;
  return oss.str();
}

inline std::string PrintDebugDeviceProperties(const DevicePropertyType& prop) {
  std::ostringstream oss;
  oss << "Hardware accelerator device properties: "
      << "\n  Device: "
      << "\n     ASCII string identifying device: " << prop.name
      << "\n     Major compute capability: " << prop.major
      << "\n     Minor compute capability: " << prop.minor
      << "\n     AMD GCN Arch Value: " << prop.gcnArch
      << "\n     PCI bus ID of the device: " << prop.pciBusID
      << "\n     PCI device ID of the device: " << prop.pciDeviceID

      << "\n  Memory limits: "
      << "\n     Constant memory available on device in bytes: "
      << prop.totalConstMem
      << "\n     Global memory available on device in bytes: "
      << prop.totalGlobalMem
      << "\n     Global memory bus width in bits: " << prop.memoryBusWidth
      << "\n     Size of L2 cache in bytes: " << prop.l2CacheSize
      << "\n     Shared memory available per block in bytes: "
      << prop.sharedMemPerBlock
      << "\n     Maximum Shared Memory Per Multiprocessor in bytes: "
      << prop.maxSharedMemoryPerMultiProcessor
      << "\n     Max global memory clock frequency in khz: "
      << prop.memoryClockRate
      << "\n     Peak global memory bandwidth (GByte/s): "
      << (prop.memoryClockRate / 1e6) * (prop.memoryBusWidth / 8) * 2

      << "\n  Thread limits: "
      << "\n     Warp size in threads: " << prop.warpSize
      << "\n     Maximum size of each dimension of a grid: "
      << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " "
      << prop.maxGridSize[2]
      << "\n     Maximum size of each dimension of a block: "
      << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " "
      << prop.maxThreadsDim[2] << "\n     Maximum number of threads per block: "
      << prop.maxThreadsPerBlock
      << "\n     Registers available per block: " << prop.regsPerBlock
      << "\n     Number of multiprocessors on device: "
      << prop.multiProcessorCount
      << "\n     Maximum resident threads per multiprocessor: "
      << prop.maxThreadsPerMultiProcessor
      << "\n     Max clock frequency of the multiProcessors in khz: "
      << prop.clockRate

      << "\n  Device features: "
      << "\n     Device can possibly execute multiple kernels concurrently: "
      << (prop.concurrentKernels ? "yes" : "no")
      << "\n     Device is on a multi-GPU board: "
      << (prop.isMultiGpuBoard ? "yes" : "no")
      << "\n     HIP can map host memory: "
      << (prop.canMapHostMemory ? "yes" : "no")
      << PrintArchFeatureFlags(prop.arch);

  return oss.str();
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

inline DeviceError CopyToDevice(
    Handle dst,
    const void* src,
    size_t size,
    StreamType stream = 0) {
  return hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream);
}

inline DeviceError CopyToHost(
    Handle dst,
    const void* src,
    size_t size,
    StreamType stream = 0) {
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

inline DeviceError FreeDeviceHostMemory(Handle src) {
  return hipHostFree(src);
}

inline DeviceError FreeDeviceMemoryAsync(
    Handle src,
    StreamType /*stream*/ = 0) {
  // hipFreeAsync is not supported in many versions of HIP
  return hipFree(src);
}

inline DeviceError DeviceMalloc(Handle* dst, size_t size) {
  return hipMalloc(dst, size);
}

inline DeviceError DeviceMallocHost(Handle* dst, size_t size) {
  return hipHostMalloc(dst, size, hipHostMallocDefault);
}

inline DeviceError DeviceMallocAsync(
    Handle* dst,
    size_t size,
    StreamType /*stream*/ = 0) {
  // hipMallocAsync is not supported in many versions of HIP
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

inline std::string GetErrorString(DeviceError err) {
  return hipGetErrorString(err);
}

inline DeviceError GetDeviceNotReady() {
  return hipErrorNotReady;
}

inline DeviceError GetDriverVersion(int* driverVersion) {
  return hipDriverGetVersion(driverVersion);
}

inline DeviceError GetRuntimeVersion(int* runtimeVersion) {
  return hipRuntimeGetVersion(runtimeVersion);
}

inline void ProfilerRangePush(const char* msg) {
  // TODO: Activate roctx header and linkage
  // roctxRangePush(msg);
}

inline void ProfilerRangePop() {
  // TODO: Activate roctx header and linkage
  // roctxRangePop();
}
} // namespace ait
