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

#include <nvtx3/nvToolsExt.h>

namespace ait {

inline thread_local bool target_has_graph_mode = true;

using DeviceError = cudaError_t;
using DevicePropertyType = cudaDeviceProp;
using StreamType = cudaStream_t;
using EventType = cudaEvent_t;
using GraphType = cudaGraph_t;
using GraphExecType = cudaGraphExec_t;
using Handle = void*;

using bfloat16 = __nv_bfloat16;

inline DeviceError GetDevice(int* device_idx) {
  return cudaGetDevice(device_idx);
}

inline DeviceError GetDeviceProperties(
    DevicePropertyType* prop,
    int device_idx) {
  return cudaGetDeviceProperties(prop, device_idx);
}

inline std::string GetUUIDToString(const char bytes[16]) {
  std::vector<std::tuple<int, int>> groups = {
      {0, 4}, {4, 6}, {6, 8}, {8, 10}, {10, 16}};
  char const hex_chars[16] = {
      '0',
      '1',
      '2',
      '3',
      '4',
      '5',
      '6',
      '7',
      '8',
      '9',
      'a',
      'b',
      'c',
      'd',
      'e',
      'f'};

  std::string result = "GPU";
  for (auto g : groups) {
    result += "-";
    for (size_t i = std::get<0>(g); i < std::get<1>(g); ++i) {
      result += hex_chars[(bytes[i] & 0xF0) >> 4];
      result += hex_chars[(bytes[i] & 0x0F)];
    }
  }
  return result;
}

inline std::string PrintDebugDeviceProperties(const DevicePropertyType& prop) {
  std::ostringstream oss;
  oss << "Hardware accelerator device properties: "
      << "\n  Device: "
      << "\n     ASCII string identifying device: " << prop.name
      << "\n     Major compute capability: " << prop.major
      << "\n     Minor compute capability: " << prop.minor
      << "\n     UUID: " << GetUUIDToString(prop.uuid.bytes)
      << "\n     Unique identifier for a group of devices on the same multi-GPU board: "
      << prop.multiGpuBoardGroupID
      << "\n     PCI bus ID of the device: " << prop.pciBusID
      << "\n     PCI device ID of the device: " << prop.pciDeviceID
      << "\n     PCI domain ID of the device: " << prop.pciDomainID

      << "\n  Memory limits: "
      << "\n     Constant memory available on device in bytes: "
      << prop.totalConstMem
      << "\n     Global memory available on device in bytes: "
      << prop.totalGlobalMem
      << "\n     Global memory bus width in bits: " << prop.memoryBusWidth
      << "\n     Size of L2 cache in bytes: " << prop.l2CacheSize
      << "\n     Device's maximum L2 persisting lines capacity in bytes: "
      << prop.persistingL2CacheMaxSize
      << "\n     Shared memory reserved by CUDA driver per block in bytes: "
      << prop.reservedSharedMemPerBlock
      << "\n     Shared memory available per block in bytes: "
      << prop.sharedMemPerBlock
      << "\n     Per device maximum shared memory per block usable by special opt in: "
      << prop.sharedMemPerBlockOptin
      << "\n     Shared memory available per multiprocessor in bytes: "
      << prop.sharedMemPerMultiprocessor
      << "\n     The maximum value of cudaAccessPolicyWindow::num_bytes: "
      << prop.accessPolicyMaxWindowSize
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
      << prop.maxThreadsDim[2]
      << "\n     Number of asynchronous engines: " << prop.asyncEngineCount
      << "\n     Maximum number of resident blocks per multiprocessor: "
      << prop.maxBlocksPerMultiProcessor
      << "\n     Maximum number of threads per block: "
      << prop.maxThreadsPerBlock
      << "\n     Maximum resident threads per multiprocessor: "
      << prop.maxThreadsPerMultiProcessor
      << "\n     Maximum pitch in bytes allowed by memory copies: "
      << prop.memPitch << "\n     Number of multiprocessors on device: "
      << prop.multiProcessorCount
      << "\n     32-bit registers available per block: " << prop.regsPerBlock
      << "\n     32-bit registers available per multiprocessor: "
      << prop.regsPerMultiprocessor
      << "\n     Max clock frequency of the multiProcessors in khz: "
      << prop.clockRate

      << "\n  Device features: "
      << "\n     Device has ECC support enabled: "
      << (prop.ECCEnabled ? "yes" : "no")
      << "\n     Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer: "
      << (prop.canMapHostMemory ? "yes" : "no")
      << "\n     Device can access host registered memory at the same virtual address as the CPU: "
      << (prop.canUseHostPointerForRegisteredMem ? "yes" : "no")
      << "\n     Device supports Compute Preemption: "
      << (prop.computePreemptionSupported ? "yes" : "no")
      << "\n     Device can possibly execute multiple kernels concurrently: "
      << (prop.concurrentKernels ? "yes" : "no")
      << "\n     Device can coherently access managed memory concurrently with the CPU: "
      << (prop.concurrentManagedAccess ? "yes" : "no")
      << "\n     Device supports launching cooperative kernels via cudaLaunchCooperativeKernel: "
      << (prop.cooperativeLaunch ? "yes" : "no")
      << "\n     Host can directly access managed memory on the device without migration: "
      << (prop.directManagedMemAccessFromHost ? "yes" : "no")
      << "\n     Device supports caching globals in L1: "
      << (prop.globalL1CacheSupported ? "yes" : "no")
      << "\n     Link between the device and the host supports native atomic operations: "
      << (prop.hostNativeAtomicSupported ? "yes" : "no")
      << "\n     Device is integrated as opposed to discrete: "
      << (prop.integrated ? "yes" : "no")
      << "\n     Device is on a multi-GPU board: "
      << (prop.isMultiGpuBoard ? "yes" : "no")
      << "\n     Device supports caching locals in L1: "
      << (prop.localL1CacheSupported ? "yes" : "no")
      << "\n     Device supports allocating managed memory on this system: "
      << (prop.managedMemory ? "yes" : "no")
      << "\n     Device supports coherently accessing pageable memory without calling cudaHostRegister on it: "
      << (prop.pageableMemoryAccess ? "yes" : "no")
      << "\n     Device accesses pageable memory via the host's page tables: "
      << (prop.pageableMemoryAccessUsesHostPageTables ? "yes" : "no")
      << "\n     Device supports stream priorities: "
      << (prop.streamPrioritiesSupported ? "yes" : "no")
      << "\n     Device is a Tesla device using TCC driver: "
      << (prop.tccDriver ? "yes" : "no")
      << "\n     Device shares a unified address space with the host: "
      << (prop.unifiedAddressing ? "yes" : "no")

      << "\n  Texture limits: "
      << "\n     Maximum 1D surface size: " << prop.maxSurface1D
      << "\n     Maximum 1D layered surface dimensions: "
      << prop.maxSurface1DLayered[0] << " " << prop.maxSurface1DLayered[1]
      << "\n     Maximum 2D surface dimensions: " << prop.maxSurface2D[0] << " "
      << prop.maxSurface2D[1]
      << "\n     Maximum 2D layered surface dimensions: "
      << prop.maxSurface2DLayered[0] << " " << prop.maxSurface2DLayered[1]
      << " " << prop.maxSurface2DLayered[2]
      << "\n     Maximum 3D surface dimensions: " << prop.maxSurface3D[0] << " "
      << prop.maxSurface3D[1] << " " << prop.maxSurface3D[2]
      << "\n     Maximum Cubemap surface dimensions: " << prop.maxSurfaceCubemap
      << "\n     Maximum Cubemap layered surface dimensions: "
      << prop.maxSurfaceCubemapLayered[0] << " "
      << prop.maxSurfaceCubemapLayered[1]
      << "\n     Maximum 1D texture size: " << prop.maxTexture1D
      << "\n     Maximum 1D layered texture dimensions "
      << prop.maxTexture1DLayered[0] << " " << prop.maxTexture1DLayered[1]
      << "\n     Maximum 1D mipmapped texture size: " << prop.maxTexture1DMipmap
      << "\n     Maximum 2D texture dimensions: " << prop.maxTexture2D[0] << " "
      << prop.maxTexture2D[1]
      << "\n     Maximum 2D texture dimensions if texture gather operations have to be performed: "
      << prop.maxTexture2DGather[0] << " " << prop.maxTexture2DGather[1]
      << "\n     Maximum 2D layered texture dimensions: "
      << prop.maxTexture2DLayered[0] << " " << prop.maxTexture2DLayered[1]
      << " " << prop.maxTexture2DLayered[2]
      << "\n     Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory: "
      << prop.maxTexture2DLinear[0] << " " << prop.maxTexture2DLinear[1] << " "
      << prop.maxTexture2DLinear[2]
      << "\n     Maximum 2D mipmapped texture dimensions: "
      << prop.maxTexture2DMipmap[0] << " " << prop.maxTexture2DMipmap[1]
      << "\n     Maximum 3D texture dimensions: " << prop.maxTexture3D[0] << " "
      << prop.maxTexture3D[1] << " " << prop.maxTexture3D[2]
      << "\n     Maximum alternate 3D texture dimensions: "
      << prop.maxTexture3DAlt[0] << " " << prop.maxTexture3DAlt[1] << " "
      << prop.maxTexture3DAlt[2]
      << "\n     Maximum Cubemap texture dimensions: " << prop.maxTextureCubemap
      << "\n     Maximum Cubemap layered texture dimensions: "
      << prop.maxTextureCubemapLayered[0] << " "
      << prop.maxTextureCubemapLayered[1]
      << "\n     Alignment requirements for surfaces: " << prop.surfaceAlignment
      << "\n     Alignment requirement for textures: " << prop.textureAlignment
      << "\n     Pitch alignment requirement for texture references bound to pitched memory: "
      << prop.texturePitchAlignment;
  return oss.str();
}

inline std::string PrintInfoDeviceProperties(const DevicePropertyType& prop) {
  std::ostringstream oss;
  oss << "Hardware accelerator device properties: "
      << "\n  Device: "
      << "\n     ASCII string identifying device: " << prop.name
      << "\n     Major compute capability: " << prop.major
      << "\n     Minor compute capability: " << prop.minor
      << "\n     UUID: " << GetUUIDToString(prop.uuid.bytes)
      << "\n     Unique identifier for a group of devices on the same multi-GPU board: "
      << prop.multiGpuBoardGroupID
      << "\n     PCI bus ID of the device: " << prop.pciBusID
      << "\n     PCI device ID of the device: " << prop.pciDeviceID
      << "\n     PCI domain ID of the device: " << prop.pciDomainID

      << "\n  Memory limits: "
      << "\n     Constant memory available on device in bytes: "
      << prop.totalConstMem
      << "\n     Global memory available on device in bytes: "
      << prop.totalGlobalMem
      << "\n     Size of L2 cache in bytes: " << prop.l2CacheSize
      << "\n     Shared memory available per block in bytes: "
      << prop.sharedMemPerBlock
      << "\n     Shared memory available per multiprocessor in bytes: "
      << prop.sharedMemPerMultiprocessor;
  return oss.str();
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

inline DeviceError FreeDeviceHostMemory(Handle src) {
  return cudaFreeHost(src);
}

inline DeviceError FreeDeviceMemoryAsync(Handle src, StreamType stream = 0) {
  return cudaFreeAsync(src, stream);
}

inline DeviceError DeviceMalloc(Handle* dst, size_t size) {
  return cudaMalloc(dst, size);
}

inline DeviceError DeviceMallocHost(Handle* dst, size_t size) {
  return cudaMallocHost(dst, size);
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

inline std::string GetErrorString(DeviceError err) {
  return cudaGetErrorString(err);
}

inline DeviceError GetDeviceNotReady() {
  return cudaErrorNotReady;
}

inline DeviceError GetDriverVersion(int* driverVersion) {
  return cudaDriverGetVersion(driverVersion);
}

inline DeviceError GetRuntimeVersion(int* runtimeVersion) {
  return cudaRuntimeGetVersion(runtimeVersion);
}

inline void ProfilerRangePush(const char* msg) {
  nvtxRangePushA(msg);
}

inline void ProfilerRangePop() {
  nvtxRangePop();
}

} // namespace ait
