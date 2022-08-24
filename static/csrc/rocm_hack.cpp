#include "library/include/ck/library/utility/device_memory.hpp"
#include "library/include/ck/library/utility/host_tensor.hpp"
#include "library/include/ck/library/utility/host_tensor_generator.hpp"

// hack for DeviceMem linking error
// TODO fix this by making CK a header-only lib
// <<< hack begin
DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size) {
  hipGetErrorString(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}
void* DeviceMem::GetDeviceBuffer() const {
  return mpDeviceBuf;
}
void DeviceMem::ToDevice(const void* p) const {
  hipGetErrorString(hipMemcpy(
      mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}
void DeviceMem::FromDevice(void* p) const {
  hipGetErrorString(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}
DeviceMem::~DeviceMem() {
  hipGetErrorString(hipFree(mpDeviceBuf));
}
struct KernelTimerImpl {
  KernelTimerImpl() {
    hipGetErrorString(hipEventCreate(&mStart));
    hipGetErrorString(hipEventCreate(&mEnd));
  }
  ~KernelTimerImpl() {
    hipGetErrorString(hipEventDestroy(mStart));
    hipGetErrorString(hipEventDestroy(mEnd));
  }
  void Start() {
    hipGetErrorString(hipDeviceSynchronize());
    hipGetErrorString(hipEventRecord(mStart, nullptr));
  }
  void End() {
    hipGetErrorString(hipEventRecord(mEnd, nullptr));
    hipGetErrorString(hipEventSynchronize(mEnd));
  }
  float GetElapsedTime() const {
    float time;
    hipGetErrorString(hipEventElapsedTime(&time, mStart, mEnd));
    return time;
  }
  hipEvent_t mStart, mEnd;
};
// >>> hack end
