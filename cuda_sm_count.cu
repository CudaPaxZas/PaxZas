#include <cstdio>
#include <cuda_runtime.h>

int main() {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess) {
    std::printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  if (count == 0) {
    std::printf("No CUDA devices found.\n");
    return 1;
  }
  cudaDeviceProp p{};
  e = cudaGetDeviceProperties(&p, 0);
  if (e != cudaSuccess) {
    std::printf("cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf("Device: %s\n", p.name);
  std::printf("Compute Capability: %d.%d\n", p.major, p.minor);
  std::printf("SM count (multiProcessorCount): %d\n", p.multiProcessorCount);
  std::printf("Max threads per block: %d\n", p.maxThreadsPerBlock);
  return 0;
}
