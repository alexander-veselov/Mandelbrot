#include "mandelbrot/core/cuda/gpu_memory_pool.h"

#include "mandelbrot/core/cuda/defines.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace {

size_t Align(size_t n, size_t a) {
  return (n + a - 1) & ~(a - 1);
}

}

namespace mandelbrot {

GPUMemoryPool::GPUMemoryPool(size_t pool_size)
  : pool_size_{pool_size},
    pool_{nullptr},
    offset_{0} {
  CUDA_CHECK(cudaMalloc(&pool_, pool_size_));
}

GPUMemoryPool::~GPUMemoryPool() {
  CUDA_CHECK(cudaFree(pool_));
}

void* GPUMemoryPool::Alloc(size_t size) {
  size = Align(size, 256);

  if (offset_ + size > pool_size_) {
    throw std::runtime_error("GPU memory pool exhausted");
  }

  void* ptr = static_cast<char*>(pool_) + offset_;
  offset_ += size;
  return ptr;
}

void GPUMemoryPool::Reset() {
  offset_ = 0;
}

}