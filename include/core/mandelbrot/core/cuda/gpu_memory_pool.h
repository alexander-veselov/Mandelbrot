#pragma once

namespace mandelbrot {

class GPUMemoryPool {
 public:
  GPUMemoryPool(size_t pool_size);
  ~GPUMemoryPool();
  void* Alloc(size_t size);
  void Reset();

 private:
  size_t offset_;
  size_t pool_size_;
  void* pool_;
};

}