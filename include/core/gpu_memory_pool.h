#pragma once

namespace MandelbrotSet {

// Note: GPUMemoryPool isn't properly implemented.
// Alloc/Free functions doesn't do anything.
// Alloc always returns pool pointer.
class GPUMemoryPool {
 public:
  GPUMemoryPool(size_t pool_size);
  ~GPUMemoryPool();
  void* Alloc(size_t size);
  void Free(void* ptr);

 private:
  size_t pool_size_;
  void* pool_;
};

}