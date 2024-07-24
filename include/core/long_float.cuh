#pragma once

#include <cuda_runtime.h>
#include "defines.cuh"

namespace MandelbrotSet {

class LongFloat {
 public:
  __device__ explicit LongFloat(double_t data);
  __device__ LongFloat(const LongFloat& other);
  __device__ LongFloat& operator=(const LongFloat& other);
  __device__ LongFloat operator+(const LongFloat& other) const;
  __device__ LongFloat operator-(const LongFloat& other) const;
  __device__ LongFloat operator*(const LongFloat& other) const;
  __device__ LongFloat operator/(const LongFloat& other) const;
  __device__ bool operator>(double_t data) const;
  __device__ explicit operator double() const;
 private:
  double_t data_;
};

}