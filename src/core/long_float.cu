#include "long_float.cuh"

#include <math.h>

namespace MandelbrotSet {

__device__ LongFloat::LongFloat(double_t data)
  : data_{ data } {
}

__device__ LongFloat::LongFloat(const LongFloat& other)
  : data_{ other.data_ } {
}

__device__ LongFloat& LongFloat::operator=(const LongFloat& other) {
  data_ = other.data_;
}

__device__ LongFloat LongFloat::operator+(const LongFloat& other) const {
  auto result = *this;
  result.data_ += other.data_;
  return result;
}

__device__ LongFloat LongFloat::operator-(const LongFloat& other) const {
  auto result = *this;
  result.data_ -= other.data_;
  return result;
}

__device__ LongFloat LongFloat::operator*(const LongFloat& other) const {
  auto result = *this;
  result.data_ *= other.data_;
  return result;
}

__device__ LongFloat LongFloat::operator/(const LongFloat& other) const {
  auto result = *this;
  result.data_ /= other.data_;
  return result;
}

__device__ bool LongFloat::operator>(double_t other) const {
  return data_ > other;
}

__device__ LongFloat::operator double() const {
  return data_;
}

}