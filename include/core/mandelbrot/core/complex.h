#pragma once

#include "mandelbrot/core/float.h"
#include "mandelbrot/core/double_double.h"

namespace mandelbrot {

template <typename T>
struct ComplexT {
  T real;
  T imag;
};

template <typename T>
bool operator==(const ComplexT<T>& left, const ComplexT<T>& right) {
  return left.real == right.real && left.imag == right.imag;
}

template <typename T>
bool operator!=(const ComplexT<T>& left, const ComplexT<T>& right) {
  return !(left == right);
}

using Complex = ComplexT<Float>;

}  // namespace mandelbrot