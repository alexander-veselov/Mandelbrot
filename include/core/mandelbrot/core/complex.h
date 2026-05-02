#pragma once

#include "mandelbrot/core/typedefs.h"
#include "mandelbrot/core/double_double.h"

namespace mandelbrot {

template <typename T>
struct ComplexT {
  using value_type = T;

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

using ComplexD = ComplexT<double_t>;
using ComplexDD = ComplexT<DoubleDouble>;
using ComplexSD = ComplexT<SuperDouble>;
using Complex = ComplexSD;

}  // namespace mandelbrot