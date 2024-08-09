#include "mandelbrot/core/complex.h"

namespace mandelbrot {

bool operator==(const Complex& left, const Complex& right) {
  return left.real == right.real && left.imag == right.imag;
}

bool operator!=(const Complex& left, const Complex& right) {
  return !(left == right);
}

}  // namespace mandelbrot