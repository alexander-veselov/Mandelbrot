#pragma once

#include "mandelbrot/core/typedefs.h"

namespace mandelbrot {

struct Complex {
  double_t real;
  double_t imag;
};

bool operator==(const Complex& left, const Complex& right);
bool operator!=(const Complex& left, const Complex& right);

}  // namespace mandelbrot