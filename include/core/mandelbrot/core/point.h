#pragma once

#include "mandelbrot/core/typedefs.h"

namespace mandelbrot {

struct Point {
  double_t x;
  double_t y;
};

bool operator==(const Point& left, const Point& right);
bool operator!=(const Point& left, const Point& right);

}  // namespace mandelbrot