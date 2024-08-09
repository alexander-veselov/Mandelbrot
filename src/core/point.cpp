#include "mandelbrot/core/point.h"

namespace mandelbrot {

bool operator==(const Point& left, const Point& right) {
  return left.x == right.x && left.y == right.y;
}

bool operator!=(const Point& left, const Point& right) {
  return !(left == right);
}

}  // namespace mandelbrot