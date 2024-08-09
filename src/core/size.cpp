#include "mandelbrot/core/size.h"

namespace mandelbrot {

bool operator==(const Size& left, const Size& right) {
  return left.width == right.width && left.height == right.height;
}

bool operator!=(const Size& left, const Size& right) {
  return !(left == right);
}

}  // namespace mandelbrot