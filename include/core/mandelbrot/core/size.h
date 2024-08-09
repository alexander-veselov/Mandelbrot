#pragma once

#include "mandelbrot/core/typedefs.h"

namespace mandelbrot {

struct Size {
  uint32_t width;
  uint32_t height;
};

bool operator==(const Size& left, const Size& right);
bool operator!=(const Size& left, const Size& right);

}  // namespace mandelbrot