#pragma once

#include "mandelbrot/core/coloring_mode.h"
#include "mandelbrot/core/palette.h"
#include "mandelbrot/core/typedefs.h"

namespace mandelbrot {

struct RenderOptions {
  ColoringMode coloring_mode;
  Palette palette;
  uint32_t max_iterations;
  bool smoothing;
};

bool operator==(const RenderOptions& left, const RenderOptions& right);
bool operator!=(const RenderOptions& left, const RenderOptions& right);

}  // namespace mandelbrot