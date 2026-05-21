#pragma once

#include "mandelbrot/core/complex.h"

#include <vector>

namespace mandelbrot {

std::vector<Complex> ComputeReferenceOrbit(
  const Complex& reference_point,
  uint32_t max_iterations,
  bool smoothing_step = false
);

}  // namespace mandelbrot