#pragma once

#include "mandelbrot/core/cuda/defines.h"
#include "mandelbrot/core/double_double.h"

#include <vector>

namespace mandelbrot {
namespace cuda {


  struct ComplexDD {
    DoubleDouble real;
    DoubleDouble imag;
  };

void Visualize(uint32_t* image, uint32_t image_width, uint32_t image_height,
               double_t center_real = -0.5, double_t center_imag = 0.0,
               double_t zoom_factor = 1.0, uint32_t max_iterations = 256u,
               uint32_t coloring_mode = 0u, uint32_t palette = 0u,
               const std::vector<ComplexDD>& orbit= {});



inline void ComputeReferenceOrbit(std::vector<ComplexDD>& orbit,
  DoubleDouble c_real,
  DoubleDouble c_imag,
  uint32_t max_iterations) {
  orbit.resize(max_iterations);

  ComplexDD z{ 0.0, 0.0 };

  for (uint32_t i = 0; i < max_iterations; ++i) {
    orbit[i] = z;

    // z = z^2 + c
    auto real2 = z.real * z.real;
    auto imag2 = z.imag * z.imag;

    auto new_real = real2 - imag2 + c_real;
    auto new_imag = DoubleDouble(2.0) * z.real * z.imag + c_imag;

    z.real = new_real;
    z.imag = new_imag;
  }
}

}
}  // namespace mandelbrot