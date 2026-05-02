#pragma once

#include "mandelbrot/core/cuda/defines.h"
#include "mandelbrot/core/double_double.h"

#include <vector>

namespace mandelbrot {
namespace cuda {


  struct ComplexDD {
    SuperDouble real;
    SuperDouble imag;
  };

void Visualize(uint32_t* image, uint32_t image_width, uint32_t image_height,
  DoubleDouble ref_real = -0.5, DoubleDouble ref_imag = 0.0,
  DoubleDouble dc_real = 0.0, DoubleDouble dc_imag = 0.0,
  DoubleDouble zoom_factor = 1.0, uint32_t max_iterations = 256u,
               uint32_t coloring_mode = 0u, uint32_t palette = 0u,
               const std::vector<ComplexDD>& orbit= {});



inline void ComputeReferenceOrbit(
  std::vector<ComplexDD>& orbit,
  SuperDouble ref_real,
  SuperDouble ref_imag,
  SuperDouble dc_real,
  SuperDouble dc_imag,
  uint32_t max_iterations) {

  orbit.clear();
  orbit.reserve(max_iterations);

  ComplexDD z{ 0.0, 0.0 };

  const SuperDouble c_real = ref_real;
  const SuperDouble c_imag = ref_imag;

  orbit.push_back(z);

  for (uint32_t i = 0; i < max_iterations; ++i) {

    const SuperDouble real2 = z.real * z.real;
    const SuperDouble imag2 = z.imag * z.imag;

    const SuperDouble new_real = real2 - imag2 + c_real;
    const SuperDouble new_imag =
      SuperDouble(2.0) * z.real * z.imag + c_imag;

    z.real = new_real;
    z.imag = new_imag;

    orbit.push_back(z);

    if (z.real * z.real + z.imag * z.imag > 4.0) {
      return;
    }
  }
}

}
}  // namespace mandelbrot