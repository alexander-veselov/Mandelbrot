#include "mandelbrot/core/algorithms.h"

namespace mandelbrot {

std::vector<Complex> ComputeReferenceOrbit(
  const Complex& reference_point,
  uint32_t max_iterations,
  bool smoothing_step) {

  auto orbit = std::vector<Complex>{};
  orbit.reserve(max_iterations);

  auto z = Complex{ 0.0, 0.0 };
  orbit.push_back(z);

  const auto c_real = reference_point.real;
  const auto c_imag = reference_point.imag;

  auto limit = 4.0;
  if (smoothing_step) {
    limit = 16.0;
  }

  for (uint32_t i = 0; i < max_iterations; ++i) {
    const auto real = z.real;
    const auto imag = z.imag;

    z.real = real * real - imag * imag + c_real;
    z.imag = Float{2.0} * real * imag + c_imag;

    orbit.push_back(z);

    if (z.real * z.real + z.imag * z.imag > limit) {
      return orbit;
    }
  }

  return orbit;
}

}  // namespace mandelbrot