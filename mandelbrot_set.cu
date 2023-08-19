#include <device_launch_parameters.h>

#include <cmath>
#include <cstdint>

__global__ void MandelbrotSet(uint8_t* data, uint32_t width, uint32_t height,
                              double_t center_real, double_t center_imag,
                              double_t zoom_factor) {

  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (pixel_index < width * height) {
    
    constexpr static auto kMaxIterations = 256;

    // Mandelbrot set parameters
    constexpr static auto kMandelbrotSetWidth  = 3.; // [-2, 1]
    constexpr static auto kMandelbrotSetHeight = 2.; // [-1, 1]

    const auto scale =
        1. / fmin(width / kMandelbrotSetWidth, height / kMandelbrotSetHeight);

    const auto x = (pixel_index % width - width / 2.) * scale;
    const auto y = (pixel_index / width - height / 2.) * scale;

    const auto real0 = center_real + x / zoom_factor;
    const auto imag0 = center_imag + y / zoom_factor;

    auto real = real0;
    auto imag = imag0;

    for (auto i = 0; i < kMaxIterations; ++i) {
      const auto r2 = real * real;
      const auto i2 = imag * imag;

      if (r2 + i2 > 4.) {
        data[pixel_index] = i * UINT8_MAX / kMaxIterations;
        return;
      }

      imag = 2. * real * imag + imag0;
      real = r2 - i2 + real0;
    }

    data[pixel_index] = UINT8_MAX;
  }
}