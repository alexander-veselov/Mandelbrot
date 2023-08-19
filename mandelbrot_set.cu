#include <device_launch_parameters.h>

#include <cmath>
#include <cstdint>

__global__ void MandelbrotSet(uint8_t* data, uint32_t width, uint32_t height) {

  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (pixel_index < width * height) {
    
    constexpr static auto kMaxIterations = 256;

    // Mandelbrot set parameters
    constexpr static auto kMinX = -2.;
    constexpr static auto kMaxX = 1.;
    constexpr static auto kMinY = -1.;
    constexpr static auto kMaxY = 1.;
    constexpr static auto kSizeX = kMaxX - kMinX;
    constexpr static auto kSizeY = kMaxY - kMinY;

    const auto scale = 1. / fmin(width / kSizeX, height / kSizeY);

    const auto x = (pixel_index % width + width * kMinX / kSizeX) * scale;
    const auto y = (pixel_index / width + height * kMinY / kSizeY) * scale;

    auto real = x;
    auto imag = y;

    for (auto i = 0; i < kMaxIterations; ++i) {
      const auto r2 = real * real;
      const auto i2 = imag * imag;

      if (r2 + i2 > 4.) {
        data[pixel_index] = i * UINT8_MAX / kMaxIterations;
        return;
      }

      imag = 2. * real * imag + y;
      real = r2 - i2 + x;
    }

    data[pixel_index] = UINT8_MAX;
  }
}