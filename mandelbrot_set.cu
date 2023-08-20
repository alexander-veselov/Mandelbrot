#include <device_launch_parameters.h>

#include <cmath>
#include <cstdint>

template <typename T>
__device__ void MandelbrotSetT(uint8_t* data, uint32_t width, uint32_t height,
                               T center_real, T center_imag, T zoom_factor) {

  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (pixel_index < width * height) {
    constexpr static auto kMaxIterations = 256;

    // Mandelbrot set parameters
    constexpr static auto kMandelbrotSetWidth  = T{3};  // [-2, 1]
    constexpr static auto kMandelbrotSetHeight = T{2};  // [-1, 1]

    const auto scale =
        T{1} / fmin(width / kMandelbrotSetWidth, height / kMandelbrotSetHeight);

    const auto x = (pixel_index % width - width  / T{2}) * scale;
    const auto y = (pixel_index / width - height / T{2}) * scale;

    const auto real0 = center_real + x / zoom_factor;
    const auto imag0 = center_imag + y / zoom_factor;

    auto real = real0;
    auto imag = imag0;

    for (auto i = 0; i < kMaxIterations; ++i) {
      const auto r2 = real * real;
      const auto i2 = imag * imag;

      if (r2 + i2 > T{4}) {
        data[pixel_index] = i * UINT8_MAX / kMaxIterations;
        return;
      }

      imag = T{2} * real * imag + imag0;
      real = r2 - i2 + real0;
    }

    data[pixel_index] = UINT8_MAX;
  }
}

__global__ void MandelbrotSet(uint8_t* data, uint32_t width, uint32_t height,
                              double_t center_real, double_t center_imag,
                              double_t zoom_factor) {
  MandelbrotSetT<double_t>(data, width, height, center_real, center_imag,
                           zoom_factor);
}