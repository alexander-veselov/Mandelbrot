#include <device_launch_parameters.h>

#include <cmath>
#include <cstdint>

__device__ constexpr uint32_t MakeRGB(uint8_t r, uint8_t g, uint8_t b) {
  return b + (g << 8) + (r << 16);
}

template <typename DataType>
__device__ void ColorPixel(DataType* data, uint32_t pixel_index,
                           uint32_t iterations, uint32_t max_iterations) {
  if constexpr (sizeof(DataType) == sizeof(uint8_t)) {
    data[pixel_index] = iterations * UINT8_MAX / max_iterations;
  }

  if constexpr (sizeof(DataType) == sizeof(uint32_t)) {
    // TODO: fix code-style
    double r, g, b;
    double r1 = 0, b1 = max_iterations / 16, gr = max_iterations / 8,
           r2 = max_iterations / 4, b2 = max_iterations / 2;

    if (iterations < 0) {
      data[pixel_index] = MakeRGB(0, 0, 0);
    }

    if (iterations == 0) {
      r = max_iterations;
      g = 0;
      b = 0;
    } else {
      if (iterations < b1) {
        r = b1 * (b1 - iterations);
        g = 0;
        b = b1 * iterations - 1;
      } else if (iterations < gr) {
        r = 0;
        g = b1 * (iterations - b1);
        b = b1 * (gr - iterations) - 1;
      } else if (iterations < r2) {
        r = 8 * (iterations - gr);
        g = 8 * (r2 - iterations) - 1;
        b = 0;
      } else {
        r = max_iterations - (iterations - r2) * 4;
        g = 0;
        b = 0;
      }
    }
    data[pixel_index] = MakeRGB(r, g, b);
  }
}

template <typename DataType, typename T>
__device__ void MandelbrotSetT(DataType* data, uint32_t width, uint32_t height,
                               T center_real, T center_imag, T zoom_factor) {

  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (pixel_index < width * height) {
    constexpr static auto kMaxIterations = 1024;

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

    for (auto i = uint32_t{0}; i < kMaxIterations; ++i) {
      const auto r2 = real * real;
      const auto i2 = imag * imag;

      if (r2 + i2 > T{4}) {
        ColorPixel(data, pixel_index, i, kMaxIterations);
        return;
      }

      imag = T{2} * real * imag + imag0;
      real = r2 - i2 + real0;
    }

    ColorPixel(data, pixel_index, kMaxIterations, kMaxIterations);
  }
}

__global__ void MandelbrotSet(uint8_t* data, uint32_t width, uint32_t height,
                              double_t center_real, double_t center_imag,
                              double_t zoom_factor) {
  MandelbrotSetT<uint8_t, double_t>(data, width, height, center_real, center_imag,
                           zoom_factor);
}

__global__ void MandelbrotSet(uint32_t* data, uint32_t width, uint32_t height,
                              double_t center_real, double_t center_imag,
                              double_t zoom_factor) {
  MandelbrotSetT<uint32_t, double_t>(data, width, height, center_real,
                                     center_imag, zoom_factor);
}