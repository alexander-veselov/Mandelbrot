#include "mandelbrot/core/cuda/coloring.h"

#include "mandelbrot/core/cuda/color.h"
#include "mandelbrot/core/cuda/palettes.h"

#include <cuda_runtime.h>

namespace mandelbrot {
namespace cuda {

__device__ uint32_t Mode0(uint32_t iterations, uint32_t max_iterations,
                          const uint32_t* palette, size_t palette_size) {
  const auto ratio = static_cast<double_t>(iterations) /
                     static_cast<double_t>(max_iterations);

  auto r = uint8_t{};
  auto g = uint8_t{};
  auto b = uint8_t{};

  r = g = b = ratio * UINT8_MAX;

  return MakeRGB(UINT8_MAX - r, UINT8_MAX - g, UINT8_MAX - b);
}

__device__ uint32_t Mode1(uint32_t iterations, uint32_t max_iterations,
                          const uint32_t* palette, size_t palette_size) {
  if (0 < iterations && iterations < max_iterations) {
    constexpr auto step = 64;
    const auto interval = 1. / palette_size;
    const auto hue = static_cast<double_t>(iterations % step) / step;
    const auto n = static_cast<uint32_t>(hue / interval);
    const auto fraction = fmod(hue, interval) / interval;
    return InterpolateColor(palette[n], palette[(n + 1) % palette_size], fraction);
  }
  else {
    return MakeRGB(0, 0, 0);
  }
}

__device__ uint32_t Mode2(uint32_t iterations, uint32_t max_iterations,
                          const uint32_t* palette, size_t palette_size) {
  if (0 < iterations && iterations < max_iterations) {
    constexpr auto step = 32;
    const auto interval = 1. / palette_size;
    const auto hue = static_cast<double_t>(iterations % step) / step;
    const auto n = static_cast<uint32_t>(hue / interval);
    const auto fraction = fmod(hue, interval) / interval;
    return InterpolateColor(palette[(n + 1) % palette_size], palette[0], fraction);
  } else {
    return MakeRGB(0, 0, 0);
  }
}

__device__ uint32_t Mode3(uint32_t iterations, uint32_t max_iterations,
                          const uint32_t* palette, size_t palette_size) {
  if (0 < iterations && iterations < max_iterations) {
    const auto step = max_iterations / 3;
    const auto interval = 1. / palette_size;
    const auto hue = static_cast<double_t>(iterations % step) / step;
    const auto n = static_cast<uint32_t>(hue / interval);
    const auto fraction = fmod(hue, interval) / interval;
    return InterpolateColor(palette[n], palette[(n + 1) % palette_size], fraction);
  }
  else {
    return MakeRGB(0, 0, 0);
  }
}

__device__ uint32_t Mode4(uint32_t iterations, uint32_t max_iterations,
                          const uint32_t* palette, size_t palette_size) {
  const auto ratio = static_cast<double_t>(iterations) /
                     static_cast<double_t>(max_iterations);

  const auto h = fmod(pow(ratio * 360., 1.5), 360.);
  const auto s = 100.;
  const auto v = ratio * 100.;

  return HSVToRGB(h, s, v);
}

__device__ uint32_t Mode5(uint32_t iterations, uint32_t max_iterations,
                          const uint32_t* palette, size_t palette_size) {
  const auto ratio = static_cast<double_t>(iterations) /
                     static_cast<double_t>(max_iterations);

  const auto t = ratio;
  const auto r = static_cast<uint8_t>(9. * (1 - t) * t * t * t * UINT8_MAX);
  const auto g = static_cast<uint8_t>(15. * (1 - t) * (1 - t) * t * t * UINT8_MAX);
  const auto b = static_cast<uint8_t>(8.5 * (1 - t) * (1 - t) * (1 - t) * t * UINT8_MAX);

  return MakeRGB(r, g, b);
}

__device__ uint32_t Mode6(uint32_t iterations, uint32_t max_iterations,
                          const uint32_t* palette, size_t palette_size) {
  const auto ratio = static_cast<double_t>(iterations) /
                     static_cast<double_t>(max_iterations);

  const auto hue = 240. * sqrt(ratio);

  const auto c = 1.;
  const auto x = c * (1. - fabs(fmod(hue / 60., 2.) - 1.));
  const auto m = 0.;

  auto r = double_t{};
  auto g = double_t{};
  auto b = double_t{};

  if (hue < 60.) {
    r = c;
    g = x;
    b = 0;
  } else if (hue < 120.) {
    r = x;
    g = c;
    b = 0;
  } else if (hue < 180.) {
    r = 0;
    g = c;
    b = x;
  } else if (hue < 240.) {
    r = 0;
    g = x;
    b = c;
  } else if (hue < 300.) {
    r = x;
    g = 0;
    b = c;
  } else {
    r = c;
    g = 0;
    b = x;
  }

  r = (r + m) * UINT8_MAX;
  g = (g + m) * UINT8_MAX;
  b = (b + m) * UINT8_MAX;

  return MakeRGB(r, g, b);
}

__device__ uint32_t Mode7(uint32_t iterations, uint32_t max_iterations,
                          const uint32_t* palette, size_t palette_size) {
  const auto ratio = static_cast<double_t>(iterations) /
    static_cast<double_t>(max_iterations);

  const auto v = 1.0 - pow(cos(CUDART_PI_F * ratio), 2.);
  const auto a = 111.;
  const auto L = a - (a * v);
  const auto C = 28. + (a - (a * v));
  const auto H = fmod(pow(360. * ratio, 1.5), 360.);

  return LchToRGB(L, C, H);
}

template <typename ColoringFunction>
__device__ void SmoothColor(ColoringFunction coloring_function, uint32_t* data,
                            uint32_t image_width, uint32_t image_height,
                            uint32_t max_iterations, const uint32_t* palette,
                            size_t palette_size) {
  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel_index < image_width * image_height) {
    const auto iterations_f = reinterpret_cast<float_t*>(data)[pixel_index];
    const auto iterations = static_cast<uint32_t>(floorf(iterations_f));
    if (iterations >= max_iterations) {
      data[pixel_index] = MakeRGB(0, 0, 0);
      return;
    }

    const auto color1 = coloring_function(iterations, max_iterations, palette, palette_size);
    const auto color2 = coloring_function(iterations + 1, max_iterations, palette, palette_size);
    const auto fraction = fmod(iterations_f, 1.f);

    data[pixel_index] = InterpolateColor(color1, color2, fraction);
  }
}

template <typename ColoringFunction>
__device__ void NativeColor(ColoringFunction coloring_function, uint32_t* data,
                            uint32_t image_width, uint32_t image_height,
                            uint32_t max_iterations, const uint32_t* palette,
                            size_t palette_size) {
  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel_index < image_width * image_height) {
    const auto iterations_f = reinterpret_cast<float_t*>(data)[pixel_index];
    const auto iterations = static_cast<uint32_t>(floorf(iterations_f));
    if (iterations >= max_iterations) {
      data[pixel_index] = MakeRGB(0, 0, 0);
      return;
    }

    data[pixel_index] = coloring_function(iterations, max_iterations, palette, palette_size);
  }
}

__global__ void KenrelColor(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations,
                            uint32_t mode, uint32_t palette) {

  typedef uint32_t (*ModeFunction)(uint32_t, uint32_t, const uint32_t*, size_t);
  ModeFunction mode_functions[] = {Mode0, Mode1, Mode2, Mode3,
                                   Mode4, Mode5, Mode6, Mode7};

  const uint32_t* palettes[] = {
      blue_palette,    pretty_palette, gradient_palette,   artistic_palette,
      natural_palette, cosmic_palette, black_white_palette};

  size_t palettes_size[] = {blue_palette_size,       pretty_palette_size,
                            gradient_palette_size,   artistic_palette_size,
                            natural_palette_size,    cosmic_palette_size,
                            black_white_palette_size};

  constexpr auto kModes = sizeof(mode_functions) / sizeof(ModeFunction);
  if (mode >= kModes) {
    mode = 0;
  }

  SmoothColor(mode_functions[mode], data, image_width, image_height,
              max_iterations, palettes[palette], palettes_size[palette]);
}


}  // namespace cuda
}  // namespace mandelbrot