#include "mandelbrot/core/cuda/coloring.h"

#include <cuda_runtime.h>

__device__ constexpr uint32_t MakeRGB(uint8_t r, uint8_t g, uint8_t b,
                                      uint8_t a = 255) {
  return r + (g << 8) + (b << 16) + (a << 24);
}

__device__ constexpr uint32_t InterpolateColor(uint32_t color1, uint32_t color2,
                                               double_t fraction) {

  const auto b1 = static_cast<uint8_t>((color1 >> 16) & 0xff);
  const auto b2 = static_cast<uint8_t>((color2 >> 16) & 0xff);
  const auto g1 = static_cast<uint8_t>((color1 >> 8) & 0xff);
  const auto g2 = static_cast<uint8_t>((color2 >> 8) & 0xff);
  const auto r1 = static_cast<uint8_t>(color1 & 0xff);
  const auto r2 = static_cast<uint8_t>(color2 & 0xff);

  const auto r = static_cast<uint32_t>((r2 - r1) * fraction + r1);
  const auto g = static_cast<uint32_t>((g2 - g1) * fraction + g1) << 8;
  const auto b = static_cast<uint32_t>((b2 - b1) * fraction + b1) << 16;
  constexpr auto kAlpha = static_cast<uint32_t>(uint8_t{255} << 24);

  return r + g + b + kAlpha;
}

__device__ void LchToRGB(double L, double C, double H, uint8_t& ROut, uint8_t& GOut,
                         uint8_t& BOut) {

  // Convert Lch to Lab
  auto A = C * cos(H * CUDART_PI_F / 180.0);
  auto B = C * sin(H * CUDART_PI_F / 180.0);

  // Convert Lab to XYZ
  auto Y = (L + 16.) / 116.;
  auto X = A / 500. + Y;
  auto Z = Y - B / 200.;

  if (pow(Y, 3) > 0.008856) {
    Y = pow(Y, 3);
  } else {
    Y = (Y - 16. / 116.) / 7.787;
  }

  if (pow(X, 3) > 0.008856) {
    X = pow(X, 3);
  }
  else {
    X = (X - 16. / 116.) / 7.787;
  }

  if (pow(Z, 3) > 0.008856) {
    Z = pow(Z, 3);
  } else {
    Z = (Z - 16. / 116.) / 7.787;
  }

  X *= 0.95047;
  Y *= 1.00000;
  Z *= 1.08883;

  // Convert XYZ to RGB
  auto R = X * +3.2406 + Y * -1.5372 + Z * -0.4986;
  auto G = X * -0.9689 + Y * +1.8758 + Z * +0.0415;
  /**/ B = X * +0.0557 + Y * -0.2040 + Z * +1.0570;

  if (R > 0.0031308) {
    R = 1.055 * pow(R, (1 / 2.4)) - 0.055;
  }
  else {
    R = 12.92 * R;
  }

  if (G > 0.0031308) {
    G = 1.055 * pow(G, (1 / 2.4)) - 0.055;
  }
  else {
    G = 12.92 * G;
  }

  if (B > 0.0031308) {
    B = 1.055 * pow(B, (1 / 2.4)) - 0.055;
  }
  else {
    B = 12.92 * B;
  }

  ROut = R * UINT8_MAX;
  GOut = G * UINT8_MAX;
  BOut = B * UINT8_MAX;
}

__device__ void HSVToRGB(double h, double s, double v, uint8_t& ROut,
                         uint8_t& GOut, uint8_t& BOut) {

  if (s <= 0.0) {
    ROut = v * UINT8_MAX;
    GOut = v * UINT8_MAX;
    BOut = v * UINT8_MAX;
    return;
  }

  auto hh = h;
  if (hh >= 360.0) {
    hh = 0.0;
  }
  hh /= 60.0;

  const auto i = static_cast<uint8_t>(hh);
  const auto ff = hh - i;
  const auto p = v * (1.0 - s);
  const auto q = v * (1.0 - (s * ff));
  const auto t = v * (1.0 - (s * (1.0 - ff)));

  auto R = double_t{};
  auto G = double_t{};
  auto B = double_t{};

  switch (i) {
    case 0:
      R = v;
      G = t;
      B = p;
      break;
    case 1:
      R = q;
      G = v;
      B = p;
      break;
    case 2:
      R = p;
      G = v;
      B = t;
      break;
    case 3:
      R = p;
      G = q;
      B = v;
      break;
    case 4:
      R = t;
      G = p;
      B = v;
      break;
    case 5:
    default:
      R = v;
      G = p;
      B = q;
      break;
  }

  ROut = R * UINT8_MAX;
  GOut = G * UINT8_MAX;
  BOut = B * UINT8_MAX;
}

namespace mandelbrot {
namespace cuda {

__device__ uint32_t DefaultMode(uint32_t iterations, uint32_t max_iterations) {

  const auto ratio = static_cast<double_t>(iterations) /
                     static_cast<double_t>(max_iterations);

  auto r = uint8_t{};
  auto g = uint8_t{};
  auto b = uint8_t{};

  r = g = b = ratio * UINT8_MAX;

  return MakeRGB(UINT8_MAX - r, UINT8_MAX - g, UINT8_MAX - b);
}

__device__ uint32_t Mode1(uint32_t iterations, uint32_t max_iterations) {

  static uint32_t palette[] = {
      MakeRGB(30,  16,  48), MakeRGB(15,  12,  94), MakeRGB(3,  28, 150),
      MakeRGB(6,  50, 188), MakeRGB(20,  80, 195), MakeRGB(57, 125, 209),
      MakeRGB(73, 155, 216), MakeRGB(134, 181, 229), MakeRGB(170, 210, 235),
      MakeRGB(240, 249, 250), MakeRGB(246, 231, 161), MakeRGB(252, 212,  70),
      MakeRGB(251, 175,  42), MakeRGB(215, 126,  41), MakeRGB(187,  79,  39),
      MakeRGB(110,  50,  53)
  };

  constexpr auto kPaletteSize = sizeof(palette) / sizeof(uint32_t);

  if (0 < iterations && iterations < max_iterations) {
    constexpr auto kStep = 64;
    constexpr auto kInterval = 1. / kPaletteSize;
    const auto hue = static_cast<double_t>(iterations % kStep) / kStep;
    const auto n = static_cast<uint32_t>(hue / kInterval);
    const auto fraction = fmod(hue, kInterval) / kInterval;
    return InterpolateColor(palette[n], palette[(n + 1) % kPaletteSize], fraction);
  }
  else {
    return MakeRGB(0, 0, 0);
  }
}

__device__ uint32_t Mode2(uint32_t iterations, uint32_t max_iterations) {

  const auto ratio = static_cast<double_t>(iterations) /
                     static_cast<double_t>(max_iterations);

  const auto h = fmod(pow(ratio * 360., 1.5), 360.);
  const auto s = 100.;
  const auto v = ratio * 100.;

  auto r = uint8_t{};
  auto g = uint8_t{};
  auto b = uint8_t{};

  HSVToRGB(h, s, v, r, g, b);

  return MakeRGB(r, g, b);
}

__device__ uint32_t Mode3(uint32_t iterations, uint32_t max_iterations) {

  const auto ratio = static_cast<double_t>(iterations) /
                     static_cast<double_t>(max_iterations);

  const auto t = ratio;
  const auto r = static_cast<uint8_t>(9. * (1 - t) * t * t * t * UINT8_MAX);
  const auto g = static_cast<uint8_t>(15. * (1 - t) * (1 - t) * t * t * UINT8_MAX);
  const auto b = static_cast<uint8_t>(8.5 * (1 - t) * (1 - t) * (1 - t) * t * UINT8_MAX);

  return MakeRGB(r, g, b);
}

__device__ uint32_t Mode4(uint32_t iterations, uint32_t max_iterations) {

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

__device__ uint32_t Mode5(uint32_t iterations, uint32_t max_iterations) {

  const auto ratio = static_cast<double_t>(iterations) /
    static_cast<double_t>(max_iterations);

  const auto v = 1.0 - pow(cos(CUDART_PI_F * ratio), 2.);
  const auto a = 111.;
  const auto L = a - (a * v);
  const auto C = 28. + (a - (a * v));
  const auto H = fmod(pow(360. * ratio, 1.5), 360.);

  auto r = uint8_t{};
  auto g = uint8_t{};
  auto b = uint8_t{};

  LchToRGB(L, C, H, r, g, b);

  return MakeRGB(r, g, b);
}

template <typename ColoringFunction>
__device__ void SmoothColor(ColoringFunction coloring_function, uint32_t* data,
                            uint32_t image_width, uint32_t image_height,
                            uint32_t max_iterations) {

  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel_index < image_width * image_height) {
    const auto iterations_f = reinterpret_cast<float_t*>(data)[pixel_index];
    const auto iterations = static_cast<uint32_t>(floorf(iterations_f));
    if (iterations >= max_iterations) {
      data[pixel_index] = MakeRGB(0, 0, 0);
      return;
    }

    const auto color1 = coloring_function(iterations, max_iterations);
    const auto color2 = coloring_function(iterations + 1, max_iterations);
    const auto fraction = fmod(iterations_f, 1.f);

    data[pixel_index] = InterpolateColor(color1, color2, fraction);
  }
}

template <typename ColoringFunction>
__device__ void NativeColor(ColoringFunction coloring_function, uint32_t* data,
                            uint32_t image_width, uint32_t image_height,
                            uint32_t max_iterations) {
  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel_index < image_width * image_height) {
    const auto iterations_f = reinterpret_cast<float_t*>(data)[pixel_index];
    const auto iterations = static_cast<uint32_t>(floorf(iterations_f));
    if (iterations >= max_iterations) {
      data[pixel_index] = MakeRGB(0, 0, 0);
      return;
    }

    data[pixel_index] = coloring_function(iterations, max_iterations);
  }
}

__global__ void KenrelDefaultMode(uint32_t* data, uint32_t image_width,
                                  uint32_t image_height,
                                  uint32_t max_iterations) {
  SmoothColor(DefaultMode, data, image_width, image_height, max_iterations);
}

__global__ void KenrelMode1(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations) {
  SmoothColor(Mode1, data, image_width, image_height, max_iterations);
}

__global__ void KenrelMode2(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations) {

  SmoothColor(Mode2, data, image_width, image_height, max_iterations);
}

__global__ void KenrelMode3(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations) {
  SmoothColor(Mode3, data, image_width, image_height, max_iterations);
}

__global__ void KenrelMode4(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations) {
  SmoothColor(Mode4, data, image_width, image_height, max_iterations);
}

__global__ void KenrelMode5(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations) {
  SmoothColor(Mode5, data, image_width, image_height, max_iterations);
}

}  // namespace cuda
}  // namespace mandelbrot