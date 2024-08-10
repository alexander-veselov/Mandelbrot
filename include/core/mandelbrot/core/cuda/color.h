#pragma once

#include "mandelbrot/core/cuda/defines.h"

#include <cuda_runtime.h>

namespace mandelbrot {
namespace cuda {

__device__ constexpr uint32_t MakeRGB(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);
__device__ constexpr uint32_t InterpolateColor(uint32_t color1, uint32_t color2, double_t fraction);
__device__ inline uint32_t LchToRGB(double L, double C, double H);
__device__ inline uint32_t HSVToRGB(double h, double s, double v);

// For some reason __device__ don't link normally.
// So "color.cu" is included
#include "color.cu"

}  // namespace cuda
}  // namespace mandelbrot