#pragma once

#include "mandelbrot/core/cuda/defines.h"

namespace mandelbrot {
namespace cuda {

__global__ void KenrelDefaultMode(uint32_t* data, uint32_t image_width,
                                  uint32_t image_height,
                                  uint32_t max_iterations);

__global__ void KenrelMode1(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations);

__global__ void KenrelMode2(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations);

__global__ void KenrelMode3(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations);

__global__ void KenrelMode4(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations);

__global__ void KenrelMode5(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations);

}  // namespace cuda
}  // namespace mandelbrot