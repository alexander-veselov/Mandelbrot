#pragma once

#include "mandelbrot/core/cuda/defines.h"

namespace mandelbrot {
namespace cuda {

__global__ void KenrelColor(uint32_t* data, uint32_t image_width,
                            uint32_t image_height, uint32_t max_iterations,
                            uint32_t mode, uint32_t palette);

}  // namespace cuda
}  // namespace mandelbrot