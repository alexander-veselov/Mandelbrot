#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>

#include "mandelbrot_set.cuh"
#include "coloring.cuh"
#include "long_float.cuh"
#include "gpu_memory_pool.h"

namespace MandelbrotSet {

template <typename T>
__global__ void KernelMandelbrotSet(float_t* data, uint32_t width,
                                    uint32_t height, T center_real,
                                    T center_imag, T zoom_factor,
                                    uint32_t max_iterations,
                                    bool smoothing_step = false) {

  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (pixel_index < width * height) {
    // Mandelbrot set parameters
    constexpr static auto kMandelbrotSetWidth  = 3.; // [-2, 1]
    constexpr static auto kMandelbrotSetHeight = 2.; // [-1, 1]

    const auto scale = fmin(width / kMandelbrotSetWidth, height / kMandelbrotSetHeight);

    const auto x = T{ (pixel_index % width - width  / 2.) / scale };
    const auto y = T{ (pixel_index / width - height / 2.) / scale };

    const auto real0 = center_real + x / zoom_factor;
    const auto imag0 = center_imag + y / zoom_factor;

    auto real = real0;
    auto imag = imag0;
    auto iterations = max_iterations;

    auto limit = 4.;
    if (smoothing_step) {
      limit = 16.;
    }

    for (auto i = uint32_t{0}; i < max_iterations; ++i) {
      const auto real_squared = real * real;
      const auto imag_squared = imag * imag;

      if (real_squared + imag_squared > limit) {
        iterations = i;
        break;
      }

      imag = T{2.} * real * imag + imag0;
      real = real_squared - imag_squared + real0;
    }

    if (smoothing_step && iterations < max_iterations) {
      const auto log_zn = log(static_cast<double_t>(real * real + imag * imag)) / 2.;
      const auto nu = static_cast<float_t>(log(log_zn / log(2.)) / log(2.));
      data[pixel_index] = static_cast<float_t>(iterations) + 1.f - nu;
    } else {
      data[pixel_index] = static_cast<float_t>(iterations);
    }
  }
}

void Visualize(uint32_t* image, uint32_t image_width, uint32_t image_height,
               double_t center_real, double_t center_imag, double_t zoom_factor,
               uint32_t max_iterations, uint32_t coloring_mode,
               bool smoothing) {

  static auto memory_pool = GPUMemoryPool{ 128 << 20 }; // 128 MB

  const auto image_size = image_width * image_height;
  const auto image_size_in_bytes = image_size * sizeof(uint32_t);

  auto device_data = memory_pool.Alloc(image_size_in_bytes);
  cudaMemcpy(device_data, image, image_size_in_bytes, cudaMemcpyHostToDevice);

  constexpr auto kThreadsPerBlock = 512;
  const auto kBlocksPerGrid = (image_size - 1) / kThreadsPerBlock + 1;

  using DataType = double_t;

  KernelMandelbrotSet<DataType><<<kBlocksPerGrid, kThreadsPerBlock>>>(
    reinterpret_cast<float_t*>(device_data), image_width, image_height,
    DataType{ center_real }, DataType{ center_imag }, DataType{ zoom_factor },
    max_iterations, smoothing);

  switch (coloring_mode) {
    case 1:
      Coloring::KenrelMode1<<<kBlocksPerGrid, kThreadsPerBlock>>>(
        reinterpret_cast<uint32_t*>(device_data), image_width,
        image_height, max_iterations);
      break;
    case 2:
      Coloring::KenrelMode2<<<kBlocksPerGrid, kThreadsPerBlock>>>(
        reinterpret_cast<uint32_t*>(device_data), image_width,
        image_height, max_iterations);
      break;
    case 3:
      Coloring::KenrelMode3<<<kBlocksPerGrid, kThreadsPerBlock>>>(
        reinterpret_cast<uint32_t*>(device_data), image_width,
        image_height, max_iterations);
      break;
    case 4:
      Coloring::KenrelMode4<<<kBlocksPerGrid, kThreadsPerBlock>>>(
        reinterpret_cast<uint32_t*>(device_data), image_width,
        image_height, max_iterations);
      break;
    case 5:
      Coloring::KenrelMode5<<<kBlocksPerGrid, kThreadsPerBlock>>>(
        reinterpret_cast<uint32_t*>(device_data), image_width,
        image_height, max_iterations);
      break;
    default:
      Coloring::KenrelDefaultMode<<<kBlocksPerGrid, kThreadsPerBlock>>>(
        reinterpret_cast<uint32_t*>(device_data), image_width,
        image_height, max_iterations);
  }

  cudaMemcpy(image, device_data, image_size_in_bytes, cudaMemcpyDeviceToHost);
  memory_pool.Free(device_data);
}

}  // namespace MandelbrotSet