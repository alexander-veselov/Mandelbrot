#include "mandelbrot/core/cuda/mandelbrot_set.h"

#include "mandelbrot/core/algorithms.h"
#include "mandelbrot/core/cuda/coloring.h"
#include "mandelbrot/core/cuda/gpu_memory_pool.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

namespace mandelbrot {
namespace cuda {

template <typename T>
__global__ void KernelMandelbrotSet(float_t* data, uint32_t width,
                                    uint32_t height, T center_real,
                                    T center_imag, T scale,
                                    uint32_t max_iterations,
                                    bool smoothing_step = false) {

  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (pixel_index < width * height) {
    const auto x = pixel_index % width - width  / T{2};
    const auto y = pixel_index / width - height / T{2};

    const auto real0 = center_real + x * scale;
    const auto imag0 = center_imag + y * scale;

    auto real = real0;
    auto imag = imag0;
    auto iterations = max_iterations;

    auto limit = T{4};
    if (smoothing_step) {
      limit = T{16};
    }

    for (auto i = uint32_t{0}; i < max_iterations; ++i) {
      const auto real_squared = real * real;
      const auto imag_squared = imag * imag;

      if (real_squared + imag_squared > limit) {
        iterations = i;
        break;
      }

      imag = T{2} * real * imag + imag0;
      real = real_squared - imag_squared + real0;
    }

    if (smoothing_step && iterations < max_iterations) {
      const auto log_zn = logf(real * real + imag * imag) * 0.5f;
      const auto nu = log2f(log_zn);
      data[pixel_index] = static_cast<float_t>(iterations) + 1.f - nu;
    } else {
      data[pixel_index] = static_cast<float_t>(iterations);
    }
  }
}

void VisualizeNaive(uint32_t* image, uint32_t image_width, uint32_t image_height,
                    double_t center_real, double_t center_imag, double_t zoom_factor,
                    uint32_t max_iterations, uint32_t coloring_mode,
                    uint32_t palette, bool smoothing) {

  constexpr auto kMemoryPoolSize = 256 << 20;
  static auto memory_pool = GPUMemoryPool{kMemoryPoolSize}; // 256 MB
  memory_pool.Reset();

  const auto image_size = image_width * image_height;

  const auto data_bytes = image_size * sizeof(float_t);
  const auto color_bytes = image_size * sizeof(uint32_t);

  auto device_data = static_cast<float_t*>(memory_pool.Alloc(data_bytes));
  auto device_color = static_cast<uint32_t*>(memory_pool.Alloc(color_bytes));

  constexpr auto kThreadsPerBlock = 512;
  const auto kBlocksPerGrid = (image_size - 1) / kThreadsPerBlock + 1;

  constexpr static auto kMandelbrotSetWidth  = 3.0;  // [-2, 1]
  constexpr static auto kMandelbrotSetHeight = 2.0;  // [-1, 1]
  const auto scale =
    1.0 / std::fmin(image_width  * zoom_factor / kMandelbrotSetWidth,
                    image_height * zoom_factor / kMandelbrotSetHeight);

  KernelMandelbrotSet<double_t><<<kBlocksPerGrid, kThreadsPerBlock>>>(
      device_data, image_width, image_height,
      center_real, center_imag, scale, max_iterations, smoothing);

  cuda::KenrelColor<<<kBlocksPerGrid, kThreadsPerBlock>>>(
      device_data, device_color, image_width, image_height,
      max_iterations, coloring_mode, palette);

  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(image, device_color, color_bytes,
                        cudaMemcpyDeviceToHost));
}

template <typename T>
struct ComplexGpuT {
  using value_type = T;
  T real;
  T imag;
};

using ComplexGPU = ComplexGpuT<double_t>;

template <typename T>
__global__ void KernelMandelbrotSetPerturbation(
  float_t* data, uint32_t width, uint32_t height,
  ComplexGPU* orbit, uint32_t orbit_size,
  T center_real, T center_imag, T scale,
  uint32_t max_iterations, bool smoothing_step = false) {

  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (pixel_index < width * height) {
    const auto real0 = (pixel_index % width - width  / T{2}) * scale;
    const auto imag0 = (pixel_index / width - height / T{2}) * scale;

    auto real = T{0};
    auto imag = T{0};
    auto iterations = max_iterations;

    auto limit = T{4};
    if (smoothing_step) {
      limit = T{16};
    }

    auto orbit_index = uint32_t{0};
    for (auto i = uint32_t{0}; i < max_iterations; ++i) {
      auto Zr = orbit[orbit_index].real;
      auto Zi = orbit[orbit_index].imag;

      ++orbit_index;

      const auto dr2 = real * real - imag * imag;
      const auto di2 = 2.0 * real * imag;

      const auto tdr = 2.0 * (Zr * real - Zi * imag);
      const auto tdi = 2.0 * (Zr * imag + Zi * real);

      real = tdr + dr2 + real0;
      imag = tdi + di2 + imag0;

      Zr = orbit[orbit_index].real;
      Zi = orbit[orbit_index].imag;

      const auto zr = Zr + real;
      const auto zi = Zi + imag;

      if (zr * zr + zi * zi > limit) {
        iterations = i;
        break;
      }

      const auto dz2 = real * real + imag * imag;
      const auto Z2 = Zr * Zr + Zi * Zi;

      if (dz2 > Z2 || orbit_index == orbit_size) {
        real = zr;
        imag = zi;
        orbit_index = 0;
      }
    }

    if (smoothing_step && iterations < max_iterations) {
      const auto zr = orbit[orbit_index].real + real;
      const auto zi = orbit[orbit_index].imag + imag;
      const auto log_zn = log(zr * zr + zi * zi) * T{0.5};
      const auto nu = log2(log_zn);
      data[pixel_index] = static_cast<float_t>(iterations) + 1.f - nu;
    } else {
      data[pixel_index] = static_cast<float_t>(iterations);
    }
  }
}

void VisualizePerturbation(
  uint32_t* image, uint32_t image_width, uint32_t image_height,
  double_t center_real, double_t center_imag, double_t zoom_factor,
  uint32_t max_iterations, uint32_t coloring_mode,
  uint32_t palette, bool smoothing) {

  constexpr auto kMemoryPoolSize = 256 << 20;
  static auto memory_pool = GPUMemoryPool{kMemoryPoolSize}; // 256 MB
  memory_pool.Reset();

  auto orbit = ComputeReferenceOrbit(Complex{ center_real, center_imag }, max_iterations, smoothing);
  auto orbit_gpu = std::vector<ComplexGPU>(orbit.size());

  for (auto i = uint32_t{0}; i < orbit.size(); ++i) {
    orbit_gpu[i].real = static_cast<double_t>(orbit[i].real);
    orbit_gpu[i].imag = static_cast<double_t>(orbit[i].imag);
  }

  const auto image_size = image_width * image_height;

  const auto data_bytes = image_size * sizeof(float_t);
  const auto color_bytes = image_size * sizeof(uint32_t);
  const auto orbit_bytes = orbit_gpu.size() * sizeof(ComplexGPU);

  auto device_data = static_cast<float_t*>(memory_pool.Alloc(data_bytes));
  auto device_color = static_cast<uint32_t*>(memory_pool.Alloc(color_bytes));
  auto device_orbit = static_cast<ComplexGPU*>(memory_pool.Alloc(orbit_bytes));

  cudaMemcpy(device_orbit, orbit_gpu.data(), orbit_bytes, cudaMemcpyHostToDevice);

  constexpr auto kThreadsPerBlock = 512;
  const auto kBlocksPerGrid = (image_size - 1) / kThreadsPerBlock + 1;

  constexpr static auto kMandelbrotSetWidth  = 3.0;  // [-2, 1]
  constexpr static auto kMandelbrotSetHeight = 2.0;  // [-1, 1]
  const auto scale =
    1.0 / std::fmin(image_width  * zoom_factor / kMandelbrotSetWidth,
                    image_height * zoom_factor / kMandelbrotSetHeight);

  KernelMandelbrotSetPerturbation<ComplexGPU::value_type><<<kBlocksPerGrid, kThreadsPerBlock>>>(
    device_data, image_width, image_height,
    device_orbit, orbit.size() - 1,
    center_real, center_imag,
    scale, max_iterations, smoothing
  );

  cuda::KenrelColor<<<kBlocksPerGrid, kThreadsPerBlock>>>(
      device_data, device_color, image_width, image_height,
      max_iterations, coloring_mode, palette);

  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(image, device_color, color_bytes,
                        cudaMemcpyDeviceToHost));
}

}  // namespace cuda
}  // namespace mandelbrot