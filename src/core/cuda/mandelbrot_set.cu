#include "mandelbrot/core/cuda/mandelbrot_set.h"

#include "mandelbrot/core/cuda/coloring.h"
#include "mandelbrot/core/cuda/gpu_memory_pool.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <vector>

namespace mandelbrot {
namespace cuda {

  struct Complex {
    double real;
    double imag;
  };

  __global__ void KernelMandelbrotSet(uint32_t* data,
    uint32_t width,
    uint32_t height,
    double center_real,
    double center_imag,
    double zoom_factor,
    uint32_t max_iterations,
    const Complex* __restrict__ ref_orbit,
    const Complex* __restrict__ delta_c, int ref_max_iterations) {

    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    constexpr double kWidth = 3.0;
    constexpr double kHeight = 2.0;

    const double scale =
      1.0 / fmin(width / kWidth, height / kHeight);

    const double x = (idx % width - width / 2.0) * scale;
    const double y = (idx / width - height / 2.0) * scale;

    const double dc_real = delta_c[idx].real;
    const double dc_imag = delta_c[idx].imag;

    double dr = 0.0;
    double di = 0.0;

    uint32_t iter = max_iterations;

    int ref = 0;
    for (uint32_t i = 0; i < max_iterations; ++i) {
      double Zr = ref_orbit[ref].real;
      double Zi = ref_orbit[ref].imag;

      ref++;

      const double dr2 = dr * dr - di * di;
      const double di2 = 2.0 * dr * di;

      const double tdr = 2.0 * (Zr * dr - Zi * di);
      const double tdi = 2.0 * (Zr * di + Zi * dr);

      dr = tdr + dr2 + dc_real;
      di = tdi + di2 + dc_imag;

      Zr = ref_orbit[ref].real;
      Zi = ref_orbit[ref].imag;

      const double zr = Zr + dr;
      const double zi = Zi + di;

      if (zr * zr + zi * zi > 4.0) {
        iter = i;
        break;
      }

      const double dz2 = dr * dr + di * di;
      const double Z2 = Zr * Zr + Zi * Zi;

      if (dz2 > Z2 || ref == ref_max_iterations) {
        dr = zr;
        di = zi;
        ref = 0;
      }
    }

    data[idx] = iter;
  }



void Visualize(uint32_t* image, uint32_t image_width, uint32_t image_height,
               double_t center_real, double_t center_imag, double_t zoom_factor,
               uint32_t max_iterations, uint32_t coloring_mode,
               uint32_t palette, const std::vector<ComplexDD>& orbit) {

  constexpr auto kMemoryPoolSize = 128 << 20;
  static auto memory_pool = GPUMemoryPool{kMemoryPoolSize};  // 128 MB

  const auto image_size = image_width * image_height;
  const auto image_size_in_bytes = image_size * sizeof(uint32_t);

  if (image_size_in_bytes > kMemoryPoolSize) {
    throw std::runtime_error{"Not enought GPU memory in pool"};
  }

  std::vector<Complex> orbit_gpu(orbit.size());

  for (uint32_t i = 0; i < orbit.size(); ++i) {
    orbit_gpu[i].real = static_cast<double_t>(orbit[i].real);
    orbit_gpu[i].imag = static_cast<double_t>(orbit[i].imag);
  }

  Complex* device_orbit;
  cudaMalloc(&device_orbit, max_iterations * sizeof(Complex));

  cudaMemcpy(device_orbit, orbit_gpu.data(),
    max_iterations * sizeof(Complex),
    cudaMemcpyHostToDevice);

  std::vector<Complex> delta_c(image_size);

  DoubleDouble dd_center_real(center_real);
  DoubleDouble dd_center_imag(center_imag);
  DoubleDouble dd_zoom(zoom_factor);

  constexpr double kWidth = 3.0;
  constexpr double kHeight = 2.0;

  const double scale =
    1.0 / fmin(image_width / kWidth, image_height / kHeight);

  for (uint32_t idx = 0; idx < image_size; ++idx) {
    const double x = (idx % image_width - image_width / 2.0) * scale;
    const double y = (idx / image_width - image_height / 2.0) * scale;

    // high precision division
    DoubleDouble dd_dx = DoubleDouble(x) / dd_zoom;
    DoubleDouble dd_dy = DoubleDouble(y) / dd_zoom;

    delta_c[idx].real = static_cast<double>(dd_dx);
    delta_c[idx].imag = static_cast<double>(dd_dy);
  }

  Complex* device_delta_c;
  cudaMalloc(&device_delta_c, image_size * sizeof(Complex));

  cudaMemcpy(device_delta_c,
    delta_c.data(),
    image_size * sizeof(Complex),
    cudaMemcpyHostToDevice);

  auto device_data = memory_pool.Alloc(image_size_in_bytes);

  constexpr auto kThreadsPerBlock = 512;
  const auto kBlocksPerGrid = (image_size - 1) / kThreadsPerBlock + 1;

  KernelMandelbrotSet<<<kBlocksPerGrid, kThreadsPerBlock>>>(
      reinterpret_cast<uint32_t*>(device_data), image_width, image_height,
      center_real, center_imag, zoom_factor, max_iterations, device_orbit, device_delta_c, orbit.size() -1);

  cuda::KenrelColor<<<kBlocksPerGrid, kThreadsPerBlock>>>(
      reinterpret_cast<uint32_t*>(device_data), image_width, image_height,
      max_iterations, coloring_mode, palette);

  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(device_delta_c);
  cudaFree(device_orbit);

  CUDA_CHECK(cudaMemcpy(image, device_data, image_size_in_bytes,
                        cudaMemcpyDeviceToHost));
  memory_pool.Free(device_data);
}

}  // namespace cuda
}  // namespace mandelbrot