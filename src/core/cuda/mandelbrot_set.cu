#include "mandelbrot/core/cuda/mandelbrot_set.h"

#include "mandelbrot/core/cuda/coloring.h"
#include "mandelbrot/core/cuda/gpu_memory_pool.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mandelbrot {
namespace cuda {

namespace {

struct DoubleDouble {
  double hi;
  double lo;
};

struct ReferenceOrbitPoint {
  double real_hi;
  double real_lo;
  double imag_hi;
  double imag_lo;
};

DoubleDouble Normalize(double hi, double lo) {
  const auto sum = hi + lo;
  const auto error = lo - (sum - hi);
  return {sum, error};
}

DoubleDouble FromDouble(double value) {
  return {value, 0.};
}

DoubleDouble Negate(DoubleDouble value) {
  return {-value.hi, -value.lo};
}

DoubleDouble Add(DoubleDouble left, DoubleDouble right) {
  const auto sum = left.hi + right.hi;
  const auto virtual_right = sum - left.hi;
  const auto virtual_left = sum - virtual_right;
  const auto right_error = right.hi - virtual_right;
  const auto left_error = left.hi - virtual_left;
  const auto error = left_error + right_error + left.lo + right.lo;
  return Normalize(sum, error);
}

DoubleDouble Subtract(DoubleDouble left, DoubleDouble right) {
  return Add(left, Negate(right));
}

DoubleDouble Multiply(DoubleDouble left, DoubleDouble right) {
  const auto product = left.hi * right.hi;
  const auto error = std::fma(left.hi, right.hi, -product) +
                     left.hi * right.lo + left.lo * right.hi;
  return Normalize(product, error);
}

DoubleDouble Square(DoubleDouble value) {
  return Multiply(value, value);
}

std::vector<ReferenceOrbitPoint> BuildReferenceOrbit(double center_real,
                                                     double center_imag,
                                                     uint32_t max_iterations) {
  auto orbit = std::vector<ReferenceOrbitPoint>{};
  orbit.reserve(max_iterations);

  const auto reference_real = FromDouble(center_real);
  const auto reference_imag = FromDouble(center_imag);
  auto real = reference_real;
  auto imag = reference_imag;

  for (auto i = uint32_t{0}; i < max_iterations; ++i) {
    orbit.push_back({real.hi, real.lo, imag.hi, imag.lo});

    const auto real_squared = Square(real);
    const auto imag_squared = Square(imag);
    const auto real_imag = Multiply(real, imag);

    const auto next_real =
        Add(Subtract(real_squared, imag_squared), reference_real);
    const auto next_imag =
        Add(Add(real_imag, real_imag), reference_imag);

    real = next_real;
    imag = next_imag;
  }

  return orbit;
}

}  // namespace

__global__ void KernelMandelbrotSet(float_t* data, uint32_t width,
                                    uint32_t height, double zoom_factor,
                                    const ReferenceOrbitPoint* reference_orbit,
                                    uint32_t max_iterations,
                                    bool smoothing_step = false) {

  const auto pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (pixel_index < width * height) {
    // Mandelbrot set parameters
    constexpr static auto kMandelbrotSetWidth  = 3.;  // [-2, 1]
    constexpr static auto kMandelbrotSetHeight = 2.;  // [-1, 1]

    const auto scale =
        1. / fmin(static_cast<double>(width) / kMandelbrotSetWidth,
                  static_cast<double>(height) / kMandelbrotSetHeight);

    const auto x = (static_cast<double>(pixel_index % width) -
                    static_cast<double>(width) / 2.) * scale;
    const auto y = (static_cast<double>(pixel_index / width) -
                    static_cast<double>(height) / 2.) * scale;

    const auto delta0_real = x / zoom_factor;
    const auto delta0_imag = y / zoom_factor;

    auto delta_real = delta0_real;
    auto delta_imag = delta0_imag;
    auto iterations = max_iterations;

    auto limit = 4.;
    if (smoothing_step) {
      limit = 16.;
    }

    auto real = 0.;
    auto imag = 0.;
    for (auto i = uint32_t{0}; i < max_iterations; ++i) {
      const auto reference = reference_orbit[i];

      real = reference.real_hi + (reference.real_lo + delta_real);
      imag = reference.imag_hi + (reference.imag_lo + delta_imag);

      const auto real_squared = real * real;
      const auto imag_squared = imag * imag;

      if (real_squared + imag_squared > limit) {
        iterations = i;
        break;
      }

      const auto delta_real_squared = delta_real * delta_real;
      const auto delta_imag_squared = delta_imag * delta_imag;
      const auto reference_delta_real =
          reference.real_hi * delta_real + reference.real_lo * delta_real -
          reference.imag_hi * delta_imag - reference.imag_lo * delta_imag;
      const auto reference_delta_imag =
          reference.real_hi * delta_imag + reference.real_lo * delta_imag +
          reference.imag_hi * delta_real + reference.imag_lo * delta_real;

      const auto next_delta_real =
          2. * reference_delta_real +
          delta_real_squared - delta_imag_squared + delta0_real;
      const auto next_delta_imag =
          2. * reference_delta_imag +
          2. * delta_real * delta_imag + delta0_imag;

      delta_real = next_delta_real;
      delta_imag = next_delta_imag;
    }

    if (smoothing_step && iterations < max_iterations) {
      const auto log_zn = log(real * real + imag * imag) / 2.;
      const auto nu = log(log_zn / log(2.)) / log(2.);
      data[pixel_index] = static_cast<float_t>(iterations) + 1.f -
                          static_cast<float_t>(nu);
    } else {
      data[pixel_index] = static_cast<float_t>(iterations);
    }
  }
}

void Visualize(uint32_t* image, uint32_t image_width, uint32_t image_height,
               double_t center_real, double_t center_imag, double_t zoom_factor,
               uint32_t max_iterations, uint32_t coloring_mode,
               uint32_t palette, bool smoothing) {

  constexpr auto kMemoryPoolSize = 128 << 20;
  static auto memory_pool = GPUMemoryPool{kMemoryPoolSize};  // 128 MB

  static_assert(sizeof(float_t) == sizeof(uint32_t));

  const auto image_size = image_width * image_height;
  const auto image_size_in_bytes = image_size * sizeof(uint32_t);

  if (image_size_in_bytes > kMemoryPoolSize) {
    throw std::runtime_error{"Not enought GPU memory in pool"};
  }

  auto device_data = memory_pool.Alloc(image_size_in_bytes);
  auto reference_orbit = BuildReferenceOrbit(center_real, center_imag,
                                             max_iterations);
  auto device_reference_orbit = static_cast<ReferenceOrbitPoint*>(nullptr);
  const auto reference_orbit_size_in_bytes =
      reference_orbit.size() * sizeof(ReferenceOrbitPoint);

  if (reference_orbit_size_in_bytes > 0) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_reference_orbit),
                          reference_orbit_size_in_bytes));
  }

  try {
    if (device_reference_orbit != nullptr) {
      CUDA_CHECK(cudaMemcpy(device_reference_orbit, reference_orbit.data(),
                            reference_orbit_size_in_bytes,
                            cudaMemcpyHostToDevice));
    }

    constexpr auto kThreadsPerBlock = 512;
    const auto kBlocksPerGrid = (image_size - 1) / kThreadsPerBlock + 1;

    KernelMandelbrotSet<<<kBlocksPerGrid, kThreadsPerBlock>>>(
        reinterpret_cast<float_t*>(device_data), image_width, image_height,
        zoom_factor, device_reference_orbit, max_iterations, smoothing);

    cuda::KenrelColor<<<kBlocksPerGrid, kThreadsPerBlock>>>(
        reinterpret_cast<uint32_t*>(device_data), image_width, image_height,
        max_iterations, coloring_mode, palette);

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(image, device_data, image_size_in_bytes,
                          cudaMemcpyDeviceToHost));
  } catch (...) {
    if (device_reference_orbit != nullptr) {
      cudaFree(device_reference_orbit);
    }
    memory_pool.Free(device_data);
    throw;
  }

  if (device_reference_orbit != nullptr) {
    CUDA_CHECK(cudaFree(device_reference_orbit));
  }
  memory_pool.Free(device_data);
}

}  // namespace cuda
}  // namespace mandelbrot
