#pragma once

#include <math_constants.h>
#include <stdint.h>
#include <string>

using double_t = double;
using float_t = float;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(error) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                \
    }                                                                          \
  } while (false)