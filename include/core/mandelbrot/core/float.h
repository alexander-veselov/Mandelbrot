#pragma once

#include <cmath>
#include <stdint.h>

#include <boost/multiprecision/cpp_bin_float.hpp>

namespace mandelbrot {

using BoostFloat = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<128>>;
using Float = BoostFloat;

}