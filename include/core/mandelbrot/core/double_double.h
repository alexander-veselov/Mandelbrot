#pragma once

#include "mandelbrot/core/typedefs.h"
#include "D:\Projects\Mandelbrot\include\core\boost/multiprecision/cpp_bin_float.hpp"
#include <string>

namespace mandelbrot {

// TODO:
// - Make better implicit types conversion
// - Make DoubleDouble usage under ifdef
// - Review json serialization/deserialization
class DoubleDouble {
public:
  DoubleDouble();
  DoubleDouble(double_t);

  operator double_t() const;

  bool operator==(const DoubleDouble& right) const;
  bool operator!=(const DoubleDouble& right) const;

  DoubleDouble operator+(const DoubleDouble& right) const;
  DoubleDouble operator-(const DoubleDouble& right) const;
  DoubleDouble operator*(const DoubleDouble& right) const;
  DoubleDouble operator/(const DoubleDouble& right) const;


  DoubleDouble(double_t hi, double_t lo);

  double_t hi;
  double_t lo;
};

using SuperDouble = boost::multiprecision::number<
  boost::multiprecision::cpp_bin_float<200>>;
}  // namespace mandelbrot