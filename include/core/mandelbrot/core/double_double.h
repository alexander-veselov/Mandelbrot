#pragma once

#include "mandelbrot/core/typedefs.h"

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

private:
  DoubleDouble(double_t hi, double_t lo);

  double_t hi;
  double_t lo;
};

}  // namespace mandelbrot