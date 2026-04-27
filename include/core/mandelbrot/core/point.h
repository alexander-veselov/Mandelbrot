#pragma once

#include "mandelbrot/core/typedefs.h"

namespace mandelbrot {

template <typename T>
struct PointT {
  using value_type = T;

  T x;
  T y;
};

template <typename T>
bool operator==(const PointT<T>& left, const PointT<T>& right) {
  return left.x == right.x && left.y == right.y;
}

template <typename T>
bool operator!=(const PointT<T>& left, const PointT<T>& right) {
  return !(left == right);
}

using PointD = PointT<double_t>;
using PointDD = PointT<DoubleDouble>;
using Point = PointDD;

}  // namespace mandelbrot