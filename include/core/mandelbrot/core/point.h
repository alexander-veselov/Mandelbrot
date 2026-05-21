#pragma once

#include "mandelbrot/core/float.h"

namespace mandelbrot {

template <typename T>
struct PointT {
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

using Point = PointT<Float>;

}  // namespace mandelbrot