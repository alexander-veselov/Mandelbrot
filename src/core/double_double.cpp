#include "mandelbrot/core/double_double.h"

#include <cmath>
#include <sstream>

namespace {

void TwoSum(double_t a, double_t b, double_t& s, double_t& e) {
  s = a + b;
  const auto v = s - a;
  e = (a - (s - v)) + (b - v);
}

// Faster version of TwoSum. Requirement: |a| >= |b|
void QuickTwoSum(double_t a, double_t b, double_t& s, double_t& e) {
  s = a + b;
  e = b - (s - a);
}

void TwoProd(double_t a, double_t b, double_t& p, double_t& e) {
  p = a * b;
  e = std::fma(a, b, -p);
}

}

namespace mandelbrot {

DoubleDouble::DoubleDouble(double_t hi, double_t lo)
  : hi{ hi }, lo{ lo }
{}

DoubleDouble::operator double_t() const {
  return hi + lo;
}

bool DoubleDouble::operator==(const DoubleDouble& r) const {
  return hi == r.hi && lo == r.lo;
}

bool DoubleDouble::operator!=(const DoubleDouble& r) const {
  return !(*this == r);
}

DoubleDouble DoubleDouble::operator+(const DoubleDouble& r) const {
  auto result = DoubleDouble{};

  auto s = double_t{};
  auto e1 = double_t{};
  TwoSum(hi, r.hi, s, e1);

  const auto e2 = lo + r.lo;

  QuickTwoSum(s, e1 + e2, result.hi, result.lo);

  return result;
}

DoubleDouble DoubleDouble::operator-(const DoubleDouble& r) const {
  return *this + DoubleDouble{ -r.hi, -r.lo };
}

DoubleDouble DoubleDouble::operator*(const DoubleDouble& r) const {
  auto result = DoubleDouble{};

  auto p = double_t{}; 
  auto e1 = double_t{};
  TwoProd(hi, r.hi, p, e1);

  const auto e2 = hi * r.lo + lo * r.hi;

  QuickTwoSum(p, e1 + e2, result.hi, result.lo);

  return result;
}

DoubleDouble DoubleDouble::operator/(const DoubleDouble& r) const {
  const auto q = hi / r.hi;

  const auto rq = r * DoubleDouble(q);
  const auto diff = *this - rq;

  const auto correction = (diff.hi + diff.lo) / r.hi;

  return DoubleDouble(q + correction);
}

}  // namespace mandelbrot