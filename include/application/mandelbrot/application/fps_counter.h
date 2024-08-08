#pragma once

#include "mandelbrot/core/typedefs.h"

namespace MandelbrotSet {

class FPSCounter {
 public:
  explicit FPSCounter(uint32_t update_rate, double_t current_time);
  void Update(double_t current_time);
  double_t GetFPS() const noexcept;

 private:
  const uint32_t update_rate_;
  int32_t frame_count_;
  double_t last_capture_time_;
  double_t fps_;
};

}  // namespace MandelbrotSet