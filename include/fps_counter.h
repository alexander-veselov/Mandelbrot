#pragma once

#include <cmath>
#include <stdint.h>

namespace MandelbrotSet {

class FPSCounter {
 public:
  explicit FPSCounter(int32_t update_rate, double_t current_time);
  void Update(double_t current_time);
  double_t GetFPS() const noexcept;

 private:
  const int32_t update_rate_;
  int32_t frame_count_;
  double_t last_capture_time_;
  double_t fps_;
};

}  // namespace MandelbrotSet