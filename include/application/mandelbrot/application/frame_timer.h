#pragma once

#include "mandelbrot/core/typedefs.h"

namespace mandelbrot {

class FrameTimer {
 public:
  explicit FrameTimer(uint32_t update_rate, double_t current_time);
  void Update(double_t current_time);
  double_t GetFPS() const noexcept;
  double_t GetDeltaTime() const noexcept;

 private:
  const uint32_t update_rate_;
  int32_t frame_count_;
  double_t last_frame_time_;
  double_t last_capture_time_;
  double_t fps_;
  double_t fps_accum_time_;
  double_t delta_time_;
};

}  // namespace mandelbrot