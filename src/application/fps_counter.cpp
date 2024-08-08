#include "mandelbrot/application/fps_counter.h"

namespace MandelbrotSet {

FPSCounter::FPSCounter(uint32_t update_rate, double_t current_time)
    : update_rate_{update_rate},
      frame_count_{0},
      last_capture_time_{current_time},
      fps_{0.} {}

void FPSCounter::Update(double_t current_time) {
  const auto delta_time = current_time - last_capture_time_;
  fps_ = static_cast<double_t>(++frame_count_) / delta_time;
  if (delta_time * update_rate_ >= 1.) {
    last_capture_time_ = current_time;
    frame_count_ = 0;
  }
}

double_t FPSCounter::GetFPS() const noexcept { return fps_; }

}  // namespace MandelbrotSet