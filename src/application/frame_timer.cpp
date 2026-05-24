#include "mandelbrot/application/frame_timer.h"

namespace mandelbrot {

FrameTimer::FrameTimer(uint32_t update_rate, double_t current_time)
    : update_rate_{update_rate},
      frame_count_{0},
      last_capture_time_{current_time},
      last_frame_time_{current_time},
      fps_{0.},
      fps_accum_time_{0.},
      delta_time_{0.} {}

void FrameTimer::Update(double_t current_time) {
  delta_time_ = current_time - last_frame_time_;
  last_frame_time_ = current_time;

  fps_accum_time_ += delta_time_;
  frame_count_++;

  if (fps_accum_time_ * update_rate_ >= 1.0) {
    fps_ = static_cast<double_t>(frame_count_) / fps_accum_time_;
    frame_count_ = 0;
    fps_accum_time_ = 0.0;
  }
}

double_t FrameTimer::GetFPS() const noexcept { return fps_; }

double_t FrameTimer::GetDeltaTime() const noexcept { return delta_time_; }

}  // namespace mandelbrot