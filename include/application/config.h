#pragma once

#include "coloring_mode.h"
#include "complex.h"
#include "size.h"
#include "window_mode.h"

#include <string>

namespace MandelbrotSet {

struct Config {
  ColoringMode coloring_mode = ColoringMode::kBlue;
  Complex default_position = Complex{-0.5, 0.0};
  double_t default_zoom = 1.0;
  bool directional_zoom = true;
  bool enable_vsync = true;
  uint32_t fps_update_rate = 10;
  uint32_t max_iterations = 1024;
  Size screenshot_size = Size{1920, 1080};
  std::string screenshots_folder = "images";
  bool smoothing = true;
  WindowMode window_mode = WindowMode::kWindowed;
  Size window_size = Size{1024, 768};
  double_t zoom_factor = 1.5;
};

const Config& GetConfig();

}  // namespace MandelbrotSet