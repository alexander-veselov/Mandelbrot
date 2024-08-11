#pragma once

#include "mandelbrot/core/coloring_mode.h"
#include "mandelbrot/core/complex.h"
#include "mandelbrot/core/palette.h"
#include "mandelbrot/core/size.h"

#include "mandelbrot/application/window_mode.h"

#include <string>

namespace mandelbrot {

struct Config {
  ColoringMode coloring_mode = ColoringMode::kMode1;
  Complex default_position = Complex{-0.5, 0.0};
  double_t default_zoom = 1.0;
  bool directional_zoom = true;
  bool enable_vsync = true;
  uint32_t fps_update_rate = 10;
  uint32_t max_iterations = 1024;
  Palette palette = Palette::kBluePalette;
  Size screenshot_size = Size{1920, 1080};
  std::string screenshots_folder = "data/images";
  bool smoothing = true;
  WindowMode window_mode = WindowMode::kWindowed;
  Size window_size = Size{1024, 768};
  double_t zoom_factor = 1.5;
};

const Config& GetConfig();

}  // namespace mandelbrot