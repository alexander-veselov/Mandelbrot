#pragma once

#include "complex.h"

namespace MandelbrotSet {

class MandelbrotRenderer {
 public:
  enum class ColoringMode {
    kBlackWhite,
    kBlue,
    kRed,
    kBlueGreen,
    kOrange,
    kWaves
  };

  struct RenderOptions {
    ColoringMode coloring_mode = ColoringMode::kBlue;
    uint32_t max_iterations = 1024;
    bool smoothing = true;
  };

  virtual void Render(const Complex& center, double_t zoom,
    const RenderOptions& render_options) = 0;
};

}  // namespace MandelbrotSet