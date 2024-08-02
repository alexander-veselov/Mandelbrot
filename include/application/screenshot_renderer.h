#pragma once

#include "image.h"
#include "complex.h"
#include "mandelbrot_renderer.h"

namespace MandelbrotSet {

// TODO: make not global
constexpr auto kScreenshotWidth = 1920u;
constexpr auto kScreenshotHeight = 1080u;

// TODO: save images into folder relative to project sources
constexpr auto kScreenshotsFolder = "images";

class ScreenshotRenderer : public MandelbrotRenderer {
 public:
  ScreenshotRenderer(uint32_t screenshot_width = kScreenshotWidth,
                     uint32_t screenshot_height = kScreenshotHeight);
  void Render(const Complex& center, double_t zoom,
              const RenderOptions& render_options) override;

 private:
  // TODO: Create Size structure
  uint32_t screenshot_width_;
  uint32_t screenshot_height_;
};

}