#pragma once

#include "image.h"
#include "complex.h"
#include "size.h"
#include "mandelbrot_renderer.h"

namespace MandelbrotSet {

// TODO: make not global
constexpr auto kScreenshotWidth = 1920u;
constexpr auto kScreenshotHeight = 1080u;

// TODO: save images into folder relative to project sources
constexpr auto kScreenshotsFolder = "images";

class ScreenshotRenderer : public MandelbrotRenderer {
 public:
  ScreenshotRenderer(const Size& size = {kScreenshotWidth, kScreenshotHeight});

 protected:
  bool IsDirty(const Complex& center, double_t zoom) const override;
  void RenderImage(const Image& image) const override;
};

}