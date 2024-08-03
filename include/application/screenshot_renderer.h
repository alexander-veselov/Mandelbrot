#pragma once

#include "image.h"
#include "complex.h"
#include "size.h"
#include "mandelbrot_renderer.h"

namespace MandelbrotSet {

class ScreenshotRenderer : public MandelbrotRenderer {
 public:
  ScreenshotRenderer(const Size& size);

 protected:
  bool IsDirty(const Complex& center, double_t zoom) const override;
  void RenderImage(const Image& image) const override;
};

}