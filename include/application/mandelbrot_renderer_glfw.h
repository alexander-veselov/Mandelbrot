#pragma once

#include "mandelbrot_renderer.h"
#include "image.h"
#include "size.h"

namespace MandelbrotSet {

class MandelbrotRendererGLFW : public MandelbrotRenderer {
 public:
  MandelbrotRendererGLFW(const Size& size);

 protected:
  bool IsDirty(const Complex& center, double_t zoom) const override;
  void RenderImage(const Image& image) const override;
};

}  // namespace MandelbrotSet