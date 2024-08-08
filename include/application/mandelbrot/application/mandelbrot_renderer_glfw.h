#pragma once

#include "mandelbrot/core/image.h"
#include "mandelbrot/core/size.h"

#include "mandelbrot/application/mandelbrot_renderer.h"

namespace MandelbrotSet {

class MandelbrotRendererGLFW : public MandelbrotRenderer {
 public:
  MandelbrotRendererGLFW(const Size& size);

 protected:
  bool IsDirty(const Complex& center, double_t zoom) const override;
  void RenderImage(const Image& image) const override;
};

}  // namespace MandelbrotSet