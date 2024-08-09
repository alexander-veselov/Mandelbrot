#pragma once

#include "mandelbrot/core/image.h"
#include "mandelbrot/core/render_options.h"
#include "mandelbrot/core/size.h"

#include "mandelbrot/application/mandelbrot_renderer.h"

namespace mandelbrot {

class MandelbrotRendererGLFW : public MandelbrotRenderer {
 public:
  MandelbrotRendererGLFW(const Size& size);

 protected:
  bool IsDirty(const Complex& center, double_t zoom,
               const RenderOptions& render_options) const override;
  void RenderImage(const Image& image) const override;
};

}  // namespace mandelbrot