#pragma once

#include "mandelbrot/core/coloring_mode.h"
#include "mandelbrot/core/complex.h"
#include "mandelbrot/core/image.h"
#include "mandelbrot/core/render_options.h"
#include "mandelbrot/core/size.h"

namespace mandelbrot {

class MandelbrotRenderer {
 public:
  MandelbrotRenderer(const Size& size);

  void Render(const Complex& ref, const Complex& dc, DoubleDouble zoom,
              const RenderOptions& render_options);

 protected:
  virtual bool IsDirty(const Complex& ref, const Complex& dc, DoubleDouble zoom,
                       const RenderOptions& render_options) const = 0;
  virtual void RenderImage(const Image& image) const = 0;

 protected:
  Image image_;
  Complex ref_;
  Complex dc_;
  DoubleDouble zoom_;
  RenderOptions render_options_;
};

}  // namespace mandelbrot