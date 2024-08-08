#pragma once

#include "mandelbrot/core/coloring_mode.h"
#include "mandelbrot/core/complex.h"
#include "mandelbrot/core/image.h"
#include "mandelbrot/core/size.h"

namespace mandelbrot {

class MandelbrotRenderer {
 public:
  MandelbrotRenderer(const Size& size);

  struct RenderOptions {
    ColoringMode coloring_mode;
    uint32_t max_iterations;
    bool smoothing;
  };

  void Render(const Complex& center, double_t zoom,
              const RenderOptions& render_options);

 protected:
  virtual bool IsDirty(const Complex& center, double_t zoom) const = 0;
  virtual void RenderImage(const Image& image) const = 0;

 protected:
  Image image_;
  Complex center_;
  double_t zoom_;
};

}  // namespace mandelbrot