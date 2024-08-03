#pragma once

#include "image.h"
#include "complex.h"
#include "coloring_mode.h"
#include "size.h"

namespace MandelbrotSet {

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

}  // namespace MandelbrotSet