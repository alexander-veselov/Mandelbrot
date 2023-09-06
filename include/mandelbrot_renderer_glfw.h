#pragma once

#include "image.h"
#include "complex.h"

namespace MandelbrotSet {

// TODO: Split MandelbrotRendererGLFW class
// At the moment it's responsible for too many things:
// 1. Mandelbrot set calculation
// 2. Storing image data
// 3. Rendering using specific GLFW implementation
class MandelbrotRendererGLFW {
 public:
  struct RenderOptions {
    uint32_t coloring_mode = 5;
    uint32_t max_iterations = 1024;
    bool smoothing = true;
  };
  MandelbrotRendererGLFW(uint32_t width, uint32_t height);
  void Render(const Complex& center, double_t zoom,
              const RenderOptions& render_options = {});

 private:
  Image image_;
  Complex center_;
  double_t zoom_;
};

}  // namespace MandelbrotSet