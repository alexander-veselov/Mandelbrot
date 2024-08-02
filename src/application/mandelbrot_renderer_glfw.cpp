#include "mandelbrot_renderer_glfw.h"

#include <GLFW/glfw3.h>

#include "mandelbrot_set.cuh"

namespace MandelbrotSet {

MandelbrotRendererGLFW::MandelbrotRendererGLFW(const Size& size)
    : MandelbrotRenderer{size} {}

bool MandelbrotRendererGLFW::IsDirty(const Complex& center,
                                     double_t zoom) const {
  return center.real != center_.real || center.imag != center_.imag ||
         zoom != zoom_;
}

void MandelbrotRendererGLFW::RenderImage(const Image& image) const {
  static_assert(std::is_same_v<Image::Type, uint32_t>,
                "Image::Type must be uint32_t");
  glDrawPixels(image.GetWidth(), image.GetHeight(), GL_RGBA, GL_UNSIGNED_BYTE,
               image.GetData());
}

}  // namespace MandelbrotSet