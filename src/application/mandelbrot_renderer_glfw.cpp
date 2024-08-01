#include "mandelbrot_renderer_glfw.h"

#include <GLFW/glfw3.h>

#include "mandelbrot_set.cuh"

namespace MandelbrotSet {

void DrawImage(const Image& image) {
  static_assert(std::is_same_v<Image::Type, uint32_t>,
                "Image::Type must be uint32_t");
  glDrawPixels(image.GetWidth(), image.GetHeight(), GL_RGBA, GL_UNSIGNED_BYTE,
               image.GetData());
}

MandelbrotRendererGLFW::MandelbrotRendererGLFW(uint32_t width, uint32_t height)
    : image_{width, height}, center_{}, zoom_{} {}

void MandelbrotRendererGLFW::Render(const Complex& center, double_t zoom,
                                    const RenderOptions& render_options) {
  const auto dirty = center.real != center_.real ||
                     center.imag != center_.imag || zoom != zoom_;

  if (dirty) {
    MandelbrotSet::Visualize(
      image_.GetData(), image_.GetWidth(), image_.GetHeight(), center.real,
      center.imag, zoom, render_options.max_iterations,
      static_cast<int32_t>(render_options.coloring_mode), render_options.smoothing);

    center_ = center;
    zoom_ = zoom;
  }

  DrawImage(image_);
}

}  // namespace MandelbrotSet