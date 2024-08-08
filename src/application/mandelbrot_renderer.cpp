#include "mandelbrot/application/mandelbrot_renderer.h"

#include "mandelbrot/core/cuda/mandelbrot_set.h"

namespace mandelbrot {

MandelbrotRenderer::MandelbrotRenderer(const Size& size)
    : image_{size}, center_{}, zoom_{} {}

void MandelbrotRenderer::Render(const Complex& center, double_t zoom,
                                const RenderOptions& render_options) {
  if (IsDirty(center, zoom)) {
    cuda::Visualize(image_.GetData(), image_.GetWidth(), image_.GetHeight(),
                    center.real, center.imag, zoom,
                    render_options.max_iterations,
                    static_cast<int32_t>(render_options.coloring_mode),
                    render_options.smoothing);

    center_ = center;
    zoom_ = zoom;
  }

  RenderImage(image_);
}

}  // namespace mandelbrot