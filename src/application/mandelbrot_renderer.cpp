#include "mandelbrot/application/mandelbrot_renderer.h"

#include "mandelbrot/core/cuda/mandelbrot_set.h"

namespace mandelbrot {

MandelbrotRenderer::MandelbrotRenderer(const Size& size)
    : image_{size}, ref_{}, dc_{}, zoom_{} {}

inline DoubleDouble ToDoubleDouble(const SuperDouble& r) {
  double hi = static_cast<double>(r);
  SuperDouble rem = r - SuperDouble(hi);
  double lo = static_cast<double>(rem);
  return DoubleDouble(hi, lo);
}

void MandelbrotRenderer::Render(const Complex& ref, const Complex& dc, DoubleDouble zoom,
                                const RenderOptions& render_options) {
  if (true) {

    std::vector<cuda::ComplexDD> orbit;
    cuda::ComputeReferenceOrbit(orbit, ref.real, ref.imag, dc.real, dc.imag, render_options.max_iterations);

    cuda::Visualize(image_.GetData(), image_.GetWidth(), image_.GetHeight(),
                    ToDoubleDouble(ref.real), ToDoubleDouble(ref.imag),
                    ToDoubleDouble(dc.real), ToDoubleDouble(dc.imag),
                    zoom,
                    render_options.max_iterations,
                    static_cast<int32_t>(render_options.coloring_mode),
                    static_cast<int32_t>(render_options.palette),
                    orbit);

    ref_ = ref;
    dc_ = dc;
    zoom_ = zoom;
    render_options_ = render_options;
  }

  RenderImage(image_);
}

}  // namespace mandelbrot