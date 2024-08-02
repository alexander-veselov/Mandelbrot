#include "screenshot_renderer.h"

#include "mandelbrot_set.cuh"
#include "utils.h"

namespace MandelbrotSet {

ScreenshotRenderer::ScreenshotRenderer(const Size& screenshot_size)
    : screenshot_size_{screenshot_size} {}

void ScreenshotRenderer::Render(const Complex& center, double_t zoom,
                                const RenderOptions& render_options) {
  auto screenshot = Image{screenshot_size_};
  MandelbrotSet::Visualize(screenshot.GetData(), screenshot.GetWidth(),
                           screenshot.GetHeight(), center.real, center.imag,
                           zoom, render_options.max_iterations,
                           static_cast<int32_t>(render_options.coloring_mode),
                           render_options.smoothing);

  // TODO: improve screenshot naming
  WriteImage(screenshot, std::filesystem::path{kScreenshotsFolder} / "screenshot.bmp");
}

}