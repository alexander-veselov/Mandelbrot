#include "screenshot_renderer.h"

#include "mandelbrot_set.cuh"
#include "utils.h"

namespace MandelbrotSet {

ScreenshotRenderer::ScreenshotRenderer(const Size& size)
    : MandelbrotRenderer{size} {}

bool ScreenshotRenderer::IsDirty(const Complex& center, double_t zoom) const {
  return true;
}

void ScreenshotRenderer::RenderImage(const Image& image) const {
  // TODO: improve screenshot naming
  WriteImage(image, std::filesystem::path{kScreenshotsFolder} / "screenshot.bmp");
}

}