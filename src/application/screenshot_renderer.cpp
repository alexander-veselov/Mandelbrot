#include "screenshot_renderer.h"

#include "mandelbrot_set.cuh"
#include "utils.h"
#include "config.h"

#include <sstream>
#include <iomanip>
#include <string>

namespace MandelbrotSet {

std::string ToString(double_t value) {
  auto stream = std::ostringstream{};
  stream.precision(std::numeric_limits<double_t>::max_digits10);
  stream << value;
  auto string = stream.str();
  if (string.empty()) {
    throw std::runtime_error{"Failed to convert double to string"};
  }
  // Replace minus with 'm' symbol to be more readable as filename format
  if (string[0] == '-') {
    string[0] = 'm';
  }
  return string;
}

std::filesystem::path NameScreenshot(const Complex& center, double_t zoom) {
  constexpr auto kExtension = ".bmp";
  constexpr auto kSeparator = "_";
  auto filename = ToString(center.real);
  filename += kSeparator;
  filename += ToString(center.imag);
  filename += kSeparator;
  filename += ToString(zoom);
  filename += kExtension;
  return std::filesystem::path{GetConfig().screenshots_folder} / filename;
}

ScreenshotRenderer::ScreenshotRenderer(const Size& size)
    : MandelbrotRenderer{size} {}

bool ScreenshotRenderer::IsDirty(const Complex& center, double_t zoom) const {
  return true;
}

void ScreenshotRenderer::RenderImage(const Image& image) const {
  WriteImage(image, NameScreenshot(center_, zoom_));
}

}