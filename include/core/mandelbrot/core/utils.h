#pragma once

#include "mandelbrot/core/image.h"

#include <filesystem>

namespace MandelbrotSet {

Image ReadImage(const std::filesystem::path& path);
void WriteImage(const Image& image, const std::filesystem::path& path);
bool CompareImages(const Image& image1, const Image& image2);

}  // namespace MandelbrotSet