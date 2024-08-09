#pragma once

#include "mandelbrot/core/image.h"

#include <filesystem>
#include <optional>

namespace mandelbrot {

Image FlipHorizontally(const Image& image);
Image ReadImage(const std::filesystem::path& path);
void WriteImage(const Image& image, const std::filesystem::path& path);
std::optional<std::filesystem::path> FindDataFile(const std::string& filename);

}  // namespace mandelbrot