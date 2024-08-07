#pragma once

#include "image.h"

#include <filesystem>

namespace MandelbrotSet {

Image ReadImage(const std::filesystem::path& path);
void WriteImage(const Image& image, const std::filesystem::path& path);

}  // namespace MandelbrotSet