#pragma once

#include "image.h"

#include <filesystem>

namespace MandelbrotSet {

void WriteImage(const Image& image, const std::filesystem::path& path);

}  // namespace MandelbrotSet