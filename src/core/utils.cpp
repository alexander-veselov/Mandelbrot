#include "utils.h"

#include <lodepng.h>

namespace MandelbrotSet {

Image ReadImage(const std::filesystem::path& path) {
  auto data = std::vector<uint8_t>{};
  auto width = uint32_t{};
  auto height = uint32_t{};
  const auto error = lodepng::decode(data, width, height, path.string());

  if (error != 0) {
    throw std::runtime_error{lodepng_error_text(error)};
  }

  return Image{reinterpret_cast<const RGBA*>(data.data()), Size{width, height}};
}

void WriteImage(const Image& image, const std::filesystem::path& path) {

  const auto width = image.GetWidth();
  const auto height = image.GetHeight();

  if (width == 0 || height == 0) {
    throw std::runtime_error{"Wrong image size"};
  }

  constexpr auto kPNGExtension = ".png";
  if (path.filename().extension() != kPNGExtension) {
    throw std::invalid_argument("Unsupported image format");
  }

  if (!std::filesystem::exists(path.parent_path())) {
    std::filesystem::create_directory(path.parent_path());
  }

  const auto error = lodepng::encode(
      path.string(), reinterpret_cast<const uint8_t*>(image.GetData()),
      image.GetWidth(), image.GetHeight());

  if (error != 0) {
    throw std::runtime_error{lodepng_error_text(error)};
  }
}

bool CompareImages(const Image& image1, const Image& image2) {
  if (image1.GetWidth() != image2.GetWidth() ||
      image1.GetHeight() != image2.GetHeight()) {
    return false;
  }
  return std::memcmp(image1.GetData(), image2.GetData(),
                     image1.GetSizeInBytes()) == 0;
}

}  // namespace MandelbrotSet