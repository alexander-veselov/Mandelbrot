#include "mandelbrot/core/utils.h"

#include <lodepng.h>

namespace mandelbrot {

Image FlipHorizontally(const Image& image) {
  auto flipped = image;
  auto flipped_data = flipped.GetData();
  const auto width = flipped.GetWidth();
  const auto height = flipped.GetHeight();
  for (auto row = 0; row < height / 2; ++row) {
    auto row_data1 = flipped_data + row * width;
    auto row_data2 = flipped_data + (height - 1 - row) * width;
    std::swap_ranges(row_data1, row_data1 + width, row_data2);
  }
  return flipped;
}

Image ReadImage(const std::filesystem::path& path) {
  auto data = std::vector<uint8_t>{};
  auto width = uint32_t{};
  auto height = uint32_t{};
  const auto error = lodepng::decode(data, width, height, path.string());

  if (error != 0) {
    throw std::runtime_error{lodepng_error_text(error)};
  }

  const auto image_data = reinterpret_cast<const RGBA*>(data.data());
  const auto image = Image{image_data, Size{width, height}};
  return FlipHorizontally(image);
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

  const auto flipped = FlipHorizontally(image);
  const auto error = lodepng::encode(
      path.string(), reinterpret_cast<const uint8_t*>(flipped.GetData()),
      flipped.GetWidth(), flipped.GetHeight());

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

std::optional<std::filesystem::path> FindDataFile(const std::string& filename) {
  const auto directory_candidate1 = std::filesystem::current_path();
  const auto directory_candidate2 = std::filesystem::current_path() / "data";
  const auto candidate1 = directory_candidate1 / filename;
  if (std::filesystem::exists(candidate1)) {
    return candidate1;
  }
  const auto candidate2 = directory_candidate2 / filename;
  if (std::filesystem::exists(candidate2)) {
    return candidate2;
  }
  return std::nullopt;
}

}  // namespace mandelbrot