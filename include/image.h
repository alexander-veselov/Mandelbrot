#pragma once

#include <vector>

namespace MandelbrotSet {

using Pixel = uint32_t;
using ImageBuffer = std::vector<Pixel>;

class Image {
 public:
  using Type = ImageBuffer::value_type;

  Image(uint32_t width, uint32_t height);
  uint32_t GetWidth() const noexcept;
  uint32_t GetHeight() const noexcept;
  const Pixel* GetData() const noexcept;
  Pixel* GetData() noexcept;

 private:
  uint32_t width_;
  uint32_t height_;
  ImageBuffer data_;
};

}  // namespace MandelbrotSet