#pragma once

#include <vector>

namespace MandelbrotSet {

using RGBA = uint32_t;

class Image {
 public:
  using ImageBuffer = std::vector<RGBA>;
  using Type = ImageBuffer::value_type;

  Image(uint32_t width, uint32_t height);
  uint32_t GetWidth() const noexcept;
  uint32_t GetHeight() const noexcept;
  const RGBA* GetData() const noexcept;
  RGBA* GetData() noexcept;

 private:
  uint32_t width_;
  uint32_t height_;
  ImageBuffer data_;
};

}  // namespace MandelbrotSet