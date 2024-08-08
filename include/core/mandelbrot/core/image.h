#pragma once

#include "mandelbrot/core/size.h"

#include <filesystem>
#include <vector>

namespace MandelbrotSet {

using RGBA = uint32_t;

class Image {
 public:
  using ImageBuffer = std::vector<RGBA>;
  using Type = ImageBuffer::value_type;

  Image(const Size& size);
  Image(const RGBA* data, const Size& size);
  uint32_t GetWidth() const noexcept;
  uint32_t GetHeight() const noexcept;
  const RGBA* GetData() const noexcept;
  size_t GetSizeInBytes() const noexcept;
  RGBA* GetData() noexcept;

 private:
  ImageBuffer data_;
  Size size_;
};

}  // namespace MandelbrotSet