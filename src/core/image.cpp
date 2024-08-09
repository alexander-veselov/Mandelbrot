#pragma once

#include "mandelbrot/core/image.h"

namespace mandelbrot {

Image::Image(const Size& size)
    : size_{size},
      data_(static_cast<size_t>(size.width * size.height), RGBA{0u}) {}

Image::Image(const RGBA* data, const Size& size) : Image{size} {
  std::memcpy(data_.data(), data, GetSizeInBytes());
}

uint32_t Image::GetWidth() const noexcept { return size_.width; }
uint32_t Image::GetHeight() const noexcept { return size_.height; }
Size Image::GetSize() const noexcept { return size_; }
const RGBA* Image::GetData() const noexcept { return data_.data(); }
RGBA* Image::GetData() noexcept { return data_.data(); }
size_t Image::GetSizeInBytes() const noexcept { return data_.size() * sizeof(Type); }

bool Image::operator==(const Image& image) const {
  if (GetSize() != image.GetSize()) {
    return false;
  }
  return std::memcmp(GetData(), image.GetData(), GetSizeInBytes()) == 0;
}

bool Image::operator!=(const Image& image) const {
  return !(*this == image);
}

}  // namespace mandelbrot