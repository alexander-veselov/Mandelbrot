#pragma once

#include "image.h"

namespace MandelbrotSet {

Image::Image(uint32_t width, uint32_t height)
    : width_{width},
      height_{height},
      data_(static_cast<size_t>(width * height), RGBA{0}) {}

uint32_t Image::GetWidth() const noexcept { return width_; }
uint32_t Image::GetHeight() const noexcept { return height_; }
const RGBA* Image::GetData() const noexcept { return data_.data(); }
RGBA* Image::GetData() noexcept { return data_.data(); }

}  // namespace MandelbrotSet