#pragma once

#include "image.h"

namespace MandelbrotSet {

Image::Image(const Size& size)
    : size_{size},
      data_(static_cast<size_t>(size.width * size.height), RGBA{0u}) {}

uint32_t Image::GetWidth() const noexcept { return size_.width; }
uint32_t Image::GetHeight() const noexcept { return size_.height; }
const RGBA* Image::GetData() const noexcept { return data_.data(); }
RGBA* Image::GetData() noexcept { return data_.data(); }

}  // namespace MandelbrotSet