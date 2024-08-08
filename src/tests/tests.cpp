#include <gtest/gtest.h>
#include <filesystem>

#include "image.h"
#include "complex.h"
#include "coloring_mode.h"
#include "utils.h"
#include "mandelbrot_set.cuh"

namespace MandelbrotSet {

std::filesystem::path GetTestDataPath() {
  return std::filesystem::absolute("images");
}

TEST(MandelbrotSet, DefaultView) {
  const auto image_path = GetTestDataPath() / "m0.5_0_1.png";
  const auto expected_image = ReadImage(image_path);

  const auto center = Complex{-0.5, 0.0};
  const auto zoom = 1.;
  const auto max_iterations = 1024;
  const auto coloring_mode = ColoringMode::kBlue;
  const auto smoothing = true;

  constexpr auto kWidth = 1920;
  constexpr auto kHeight = 1080;
  auto image = Image{Size{kWidth, kHeight}};
  MandelbrotSet::Visualize(image.GetData(), image.GetWidth(), image.GetHeight(),
                           center.real, center.imag, zoom, max_iterations,
                           static_cast<int32_t>(coloring_mode), smoothing);

  EXPECT_TRUE(CompareImages(expected_image, image));
}

}  // namespace MandelbrotSet