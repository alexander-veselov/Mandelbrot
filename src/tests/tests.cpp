#include <gtest/gtest.h>
#include <filesystem>

#include "mandelbrot/core/coloring_mode.h"
#include "mandelbrot/core/complex.h"
#include "mandelbrot/core/image.h"
#include "mandelbrot/core/palette.h"
#include "mandelbrot/core/utils.h"
#include "mandelbrot/core/cuda/mandelbrot_set.h"

namespace mandelbrot {

std::filesystem::path GetTestDataPath() {
  return std::filesystem::absolute("data/images");
}

TEST(MandelbrotSet, DefaultView) {
  const auto image_path = GetTestDataPath() / "m0.5_0_1.png";
  const auto expected_image = ReadImage(image_path);

  const auto center = Complex{-0.5, 0.0};
  const auto zoom = 1.;
  const auto max_iterations = 1024;
  const auto coloring_mode = ColoringMode::kMode1;
  const auto palette = Palette::kBluePalette;
  const auto smoothing = true;

  constexpr auto kWidth = 1920;
  constexpr auto kHeight = 1080;
  auto image = Image{Size{kWidth, kHeight}};
  cuda::Visualize(image.GetData(), image.GetWidth(), image.GetHeight(),
                  center.real, center.imag, zoom, max_iterations,
                  static_cast<int32_t>(coloring_mode),
                  static_cast<int32_t>(palette), smoothing);

  EXPECT_TRUE(expected_image == image);
}

}  // namespace mandelbrot