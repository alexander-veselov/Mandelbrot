#pragma once

#include "application.h"

#include <vector>

// Forward declarations
struct GLFWwindow;

namespace MandelbrotSet {

using Image = std::vector<uint32_t>;

constexpr auto kEnableVSync = false;
constexpr auto kFullscreen = false;
constexpr auto kWindowWidth = kFullscreen ? (16 * 80) : 1024;
constexpr auto kWindowHeight = kFullscreen ? (9 * 80) : 768;

constexpr auto kWinwowName = "Mandelbrot set";
constexpr auto kSize = kWindowWidth * kWindowHeight;
constexpr auto kSizeInBytes = kSize * sizeof(Image::value_type);

constexpr auto kColoringMode = 5;
constexpr auto kMaxIterations = 1024;
constexpr auto kFPSUpdateRate = 10;  // 10 times per second
constexpr auto kSmoothing = true;

class ApplicationGLFW : public Application {
 public:
  ApplicationGLFW();
  ~ApplicationGLFW() override;
  int Run() override;

 private:
  GLFWwindow* window_;
};

}  // namespace MandelbrotSet