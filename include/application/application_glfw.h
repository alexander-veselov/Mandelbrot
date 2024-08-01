#pragma once

#include "application.h"

// Forward declarations
struct GLFWwindow;

namespace MandelbrotSet {

// TODO: make not global
constexpr auto kEnableVSync = false;
constexpr auto kFullscreen = false; // TODO: fix fullscreen mode, add borderless fullscreen option
constexpr auto kWinwowName = "Mandelbrot set";

class ApplicationGLFW : public Application {
 public:
  ApplicationGLFW(uint32_t window_width, uint32_t window_height);
  ~ApplicationGLFW() override;
  bool ShouldClose() const override;
  void SwapBuffers() override;
  void PollEvents() override;
  void GetCursorPosition(double_t& x_pos, double_t& y_pos) const override;
  double_t GetTime() const override;

 private:
  GLFWwindow* window_;
};

}  // namespace MandelbrotSet