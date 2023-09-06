#pragma once

#include "application.h"
#include "explorer.h"
#include "mandelbrot_renderer_glfw.h"

#include <vector>

// Forward declarations
struct GLFWwindow;

namespace MandelbrotSet {

constexpr auto kEnableVSync = false;
constexpr auto kFullscreen = false;

constexpr auto kWinwowName = "Mandelbrot set";

constexpr auto kFPSUpdateRate = 10;  // 10 times per second

constexpr auto kDefaultPosition = Complex{-0.5, 0.0};
constexpr auto kDefaultZoom = 1.0;


class ApplicationGLFW : public Application {
 public:
  ApplicationGLFW(uint32_t window_width, uint32_t window_height);
  ~ApplicationGLFW() override;
  int Run() override;

 private:
  void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
  void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
  void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

 private:
  uint32_t window_width_;
  uint32_t window_height_;
  GLFWwindow* window_;
  Explorer explorer_;
  MandelbrotRendererGLFW renderer_;
};

}  // namespace MandelbrotSet