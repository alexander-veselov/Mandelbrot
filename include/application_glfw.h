#pragma once

#include "application.h"
#include "explorer.h"

#include <vector>

// Forward declarations
struct GLFWwindow;

namespace MandelbrotSet {

constexpr auto kEnableVSync = false;
constexpr auto kFullscreen = false;
constexpr auto kWindowWidth = kFullscreen ? (16 * 80) : 1024;
constexpr auto kWindowHeight = kFullscreen ? (9 * 80) : 768;

constexpr auto kWinwowName = "Mandelbrot set";

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
  void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
  void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
  void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

 private:
  GLFWwindow* window_;
  Explorer explorer_;
};

}  // namespace MandelbrotSet