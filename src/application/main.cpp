#include "application_glfw.h"

#include <memory>

constexpr auto kWindowWidth = 1024;
constexpr auto kWindowHeight = 768;

int main() {
  constexpr auto kWindowSize = MandelbrotSet::Size{kWindowWidth, kWindowHeight};
  auto application = std::make_unique<MandelbrotSet::ApplicationGLFW>(kWindowSize);
  return application->Run();
}