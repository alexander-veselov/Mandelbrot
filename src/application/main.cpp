#include "application_glfw.h"

#include <memory>

constexpr auto kWindowWidth = 1024;
constexpr auto kWindowHeight = 768;

int main() {
  auto application = std::make_unique<MandelbrotSet::ApplicationGLFW>(kWindowWidth, kWindowHeight);
  return application->Run();
}