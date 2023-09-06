#include "application.h"
#include "application_glfw.h"

#include <memory>

constexpr auto kWindowWidth = 1024;
constexpr auto kWindowHeight = 768;

int main() {
  auto application = std::shared_ptr<MandelbrotSet::Application>(
      new MandelbrotSet::ApplicationGLFW(kWindowWidth, kWindowHeight));
  return application->Run();
}