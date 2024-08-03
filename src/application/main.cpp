#include "config.h"
#include "application_glfw.h"

#include <memory>

int main() {
  const auto window_size = MandelbrotSet::GetConfig().window_size;
  auto application = std::make_unique<MandelbrotSet::ApplicationGLFW>(window_size);
  return application->Run();
}