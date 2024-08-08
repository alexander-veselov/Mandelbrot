#include "mandelbrot/application/application_glfw.h"
#include "mandelbrot/application/config.h"

#include <memory>

int main() {
  const auto window_size = mandelbrot::GetConfig().window_size;
  auto application = std::make_unique<mandelbrot::ApplicationGLFW>(window_size);
  return application->Run();
}