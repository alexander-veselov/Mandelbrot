#include "application.h"
#include "application_glfw.h"

#include <memory>

int main() {
  auto application = std::shared_ptr<MandelbrotSet::Application>(
      new MandelbrotSet::ApplicationGLFW());
  return application->Run();
}