#include "application_glfw.h"
#include "mandelbrot_renderer_glfw.h"

#include <GLFW/glfw3.h>
#include <stdexcept>

namespace MandelbrotSet {

static MouseButton ConvertMouseButton(int32_t button) {
  switch (button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      return MouseButton::kLeft;
    case GLFW_MOUSE_BUTTON_RIGHT:
      return MouseButton::kRight;
    default:
      throw std::runtime_error{"Unexpected button code"};
  }
}

static MouseAction ConvertMouseAction(int32_t action) {
  switch (action) {
    case GLFW_PRESS:
      return MouseAction::kPress;
    case GLFW_RELEASE:
      return MouseAction::kRelease;
    default:
      throw std::runtime_error{"Unexpected action code"};
  }
}

ApplicationGLFW::ApplicationGLFW(uint32_t window_width, uint32_t window_height)
    : Application{window_width, window_height,
                  std::make_unique<MandelbrotRendererGLFW>(window_width,
                                                           window_height)} {

  if (glfwInit() != GLFW_TRUE) {
    glfwTerminate();
  }

  auto monitor = static_cast<GLFWmonitor*>(nullptr);
  if (kFullscreen) {
    monitor = glfwGetPrimaryMonitor();
  }

  window_ = glfwCreateWindow(window_width_, window_height_, kWinwowName,
                             monitor, nullptr);

  glfwSetWindowUserPointer(window_, this);
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(kEnableVSync);

  glViewport(0, 0, window_width_, window_height_);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glOrtho(0.0, window_width_, 0.0, window_height_, 0.0, 1.0);

  auto MouseButtonCallback = [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
    const auto app_button = ConvertMouseButton(button);
    const auto app_action = ConvertMouseAction(action);
    const auto application = static_cast<Application*>(glfwGetWindowUserPointer(window));
    application->MouseButtonCallback(app_button, app_action);
  };

  auto CursorPosCallback = [](GLFWwindow* window, double_t x_pos, double_t y_pos) {
    const auto application = static_cast<Application*>(glfwGetWindowUserPointer(window));
    application->CursorPositionCallback(x_pos, y_pos);
  };

  auto ScrollCallback = [](GLFWwindow* window, double_t x_offset, double_t y_offset) {
    const auto application = static_cast<Application*>(glfwGetWindowUserPointer(window));
    application->ScrollCallback(x_offset, y_offset);
  };

  glfwSetMouseButtonCallback(window_, MouseButtonCallback);
  glfwSetCursorPosCallback(window_, CursorPosCallback);
  glfwSetScrollCallback(window_, ScrollCallback);
}

ApplicationGLFW::~ApplicationGLFW() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

bool ApplicationGLFW::ShouldClose() const {
  return glfwWindowShouldClose(window_);
}

void ApplicationGLFW::SwapBuffers() {
  glfwSwapBuffers(window_);
}

void ApplicationGLFW::PollEvents() {
  glfwPollEvents();
}

void ApplicationGLFW::GetCursorPosition(double_t& x_pos,
                                        double_t& y_pos) const {
  glfwGetCursorPos(window_, &x_pos, &y_pos);
}

double_t ApplicationGLFW::GetTime() const {
  return glfwGetTime();
}

}  // namespace MandelbrotSet