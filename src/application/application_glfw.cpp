#include "application_glfw.h"

#include <GLFW/glfw3.h>

#include <iostream> // TODO: make logger

#include "explorer.h"
#include "fps_counter.h"

namespace MandelbrotSet {

Complex ScreenToComplex(double_t screen_x, double_t screen_y,
                        double_t screen_width, double_t screen_height,
                        const Complex& center, double_t zoom_factor) {
  // Mandelbrot set parameters
  constexpr static auto kMandelbrotSetWidth  = 3.;   // [-2, 1]
  constexpr static auto kMandelbrotSetHeight = 2.;  // [-1, 1]

  const auto scale = 1. / std::min(screen_width / kMandelbrotSetWidth,
                                   screen_height / kMandelbrotSetHeight);

  const auto real = (screen_x - screen_width / 2.) * scale;
  const auto imag = (screen_y - screen_height / 2.) * scale;

  return {center.real + real / zoom_factor, center.imag - imag / zoom_factor};
}

Complex GetCurrentCursorComplex(GLFWwindow* window, double_t screen_width,
                                double_t screen_height, const Complex& center,
                                double_t zoom_factor) {
  auto xpos = double_t{};
  auto ypos = double_t{};
  glfwGetCursorPos(window, &xpos, &ypos);
  return ScreenToComplex(xpos, ypos, screen_width, screen_height, center,
                         zoom_factor);
}

void ApplicationGLFW::MouseButtonCallback(GLFWwindow* window, int button,
  int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    const auto mouse_position = GetCurrentCursorComplex(
      window, window_width_, window_height_, explorer_.GetCenterPosition(),
      explorer_.GetZoom());
    if (action == GLFW_PRESS) {
      explorer_.MouseClickedEvent(mouse_position);
    }
    else if (action == GLFW_RELEASE) {
      explorer_.MouseReleasedEvent(mouse_position);
    }
  }
}

void ApplicationGLFW::CursorPosCallback(GLFWwindow* window, double xpos,
                                        double ypos) {
  const auto mouse_position =
      ScreenToComplex(xpos, ypos, window_width_, window_height_,
                      explorer_.GetCenterPosition(), explorer_.GetZoom());

  explorer_.MouseMovedEvent(mouse_position);
}

void ApplicationGLFW::ScrollCallback(GLFWwindow* window, double xoffset,
  double yoffset) {
  const auto mouse_position = GetCurrentCursorComplex(
    window, window_width_, window_height_, explorer_.GetCenterPosition(),
    explorer_.GetZoom());
  if (yoffset > 0.) {
    explorer_.MouseScrollEvent(mouse_position, Explorer::ScrollEvent::kScrollUp);
  }
  if (yoffset < 0.) {
    explorer_.MouseScrollEvent(mouse_position, Explorer::ScrollEvent::kScrollDown);
  }
}

ApplicationGLFW::ApplicationGLFW(uint32_t window_width, uint32_t window_height)
    : window_width_{window_width},
      window_height_{window_height},
      explorer_{kDefaultPosition, kDefaultZoom},
      renderer_{window_width, window_height} {

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

  auto MouseButtonCallback = [](GLFWwindow* window, int button, int action,
                                int mods) {
    static_cast<ApplicationGLFW*>(glfwGetWindowUserPointer(window))
        ->MouseButtonCallback(window, button, action, mods);
  };

  auto CursorPosCallback = [](GLFWwindow* window, double xpos, double ypos) {
    static_cast<ApplicationGLFW*>(glfwGetWindowUserPointer(window))
        ->CursorPosCallback(window, xpos, ypos);
  };

  auto ScrollCallback = [](GLFWwindow* window, double xoffset, double yoffset) {
    static_cast<ApplicationGLFW*>(glfwGetWindowUserPointer(window))
        ->ScrollCallback(window, xoffset, yoffset);
  };

  glfwSetMouseButtonCallback(window_, MouseButtonCallback);
  glfwSetCursorPosCallback(window_, CursorPosCallback);
  glfwSetScrollCallback(window_, ScrollCallback);
}

ApplicationGLFW::~ApplicationGLFW() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

int ApplicationGLFW::Run() {
  auto fps_counter = FPSCounter{kFPSUpdateRate, glfwGetTime()};

  while (!glfwWindowShouldClose(window_)) {

    // TODO: pass RenderOptions explicitly
    renderer_.Render(explorer_.GetDisplayPosition(), explorer_.GetZoom());

    glfwSwapBuffers(window_);
    glfwPollEvents();

    fps_counter.Update(glfwGetTime());
    std::cout << "FPS: " << fps_counter.GetFPS() << std::endl; // TODO: make logger
  }

  return 0;
}

}  // namespace MandelbrotSet