#include "application_glfw.h"

#include <GLFW/glfw3.h>

#include <iostream>  // TODO: make logger

#include "explorer.h"
#include "fps_counter.h"
#include "image.h"
#include "mandelbrot_set.cuh"

namespace MandelbrotSet {

void DrawImage(const Image& image) {
  static_assert(std::is_same_v<Image::Type, uint32_t>,
                "Image::Type must be uint32_t");
  glDrawPixels(image.GetWidth(), image.GetHeight(), GL_RGBA, GL_UNSIGNED_BYTE,
               image.GetData());
}

Complex ScreenToComplex(double_t screen_x, double_t screen_y,
                        double_t screen_width, double_t screen_height,
                        const Complex& center, double_t zoom_factor) {
  // Mandelbrot set parameters
  constexpr static auto kMandelbrotSetWidth = 3.;   // [-2, 1]
  constexpr static auto kMandelbrotSetHeight = 2.;  // [-1, 1]

  const auto scale = 1. / std::min(screen_width / kMandelbrotSetWidth,
                                   screen_height / kMandelbrotSetHeight);

  const auto real = (screen_x - screen_width / 2.) * scale;
  const auto imag = (screen_y - screen_height / 2.) * scale;

  return {center.real + real / zoom_factor, center.imag + imag / zoom_factor};
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
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    const auto mouse_position = GetCurrentCursorComplex(
        window, kWindowWidth, kWindowHeight, explorer_.GetCenterPosition(),
        explorer_.GetZoom());

    explorer_.MouseClickedEvent(mouse_position);
  }

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
    const auto mouse_position = GetCurrentCursorComplex(
        window, kWindowWidth, kWindowHeight, explorer_.GetCenterPosition(),
        explorer_.GetZoom());

    explorer_.MouseReleasedEvent(mouse_position);
  }
}

void ApplicationGLFW::CursorPosCallback(GLFWwindow* window, double xpos,
                                        double ypos) {
  const auto mouse_position =
      ScreenToComplex(xpos, ypos, kWindowWidth, kWindowHeight,
                      explorer_.GetCenterPosition(), explorer_.GetZoom());

  explorer_.MouseMovedEvent(mouse_position);
}

void ApplicationGLFW::ScrollCallback(GLFWwindow* window, double xoffset,
                                     double yoffset) {
  if (yoffset > 0.) {
    explorer_.MouseScrollEvent(Explorer::ScrollEvent::kScrollUp);
  }
  if (yoffset < 0.) {
    explorer_.MouseScrollEvent(Explorer::ScrollEvent::kScrollDown);
  }
}

ApplicationGLFW::ApplicationGLFW() : explorer_{{-0.5, 0.0}, 1.0} {
  if (glfwInit() != GLFW_TRUE) {
    glfwTerminate();
  }

  auto monitor = static_cast<GLFWmonitor*>(nullptr);
  if (kFullscreen) {
    monitor = glfwGetPrimaryMonitor();
  }

  window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, kWinwowName, monitor,
                             nullptr);

  glfwSetWindowUserPointer(window_, this);
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(kEnableVSync);

  glViewport(0, 0, kWindowWidth, kWindowHeight);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glOrtho(0.0, kWindowWidth, 0.0, kWindowHeight, 0.0, 1.0);

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
  const auto current_time = glfwGetTime();
  auto fps_counter = FPSCounter{kFPSUpdateRate, current_time};

  auto image = Image(kWindowWidth, kWindowHeight);

  auto current_center = Complex{};
  auto current_zoom = double_t{};

  while (!glfwWindowShouldClose(window_)) {
    const auto center = explorer_.GetDisplayPosition();
    const auto zoom = explorer_.GetZoom();

    const auto dirty = current_center.real != center.real ||
                       current_center.imag != center.imag ||
                       current_zoom != zoom;

    if (dirty) {
      MandelbrotSet::Visualize(image.GetData(), kWindowWidth, kWindowHeight,
                               center.real, center.imag, zoom, kMaxIterations,
                               kColoringMode, kSmoothing);
      current_center = center;
      current_zoom = zoom;
    }

    DrawImage(image);

    glfwSwapBuffers(window_);
    glfwPollEvents();

    fps_counter.Update(glfwGetTime());
    std::cout << "FPS: " << fps_counter.GetFPS() << std::endl;
  }

  return 0;
}

}  // namespace MandelbrotSet