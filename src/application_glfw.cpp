#include "application_glfw.h"
#include "mandelbrot_set.cuh"
#include "fps_counter.h"
#include "explorer.h"

#include <GLFW/glfw3.h>
#include <iostream> // TODO: make logger

namespace MandelbrotSet {


void ToRGB(Image::value_type pixel, uint8_t& r, uint8_t& g, uint8_t& b) {
  r = (pixel >> 16) & 0xFF;
  g = (pixel >> 8) & 0xFF;
  b = (pixel >> 0) & 0xFF;
}

void DrawImage(const Image& image) {
  glDrawPixels(kWindowWidth, kWindowHeight, GL_RGBA, GL_UNSIGNED_BYTE,
               image.data());
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

// TODO: make not global
auto explorer = Explorer{{-0.5, 0.0}, 1.0};

// TODO: Figure out how to pass non-global objects to callbacks
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    const auto mouse_position = GetCurrentCursorComplex(
        window, kWindowWidth, kWindowHeight, explorer.GetCenterPosition(),
        explorer.GetZoom());

    explorer.MouseClickedEvent(mouse_position);
  }

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
    const auto mouse_position = GetCurrentCursorComplex(
        window, kWindowWidth, kWindowHeight, explorer.GetCenterPosition(),
        explorer.GetZoom());

    explorer.MouseReleasedEvent(mouse_position);
  }
}

void CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
  const auto mouse_position =
      ScreenToComplex(xpos, ypos, kWindowWidth, kWindowHeight,
                      explorer.GetCenterPosition(), explorer.GetZoom());

  explorer.MouseMovedEvent(mouse_position);
}

void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  if (yoffset > 0.) {
    explorer.MouseScrollEvent(Explorer::ScrollEvent::kScrollUp);
  }
  if (yoffset < 0.) {
    explorer.MouseScrollEvent(Explorer::ScrollEvent::kScrollDown);
  }
}

ApplicationGLFW::ApplicationGLFW() {
  if (glfwInit() != GLFW_TRUE) {
    glfwTerminate();
  }

  auto monitor = static_cast<GLFWmonitor*>(nullptr);
  if (kFullscreen) {
    monitor = glfwGetPrimaryMonitor();
  }

  window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, kWinwowName, monitor,
                             nullptr);

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(kEnableVSync);

  glViewport(0, 0, kWindowWidth, kWindowHeight);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glOrtho(0.0, kWindowWidth, 0.0, kWindowHeight, 0.0, 1.0);

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

  auto image = std::vector<Image::value_type>(kSize);

  auto current_center = Complex{};
  auto current_zoom = double_t{};

  while (!glfwWindowShouldClose(window_)) {
    const auto center = explorer.GetDisplayPosition();
    const auto zoom = explorer.GetZoom();

    const auto dirty = current_center.real != center.real ||
                       current_center.imag != center.imag ||
                       current_zoom != zoom;

    if (dirty) {
      MandelbrotSet::Visualize(image.data(), kWindowWidth, kWindowHeight,
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