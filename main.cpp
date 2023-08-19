#include <cmath>

#include <GLFW/glfw3.h>

constexpr auto kWindowWidth = 1024;
constexpr auto kWindowHeight = 768;

void drawGradient() {
  glBegin(GL_POINTS);
  for (auto x = 0; x < kWindowWidth; ++x) {
    for (auto y = 0; y < kWindowHeight; ++y) {
      const auto r = float_t(x) / kWindowWidth;
      const auto g = float_t(y) / kWindowHeight;
      const auto b = (x * y) / (kWindowWidth * kWindowHeight);
      glColor3f(r, g, b);
      glVertex2d(x, y);
    }
  }
  glEnd();
}

int main() {

  if (glfwInit() != GLFW_TRUE) {
    glfwTerminate();
  }

  const auto window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Mandelbrot set", nullptr, nullptr);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  glViewport(0, 0, kWindowWidth, kWindowHeight);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glOrtho(0.0, kWindowWidth, 0.0, kWindowHeight, 0.0, 1.0);

  while (!glfwWindowShouldClose(window)) {
    drawGradient();
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}