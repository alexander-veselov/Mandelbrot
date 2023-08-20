#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <type_traits>

#include "mandelbrot_set.cuh"


// TODO: remove uint8_t support? Measure performance
using Image = std::vector<uint32_t>;

constexpr auto kEnableVSync = false;

// TODO: make not global
constexpr auto kWindowWidth = 1024;
constexpr auto kWindowHeight = 768;
constexpr auto kSize = kWindowWidth * kWindowHeight;
constexpr auto kSizeInBytes = kSize * sizeof(Image::value_type);

struct Complex {
  double_t real;
  double_t imag;
};

class FPSCounter {
 public:
  explicit FPSCounter(int32_t update_rate, double_t current_time)
      : update_rate_{update_rate},
        frame_count_{0},
        last_capture_time_{current_time},
        fps_{0.} {}

  void Update(double current_time) {
    const auto delta_time = current_time - last_capture_time_;
    fps_ = static_cast<double_t>(++frame_count_) / delta_time;
    if (delta_time * update_rate_ >= 1.) {
      last_capture_time_ = current_time;
      frame_count_ = 0;
    }
  }

  double_t GetFPS() const noexcept { return fps_; }

 private:
  const int32_t update_rate_;
  int32_t frame_count_;
  double_t last_capture_time_;
  double_t fps_;
};

void ToRGB(Image::value_type pixel, uint8_t& r, uint8_t& g, uint8_t& b) {
  if constexpr (std::is_same_v<Image::value_type, uint8_t>) {
    r = pixel;
    g = pixel;
    b = pixel;
  }

  if constexpr (std::is_same_v<Image::value_type, uint32_t>) {
    r = (pixel >> 16) & 0xFF;
    g = (pixel >>  8) & 0xFF;
    b = (pixel >>  0) & 0xFF;
  }
}

void DrawImage(const Image& image) {
  glBegin(GL_POINTS);
  auto r = uint8_t{};
  auto g = uint8_t{};
  auto b = uint8_t{};
  for (auto y = 0; y < kWindowHeight; ++y) {
    for (auto x = 0; x < kWindowWidth; ++x) {
      const auto pixel = image[y * kWindowWidth + x];
      ToRGB(pixel, r, g, b);
      glColor3ub(r, g, b);
      glVertex2d(x, y);
    }
  }
  glEnd();
}

void MandelbrotSet(Image& image, const Complex& center, double_t zoom_factor) {
  auto device_data = static_cast<Image::value_type*>(nullptr);
  cudaMalloc(&device_data, kSizeInBytes);
  cudaMemcpy(device_data, image.data(), kSizeInBytes, cudaMemcpyHostToDevice);

  constexpr auto kThreadsPerBlock = 512;
  constexpr auto kBlocksPerGrid = (kSize - 1) / kThreadsPerBlock + 1;

  MandelbrotSet<<<kBlocksPerGrid, kThreadsPerBlock>>>(
    device_data, kWindowWidth, kWindowHeight,
    center.real,center.imag, zoom_factor
  );

  cudaMemcpy(image.data(), device_data, kSizeInBytes, cudaMemcpyDeviceToHost);
}

class Explorer
{
 public:
  Explorer(const Complex& position, double_t zoom)
      : center_position_{position},
        display_position_{position},
        click_position_{},
        zoom_{zoom},
        moving_{false} {}

  enum class ZoomEvent {
     kZoomIn,
     kZoomOut
  };
  
  void MouseClickedEvent(const Complex& position) {
    moving_ = true;
    click_position_ = position;
  }

  void MouseReleasedEvent(const Complex& position) {
    moving_ = false;
    center_position_.real += click_position_.real - position.real;
    center_position_.imag -= click_position_.imag - position.imag;
  }

  void MouseMovedEvent(const Complex& position) {
    if (moving_) {
      display_position_.real =
          center_position_.real + (click_position_.real - position.real);
      display_position_.imag =
          center_position_.imag - (click_position_.imag - position.imag);
    }
  }

  void ZoomEvent(ZoomEvent event) {
    constexpr static auto kZoomFactor = 1.2;
    if (event == ZoomEvent::kZoomIn) {
      zoom_ *= kZoomFactor;
    }

    if (event == ZoomEvent::kZoomOut) {
      zoom_ /= kZoomFactor;
    }
  }

  Complex GetCenterPosition() const noexcept { return center_position_; }
  Complex GetDisplayPosition() const noexcept { return display_position_; }
  double_t GetZoom() const noexcept { return zoom_; }

 private:
  Complex center_position_;
  Complex display_position_;
  Complex click_position_;
  double_t zoom_;
  bool moving_;
};

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
    explorer.ZoomEvent(Explorer::ZoomEvent::kZoomIn);
  }
  if (yoffset < 0.) {
    explorer.ZoomEvent(Explorer::ZoomEvent::kZoomOut);
  }
}


int main() {

  if (glfwInit() != GLFW_TRUE) {
    glfwTerminate();
  }

  const auto window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Mandelbrot set", nullptr, nullptr);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(kEnableVSync);

  glViewport(0, 0, kWindowWidth, kWindowHeight);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glOrtho(0.0, kWindowWidth, 0.0, kWindowHeight, 0.0, 1.0);

  glfwSetMouseButtonCallback(window, MouseButtonCallback);
  glfwSetCursorPosCallback(window, CursorPosCallback);
  glfwSetScrollCallback(window, ScrollCallback);

  constexpr auto kFPSUpdateRate = 10; // 10 times per second
  const auto current_time = glfwGetTime();
  auto fps_counter = FPSCounter{kFPSUpdateRate, current_time};

  auto image = std::vector<Image::value_type>(kSize);

  while (!glfwWindowShouldClose(window)) {

    MandelbrotSet(image, explorer.GetDisplayPosition(), explorer.GetZoom());
    DrawImage(image);

    glfwSwapBuffers(window);
    glfwPollEvents();

    fps_counter.Update(glfwGetTime());
    std::cout << "FPS: " << fps_counter.GetFPS()<< std::endl;
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}