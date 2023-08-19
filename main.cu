#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "mandelbrot_set.cuh"

using ImageType = uint8_t;
using Image = std::vector<ImageType>;

constexpr auto kEnableVSync = false;
constexpr auto kWindowWidth = 1024;
constexpr auto kWindowHeight = 768;
constexpr auto kSize = kWindowWidth * kWindowHeight;
constexpr auto kSizeInBytes = kSize * sizeof(ImageType);

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

void DrawImage(const Image& image) {
  glBegin(GL_POINTS);
  for (auto y = 0; y < kWindowHeight; ++y) {
    for (auto x = 0; x < kWindowWidth; ++x) {
      const auto pixel = image[y * kWindowWidth + x];
      glColor3ub(pixel, pixel, pixel);
      glVertex2d(x, y);
    }
  }
  glEnd();
}

void MandelbrotSet(Image& image) {
  auto device_data = static_cast<uint8_t*>(nullptr);
  cudaMalloc(&device_data, kSizeInBytes);
  cudaMemcpy(device_data, image.data(), kSizeInBytes, cudaMemcpyHostToDevice);

  constexpr auto kThreadsPerBlock = 512;
  constexpr auto kBlocksPerGrid = (kSize - 1) / kThreadsPerBlock + 1;

  constexpr static auto kDx = -0.87591;
  constexpr static auto kDy = 0.20464;
  constexpr static auto kZoom = 5.7854;

  MandelbrotSet<<<kBlocksPerGrid, kThreadsPerBlock>>>(
      device_data, kWindowWidth, kWindowHeight, kDx, kDy, kZoom
  );

  cudaMemcpy(image.data(), device_data, kSizeInBytes, cudaMemcpyDeviceToHost);
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

  constexpr auto kFPSUpdateRate = 10; // 10 times per second
  const auto current_time = glfwGetTime();
  auto fps_counter = FPSCounter{kFPSUpdateRate, current_time};

  auto image = std::vector<uint8_t>(kSize);

  while (!glfwWindowShouldClose(window)) {

    MandelbrotSet(image);
    DrawImage(image);

    glfwSwapBuffers(window);
    glfwPollEvents();

    fps_counter.Update(glfwGetTime());
    std::cout << fps_counter.GetFPS() << std::endl;
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}