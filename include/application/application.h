#pragma once

#include "complex.h"
#include "explorer.h"
#include "screenshot_renderer.h"
#include "mandelbrot_renderer.h"

#include <memory>

namespace MandelbrotSet {

// TODO: make not global
constexpr auto kFPSUpdateRate = 10;  // 10 times per second
constexpr auto kDefaultPosition = Complex{-0.5, 0.0};
constexpr auto kDefaultZoom = 1.0;

enum class MouseButton {
  kLeft,
  kRight
};

enum class MouseAction {
  kPress,
  kRelease
};

class Application {
 public:
  Application(uint32_t window_width, uint32_t window_height,
              std::unique_ptr<MandelbrotRenderer> renderer);
  virtual ~Application() = default;
  virtual bool ShouldClose() const = 0;
  virtual void SwapBuffers() = 0;
  virtual void PollEvents() = 0;
  virtual void GetCursorPosition(double_t& x_pos, double_t& y_pos) const = 0;
  virtual double_t GetTime() const = 0;

  void MouseButtonCallback(MouseButton button, MouseAction action);
  void CursorPositionCallback(double_t x_pos, double_t y_pos);
  void ScrollCallback(double_t x_offset, double_t y_offset);

  int Run();

 protected:
  Complex GetCurrentCursorComplex(double_t screen_width, double_t screen_height,
                                  const Complex& center, double_t zoom_factor);

 protected:
  uint32_t window_width_;
  uint32_t window_height_;
  Explorer explorer_;
  ScreenshotRenderer screenshot_renderer_;
  std::unique_ptr<MandelbrotRenderer> renderer_;
  MandelbrotRenderer::RenderOptions render_options_;
};

}  // namespace MandelbrotSet