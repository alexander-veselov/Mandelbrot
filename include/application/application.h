#pragma once

#include "complex.h"
#include "explorer.h"
#include "size.h"
#include "point.h"
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
  Application(const Size& window_size,
              std::unique_ptr<MandelbrotRenderer> renderer);
  virtual ~Application() = default;
  virtual bool ShouldClose() const = 0;
  virtual void SwapBuffers() = 0;
  virtual void PollEvents() = 0;
  virtual Point GetCursorPosition() const = 0;
  virtual double_t GetTime() const = 0;

  void MouseButtonCallback(MouseButton button, MouseAction action);
  void CursorPositionCallback(const Point& cursor_position);
  void ScrollCallback(double_t x_offset, double_t y_offset);

  int Run();

 protected:
  Complex GetCurrentCursorComplex(const Size& screen_size,
                                  const Complex& center, double_t zoom_factor);

 protected:
  Size window_size_;
  Explorer explorer_;
  std::unique_ptr<MandelbrotRenderer> renderer_;
  std::unique_ptr<MandelbrotRenderer> screenshot_renderer_;
  MandelbrotRenderer::RenderOptions render_options_;
};

}  // namespace MandelbrotSet