#pragma once

#include "actions.h"
#include "buttons.h"
#include "complex.h"
#include "explorer.h"
#include "mandelbrot_renderer.h"
#include "point.h"
#include "size.h"
#include "window_mode.h"

#include <memory>

namespace MandelbrotSet {

class Application {
 public:
  Application(const Size& window_size,
              std::unique_ptr<MandelbrotRenderer> renderer);
  virtual ~Application() = default;
  virtual void Close() = 0;
  virtual bool ShouldClose() const = 0;
  virtual void SwapBuffers() = 0;
  virtual void PollEvents() = 0;
  virtual Point GetCursorPosition() const = 0;
  virtual double_t GetTime() const = 0;

  void MouseButtonCallback(MouseButton button, MouseAction action);
  void CursorPositionCallback(const Point& cursor_position);
  void ScrollCallback(ScrollAction action);
  void KeyCallback(KeyButton key_button, KeyAction action);

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