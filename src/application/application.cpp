#include "application.h"
#include "screenshot_renderer.h"
#include "fps_counter.h"
#include "logger.h"

namespace MandelbrotSet {

static Complex ScreenToComplex(const Point& cursor_position,
                               const Size& screen_size,
                               const Complex& center,
                               double_t zoom_factor) {
  // Mandelbrot set parameters
  constexpr static auto kMandelbrotSetWidth = 3.;   // [-2, 1]
  constexpr static auto kMandelbrotSetHeight = 2.;  // [-1, 1]

  const auto scale = 1. / std::min(screen_size.width / kMandelbrotSetWidth,
                                   screen_size.height / kMandelbrotSetHeight);

  const auto real = (cursor_position.x - screen_size.width / 2.) * scale;
  const auto imag = (cursor_position.y - screen_size.height / 2.) * scale;

  return {center.real + real / zoom_factor, center.imag - imag / zoom_factor};
}

Complex Application::GetCurrentCursorComplex(const Size& screen_size,
                                             const Complex& center,
                                             double_t zoom_factor) {
  const auto& cursor_position = GetCursorPosition();
  return ScreenToComplex(cursor_position, screen_size, center, zoom_factor);
}

void Application::MouseButtonCallback(MouseButton button, MouseAction action) {
  if (button == MouseButton::kLeft) {
    const auto mouse_position = GetCurrentCursorComplex(
        window_size_, explorer_.GetCenterPosition(), explorer_.GetZoom());
    if (action == MouseAction::kPress) {
      explorer_.MouseClickedEvent(mouse_position);
    } else if (action == MouseAction::kRelease) {
      explorer_.MouseReleasedEvent(mouse_position);
    }
  } else if (button == MouseButton::kRight) {
    // TODO: create keyboard click event
    screenshot_renderer_->Render(explorer_.GetDisplayPosition(),
                                 explorer_.GetZoom(), render_options_);
  }
}

void Application::CursorPositionCallback(const Point& cursor_position) {
  const auto mouse_position =
      ScreenToComplex(cursor_position, window_size_,
                      explorer_.GetCenterPosition(), explorer_.GetZoom());
  explorer_.MouseMovedEvent(mouse_position);
}

void Application::ScrollCallback(double_t x_offset, double_t y_offset) {
  const auto mouse_position = GetCurrentCursorComplex(
      window_size_, explorer_.GetCenterPosition(), explorer_.GetZoom());
  if (y_offset > 0.) {
    explorer_.MouseScrollEvent(mouse_position, Explorer::ScrollEvent::kScrollUp);
  } else if (y_offset < 0.) {
    explorer_.MouseScrollEvent(mouse_position, Explorer::ScrollEvent::kScrollDown);
  }
}

static void LogInformation(const Logger& logger, const Complex& position,
                           double_t zoom, double_t fps) {
  logger.ResetCursor();
  logger << logger.SetPrecision(15) << "Center: " << position.real
         << logger.ShowSign(true) << position.imag << logger.ShowSign(false)
         << "i" << logger.NewLine() << "Zoom: " << zoom << logger.NewLine()
         << "FPS: " << static_cast<int32_t>(fps) << logger.NewLine();
}

Application::Application(const Size& window_size,
                         std::unique_ptr<MandelbrotRenderer> renderer)
    : window_size_{window_size},
      explorer_{kDefaultPosition, kDefaultZoom},
      screenshot_renderer_{std::make_unique<ScreenshotRenderer>()},
      renderer_{std::move(renderer)},
      render_options_{} {}

int Application::Run() {
  const auto& logger = Logger::Instance();
  auto fps_counter = FPSCounter{kFPSUpdateRate, GetTime()};

  while (!ShouldClose()) {
    const auto position = explorer_.GetDisplayPosition();
    const auto zoom = explorer_.GetZoom();

    renderer_->Render(position, zoom, render_options_);

    SwapBuffers();
    PollEvents();

    fps_counter.Update(GetTime());

    LogInformation(logger, position, zoom, fps_counter.GetFPS());
  }

  return 0;
}

}  // namespace MandelbrotSet