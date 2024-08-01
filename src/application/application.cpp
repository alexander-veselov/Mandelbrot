#include "application.h"
#include "fps_counter.h"
#include "logger.h"

namespace MandelbrotSet {

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

  return {center.real + real / zoom_factor, center.imag - imag / zoom_factor};
}

Complex Application::GetCurrentCursorComplex(double_t screen_width,
                                             double_t screen_height,
                                             const Complex& center,
                                             double_t zoom_factor) {
  auto xpos = double_t{};
  auto ypos = double_t{};
  GetCursorPosition(xpos, ypos);
  return ScreenToComplex(xpos, ypos, screen_width, screen_height, center,
                         zoom_factor);
}

void Application::MouseButtonCallback(MouseButton button, MouseAction action) {
  if (button == MouseButton::kLeft) {
    const auto mouse_position = GetCurrentCursorComplex(
        window_width_, window_height_, explorer_.GetCenterPosition(),
        explorer_.GetZoom());
    if (action == MouseAction::kPress) {
      explorer_.MouseClickedEvent(mouse_position);
    } else if (action == MouseAction::kRelease) {
      explorer_.MouseReleasedEvent(mouse_position);
    }
  }
}

void Application::CursorPositionCallback(double_t x_pos, double_t y_pos) {
  const auto mouse_position =
      ScreenToComplex(x_pos, y_pos, window_width_, window_height_,
                      explorer_.GetCenterPosition(), explorer_.GetZoom());
  explorer_.MouseMovedEvent(mouse_position);
}

void Application::ScrollCallback(double_t x_offset, double_t y_offset) {
  const auto mouse_position = GetCurrentCursorComplex(
    window_width_, window_height_, explorer_.GetCenterPosition(),
    explorer_.GetZoom());
  if (y_offset > 0.) {
    explorer_.MouseScrollEvent(mouse_position, Explorer::ScrollEvent::kScrollUp);
  }
  if (y_offset < 0.) {
    explorer_.MouseScrollEvent(mouse_position, Explorer::ScrollEvent::kScrollDown);
  }
}

Application::Application(uint32_t window_width, uint32_t window_height,
                         std::unique_ptr<MandelbrotRenderer> renderer)
    : window_width_{window_width},
      window_height_{window_height},
      explorer_{kDefaultPosition, kDefaultZoom},
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

    logger.ResetCursor();
    logger <<
      logger.SetPrecision(15) <<
      "Center: " << position.real << logger.ShowSign(true) << position.imag << logger.ShowSign(false) << "i" << logger.NewLine() <<
      "Zoom: " << zoom << logger.NewLine() <<
      "FPS: " << static_cast<int32_t>(fps_counter.GetFPS()) << logger.NewLine();
  }

  return 0;
}

}  // namespace MandelbrotSet