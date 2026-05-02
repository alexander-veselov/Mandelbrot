#include "mandelbrot/application/application.h"

#include "mandelbrot/application/config.h"
#include "mandelbrot/application/fps_counter.h"
#include "mandelbrot/application/logger.h"
#include "mandelbrot/application/screenshot_renderer.h"

namespace mandelbrot {

static Complex ScreenToComplex(
  const Point& p,
  const Size& size,
  const Complex& center,
  double zoom) {

  constexpr double kWidth = 3.0;
  constexpr double kHeight = 2.0;

  const Point::value_type scale =
    Point::value_type{ std::min(kWidth / size.width, kHeight / size.height) };

  const Point::value_type x = (p.x - Point::value_type{ (double)size.width } * Point::value_type{ 0.5 }) * scale;
  const Point::value_type y = (p.y - Point::value_type{ (double)size.height } * Point::value_type{ 0.5 }) * scale;

  return {
    center.real + x / Point::value_type{zoom},
    center.imag + y / Point::value_type{zoom}
  };
}

Complex Application::GetCurrentCursorComplex(const Size& screen_size,
                                             const Complex& center,
                                             double_t zoom_factor) {
  const auto& cursor_position = GetCursorPosition();
  return ScreenToComplex(cursor_position, screen_size, center, zoom_factor);
}

void Application::MouseButtonCallback(MouseButton button, MouseAction action) {
  if (button == MouseButton::kLeft) {
    const auto cursor = GetCursorPosition();
    if (action == MouseAction::kPress) {
      explorer_.MouseClickedEvent(cursor);
    }
    else if (action == MouseAction::kRelease) {
      explorer_.MouseReleasedEvent(cursor);
    }
  }
}

void Application::CursorPositionCallback(const Point& cursor_position) {
  explorer_.MouseMovedEvent(cursor_position);
}

void Application::ScrollCallback(ScrollAction action) {
  const auto cursor = GetCursorPosition();
  explorer_.MouseScrollEvent(cursor, action);
}

void Application::KeyCallback(KeyButton key_button, KeyAction action) {
  if (action == KeyAction::kRelease) {
    return;
  }

  constexpr auto kIterationsStep = 128u;
  if (key_button == KeyButton::kPrintScreen) {
    screenshot_renderer_->Render(explorer_.GetReferenceCenter(), explorer_.GetOffset(),
                                 explorer_.GetZoom(), render_options_);
  } else if (key_button == KeyButton::kLeft) {
    const auto& bookmark = bookmarks_.Previous();
    explorer_.Navigate(bookmark.position, bookmark.zoom);
  } else if (key_button == KeyButton::kRight) {
    const auto& bookmark = bookmarks_.Next();
    explorer_.Navigate(bookmark.position, bookmark.zoom);
  } else if (key_button == KeyButton::kUp) {
    const auto& bookmark = bookmarks_.Current();
    explorer_.Navigate(bookmark.position, bookmark.zoom);
  } else if (key_button == KeyButton::kDown) {
    bookmarks_.Add(explorer_.GetReferenceCenter(), explorer_.GetZoom());
  } else if (key_button == KeyButton::kComma) {
    auto& max_iterations = render_options_.max_iterations; 
    max_iterations = std::max(kIterationsStep, max_iterations - kIterationsStep);
  } else if (key_button == KeyButton::kPeriod) {
    render_options_.max_iterations += kIterationsStep;
  } else if (key_button == KeyButton::kP) {
    explorer_.Chop();
  } else if (key_button == KeyButton::kM) {
    const auto mode = static_cast<uint32_t>(render_options_.coloring_mode);
    const auto modes = static_cast<uint32_t>(ColoringMode::kCount);
    const auto new_mode = (mode + 1) % modes;
    render_options_.coloring_mode = static_cast<ColoringMode>(new_mode);
  } else if (key_button == KeyButton::kEscape) {
    Close();
  }
}

static void LogInformation(const Logger& logger, const Complex& ref, const Complex& dc,
                           double_t zoom, const RenderOptions& render_options,
                           double_t fps) {
  logger.ResetCursor();
  logger << logger.SetPrecision(15) << "Ref: " << (double)ref.real
         << logger.ShowSign(true) << (double)ref.imag << logger.ShowSign(false)
         << "dc: " << (double)dc.real
         << logger.ShowSign(true) << (double)dc.imag << logger.ShowSign(false)
         << "i" << logger.NewLine() << "Zoom: " << zoom << logger.NewLine()
         << "Max iterations: " << render_options.max_iterations << logger.NewLine()
         << "Coloring mode: " << static_cast<uint32_t>(render_options.coloring_mode) << logger.NewLine()
         << "Palette: " << static_cast<uint32_t>(render_options.palette) << logger.NewLine()
         << "FPS: " << static_cast<int32_t>(fps) << logger.NewLine();
}

Application::Application(const Size& window_size,
                         std::unique_ptr<MandelbrotRenderer> renderer)
    : window_size_{window_size},
      explorer_{GetConfig().default_position, GetConfig().default_zoom},
      bookmarks_{},
      screenshot_renderer_{
          std::make_unique<ScreenshotRenderer>(GetConfig().screenshot_size)},
      renderer_{std::move(renderer)},
      render_options_{
          RenderOptions{GetConfig().coloring_mode, GetConfig().palette,
                        GetConfig().max_iterations, GetConfig().smoothing}} {}

int Application::Run() {
  const auto& logger = Logger::Instance();
  auto fps_counter = FPSCounter{GetConfig().fps_update_rate, GetTime()};

  while (!ShouldClose()) {
    try {
      const auto ref = explorer_.GetReferenceCenter();
      const auto dc = explorer_.GetOffset();
      const auto zoom = explorer_.GetZoom();

      renderer_->Render(ref, dc, zoom, render_options_);

      SwapBuffers();
      PollEvents();

      fps_counter.Update(GetTime());

      LogInformation(logger, ref, dc, zoom, render_options_,
                     fps_counter.GetFPS());
    } catch (const std::exception& e) {
      logger << e.what();
      return -1;
    }
  }

  return 0;
}

}  // namespace mandelbrot