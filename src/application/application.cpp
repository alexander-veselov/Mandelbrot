#include "mandelbrot/application/application.h"

#include "mandelbrot/application/config.h"
#include "mandelbrot/application/fps_counter.h"
#include "mandelbrot/application/logger.h"
#include "mandelbrot/application/screenshot_renderer.h"

namespace mandelbrot {

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
  }
}

void Application::CursorPositionCallback(const Point& cursor_position) {
  const auto mouse_position =
      ScreenToComplex(cursor_position, window_size_,
                      explorer_.GetCenterPosition(), explorer_.GetZoom());
  explorer_.MouseMovedEvent(mouse_position);
}

void Application::ScrollCallback(ScrollAction action) {
  const auto mouse_position = GetCurrentCursorComplex(
      window_size_, explorer_.GetCenterPosition(), explorer_.GetZoom());
  explorer_.MouseScrollEvent(mouse_position, action);
}

void Application::KeyCallback(KeyButton key_button, KeyAction action) {
  if (action == KeyAction::kRelease) {
    return;
  }

  constexpr auto kIterationsStep = 32u;
  if (key_button == KeyButton::kPrintScreen) {
    screenshot_renderer_->Render(explorer_.GetDisplayPosition(),
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
    bookmarks_.Add(explorer_.GetDisplayPosition(), explorer_.GetZoom());
  } else if (key_button == KeyButton::kComma) {
    auto& max_iterations = render_options_.max_iterations; 
    max_iterations = std::max(kIterationsStep, max_iterations - kIterationsStep);
  } else if (key_button == KeyButton::kPeriod) {
    render_options_.max_iterations += kIterationsStep;
  } else if (key_button == KeyButton::kP) {
    const auto palette = static_cast<uint32_t>(render_options_.palette);
    const auto palettes = static_cast<uint32_t>(Palette::kCount);
    const auto new_palette = (palette + 1) % palettes;
    render_options_.palette = static_cast<Palette>(new_palette);
  } else if (key_button == KeyButton::kM) {
    const auto mode = static_cast<uint32_t>(render_options_.coloring_mode);
    const auto modes = static_cast<uint32_t>(ColoringMode::kCount);
    const auto new_mode = (mode + 1) % modes;
    render_options_.coloring_mode = static_cast<ColoringMode>(new_mode);
  } else if (key_button == KeyButton::kEscape) {
    Close();
  }
}

static void LogInformation(const Logger& logger, const Complex& position,
                           double_t zoom, const RenderOptions& render_options,
                           double_t fps) {
  logger.ResetCursor();
  logger << logger.SetPrecision(15) << "Center: " << position.real
         << logger.ShowSign(true) << position.imag << logger.ShowSign(false)
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
      const auto position = explorer_.GetDisplayPosition();
      const auto zoom = explorer_.GetZoom();

      renderer_->Render(position, zoom, render_options_);

      SwapBuffers();
      PollEvents();

      fps_counter.Update(GetTime());

      LogInformation(logger, position, zoom, render_options_,
                     fps_counter.GetFPS());
    } catch (const std::exception& e) {
      logger << e.what();
      return -1;
    }
  }

  return 0;
}

}  // namespace mandelbrot